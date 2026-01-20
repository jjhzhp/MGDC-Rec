# coding=utf-8

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from utils import haversine_distance
from config import DEFAULT_MODEL_CONFIG, TEMPORAL_CONFIG

# Import components from model_components.py
from model_components import (
    ContrastiveLearningModule,
    FocalLoss,
    NegativeSamplingLoss,
    MultiViewHyperConvNetwork,
    DirectedHyperConvNetwork,
    SeqGraphRepNetwork
)

class MGDC(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        super(MGDC, self).__init__()

        # definition
        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim

        # embedding
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)
        
        # add distance interval embedding
        self.interval = getattr(args, 'interval', 100)
        self.delta_dis_embs = nn.Parameter(torch.empty(self.interval, self.emb_dim))
        nn.init.xavier_normal_(self.delta_dis_embs)

        # embedding initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # network
        self.mv_hconv_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim, 0, device)
        self.di_hconv_network = DirectedHyperConvNetwork(args.num_di_layers, device, args.dropout)
        # add sequence graph representation network
        num_seq_heads = getattr(args, 'num_seq_heads', 4)
        self.seq_graph_network = SeqGraphRepNetwork(args.emb_dim, num_heads=num_seq_heads, dropout=args.dropout)

        # gate for adaptive fusion with POI embeddings
        self.hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.seq_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # gate for adaptive fusion with user embeddings
        self.user_hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_seq_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # temporal-augmentation
        max_pos_emb = getattr(args, 'max_position_embeddings', TEMPORAL_CONFIG['max_position_embeddings'])
        self.pos_embeddings = nn.Embedding(max_pos_emb, self.emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # gating before disentangled learning
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # dropout
        self.dropout = nn.Dropout(args.dropout)

        # contrastive learning module
        contrastive_temp = getattr(args, 'contrastive_temperature', 0.1)
        self.contrastive_module = ContrastiveLearningModule(self.emb_dim, temperature=contrastive_temp, device=device)
        
        # contrastive learning loss functions
        focal_alpha = getattr(args, 'focal_alpha', 0.5)
        focal_gamma = getattr(args, 'focal_gamma', 1.5)
        num_neg_samples = getattr(args, 'num_neg_samples', 8)
        neg_weight = getattr(args, 'neg_weight', 0.2)
        hard_ratio = getattr(args, 'hard_ratio', 0.25)
        popular_ratio = getattr(args, 'popular_ratio', 0.25)
        
        self.contrastive_focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.contrastive_neg_sampling_loss = NegativeSamplingLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            num_neg_samples=num_neg_samples,
            neg_weight=neg_weight,
            hard_ratio=hard_ratio,
            popular_ratio=popular_ratio
        )
        
        # contrastive learning weight
        self.contrastive_weight = getattr(args, 'contrastive_weight', 0.1)

        # Multi-loss mixing weights (should sum to ~1.0)
        self.info_nce_weight = getattr(args, 'info_nce_weight', 0.33)
        self.focal_loss_weight = getattr(args, 'focal_loss_weight', 0.33)
        self.neg_sampling_weight = getattr(args, 'neg_sampling_weight', 0.34)

    def _build_user_trajectory_graph(self, batch, pois_coos_dict=None):
        """
        build user trajectory graph G_u
        """
        
        user_seqs = batch["user_seq"]  # [BS, MAX_SEQ_LEN]
        user_seq_lens = batch["user_seq_len"]  # [BS]
        batch_size = user_seqs.size(0)
        
        # collect all nodes and edges
        all_node_ids = []
        all_edge_index = []
        all_edge_dist = []
        all_batch = []
        node_offset = 0
        
        for i in range(batch_size):
            seq_len = user_seq_lens[i].item()
            seq = user_seqs[i][:seq_len]  # actual sequence
            
            if seq_len < 2:  # sequence too short, cannot build graph
                # at least add one node
                all_node_ids.append(seq[0].item())
                all_batch.append(i)
                continue
            
            # add nodes
            for j in range(seq_len):
                all_node_ids.append(seq[j].item())
                all_batch.append(i)
            
            # add edges
            for j in range(seq_len - 1):
                src_idx = node_offset + j
                dst_idx = node_offset + j + 1
                all_edge_index.append([src_idx, dst_idx])
                
                # compute distance interval
                if pois_coos_dict is not None:
                    poi_src = seq[j].item()
                    poi_dst = seq[j + 1].item()
                    
                    # check if POI coordinates exist
                    if poi_src not in pois_coos_dict or poi_dst not in pois_coos_dict:
                        # if coordinates do not exist, use default index difference
                        dist_idx = min(j, self.interval - 1)
                    else:
                        lat1, lon1 = pois_coos_dict[poi_src]
                        lat2, lon2 = pois_coos_dict[poi_dst]
                        dist = haversine_distance(lon1, lat1, lon2, lat2)
                        # discretize distance to interval range
                        dist_idx = min(int(dist * 10) % self.interval, self.interval - 1)
                    all_edge_dist.append(dist_idx)
                else:
                    # use simple index difference
                    all_edge_dist.append(min(j, self.interval - 1))
            
            node_offset += seq_len
        
        if len(all_edge_index) == 0:
            # if no edges, create an empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_dist = torch.empty((0,), dtype=torch.long, device=self.device)
        else:
            edge_index = torch.tensor(all_edge_index, dtype=torch.long, device=self.device).t()
            edge_dist = torch.tensor(all_edge_dist, dtype=torch.long, device=self.device)
        
        x = torch.tensor(all_node_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
        batch_tensor = torch.tensor(all_batch, dtype=torch.long, device=self.device)
        
        G_u = Data(x=x, edge_index=edge_index, edge_dist=edge_dist, batch=batch_tensor)
        return G_u

    def compute_contrastive_loss(self, batch, fusion_batch_users_embs, fusion_pois_embs, 
                                norm_hg_batch_users_embs, norm_trans_batch_users_embs, 
                                norm_seq_batch_users_embs):
        """
        compute contrastive learning loss
        mixed use of InfoNCE, FocalLoss and NegativeSamplingLoss
        each view pair uses three loss methods
        """
        batch_size = fusion_batch_users_embs.size(0)
        positive_labels = batch["label"].to(self.device)  # [BS]
        
        # view embeddings dictionary
        view_embeddings = {
            'hypergraph': norm_hg_batch_users_embs,
            'transition': norm_trans_batch_users_embs,
            'sequence': norm_seq_batch_users_embs
        }
        
        view_names = list(view_embeddings.keys())
        
        # ========== 1. multi-view contrastive learning loss ==========
        multi_view_info_nce_loss = 0.0
        multi_view_focal_loss = 0.0
        multi_view_neg_sampling_loss = 0.0
        
        for i in range(len(view_names)):
            for j in range(i + 1, len(view_names)):
                view1_name = view_names[i]
                view2_name = view_names[j]
                view1_emb = view_embeddings[view1_name]
                view2_emb = view_embeddings[view2_name]
                
                # 1.1 InfoNCE loss
                view1_proj = self.contrastive_module(view1_emb)
                view2_proj = self.contrastive_module(view2_emb)
                info_nce_loss_1to2 = self.contrastive_module.info_nce_loss(view1_proj, view2_proj, batch_size)
                info_nce_loss_2to1 = self.contrastive_module.info_nce_loss(view2_proj, view1_proj, batch_size)
                multi_view_info_nce_loss += (info_nce_loss_1to2 + info_nce_loss_2to1) / 2
                
                # 1.2 FocalLoss loss
                similarity_logits = torch.mm(view1_emb, view2_emb.T) / self.contrastive_module.temperature
                labels = torch.arange(similarity_logits.size(0)).to(self.device)
                focal_loss = self.contrastive_focal_loss(similarity_logits, labels)
                multi_view_focal_loss += focal_loss
                
                # 1.3 NegativeSamplingLoss
                # build similarity matrix [BS, BS]
                view1_to_view2_logits = torch.mm(view1_emb, view2_emb.T) / self.contrastive_module.temperature
                # diagonal elements are positive pairs, others are negative pairs
                view_labels = torch.arange(batch_size).to(self.device)
                neg_sampling_loss, _, _ = self.contrastive_neg_sampling_loss(view1_to_view2_logits, view_labels)
                multi_view_neg_sampling_loss += neg_sampling_loss
        
        # ========== 2. user-POI contrastive learning loss ==========
        user_poi_logits = torch.mm(fusion_batch_users_embs, fusion_pois_embs.T) / self.contrastive_module.temperature
        
        # 2.1 InfoNCE loss
        user_proj = self.contrastive_module(fusion_batch_users_embs)
        poi_proj = self.contrastive_module(fusion_pois_embs)
        # use projected embeddings to compute similarity matrix
        user_poi_proj_logits = torch.mm(user_proj, poi_proj.T) / self.contrastive_module.temperature
        # cross entropy loss
        user_poi_info_nce_loss = F.cross_entropy(user_poi_proj_logits, positive_labels)
        
        # 2.2 FocalLoss loss
        user_poi_focal_loss = self.contrastive_focal_loss(user_poi_logits, positive_labels)
        
        # 2.3 NegativeSamplingLoss loss
        user_poi_neg_sampling_loss, _, _ = self.contrastive_neg_sampling_loss(user_poi_logits, positive_labels)
        
        # ========== 3. cross-view contrastive learning loss ==========
        cross_view_logits = torch.mm(fusion_batch_users_embs, fusion_pois_embs.T) / self.contrastive_module.temperature
        
        # 3.1 InfoNCE loss
        cross_view_proj_logits = torch.mm(user_proj, poi_proj.T) / self.contrastive_module.temperature
        cross_view_info_nce_loss = F.cross_entropy(cross_view_proj_logits, positive_labels)
        
        # 3.2 FocalLoss loss
        cross_view_focal_loss = self.contrastive_focal_loss(cross_view_logits, positive_labels)
        
        # 3.3 NegativeSamplingLoss loss
        cross_view_neg_sampling_loss, _, _ = self.contrastive_neg_sampling_loss(cross_view_logits, positive_labels)
        
        # ========== 4. mixed loss: weighted combination of three loss methods ==========
        # multi-view loss
        multi_view_loss = (
            self.info_nce_weight * multi_view_info_nce_loss +
            self.focal_loss_weight * multi_view_focal_loss +
            self.neg_sampling_weight * multi_view_neg_sampling_loss
        )
        
        # user-POI loss
        user_poi_loss = (
            self.info_nce_weight * user_poi_info_nce_loss +
            self.focal_loss_weight * user_poi_focal_loss +
            self.neg_sampling_weight * user_poi_neg_sampling_loss
        )
        
        # cross-view loss
        cross_view_loss = (
            self.info_nce_weight * cross_view_info_nce_loss +
            self.focal_loss_weight * cross_view_focal_loss +
            self.neg_sampling_weight * cross_view_neg_sampling_loss
        )
        
        # total contrastive learning loss
        total_contrastive_loss = multi_view_loss + user_poi_loss + cross_view_loss
        
        return total_contrastive_loss

    def forward(self, dataset, batch):
        """
        forward propagation, containing 3 views
        """
        # self-gating input
        seq_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_seq) + self.b_gate_seq))
        col_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))

        # multi-view hypergraph convolutional network
        hg_pois_embs = self.mv_hconv_network(col_gate_pois_embs, dataset.pad_all_train_sessions, dataset.HG_up, dataset.HG_pu)
        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_pois_embs)
        hg_batch_users_embs = hg_structural_users_embs[batch["user_idx"]]

        # poi-poi directed hypergraph (transition)
        trans_pois_embs = self.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)
        trans_structural_users_embs = torch.sparse.mm(dataset.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[batch["user_idx"]]

        # sequence graph representation
        pois_coos_dict = getattr(dataset, 'pois_coos_dict', None)
        G_u = self._build_user_trajectory_graph(batch, pois_coos_dict)
        
        if G_u.edge_index.size(1) > 0:
            seq_user_embs = self.seq_graph_network(
                self.poi_embedding.weight[:-1], 
                self.delta_dis_embs, 
                G_u,
                use_dropout=(self.training and hasattr(self.args, 'dropout') and self.args.dropout > 0)
            )
        else:
            batch_user_seqs = batch["user_seq"]
            batch_seq_lens = batch["user_seq_len"]
            seq_user_embs = []
            for i in range(batch_user_seqs.size(0)):
                seq_len = batch_seq_lens[i].item()
                seq = batch_user_seqs[i][:seq_len]
                if seq_len > 0:
                    seq_emb = self.poi_embedding(seq).mean(dim=0)
                else:
                    seq_emb = torch.zeros(self.emb_dim, device=self.device)
                seq_user_embs.append(seq_emb)
            seq_user_embs = torch.stack(seq_user_embs, dim=0)

        # Normalize embeddings
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)
        norm_seq_pois_embs = F.normalize(self.poi_embedding.weight[:-1], p=2, dim=1)
        norm_seq_batch_users_embs = F.normalize(seq_user_embs, p=2, dim=1)

        # Adaptive fusion for 3 views
        user_hyper_coef = self.user_hyper_gate(norm_hg_batch_users_embs)
        user_trans_coef = self.user_trans_gate(norm_trans_batch_users_embs)
        user_seq_coef = self.user_seq_gate(norm_seq_batch_users_embs)
        
        # normalize weights
        user_coef_sum = user_hyper_coef + user_trans_coef + user_seq_coef
        user_hyper_coef = user_hyper_coef / user_coef_sum
        user_trans_coef = user_trans_coef / user_coef_sum
        user_seq_coef = user_seq_coef / user_coef_sum
        
        fusion_batch_users_embs = (
            user_hyper_coef * norm_hg_batch_users_embs + 
            user_trans_coef * norm_trans_batch_users_embs + 
            user_seq_coef * norm_seq_batch_users_embs
        )
        
        # POI embeddings
        poi_hyper_coef = self.hyper_gate(norm_hg_pois_embs)
        poi_trans_coef = self.trans_gate(norm_trans_pois_embs)
        poi_seq_coef = self.seq_gate(norm_seq_pois_embs)
        
        # normalize weights
        poi_coef_sum = poi_hyper_coef + poi_trans_coef + poi_seq_coef
        poi_hyper_coef = poi_hyper_coef / poi_coef_sum
        poi_trans_coef = poi_trans_coef / poi_coef_sum
        poi_seq_coef = poi_seq_coef / poi_coef_sum
        
        fusion_pois_embs = (
            poi_hyper_coef * norm_hg_pois_embs + 
            poi_trans_coef * norm_trans_pois_embs + 
            poi_seq_coef * norm_seq_pois_embs
        )

        # Final prediction
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T
        
        # Auxiliary regularization (scaled to 0 for stability)
        aux_reg = (self.pos_embeddings.weight.sum() + self.w_1.weight.sum() + 
                   self.w_2.sum() + self.glu1.weight.sum() + self.glu2.weight.sum()) * 0.0
        prediction = prediction + aux_reg
        
        contrastive_loss = self.compute_contrastive_loss(
            batch, fusion_batch_users_embs, fusion_pois_embs,
            norm_hg_batch_users_embs, norm_trans_batch_users_embs, 
            norm_seq_batch_users_embs
        )
        return prediction, contrastive_loss
