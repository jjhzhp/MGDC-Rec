# coding=utf-8

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from utils import haversine_distance
import numpy as np


class ContrastiveLearningModule(nn.Module):
    """
    Contrastive learning module for multi-view contrastive learning
    """
    def __init__(self, emb_dim, temperature=0.1, device='cuda'):
        super(ContrastiveLearningModule, self).__init__()
        self.emb_dim = emb_dim
        self.temperature = temperature
        self.device = device
        
        # projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # initialize projection head
        for layer in self.projection_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, embeddings):
        """
        Project embeddings
        """
        return self.projection_head(embeddings)
    
    def info_nce_loss(self, pos_emb, neg_emb, batch_size=None):
        """
        Compute InfoNCE loss
        """
        # normalize embeddings
        pos_emb = F.normalize(pos_emb, dim=1)
        neg_emb = F.normalize(neg_emb, dim=1)
        
        # compute similarity matrix
        logits = torch.mm(pos_emb, neg_emb.T) / self.temperature
        
        # diagonal elements are positive pairs
        labels = torch.arange(logits.size(0)).to(self.device)
        
        # compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FGM:
    """
    Fast Gradient Method for Adversarial Training
    Adds adversarial perturbations to embeddings during training
    """
    def __init__(self, model, epsilon=1.0, emb_names=['embedding']):
        self.model = model
        self.epsilon = epsilon
        self.emb_names = emb_names
        self.backup = {}
        
    def attack(self):
        """
        Add adversarial perturbation to embeddings
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(emb_name in name for emb_name in self.emb_names):
                # Check if gradient exists
                if param.grad is None:
                    continue
                    
                # Save original embedding
                self.backup[name] = param.data.clone()
                
                # Calculate perturbation
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # r_adv = epsilon * grad / ||grad||
                    r_adv = self.epsilon * param.grad / norm
                    # Add perturbation
                    param.data.add_(r_adv)
    
    def restore(self):
        """
        Restore original embeddings
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class NegativeSamplingLoss(nn.Module):
    """
    Negative Sampling Enhanced Loss
    Combines Focal Loss with explicit negative sampling for better ranking
    """
    def __init__(self, focal_alpha=0.5, focal_gamma=1.5, 
                 num_neg_samples=10, neg_weight=0.3,
                 hard_ratio=0.3, popular_ratio=0.2):
        super(NegativeSamplingLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.num_neg_samples = num_neg_samples
        self.neg_weight = neg_weight
        self.hard_ratio = hard_ratio
        self.popular_ratio = popular_ratio
        
        # Calculate number of each type of negatives
        self.num_hard = max(1, int(num_neg_samples * hard_ratio))
        self.num_popular = max(1, int(num_neg_samples * popular_ratio))
        self.num_random = num_neg_samples - self.num_hard - self.num_popular
        
        # POI popularity will be computed on-the-fly
        self.poi_popularity = None
        
    def sample_negatives(self, predictions, targets):
        """
        Sample negative items using mixed strategy
        """
        batch_size = predictions.size(0)
        num_pois = predictions.size(1)
        device = predictions.device
        
        neg_indices_list = []
        
        for i in range(batch_size):
            pos_idx = targets[i].item()
            scores = predictions[i]
            
            # Create mask excluding positive item
            mask = torch.ones(num_pois, dtype=torch.bool, device=device)
            mask[pos_idx] = False
            valid_indices = torch.arange(num_pois, device=device)[mask]
            
            neg_samples = []
            
            # 1. Hard negatives: high-scored but wrong items
            if self.num_hard > 0:
                masked_scores = scores.clone()
                masked_scores[pos_idx] = -float('inf')
                # Get top-K high scores as hard negatives
                if len(valid_indices) >= self.num_hard:
                    hard_neg = torch.topk(masked_scores, self.num_hard).indices
                    neg_samples.append(hard_neg)
                else:
                    # Not enough items, sample with replacement
                    hard_neg = valid_indices[torch.randint(0, len(valid_indices), (self.num_hard,), device=device)]
                    neg_samples.append(hard_neg)
            
            # 2. Popular negatives: sample from high-frequency items
            if self.num_popular > 0:
                # Use score as proxy for popularity if popularity not available
                if self.poi_popularity is None:
                    # Sample from top 30% scored items (excluding positive and already sampled)
                    masked_scores = scores.clone()
                    masked_scores[pos_idx] = -float('inf')
                    if len(neg_samples) > 0:
                        for idx in neg_samples[0]:
                            masked_scores[idx] = -float('inf')
                    
                    top_30_percent = max(1, int(num_pois * 0.3))
                    popular_candidates = torch.topk(masked_scores, min(top_30_percent, len(valid_indices))).indices
                    
                    if len(popular_candidates) >= self.num_popular:
                        perm = torch.randperm(len(popular_candidates), device=device)[:self.num_popular]
                        popular_neg = popular_candidates[perm]
                    else:
                        popular_neg = popular_candidates[torch.randint(0, len(popular_candidates), (self.num_popular,), device=device)]
                    neg_samples.append(popular_neg)
            
            # 3. Random negatives: uniform sampling
            if self.num_random > 0:
                # Exclude positive and already sampled
                remaining_mask = mask.clone()
                if len(neg_samples) > 0:
                    for sampled in neg_samples:
                        remaining_mask[sampled] = False
                
                remaining_indices = torch.arange(num_pois, device=device)[remaining_mask]
                
                if len(remaining_indices) >= self.num_random:
                    perm = torch.randperm(len(remaining_indices), device=device)[:self.num_random]
                    random_neg = remaining_indices[perm]
                else:
                    # Sample with replacement if not enough
                    random_neg = remaining_indices[torch.randint(0, len(remaining_indices), (self.num_random,), device=device)]
                neg_samples.append(random_neg)
            
            # Concatenate all negatives
            all_neg = torch.cat(neg_samples)
            neg_indices_list.append(all_neg)
        
        return torch.stack(neg_indices_list)
    
    def forward(self, predictions, targets):
        # 1. Focal loss for classification
        focal = self.focal_loss(predictions, targets)
        
        # 2. Negative sampling loss
        # Get positive scores
        pos_scores = predictions.gather(1, targets.view(-1, 1))  # [BS, 1]
        
        # Sample negatives
        neg_indices = self.sample_negatives(predictions, targets)  # [BS, num_neg]
        neg_scores = predictions.gather(1, neg_indices)  # [BS, num_neg]
        
        # Ranking loss: positive should score higher than negatives
        margin = 1.0
        diff = pos_scores - neg_scores  # [BS, num_neg]
        ranking_loss = torch.clamp(margin - diff, min=0).mean()
        
        # Combine losses
        total_loss = (1 - self.neg_weight) * focal + self.neg_weight * ranking_loss
        
        return total_loss, focal, ranking_loss


class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view hypergraph convolution layer
    """

    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()

        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        # 1. node -> hyperedge message passing
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]

        # 2. propagation: hyperedge -> node
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]
        propag_pois_embs = F.relu(propag_pois_embs)  # add non-linear activation
        propag_pois_embs = self.dropout(propag_pois_embs)  # add dropout

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """Directed hypergraph convolution layer"""

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()
        self.dropout = nn.Dropout(0.3)

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)
        msg_src = F.relu(msg_src)  # add non-linear activation
        msg_src = self.dropout(msg_src)  # add dropout

        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view hypergraph convolution network
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout
        # learnable attention weights
        self.layer_attention = nn.Parameter(torch.ones(num_layers + 1))

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)  # [L, d]
            # add residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        # use learnable attention weights for weighted sum
        attention_weights = F.softmax(self.layer_attention, dim=0)
        final_pois_embs = torch.sum(torch.stack(final_pois_embs) * attention_weights[:, None, None], dim=0)

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()
        # learnable attention weights
        self.layer_attention = nn.Parameter(torch.ones(num_layers + 1))

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # add residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        # use learnable attention weights for weighted sum
        attention_weights = F.softmax(self.layer_attention, dim=0)
        final_pois_embs = torch.sum(torch.stack(final_pois_embs) * attention_weights[:, None, None], dim=0)

        return final_pois_embs


class PFFN(nn.Module):
    """
    Point-wise feedforward network
    """
    def __init__(self, hid_size, dropout_rate):
        super(PFFN, self).__init__()
        self.conv1 = nn.Conv1d(hid_size, hid_size, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hid_size, hid_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class BiSeqGCN(MessagePassing):
    """
    Bidirectional sequence graph convolution network
    """
    def __init__(self, hid_dim, flow="source_to_target"):
        super(BiSeqGCN, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)
        
        # attention weights
        self.attention_weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.attention_weight.data)
        
        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)
        self.act = nn.LeakyReLU()

    def forward(self, embs, G_u):
        POI_embs, delta_dis_embs = embs
        sess_idx = G_u.x.squeeze() if G_u.x.dim() > 1 else G_u.x
        edge_index = G_u.edge_index
        edge_dist = G_u.edge_dist
        
        x = POI_embs[sess_idx]
        edge_l = delta_dis_embs[edge_dist]
        # create bidirectional edges
        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)
        
        H_u = self.propagate(all_edges, x=x, edge_l=edge_l, edge_size=edge_index.size(1))
        return H_u

    def message(self, x_j, x_i, edge_index_j, edge_index_i, edge_l, edge_size):
        # edge_l's length is edge_size, corresponding to original edges
        # x_i's first edge_size corresponds to original edges, the rest corresponds to reverse edges
        # compute attention coefficients
        # for original edges: use x_i + edge_l
        # for reverse edges: use x_i
        forward_attn_input = x_i[:edge_size] + edge_l
        backward_attn_input = x_i[edge_size:]
        
        forward_coeff = torch.matmul(forward_attn_input, self.attention_weight.t())
        backward_coeff = torch.matmul(backward_attn_input, self.attention_weight.t())
        
        src_attention = self.alpha_src(forward_coeff).squeeze(-1)
        dst_attention = self.alpha_dst(backward_coeff).squeeze(-1)
        
        # softmax on tot_attention
        tot_attention = torch.cat((src_attention, dst_attention), dim=0)
        attn_weight = softmax(tot_attention, edge_index_i)

        # attention weights on neighbor node features
        updated_rep = x_j * attn_weight.unsqueeze(-1)
        return updated_rep


class SeqGraphEncoder(nn.Module):
    """
    Sequence graph encoder
    """
    def __init__(self, hid_dim):
        super(SeqGraphEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = BiSeqGCN(hid_dim)

    def encode(self, embs, G_u):
        return self.encoder(embs, G_u)


def Seq_MASK(lengths, max_len=None):
    """identify the actual length and padding of user check-in trajectory sequence"""
    lengths_shape = lengths.shape
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len).lt(lengths.unsqueeze(1))).reshape(lengths_shape)


class SeqGraphRepNetwork(nn.Module):
    """
    Sequence graph representation network
    """
    def __init__(self, emb_dim, num_heads=4, dropout=0.2):
        super(SeqGraphRepNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.seq_Rep = SeqGraphEncoder(emb_dim)
        
        self.seq_layernorm = nn.LayerNorm(emb_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(emb_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=dropout
        )
        self.seq_PFFN = PFFN(emb_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, POI_embs, delta_dis_embs, G_u, use_dropout=False):
        # encode sequence graph
        seq_embs = self.seq_Rep.encode((POI_embs, delta_dis_embs), G_u)
        
        if use_dropout:
            seq_embs = self.dropout(seq_embs)

        seq_lengths = torch.bincount(G_u.batch)
        seq_embs = torch.split(seq_embs, seq_lengths.cpu().numpy().tolist())

        seq_embs_pad = pad_sequence(seq_embs, batch_first=True, padding_value=0)
        pad_mask = Seq_MASK(seq_lengths)
        
        # multi-head attention
        Q = self.seq_layernorm(seq_embs_pad)
        K = seq_embs_pad
        V = seq_embs_pad
        output, att_weights = self.seq_attn(Q, K, V, key_padding_mask=~pad_mask)

        output = output + Q
        output = self.seq_attn_layernorm(output)

        # PFNN
        output = self.seq_PFFN(output)
        output = [seq[:seq_len] for seq, seq_len in zip(output, seq_lengths)]

        S_u = torch.stack([seq.mean(dim=0) for seq in output], dim=0)
        return S_u


class DCHL(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        super(DCHL, self).__init__()

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
        self.pos_embeddings = nn.Embedding(1500, self.emb_dim, padding_idx=0)
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

        self.info_nce_weight = 0.33
        self.focal_loss_weight = 0.33
        self.neg_sampling_weight = 0.34

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
        
        contrastive_loss = self.compute_contrastive_loss(
            batch, fusion_batch_users_embs, fusion_pois_embs,
            norm_hg_batch_users_embs, norm_trans_batch_users_embs, 
            norm_seq_batch_users_embs
        )
        return prediction, contrastive_loss
