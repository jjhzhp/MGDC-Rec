# coding=utf-8

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.utils.rnn import pad_sequence
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

