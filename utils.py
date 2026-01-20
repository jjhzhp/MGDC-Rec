# coding=utf-8

import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
import torch


def load_list_with_pkl(filename):
    with open(filename, 'rb') as f:
        list_obj = pickle.load(f)

    return list_obj


def load_dict_from_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        dict_obj = pickle.load(f)

    return dict_obj


def get_user_complete_traj(sessions_dict):
    """Get each user's complete trajectory from her sessions"""
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for userID, sessions in sessions_dict.items():
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[userID] = traj
        users_trajs_lens_dict[userID] = len(traj)

    return users_trajs_dict, users_trajs_lens_dict


def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine distance"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r


def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    """Generate sparse incidence matrix for hypergraph"""
    H = np.zeros(shape=(num_pois, num_users))

    for userID, sessions in sessions_dict.items():
        seq = []
        for session in sessions:
            seq.extend(session)
        for poi in seq:
            H[poi, userID] = 1

    H = sp.csr_matrix(H)

    return H


def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    """
    Generate directed poi-poi incidence matrix for hypergraph
    Rows: source POIs
    Columns: target POIs
    """
    H = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] = 1
    H = sp.csr_matrix(H)

    return H


def get_hyper_deg(incidence_matrix):
    """
    Compute degree matrix for hypergraph
    Returns diagonal matrix with inverse of row sums
    """
    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.divide(1.0, rowsum, out=np.zeros_like(rowsum, dtype=float), where=rowsum!=0).flatten()
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv


def transform_csr_matrix_to_tensor(csr_matrix):
    """Transform csr matrix to tensor"""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sp_tensor


def get_all_users_seqs(users_trajs_dict):
    """Get all users' sequences"""
    all_seqs = []
    for userID, traj in users_trajs_dict.items():
        all_seqs.append(torch.tensor(traj))

    return all_seqs


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    """Drop edge on scipy.sparse.csr_matrix"""
    if keep_rate == 1.0:
        return csr_adj_matrix

    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]

    # generate edge mask
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(bool)

    # get new values and indices
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=np.float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix


def hit_k(y_pred, y_true, k):
    """Calculate Hit@K metric"""
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        return 1
    else:
        return 0


def ndcg_k(y_pred, y_true, k):
    """Calculate NDCG@K metric"""
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        position = y_pred_indices.index(y_true) + 1
        return 1 / np.log2(1 + position)
    else:
        return 0


def batch_performance(batch_y_pred, batch_y_true, k):
    """Calculate batch recall and NDCG"""
    batch_size = batch_y_pred.size(0)
    batch_recall = 0
    batch_ndcg = 0
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_recall += hit
        ndcg = ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_ndcg += ndcg

    recall = batch_recall / batch_size
    ndcg = batch_ndcg / batch_size

    return recall, ndcg


def smart_memory_management(threshold_ratio=0.85):
    """
    Smart GPU memory management - only clear cache when necessary
    
    Args:
        threshold_ratio: Memory usage ratio threshold (0-1)
                        If memory usage > threshold, clear cache
    
    Returns:
        bool: True if cache was cleared, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    
    if reserved > 0:
        usage_ratio = allocated / reserved
        if usage_ratio > threshold_ratio:
            torch.cuda.empty_cache()
            return True
    
    return False
