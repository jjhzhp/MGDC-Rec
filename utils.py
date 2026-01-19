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


def get_user_reverse_traj(users_trajs_dict):
    """Get each user's reversed trajectory according to her complete trajectory"""
    users_rev_trajs_dict = {}
    for userID, traj in users_trajs_dict.items():
        rev_traj = traj[::-1]
        users_rev_trajs_dict[userID] = rev_traj

    return users_rev_trajs_dict


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
    '''
    # incidence_matrix = [num_nodes, num_hyperedges]
    hyper_deg = np.array(incidence_matrix.sum(axis=axis)).squeeze()
    hyper_deg[hyper_deg == 0.] = 1
    hyper_deg = sp.diags(1.0 / hyper_deg)
    '''

    # H  = [num_node, num_edge]
    # DV = [num_node, num_node]
    # DV * H = [num_node, num_edge]

    # HT = [num_edge, num_node]
    # DE = [num_edge, num_edge]
    # DE * HT = [num_edge, num_node]

    # hyper_deg = incidence_matrix.sum(1)
    # inv_hyper_deg = hyper_deg.power(-1)
    # inv_hyper_deg_diag = sp.diags(inv_hyper_deg.toarray()[0])

    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
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
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)

    # get new values and indices
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=np.float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix
