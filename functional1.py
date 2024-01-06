import torch
from torch_geometric.utils import degree

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    sel_mask = torch.cat([sel_mask,sel_mask], dim=0)
    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index1,edge_index2,n_users, n_baskets, n_items):
    deg1 = degree(edge_index1[0])
    deg2 = degree(edge_index2[0])

    cons_deg = deg1[n_users:n_users+n_items] + deg2[n_baskets:n_baskets+n_items]
    
    deg1 = torch.cat((deg1[:n_users],cons_deg))
    deg2 = torch.cat((deg2[:n_baskets] , cons_deg))
    deg_col1_0 = deg1[edge_index1[0][:int(len(edge_index1[0])/2)]].to(torch.float32)
    deg_col1_1 = deg1[edge_index1[1][:int(len(edge_index1[1])/2)]].to(torch.float32)
    deg_col1 = (deg_col1_0 + deg_col1_1)/3
    s_col1 = torch.log(deg_col1)
    weights1 = (s_col1.max()-s_col1) / (s_col1.max() - s_col1.mean())
    deg_col2_0 = deg2[edge_index2[0][:int(len(edge_index2[0])/2)]].to(torch.float32)
    deg_col2_1 = deg2[edge_index2[1][:int(len(edge_index2[1])/2)]].to(torch.float32)
    deg_col2 = (deg_col2_0 + deg_col2_1)/3
    s_col2 = torch.log(deg_col2)
    weights2 = (s_col2.max()-s_col2) / (s_col2.max() - s_col2.mean())
   
    return weights1,weights2
