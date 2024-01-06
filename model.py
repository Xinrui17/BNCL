import torch
from torch import nn, Tensor
from torch import linalg as LA

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import MessagePassing

# model for lightgcn + dropedge augmenation on ui graph
class HGN_Basket(torch.nn.Module):
    def __init__(self, data, args):
        super(HGN_Basket, self).__init__()
        self.num_users, self.num_items, self.num_baskets= data.n_users, data.n_items, data.n_baskets
        self.n_layers = args.n_layers
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.reg = args.reg
        self.linear = not args.not_linear
        # edge_index
        self.u2i_edge_index = torch.tensor(data.u2i_edge_index).cuda()
        self.i2u_edge_index = torch.stack(
            [self.u2i_edge_index[1], self.u2i_edge_index[0]], dim=0).cuda()
        self.u2i_edge_index = torch.cat([self.u2i_edge_index, self.i2u_edge_index], dim=1)
        self.b2i_edge_index = torch.tensor(data.b2i_edge_index).cuda()
        self.i2b_edge_index = torch.stack(
            [self.b2i_edge_index[1], self.b2i_edge_index[0]], dim=0).cuda()
        self.b2i_edge_index = torch.cat([self.b2i_edge_index, self.i2b_edge_index], dim=1)
        # initial embedding
        self.ui_items_emb = nn.Embedding(self.num_items, self.embed_size)
        self.baskets_emb = nn.Embedding(self.num_baskets, self.embed_size)
        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embed_size) # e_u^0
        self.bi_items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embed_size) # e_i^0
        self.reset_parameters()
       
        self.lgcn_hyper = LightGCN_hyper(data, args, self.b2i_edge_index)
        self.lgcn = LightGCN(data,args, self.u2i_edge_index)
        self.projector_user = nn.Linear(args.embed_size, args.embed_size)
        self.projector_item = nn.Linear(args.embed_size, args.embed_size)
        self.projector_item1 = nn.Linear(args.embed_size, args.embed_size)
        self.projector_basket = nn.Linear(args.embed_size, args.embed_size)

        self.ssl_reg = args.ssl_reg
        self.cl_reg = args.cl_reg
        self.cl_t = args.cl_t
        self.cl_reg_cro = args.cl_reg_cro
        self.cl_t_cro = args.cl_t_cro
        self.ssl_t = args.ssl_t

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.users_emb.weight)
        torch.nn.init.xavier_uniform_(self.ui_items_emb.weight)
        torch.nn.init.xavier_uniform_(self.baskets_emb.weight)
        torch.nn.init.xavier_uniform_(self.bi_items_emb.weight)
    
    def forward(self, baskets, pos_items, neg_items, users, u2i_edge_index_aug, b2i_edge_index_aug):
        users_emb, ui_items_emb = self.users_emb.weight,self.ui_items_emb.weight
        baskets_emb, bi_items_emb = self.baskets_emb.weight,self.ui_items_emb.weight

        users_emb1, ui_items_emb1,users_emb2, ui_items_emb2 = self.lgcn(users_emb, ui_items_emb,u2i_edge_index_aug)
        baskets_emb1, bi_items_emb1,baskets_emb2, bi_items_emb2 = self.lgcn_hyper(baskets_emb, bi_items_emb, b2i_edge_index_aug)

        pos_item_embed = ui_items_emb1[pos_items] 
        neg_item_embed = ui_items_emb1[neg_items] 
        pos_item_embed1 = bi_items_emb1[pos_items] 
        neg_item_embed1 = bi_items_emb1[neg_items] 
        batch_user_embed = users_emb[users]
        batch_basket_embed = baskets_emb1[baskets]
        
        cross_loss = 0
        cross_loss = self.CL_loss(ui_items_emb2[pos_items],bi_items_emb2[pos_items] ,self.cl_t_cro,self.cl_reg_cro) 
        loss, reg_loss = self.BPR_loss(batch_user_embed, batch_basket_embed, pos_item_embed, neg_item_embed,pos_item_embed1, neg_item_embed1)
        ui_ssl_loss_item = self.CL_loss(ui_items_emb1[pos_items],ui_items_emb2[pos_items], self.cl_t,self.cl_reg)
        ssl_loss_user = self.CL_loss(users_emb1[users],users_emb2[users], self.cl_t,self.cl_reg)
        bi_ssl_loss_item = self.CL_loss(bi_items_emb1[pos_items],bi_items_emb2[pos_items], self.cl_t,self.cl_reg)
        ssl_loss_basket = self.CL_loss(baskets_emb1[baskets],baskets_emb2[baskets], self.cl_t,self.cl_reg)
        
        loss = loss  + self.reg*reg_loss + ui_ssl_loss_item +ssl_loss_user + bi_ssl_loss_item + ssl_loss_basket + cross_loss
        return loss
    
    def forward_test(self, baskets, pos_items, users, edge_index1=None,edge_index2=None):
        users_emb, ui_items_emb = self.users_emb.weight,self.ui_items_emb.weight
        baskets_emb, bi_items_emb = self.baskets_emb.weight,self.bi_items_emb.weight

        users_emb1, ui_items_emb1 = self.lgcn(users_emb, ui_items_emb)
        baskets_emb1, bi_items_emb1 = self.lgcn_hyper(baskets_emb, bi_items_emb)
        
        batch_user_embed = users_emb1[users]
        batch_basket_embed = baskets_emb1[baskets]
    
        pos_item_embed = ui_items_emb1[pos_items] 
        pos_item_embed1 = bi_items_emb1[pos_items] 
      
        # loss
        combined_user_embed =  batch_user_embed 
        batch_ratings = 0.9 * torch.matmul(combined_user_embed, pos_item_embed.T)+ 0.1 * torch.matmul(batch_basket_embed, pos_item_embed1.T)
        return batch_ratings


    def BPR_loss(self, users, baskets=None, pos_items=None, neg_items=None, pos_items1=None, neg_items1=None):
        if baskets==None:
            combined_user = users
            pos_scores = torch.sum(combined_user*pos_items, dim=1)
            neg_scores = torch.sum(combined_user*neg_items, dim=1)
        elif pos_items1 == None:
            combined_user = users + baskets
            pos_scores = torch.sum(combined_user*pos_items, dim=1)
            neg_scores = torch.sum(combined_user*neg_items, dim=1)
        else:
            pos_item_embed = pos_items + pos_items1
            neg_item_embed = neg_items + neg_items1
            pos_scores = torch.sum(users*pos_item_embed, dim=1) #+ torch.sum(baskets*pos_item_embed, dim=1)
            neg_scores = torch.sum(users*neg_item_embed, dim=1) #+ torch.sum(baskets*neg_item_embed, dim=1)
        regularizer = LA.matrix_norm(self.baskets_emb.weight)+LA.matrix_norm(self.users_emb.weight)+LA.matrix_norm(self.ui_items_emb.weight)
        regularizer = regularizer/pos_scores.shape[0]
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores-neg_scores)))
        return bpr_loss, regularizer
    
    def CL_loss(self, embedding1, embedding2, ssl_temp = 0.1, ssl_reg=0.1):
        embedding1 = nn.functional.normalize(embedding1)
        embedding2 = nn.functional.normalize(embedding2)
        embedding_neg = embedding2
        pos_score = torch.sum(embedding1*embedding2, dim=1)
        ttl_score = torch.matmul(embedding1, embedding_neg.T)
        pos_score = torch.exp(pos_score / ssl_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / ssl_temp), dim=1)
        ssl_loss = -torch.mean(torch.log(pos_score / (ttl_score-pos_score)))
        return ssl_reg * ssl_loss




class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, data, args,edge_index):
        """
        Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items, self.num_baskets = data.n_users, data.n_items, data.n_baskets
        self.embedding_dim, self.K = args.embed_size, 2
        self.add_self_loops = args.self_loop
        self.reg = args.reg

        self.u2i_edge_index = edge_index
        self.u2i_edge_index = SparseTensor(row=self.u2i_edge_index[0], col=self.u2i_edge_index[1], sparse_sizes=(
    self.num_users + self.num_items, self.num_users + self.num_items)).cuda()
        

        
        
    def forward(self,users_emb,items_emb,edge_index_aug=None):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        
        if edge_index_aug != None:
            edge_index_norm1 = gcn_norm(
                self.u2i_edge_index, add_self_loops=self.add_self_loops)

            edge_index_norm2 = gcn_norm(
                edge_index_aug, add_self_loops=self.add_self_loops)

            emb_0 = torch.cat([users_emb, items_emb]) # E^0
            embs1 = [emb_0]
            embs2 = [emb_0]
            emb_k1 = emb_0
            emb_k2 = emb_0

            # multi-scale diffusion
            for i in range(self.K):
                emb_k1 = self.propagate(edge_index_norm1, x=emb_k1)
                embs1.append(emb_k1)

            for i in range(self.K):
                emb_k2 = self.propagate(edge_index_norm2, x=emb_k2)
                embs2.append(emb_k2)
            embs1 = torch.stack(embs1, dim=1)
            emb_final1 = torch.mean(embs1, dim=1) # E^K

            users_emb_final1, items_emb_final1 = torch.split(
                emb_final1, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

            embs2 = torch.stack(embs2, dim=1)
            emb_final2 = torch.mean(embs2, dim=1) # E^K

            users_emb_final2, items_emb_final2 = torch.split(
                emb_final2, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K
            
            return users_emb_final1, items_emb_final1,users_emb_final2, items_emb_final2

        else:
            edge_index_norm = gcn_norm(
            self.u2i_edge_index, add_self_loops=self.add_self_loops)
            emb_0 = torch.cat([users_emb, items_emb]) # E^0
            embs = [emb_0]
            emb_k = emb_0

            # multi-scale diffusion
            for i in range(self.K):
                emb_k = self.propagate(edge_index_norm, x=emb_k)
                embs.append(emb_k)
            embs = torch.stack(embs, dim=1)
            emb_final = torch.mean(embs, dim=1) # E^K
            users_emb_final, items_emb_final = torch.split(
                emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K
            return users_emb_final, items_emb_final

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)

class LightGCN_hyper(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, data, args,edge_index):
        """
        Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items, self.num_baskets = data.n_users, data.n_items, data.n_baskets
        self.embedding_dim, self.K = args.embed_size, 3
        self.add_self_loops = args.self_loop
        self.reg = args.reg

        self.b2i_edge_index = edge_index
        self.b2i_edge_index = SparseTensor(row=self.b2i_edge_index[0], col=self.b2i_edge_index[1], sparse_sizes=(
    self.num_baskets + self.num_items, self.num_baskets + self.num_items)).cuda()
        

        
        
    def forward(self,users_emb,items_emb,edge_index_aug=None):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        
        if edge_index_aug != None:
            edge_index_norm1 = gcn_norm(
                self.b2i_edge_index, add_self_loops=self.add_self_loops)

            edge_index_norm2 = gcn_norm(
                edge_index_aug, add_self_loops=self.add_self_loops)

            emb_0 = torch.cat([users_emb, items_emb]) # E^0
            embs1 = [emb_0]
            embs2 = [emb_0]
            emb_k1 = emb_0
            emb_k2 = emb_0

            # multi-scale diffusion
            for i in range(self.K):
                emb_k1 = self.propagate(edge_index_norm1, x=emb_k1)
                embs1.append(emb_k1)

            for i in range(self.K):
                emb_k2 = self.propagate(edge_index_norm2, x=emb_k2)
                embs2.append(emb_k2)

            embs1 = torch.stack(embs1, dim=1)
            emb_final1 = torch.mean(embs1, dim=1) # E^K

            users_emb_final1, items_emb_final1 = torch.split(
                emb_final1, [self.num_baskets, self.num_items]) # splits into e_u^K and e_i^K

            embs2 = torch.stack(embs2, dim=1)
            emb_final2 = torch.mean(embs2, dim=1) # E^K

            users_emb_final2, items_emb_final2 = torch.split(
                emb_final2, [self.num_baskets, self.num_items]) # splits into e_u^K and e_i^K
            
            return users_emb_final1, items_emb_final1,users_emb_final2, items_emb_final2

        else:
            edge_index_norm = gcn_norm(
            self.b2i_edge_index, add_self_loops=self.add_self_loops)

            emb_0 = torch.cat([users_emb, items_emb]) # E^0
            embs = [emb_0]
            emb_k = emb_0

            # multi-scale diffusion
            for i in range(self.K):
                emb_k = self.propagate(edge_index_norm, x=emb_k)
                embs.append(emb_k)

            embs = torch.stack(embs, dim=1)
            emb_final = torch.mean(embs, dim=1) # E^K

            users_emb_final, items_emb_final = torch.split(
                emb_final, [self.num_baskets, self.num_items]) # splits into e_u^K and e_i^K
            
            return users_emb_final, items_emb_final

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)