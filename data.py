import numpy as np
import random as rd
from torch.utils.data import Dataset

class Basket_Data(Dataset):
    def __init__(self, path, args):
        super(Basket_Data, self).__init__()
        self.path = path
        print(path)
        train_file_u2b = path + '/train_u2b.txt'
        train_file_b2i = path + '/train_b2i.txt'
        test_file_b2i = path + '/test_b2i.txt'

        self.n_users, self.n_items, self.n_baskets = 0, 0, 0
        self.n_train_u2b, self.n_train_b2i, self.n_test_b2i = 0, 0, 0
        self.u2i_edge_index, self.b2i_edge_index= [], []
        self.neg_pools = {}

        self.exist_users = []
        self.exist_baskets = []
        self.u2b_dict = {}
        self.b2u_dict = {}
        self.u2i_dict = {}
        count = 0
        # load training files
        with open(train_file_u2b) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    bids = [int(i) for i in l[1:]]
                    
                    uid = int(l[0])
                    self.u2b_dict[uid] = bids
                    for bid in bids:
                        self.b2u_dict[bid] = uid
                    self.exist_users.append(uid)
                    self.exist_baskets.extend(bids)
                    self.n_baskets = max(self.n_baskets, max(bids))
                    self.n_users = max(self.n_users, uid)
                    self.n_train_u2b += len(bids)
        print(self.n_train_u2b)
        self.b2i_dict = {}
        with open(train_file_b2i) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    iids = [int(i) for i in l[1:]]
                    bid = int(l[0])
                    self.b2i_dict[bid] = iids
                    for iid in iids:
                        self.b2i_edge_index.append([bid, iid+self.n_baskets+1])
                    if len(iids)==0:
                        print(bid)
                    self.n_items = max(self.n_items, max(iids))
                    self.n_train_b2i += len(iids)
        self.b2i_edge_index = np.transpose(np.array(self.b2i_edge_index))
        # create u2i edge index
        for user, baskets in self.u2b_dict.items():
            self.u2i_dict[user]=[]
            for basket in baskets:
                items = self.b2i_dict[basket]
                for item in items:
                    self.u2i_edge_index.append([user, item+self.n_users+1])
                    self.u2i_dict[user].append(item)
        self.u2i_edge_index = np.transpose(np.array(self.u2i_edge_index))
        self.test_b2i_dict = {}
        with open(test_file_b2i) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        l = l.strip('\n').split(' ')
                        iids = [int(i) for i in l[1:]]
                        bid = int(l[0])
                        self.test_b2i_dict[bid] = iids
                    except Exception:
                        continue
                    
                    self.n_items = max(self.n_items, max(iids))
                    self.n_test_b2i += len(iids)
        self.n_items += 1
        self.n_users += 1
        self.n_baskets += 1

    def __getitem__(self, basket):
        # user
        user = self.b2u_dict[basket]
        # positive sampling
        items_in_b = self.b2i_dict[basket]
        n_pos_items = len(items_in_b)
        pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        pos_item = items_in_b[pos_id]
        # negative sampling
        neg_items = []
        while True:
            if len(neg_items) == 1: break
            neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
            if neg_id not in self.b2i_dict[basket] and (neg_id not in self.u2i_dict[user]):
                neg_items.append(neg_id)
        neg_item = neg_items[0]
        return basket, pos_item, neg_item, user
        
    def __len__(self):
        return self.n_baskets
