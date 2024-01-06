from utils import *
from torch.utils.data import DataLoader
from model import *
from eval import *
from parser import parse
import logging
from torch_geometric.utils import dropout_adj
from functional1 import *

n_users = data.n_users
n_items = data.n_items
n_baskets = data.n_baskets
args = parse()
def drop_edge(edge_index,p=0.3):
    edge_index = dropout_adj(edge_index=edge_index,p=0.3)[0]
    return edge_index

def drop_edge2(edge_index1,edge_index2, aug_type='random' ,p=0.3):
        if aug_type == 'degree':
            drop_weights1,drop_weights2 = degree_drop_weights(edge_index1,edge_index2, n_users, n_baskets, n_items)
            drop_weights1 = drop_weights1.cuda()
            drop_weights2 = drop_weights2.cuda()

        if aug_type == 'random':
            u2i_edge_index_aug = dropout_adj(edge_index=edge_index1,p=p)[0]
            b2i_edge_index_aug = dropout_adj(edge_index=edge_index2,p=p)[0]
            return u2i_edge_index_aug,b2i_edge_index_aug
        elif aug_type in ['degree']:
            u2i_edge_index_aug = drop_edge_weighted(edge_index1, drop_weights1, p=p, threshold=1)
            b2i_edge_index_aug = drop_edge_weighted(edge_index2, drop_weights2, p=p, threshold=1)
            return u2i_edge_index_aug,b2i_edge_index_aug
        else:
            raise Exception(f'undefined drop scheme: args.drop_type')



if __name__ == '__main__':
    pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
    
    filename = "log_VS2/{}.log".format(args.name)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(filename=filename)
    logging.info(str(args))
    fix_seed(args.seed)
    num_users = data.n_users
    num_items = data.n_items
    num_baskets = data.n_baskets
    train_loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers)
    model = HGN_Basket(data, args).cuda()
    
    b2i_edge_index_ori = torch.tensor(data.b2i_edge_index).cuda()
    u2i_edge_index_ori = torch.tensor(data.u2i_edge_index).cuda()

    i2u_edge_index_ori = torch.stack(
        [u2i_edge_index_ori[1], u2i_edge_index_ori[0]], dim=0).cuda()
    u2i_edge_index_ori = torch.cat([u2i_edge_index_ori, i2u_edge_index_ori], dim=1)
    i2b_edge_index_ori = torch.stack(
        [b2i_edge_index_ori[1], b2i_edge_index_ori[0]], dim=0).cuda()
    b2i_edge_index_ori = torch.cat([b2i_edge_index_ori, i2b_edge_index_ori], dim=1)
    optim = torch.optim.Adam(model.parameters(), args.lr)
    for epoch in range(args.epoch):
        # train
        if args.aug_type in ['degree','random']:
            u2i_edge_index_aug, b2i_edge_index_aug = drop_edge2(edge_index1=u2i_edge_index_ori,edge_index2=b2i_edge_index_ori, aug_type=args.aug_type ,p=args.aug_ratio)
            b2i_edge_index_aug = SparseTensor(row=b2i_edge_index_aug[0], col=b2i_edge_index_aug[1], sparse_sizes=(num_baskets + num_items, num_baskets + num_items)).cuda()
            u2i_edge_index_aug = SparseTensor(row=u2i_edge_index_aug[0], col=u2i_edge_index_aug[1], sparse_sizes=(
            num_users + num_items, num_users + num_items)).cuda()
        else:
            b2i_edge_index_aug = b2i_edge_index_ori
            u2i_edge_index_aug = u2i_edge_index_ori


        model.train()
        for data_batch in train_loader:
            baskets, pos_items, neg_items, users = data_batch
            baskets, pos_items, neg_items, users = baskets.cuda(), pos_items.cuda(), neg_items.cuda(), users.cuda()
            loss = model(baskets, pos_items, neg_items, users, u2i_edge_index_aug, b2i_edge_index_aug)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print("epoch {}: {}".format(epoch, loss.item()))
        logging.info("epoch {}: {}".format(epoch, loss.item()))
        best_ndcg = []
        
        if epoch%args.interval==0:
            # test
            model.eval()
            baskets_to_test = list(data.test_b2i_dict.keys())
            ret = test(model, baskets_to_test, args, data, batch_test_flag=False)
            perf_str = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            print(perf_str)
            logging.info(perf_str)
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    idx = np.argmax(recs[:, -1])
    final_perf = "Best Iter=[%d]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx*args.interval, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    logging.info(perf_str)




