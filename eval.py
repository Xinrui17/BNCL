from utils import *
import multiprocessing
import heapq
from parser import parse
from data import Basket_Data
cores = multiprocessing.cpu_count() // 2

args = parse()
data_path = os.path.join("Data", args.dataset)
data = Basket_Data(data_path, args)
Ks = eval(args.Ks)

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data.b2i_dict[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data.test_b2i_dict[u]
    all_items = set(range(data.n_items))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, baskets_to_test, args, data, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    pool = multiprocessing.Pool(cores)
    BATCH_SIZE = args.batch_size
    ITEM_NUM = data.n_items
    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_baskets = baskets_to_test
    n_test_baskets = len(test_baskets)
    n_basket_batchs = n_test_baskets // u_batch_size + 1
    b2u_dict = data.b2u_dict
    count = 0

    for u_batch_id in range(n_basket_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        basket_batch = test_baskets[start: end]
        c_users_batch = [b2u_dict[b] for b in basket_batch]
        basket_batch = torch.tensor(basket_batch, dtype=torch.long).cuda()
        c_users_batch = torch.tensor(c_users_batch, dtype=torch.long).cuda()
        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(basket_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)
                item_batch = torch.tensor(item_batch, dtype=torch.long).cuda()

                i_rate_batch = model.forward_test(basket_batch, item_batch, c_users_batch)
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)
            item_batch = torch.tensor(item_batch, dtype=torch.long).cuda()
            rate_batch = model.forward_test(basket_batch, item_batch, c_users_batch)
        basket_batch_rating_bid = zip(rate_batch.detach().cpu().numpy(), basket_batch.detach().cpu().numpy())
        batch_result = pool.map(test_one_user, basket_batch_rating_bid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_baskets
            result['recall'] += re['recall']/n_test_baskets
            result['ndcg'] += re['ndcg']/n_test_baskets
            result['hit_ratio'] += re['hit_ratio']/n_test_baskets
            result['auc'] += re['auc']/n_test_baskets


    assert count == n_test_baskets
    pool.close()
    return result