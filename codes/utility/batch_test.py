import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import Data
import multiprocessing
import heapq
import torch
import pickle
import numpy as np
from time import time
from tqdm import tqdm

cores = multiprocessing.cpu_count() // 5

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = 16

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
    auc = metrics.auc(ground_truth=r, prediction=posterior)
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
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    is_val = x[-1]
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    if is_val:
        user_pos_test = data_generator.val_set[u]
    else:
        user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

def test_torch(user_ice, item_ice, user_mce, item_mce, img_query, txt_query, uv_agg, ut_agg, image_feats, text_feats, v_rel_mlp, t_rel_mlp, users_to_test, is_val, adj, alpha, beta, gamma):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
    pool = multiprocessing.Pool(cores)
    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0
    item_item = torch.mm(item_ice, item_ice.T)

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
 
        n_item_batchs = ITEM_NUM // i_batch_size + 1
        rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

        i_count = 0
        for i_batch_id in range(n_item_batchs):
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

            item_batch = range(i_start, i_end)
            batch_user_ice = user_ice[user_batch] # (batch_size, dim)
            batch_item_ice = item_ice[item_batch] # (batch_size, dim)
            batch_user_mce = user_mce[user_batch] # (batch_size, dim)
            batch_item_mce = item_mce[item_batch] # (batch_size, dim)

            user_batch_ref = [i for i in user_batch for _ in range(len(item_batch))]
            batch_uv_agg = uv_agg[user_batch_ref]
            batch_ut_agg = ut_agg[user_batch_ref]
            batch_img_query = img_query[user_batch_ref]
            batch_txt_query = txt_query[user_batch_ref]
            item_batch_ref = [elem for _ in range(len(user_batch)) for elem in item_batch]
            batch_image_feats = image_feats[item_batch_ref]
            batch_text_feats = text_feats[item_batch_ref]

            img_rel = torch.cat([batch_uv_agg, batch_image_feats], dim=1)
            img_rel_emb = torch.mm(img_rel,v_rel_mlp)

            txt_rel = torch.cat([batch_ut_agg, batch_text_feats], dim=1)
            txt_rel_emb = torch.mm(txt_rel, t_rel_mlp)

            batch_total_query = torch.cat([batch_img_query, batch_txt_query], dim=1) 
            batch_total_rel_emb = torch.cat([img_rel_emb, txt_rel_emb], dim=1) 
            total_pos_rel_score = torch.sum(torch.mul(batch_total_rel_emb, batch_total_query), dim=1)
            total_pos_rel_score_array = total_pos_rel_score.reshape(len(user_batch),len(item_batch))

            # target-aware 
            item_query = item_item[item_batch, :] # (item_batch_size, n_items)
            item_target_user_alpha = torch.softmax(torch.multiply(item_query.unsqueeze(1), adj[user_batch, :].unsqueeze(0)).masked_fill(adj[user_batch, :].repeat(len(item_batch), 1, 1) == 0, -1e9), dim=2) # (item_batch_size, user_batch_size, n_items)
            item_target_user = torch.matmul(item_target_user_alpha, item_ice) # (item_batch_size, user_batch_size, dim)
            
            # target-aware 
            i_rate_batch = (1 - gamma) * torch.matmul(batch_user_ice, torch.transpose(batch_item_ice, 0, 1)) + gamma * torch.sum(torch.mul(item_target_user.permute(1, 0, 2).contiguous(), batch_item_ice), dim=2) + alpha*torch.matmul(batch_user_mce, torch.transpose(batch_item_mce, 0, 1))
            rate_batch[:, i_start: i_end] = i_rate_batch.detach().cpu().numpy()+beta*total_pos_rel_score_array.detach().cpu().numpy()
            i_count += i_rate_batch.shape[1]

            del item_query, item_target_user_alpha, item_target_user, batch_user_ice, batch_item_ice, batch_user_mce, batch_item_mce, img_rel, txt_rel, img_rel_emb, txt_rel_emb, batch_img_query, batch_txt_query, batch_uv_agg, batch_ut_agg
            torch.cuda.empty_cache()

        assert i_count == ITEM_NUM

        user_batch_rating_uid = zip(rate_batch, user_batch, [is_val] * len(user_batch))
                            
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result