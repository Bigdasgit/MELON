import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
from utility.parser import parse_args
from collections import defaultdict
import os
import random
import pandas as pd
from urllib3 import add_stderr_logger
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sim_ui(preference, context):
    preference_norm = F.normalize(preference, p=2, dim=1)
    context_norm = F.normalize(context.weight, p=2, dim=1)
    sim = torch.mm(preference_norm, context_norm.transpose(1, 0)) 
    return sim

def find_k_ui(sim, largest_k):
    _, top_ind = torch.topk(sim, largest_k, dim=-1)

    return top_ind

class Data(object):
    def __init__(self, path, batch_size):

        args = parse_args()
        self.path = path + '/5-core'
        self.batch_size = batch_size

        train_file = path + '/5-core/train.json'
        val_file = path + '/5-core/val.json'
        test_file = path + '/5-core/test.json'

        #get number of users and items
        self.n_users, self.n_items, self.largest_k = 0, 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.largest_k = max(self.largest_k, len(items))
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R[uid, i] = 1.

            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue  

        if args.ui_k == 'int':
            if os.path.exists(path + '/image_knn_int.txt'):
                image_knn_path = path + '/image_knn_int.txt'
                df_image = pd.read_csv(image_knn_path, names= ['user', 'item'], sep='\t')
            else:
                image_feats = np.load('data/{}/image_feat.npy'.format(args.dataset))
                image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
                _, image_dim = image_embedding.weight.size()
                image_agg = torch.zeros(self.n_users, image_dim)
                for uid, items in self.train_items.items():
                    temp_image_agg = image_embedding(torch.LongTensor(items))
                    temp_image_agg = torch.mean(temp_image_agg, dim=0)
                    image_agg[torch.tensor(uid)] = temp_image_agg
                image_sim = build_sim_ui(image_agg, image_embedding)
                image_top_k = find_k_ui(image_sim, self.largest_k)
                image_items_dict = defaultdict(list)
                for i, image in enumerate(image_top_k):
                    i = int(i)
                    image = image[:len(self.train_items[i])]
                    image_items_dict[i] = image

                image_user = []
                image_items = []

                for i, items in image_items_dict.items():
                    if len(items) > 0:         
                        uid = int(i)
                        items = [int(i) for i in items]
                        image_user.extend([uid] * len(items)) 
                        image_items.extend(items) 

                image_info = []

                for i in range(len(image_user)):
                    info = [image_user[i], image_items[i]]
                    image_info.append(info)

                df_image = pd.DataFrame(image_info, columns=['user', 'item'])
                df_image.to_csv(path_or_buf= path + '/image_knn_int.txt', index=False, header=None, sep='\t' )
                                                       
            if os.path.exists(path + '/text_knn_int.txt'):
                text_path = path + '/text_knn_int.txt'
                df_text = pd.read_csv(text_path, names= ['user', 'item'], sep='\t') 
            else:
                text_feats = np.load('data/{}/text_feat.npy'.format(args.dataset))
                text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)
                _, text_dim = text_embedding.weight.size()
                text_agg = torch.zeros(self.n_users, text_dim)
                for uid, items in self.train_items.items():
                    temp_text_agg = text_embedding(torch.LongTensor(items))
                    temp_text_agg = torch.mean(temp_text_agg, dim=0)
                    text_agg[torch.tensor(uid)] = temp_text_agg
                
                text_sim = build_sim_ui(text_agg, text_embedding)
                text_top_k = find_k_ui(text_sim, self.largest_k)
                text_items_dict = defaultdict(list)
                for i, text in enumerate(text_top_k):
                    i = int(i)
                    text = text[:len(self.train_items[i])]
                    text_items_dict[i] = text

                text_user = []
                text_items = []

                for i, items in text_items_dict.items():
                    if len(items) > 0:         
                        uid = int(i)
                        items = [int(i) for i in items]
                        text_user.extend([uid] * len(items)) 
                        text_items.extend(items) 

                text_info = []

                for i in range(len(text_user)):
                    info = [text_user[i], text_items[i]]
                    text_info.append(info)
                
                df_text = pd.DataFrame(text_info, columns=['user', 'item'])
                df_text.to_csv(path_or_buf= path + '/text_knn_int.txt', index=False, header=None, sep='\t' )

        else: 
            if os.path.exists(path + '/image_knn_'+args.ui_k+'.txt'):
                image_path = path + '/image_knn_'+args.ui_k+'.txt'
                df_image = pd.read_csv(image_path, names= ['user', 'item'], sep='\t')
            else:
                image_feats = np.load('data/{}/image_feat.npy'.format(args.dataset))
                image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
                _, image_dim = image_embedding.weight.size()
                image_agg = torch.zeros(self.n_users, image_dim)
                for uid, items in self.train_items.items():
                    temp_image_agg = image_embedding(torch.LongTensor(items))
                    temp_image_agg = torch.mean(temp_image_agg, dim=0)
                    image_agg[torch.tensor(uid)] = temp_image_agg
                image_sim = build_sim_ui(image_agg, image_embedding)
                image_top_k = find_k_ui(image_sim, self.largest_k)
                image_items_dict = defaultdict(list)
                for i, image in enumerate(image_top_k):
                    i = int(i)
                    image = image[:int(args.ui_k)]
                    image_items_dict[i] = image

                image_user = []
                image_items = []

                for i, items in image_items_dict.items():
                    if len(items) > 0:         
                        uid = int(i)
                        items = [int(i) for i in items]
                        image_user.extend([uid] * len(items)) 
                        image_items.extend(items) 

                image_info = []

                for i in range(len(image_user)):
                    info = [image_user[i], image_items[i]]
                    image_info.append(info)

                df_image = pd.DataFrame(image_info, columns=['user', 'item'])
                df_image.to_csv(path_or_buf= path + '/imag_knn_'+str(args.ui_k)+'.txt', index=False, header=None, sep='\t' )
                                                       
            if os.path.exists(path + '/text_knn_'+args.ui_k+'.txt'):
                text_path = path + '/text_knn_'+args.ui_k+'.txt'
                df_text = pd.read_csv(text_path, names= ['user', 'item'], sep='\t')
            else:
                text_feats = np.load('data/{}/text_feat.npy'.format(args.dataset))
                text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)
                _, text_dim = text_embedding.weight.size()
                text_agg = torch.zeros(self.n_users, text_dim)
                for uid, items in self.train_items.items():
                    temp_text_agg = text_embedding(torch.LongTensor(items))
                    temp_text_agg = torch.mean(temp_text_agg, dim=0)
                    text_agg[torch.tensor(uid)] = temp_text_agg
                
                text_sim = build_sim_ui(text_agg, text_embedding)
                text_top_k = find_k_ui(text_sim, self.largest_k)
                text_items_dict = defaultdict(list)
                for i, text in enumerate(text_top_k):
                    i = int(i)
                    text = text[:int(args.ui_k)]
                    text_items_dict[i] = text

                text_user = []
                text_items = []

                for i, items in text_items_dict.items():
                    if len(items) > 0:         
                        uid = int(i)
                        items = [int(i) for i in items]
                        text_user.extend([uid] * len(items)) 
                        text_items.extend(items) 

                text_info = []

                for i in range(len(text_user)):
                    info = [text_user[i], text_items[i]]
                    text_info.append(info)
                
                df_text = pd.DataFrame(text_info, columns=['user', 'item'])
                df_text.to_csv(path_or_buf= path + '/text_knn_'+str(args.ui_k)+'.txt', index=False, header=None, sep='\t' )

        image_knn = defaultdict(list)
        for index, row in df_image.iterrows():
            user, item = int(row['user']), int(row['item'])
            image_knn[user].append(item)

        text_knn = defaultdict(list)
        for index, row in df_text.iterrows():
            user, item = int(row['user']), int(row['item'])
            text_knn[user].append(item)

        self.R_image = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_text = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        for uid, k_items in image_knn.items():
            if len(k_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(k_items):
                self.R_image[uid, i] = 1.

        for uid, k_items in text_knn.items():
            if len(k_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(k_items):
                self.R_text[uid, i] = 1.
        
    def nonzero_idx(self):
        r, c = self.R.nonzero()
        idx = list(zip(r, c))
        return idx

    def nonzero_idx_img(self):
        r, c = self.R_image.nonzero()
        idx = list(zip(r, c))
        return idx
    
    def nonzero_idx_txt(self):
        r, c = self.R_text.nonzero()
        idx = list(zip(r, c))
        return idx

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_val + self.n_test))
        print('n_train=%d, n_val=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_val, self.n_test, (self.n_train + self.n_val + self.n_test)/(self.n_users * self.n_items)))

def dataset_merge_and_split(path):
    df = pd.read_csv(path + "/train.csv", index_col=None, usecols=None)
    ui = defaultdict(list)
    for index, row in df.iterrows():
        user, item = int(row['userID']), int(row['itemID'])
        ui[user].append(item)

    df = pd.read_csv(path + "/test.csv", index_col=None, usecols=None)
    for index, row in df.iterrows():
        user, item = int(row['userID']), int(row['itemID'])
        ui[user].append(item)

    train_json = {}
    val_json = {}
    test_json = {}
    for u, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[:len(testval)//2]
        val = testval[len(testval)//2:]
        train = [i for i in list(range(len(items))) if i not in testval]
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    with open(path + '/5-core/train.json', 'w') as f:
        json.dump(train_json, f)
    with open(path + '/5-core/val.json', 'w') as f:
        json.dump(val_json, f)
    with open(path + '/5-core/test.json', 'w') as f:
        json.dump(test_json, f)


def load_textual_image_features(data_path):
    import os, json
    from gensim.models.doc2vec import Doc2Vec
    asin_dict = json.load(open(os.path.join(data_path, 'asin_sample.json'), 'r'))

    # Prepare textual feture data.
    doc2vec_model = Doc2Vec.load(os.path.join(data_path, 'doc2vecFile'))
    vis_vec = np.load(os.path.join(data_path, 'image_feature.npy'), allow_pickle=True).item()
    text_vec = {}
    for asin in asin_dict:
        text_vec[asin] = doc2vec_model.docvecs[asin]

    all_dict = {}
    num_items = 0
    filename = data_path + '/train.csv'
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for index, row in df.iterrows():
        asin, i = row['asin'], int(row['itemID'])
        all_dict[i] = asin
        num_items = max(num_items, i)
    filename = data_path + '/test.csv'
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for index, row in df.iterrows():
        asin, i = row['asin'], int(row['itemID'])
        all_dict[i] = asin
        num_items = max(num_items, i)

    t_features = []
    v_features = []
    for i in range(num_items+1):
        t_features.append(text_vec[all_dict[i]])
        v_features.append(vis_vec[all_dict[i]])

    np.save(data_path+'/text_feat.npy', np.asarray(t_features,dtype=np.float32))
    np.save(data_path+'/image_feat.npy', np.asarray(v_features,dtype=np.float32))