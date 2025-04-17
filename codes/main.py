from audioop import add
from distutils.command.build import build
import math
import random
import sys
from time import time

import numpy as np
import torch
import torch.optim as optim

from utility.parser import parse_args
from utility.batch_test import *
from Models import *

outer_args = parse_args()


class Trainer(object):
    def __init__(self, data_config, args):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.train_items = data_config['train_items']
        self.nonzero_idx_img = data_config['nonzero_idx_img']
        self.nonzero_idx_txt = data_config['nonzero_idx_txt']        
        self.ui_index = {}
        index = 0
        for k in range(len(self.train_items)):
            sorted_list = sorted(self.train_items[k])           
            for v in sorted_list:
                pair = str(k)+'_'+str(v)
                self.ui_index[pair] = index
                index += 1
        self.feat_embed_dim = args.feat_embed_dim
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers
        self.has_norm = args.has_norm
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.lamb = self.regs[1]
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.delta = args.delta
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.nonzero_idx = data_config['nonzero_idx']

        self.image_feats = np.load('data/{}/image_feat.npy'.format(self.dataset))
        self.text_feats = np.load('data/{}/text_feat.npy'.format(self.dataset))
        self.model = MELON(self.n_users, self.n_items, self.feat_embed_dim, self.nonzero_idx, self.nonzero_idx_img, self.nonzero_idx_txt, self.has_norm, self.image_feats, self.text_feats, self.train_items, self.n_layers, self.alpha, self.beta, self.gamma, self.delta)

        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            users, neg_items = [1], [1]
            user_ice, item_ice, user_mce, item_mce, img_query, txt_query, _, _, uv_agg, ut_agg, v_rel_mlp, t_rel_mlp, image_features, text_features = self.model(users, neg_items)
        result = test_torch(user_ice, item_ice, user_mce, item_mce, img_query, txt_query, uv_agg, ut_agg, image_features, text_features, v_rel_mlp, t_rel_mlp, users_to_test, is_val, self.adj, self.alpha, self.beta, self.gamma)
        return result

    def train(self):
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.adj = torch.sparse.FloatTensor(nonzero_idx, torch.ones((nonzero_idx.size(1))).cuda(), (self.n_users, self.n_items)).to_dense().cuda()
        stopping_step = 0
        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.  
            n_batch = data_generator.n_train // args.batch_size + 1
            for idx in range(n_batch):

                self.model.train()
                self.optimizer.zero_grad()

                users, pos_items, neg_items = data_generator.sample()
                pos_pairs = []
                for i in range(len(users)):
                    pos_pair = str(users[i])+'_'+str(pos_items[i])
                    pos_pairs.append(self.ui_index[pos_pair])

                user_ice, item_ice, user_mce, item_mce, img_query, txt_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, _, _, _, _ = self.model(users, neg_items)
 
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model.bpr_loss(user_ice, item_ice, user_mce, item_mce, img_query, txt_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg, users, pos_items, neg_items, pos_pairs)

                batch_emb_loss = self.decay * batch_emb_loss 
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss 

                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

                del user_ice, item_ice, user_mce, item_mce, img_query, txt_query, comp_rel_v, comp_rel_t, comp_rel_v_neg, comp_rel_t_neg
                torch.cuda.empty_cache()
                
            self.lr_scheduler.step()


            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            perf_str = 'Pre_Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)

            print(perf_str)

            if epoch % args.verbose != 0:
                continue


            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)

            t3 = time()

            if args.verbose > 0:
                perf_str = 'Pre_Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f]' % \
                        (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss)
                perf_str_value10 = 'precision@10=[%.5f], recall@10=[%.5f] , ndcg@10=[%.5f]' % (ret['precision'][0], ret['recall'][0], ret['ndcg'][0])
                perf_str_value20 = 'precision@20=[%.5f], recall@20=[%.5f] , ndcg@20=[%.5f]' % (ret['precision'][-1], ret['recall'][-1], ret['ndcg'][-1])
                print(perf_str)
                print(perf_str_value10)
                print(perf_str_value20)
            if ret['recall'][-1] > best_recall: 
                best_recall = ret['recall'][-1]
                stopping_step = 0
                torch.save({self.model_name: self.model.state_dict()}, './models/' + self.dataset + '_' + self.model_name)
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print('#####Early stopping steps: %d #####' % stopping_step)
            else:
                print('#####Early stop! #####')
                break
        self.model = MELON(self.n_users, self.n_items, self.feat_embed_dim, self.nonzero_idx, self.nonzero_idx_img, self.nonzero_idx_txt, self.has_norm, self.image_feats, self.text_feats, self.train_items, self.n_layers, self.alpha, self.beta, self.gamma, self.delta)
        
        self.model.load_state_dict(torch.load('./models/' + self.dataset + '_' + self.model_name, map_location=torch.device('cpu'))[self.model_name])
        self.model.cuda()
        test_ret = self.test(users_to_test, is_val=False)
        t4 = time()
        print('Final ', test_ret)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu

if __name__ == '__main__':
    args = parse_args(True)
    set_seed(args.seed)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['train_items'] = data_generator.train_items

    nonzero_idx = data_generator.nonzero_idx()
    nonzero_idx_img = data_generator.nonzero_idx_img()
    nonzero_idx_txt = data_generator.nonzero_idx_txt()
    config['nonzero_idx'] = nonzero_idx
    config['nonzero_idx_img'] = nonzero_idx_img
    config['nonzero_idx_txt'] = nonzero_idx_txt

    trainer = Trainer(config, args)
    trainer.train()