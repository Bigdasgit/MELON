import argparse

def parse_args(flags=False):
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--seed', type=int, default=978,
                        help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='WomenClothing',
                        help='Choose a dataset from {WomenClothing, MenClothing, Toys_and_Games, Sports}')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--feat_embed_dim', type=int, default=64,
                        help='Feature embedding size.')
    parser.add_argument('--rel_embed_dim', type=int, default=64,
                        help='relational embedding size.')
    parser.add_argument('--weight_size', nargs='?', default='[64,64]',
                    help='Output sizes of every layer')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Coefficient of mce module.')
    parser.add_argument('--beta', type=float, default=0.6,
                        help='Coefficient of rce module.')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='Coefficient of fine-grained interest matching.')
    parser.add_argument('--delta', type=float, default=0.9,
                        help='Coefficient of self node features. 0.9 for Women&Men Clothing, 1.0 for the others')
    parser.add_argument('--ui_k', type=str, default='5',
                        help='user-item top/bottom-k graph')
    parser.add_argument('--core', type=int, default=5,
                        help='5-core for warm-start; 0-core for cold start.')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of graph conv layers.')
    parser.add_argument('--has_norm', default=True, action='store_false')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='') 
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20]',
                        help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    if flags:
        attribute_dict = dict(vars(parser.parse_args()))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for k, v in attribute_dict.items():
            print(k + ' : ' + str(v))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
    return parser.parse_args()
