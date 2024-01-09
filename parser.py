import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aug_type', nargs='?', default='degree')
    parser.add_argument('--aug_ratio', type=float, default=0.5)
    parser.add_argument('--not_linear', action="store_true", default="False")
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--epoch', type=int, default=5000,
                        help='Number of epoch.')
    parser.add_argument('--dataset', nargs='?', default='instacart')
    parser.add_argument('--reg', type=float, default=1e-3,
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout ratio.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers.')
    parser.add_argument('--Ks', nargs='?', default='[5,10,20, 40, 60,80,100]',
                        help='Output sizes of every layer')
    parser.add_argument('--interval', type=int, default=20,
                        help='Eval Interval.')
    parser.add_argument('--cuda', nargs='?', default='0',
                        help='the default gpu to run the code')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--name', type=str, default="test-ins9-1")
    parser.add_argument('--self_loop', action="store_true", default="False")

    parser.add_argument('--ssl_reg', type=float, default=1e-1)
    parser.add_argument('--cl_reg', type=float, default=0.05)
    parser.add_argument('--ssl_t', type=float, default=1e-1)
    parser.add_argument('--cl_t', type=float, default=0.1)
    parser.add_argument('--cl_reg_cro', type=float, default=1e-1)
    parser.add_argument('--cl_t_cro', type=float, default=1e-1)
    parser.add_argument('--noise_ratio', type=float, default=0)
    parser.add_argument('--model', nargs='?', default='sgl_fusion_ui',
     help='Choose a model from {mf, lgcn,sgl,lgcn_basket,sgl_basket,sgl_fusion,sgl_fusion_ui}')
    args = parser.parse_args()
    # args = parser.parse_args()
    return args
                    