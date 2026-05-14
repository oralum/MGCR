import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=4096, type=int, help='batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
	parser.add_argument('--latdim', default=128, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--topk', default=20, type=int, help='K of top k')
	parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
	parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for contrative learning')
	parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')  # 0.2明显的比不上原文的0.5，0.6或0.65的效果等同于0.5
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default=-1, type=int, help='indicates which gpu to use')
	parser.add_argument('--lambda0', type=float, default=1e-4, help='weight for L0 loss on laplacian matrix.')
	parser.add_argument('--gamma', type=float, default=-0.45)
	parser.add_argument('--zeta', type=float, default=1.05)
	parser.add_argument('--init_temperature', type=float, default=2.0)
	parser.add_argument('--temperature_decay', type=float, default=0.98)
	parser.add_argument("--eps", type=float, default=1e-3)   # 原来是1e-8
	parser.add_argument("--dropout", type=float, default=0.0)

	return parser.parse_args()
args = ParseArgs()
