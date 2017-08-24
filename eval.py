from argparse import ArgumentParser

from neuralknn.classifier import NeuralKNN


parser = ArgumentParser()
parser.add_argument(dest='MODEL',
                    help='The path to a model\'s directory')
parser.add_argument('--set', dest='SET', default='train',
                    help='''The name of the dataset within the loader to be used for kNN inference.
                    A typical loader will contain a train and test set.''')
parser.add_argument('--val-set', dest='VAL_SET', default='test',
                    help='''The name of the dataset within the loader to be used for evalutaion.''')
parser.add_argument('-k', dest='K', default=1, type=int,
                    help='Number of nearest neighbor to be used for the NeuralKNN.')
args = parser.parse_args()

try:
    print('Loading model from %s ...' % args.MODEL)
    model = NeuralKNN(args.MODEL, k=args.K)
    model.fit(key=args.SET)

    print('Evaluating...')
    acc = model.evaluate(key=args.VAL_SET)

    print('Accuracy: %.3f' % acc)
except FileNotFoundError:
    print('Invalid model directory')
    quit()
