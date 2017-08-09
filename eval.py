from argparse import ArgumentParser

from neuralknn.classifier import NeuralKNN


parser = ArgumentParser()
parser.add_argument(dest='MODEL',
                    help='The path to a model\'s directory')
parser.add_argument('-s', '--set', dest='SET', default='test',
                    help='''The name of the dataset within the loader yo be used for evalutaion.
                    A typical loader will contain a train and test set.''')
parser.add_argument('-k', dest='K', default=1, type=int,
                    help='Number of nearest neighbor to be used for the NeuralKNN.')
args = parser.parse_args()

try:
    print('Loading model from %s ...' % args.MODEL)
    model = NeuralKNN(args.MODEL, k=args.K)
    model.fit()

    print('Evaluating...')
    acc = model.evaluate(X_key=args.SET, y_key=args.SET)

    print('Accuracy: %.3f' % acc)
except FileNotFoundError:
    print('Invalid model directory')
    quit()
