from argparse import ArgumentParser

from neuralsvm.classifier import NeuralSVM


parser = ArgumentParser()
parser.add_argument(dest='MODEL',
                    help='The path to a model\'s directory')
parser.add_argument('--set', dest='SET', default='train',
                    help='''The name of the dataset within the loader to be used for kNN inference.
                    A typical loader will contain a train and test set.''')
parser.add_argument('--val-set', dest='VAL_SET', default='test',
                    help='''The name of the dataset within the loader to be used for evalutaion.''')
args = parser.parse_args()

try:
    print('Loading model from %s ...' % args.MODEL)
    model = NeuralSVM(args.MODEL)
    model.fit(key=args.SET)

    evaluating = True
    print('Evaluating...')
    train_acc = model.evaluate(key=args.SET)
    acc = model.evaluate(key=args.VAL_SET)

    print('\nTraining accuracy: %.3f' % train_acc)
    print('\nAccuracy: %.3f' % acc)
except FileNotFoundError:
    print('Invalid model directory')
    quit()
except KeyboardInterrupt:
    print('\nEvaluation interrupted.')

