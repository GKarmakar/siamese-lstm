from argparse import ArgumentParser
from threading import Thread
import time

from neuralknn.classifier import NeuralKNN

evaluating = False


def progress():
    global evaluating
    elapsed = 0
    while evaluating:
        print('Time elapsed: %ds...' % elapsed, end='\r')
        time.sleep(1)
        elapsed += 1

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

    thread = Thread(target=progress)

    evaluating = True
    print('Evaluating...')
    thread.daemon = True
    thread.start()
    acc = model.evaluate(key=args.VAL_SET)
    evaluating = False

    print('\nAccuracy: %.3f' % acc)
except FileNotFoundError:
    print('Invalid model directory')
    quit()
except KeyboardInterrupt:
    evaluating = False
    print('\nEvaluation interrupted.')
