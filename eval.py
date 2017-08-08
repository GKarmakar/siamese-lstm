from argparse import ArgumentParser

from neuralknn.classifier import NeuralKNN


parser = ArgumentParser()
parser.add_argument(dest='MODEL',
                    help='The path to a model\'s directory')
args = parser.parse_args()

try:
    print('Loading model from %s ...' % args.MODEL)
    model = NeuralKNN(args.MODEL)

    print('Evaluating...')
    acc = model.evaluate()

    print('Accuracy: %.3f' % acc)
except FileNotFoundError:
    print('Invalid model directory')
    quit()
