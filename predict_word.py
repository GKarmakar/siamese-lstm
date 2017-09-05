from argparse import ArgumentParser
import sys

from neuralknn.classifier import NeuralWordKNN


def main(argv):
    parser = ArgumentParser()
    parser.add_argument(dest='MODEL',
                        help='The path to a model\'s directory')
    parser.add_argument('-i', '--input', dest='INPUT', default=None,
                        help='The optional input file containing one sentence per line')
    args = parser.parse_args(argv)

    try:
        print('Loading model from %s ...' % args.MODEL)
        model = NeuralWordKNN(args.MODEL)
    except FileNotFoundError:
        print('Invalid model directory')
        return

    if args.INPUT is None:
        print('No input specified. Entering shell mode.')
        print('Type \'q\' or Ctrl+D to exit.')

        try:
            while True:
                text = input('nknn> ')
                if text == 'q':
                    break
                label = model.predict(text)
                print('Prediction: %s' % label)
        except (EOFError, KeyboardInterrupt):
            return
    else:
        f = open(args.INPUT, 'rt')
        for line in f:
            text = line.strip()
            if text == '':
                continue
            label = model.predict(text)
            print('%s\t%s' % (label, text))
        f.close()

if __name__ == '__main__':
    main(sys.argv[1:])
