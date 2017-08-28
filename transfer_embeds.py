from argparse import ArgumentParser

from charlm.model.lstm import LSTMLanguageModel
from siamese.model import LSTMSiameseNet
from siamese.loader import TwinLoader
from siamese.settings.default import *
from charlm.utils.settings import *


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(dest='FROM_MODEL',
                        help='The path to the model from which embeddings are transferred.')
    parser.add_argument(dest='TO_MODEL',
                        help='The path to a directory where the new transferred model will be saved.')
    return parser.parse_args()


def main(args):
    os.makedirs(args.TO_MODEL, exist_ok=True)

    m1 = LSTMLanguageModel.load(args.FROM_MODEL)
    m2 = LSTMSiameseNet()


if __name__ == '__main__':
    main(parse_arguments())
