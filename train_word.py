from argparse import ArgumentParser

import shutil
import sys
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from siamese.loader import TwinWordLoader
from siamese.model import LSTMSiameseWord
from siamese.settings.word_default import *
from charlm.utils.settings import *


def main(argv):
    parser = ArgumentParser()
    parser.add_argument('--name', required=False, default='default', dest='MODEL_NAME',
                        help='The directory within trained/ where the new model and logs will be stored.')
    parser.add_argument('-r', '--resume', dest='RESUME_MODEL', default=None,
                        help='''The path to a trained model to be resumed. 
                        The model will  be saved in the same directory, overriding the --name parameter.''')
    parser.add_argument('--start-from', dest='FROM', default=0, type=int,
                        help='The epoch to start training from. Useful when resuming.')
    parser.add_argument('--balance', dest='BALANCE', action='store_true',
                        help='Whether to balance the dataset before training.')

    args = parser.parse_args(argv)

    ALIAS = ('train', 'test', 'ignore')
    DIRECTORY = 'trained/' + args.MODEL_NAME
    SETTINGS_MAP = {
        'START_FROM': args.FROM,
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE
    }

    if args.RESUME_MODEL is None:
        shutil.rmtree(os.path.join(DIRECTORY, 'tensorboard'), ignore_errors=True)
        os.makedirs(DIRECTORY, exist_ok=True)

        print('Loading data...')
        loader = TwinWordLoader(paths=PATHS, path_alias=ALIAS, nlines=LINE_LIMIT,
                                sentence_len=SENTENCE_LEN, embed_dims=EMBED_DIMS,
                                fasttext_path=FASTTEXT_PATH)
        loader.load_data(skip_gen=True)
        print('Loading complete.')
        print('\tTotal document count: %d' % loader.doc_count)
        print('\tUnique characters: %d' % (len(loader.index_to_char) - 2))

        print('Creating model...')
        model = LSTMSiameseWord(loader, dense_units=DENSE_UNITS, recurrent_neurons=RECURRENT_NEURONS,
                                dropout=DROPOUT, recurrent_reg=RECURRENT_REGULARIZER,
                                dense_reg=DENSE_REGULARIZER, directory=DIRECTORY,
                                merge_layer=MERGE_LAYER)
    else:
        print('Loading model from  %s ...' % args.RESUME_MODEL)
        model = LSTMSiameseWord.load(args.RESUME_MODEL)

    SETTINGS_MAP.update({
        'FAST_TEXT_PATH': model.loader.fasttext_path,
        'SENTENCE_LEN': model.loader.sentence_len,
        'PATHS': model.loader.paths,
        'EMBED_DIM': model.loader.embed_dims,
        'LINE_LIMIT': model.loader.nlines,
        'RECURRENT_NEURONS': model.recurrent_neurons,
        'DENSE_NEURONS': model.dense_units,
        'DROPOUT': model.dropout,
        'RECURRENT_REGULARIZER': model.recurrent_reg,
        'DENSE_REGULARIZER': model.dense_reg,
        'MERGE_LAYER': model.merge_layer
    })
    print('Current model settings:')
    print_settings(SETTINGS_MAP)
    write_settings(os.path.join(DIRECTORY, 'settings.csv'), SETTINGS_MAP)

    print('Compiling...')
    model.compile(learning_rate=LEARNING_RATE)
    print(model.model.summary())
    print('Starting training...')
    try:
        if not args.BALANCE:
            model.train(epochs=EPOCHS, batch_size=BATCH_SIZE, start_from=args.FROM, generate=True,
                        callback=True)
        else:
            model.loader.balance()
            model.train_balanced(epochs=EPOCHS, batch_size=BATCH_SIZE, start_from=args.FROM, generate=True)
    except KeyboardInterrupt:
        print('\nTraining interrupted...')


if __name__ == '__main__':
    main(sys.argv[1:])
