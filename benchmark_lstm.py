from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Masking
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras import regularizers

from siamese.settings.word_default import *

LABEL_COUNT = 2
CALLBACKS = [EarlyStopping(monitor='val_loss', patience=4)]


def create_model():
    model = Sequential()
    model.add(Masking(mask_value=0,
                      input_shape=(SENTENCE_LEN, EMBED_DIMS)))
    for n in RECURRENT_NEURONS[:-1]:
        model.add(Bidirectional(LSTM(n, implementation=1,
                                     return_sequences=True,
                                     dropout=0.0,
                                     activation='tanh',
                                     recurrent_dropout=DROPOUT,
                                     kernel_regularizer=regularizers.l2(RECURRENT_REGULARIZER))))

    if DENSE_UNITS > 0:
        model.add(LSTM(RECURRENT_NEURONS[-1], implementation=1,
                       return_sequences=False,
                       dropout=DROPOUT,
                       activation='hard_sigmoid',
                       recurrent_dropout=DROPOUT,
                       kernel_regularizer=regularizers.l2(RECURRENT_REGULARIZER)))
        model.add(Dense(DENSE_UNITS, activation='relu',
                        kernel_regularizer=regularizers.l2(DENSE_REGULARIZER)))
    else:
        model.add(LSTM(RECURRENT_NEURONS[-1], implementation=1,
                       return_sequences=False,
                       dropout=DROPOUT,
                       activation='linear',
                       recurrent_dropout=DROPOUT,
                       kernel_regularizer=regularizers.l2(RECURRENT_REGULARIZER)))

    model.add(Dense(LABEL_COUNT, activation='softmax',
                    kernel_regularizer=regularizers.l2(DENSE_REGULARIZER)))

    model.compile(RMSprop(lr=LEARNING_RATE), loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def generate_data(X, y, batch_size):
    while True:
        for i in range(batch_size, len(X), batch_size):
            yield None


def main(args):
    model = create_model()
    print(model.summary())


if __name__ == '__main__':
    main(None)
