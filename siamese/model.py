from charlm.model.callbacks import LSTMCallback
from charlm import MASK_TOKEN
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding
from keras.layers.wrappers import Bidirectional
from keras import regularizers

import os
import numpy as np
import pickle
import keras

from siamese.loader import TwinLoader
from charlm.model.lstm import LSTMLanguageModel
from keras.optimizers import RMSprop

from siamese.loss import Abs


class LSTMSiameseNet(LSTMLanguageModel):
    def __init__(self, loader, dense_units=128, **kwargs):
        self.dense_units = dense_units
        LSTMLanguageModel.__init__(self, loader, **kwargs)

    def _create_model(self):
        try:
            embed_matrix = self.loader.embed_matrix
        except AttributeError:
            embed_matrix = np.zeros((len(self.loader.index_to_char), self.loader.embed_dims))

        twin = Sequential()
        twin.add(Embedding(input_dim=np.size(embed_matrix, 0),
                           output_dim=np.size(embed_matrix, 1),
                           input_shape=(self.loader.sentence_len,),
                           weights=[embed_matrix],
                           trainable=True, mask_zero=True,
                           name='Embedding'))
        for n in self.recurrent_neurons[:-1]:
            twin.add(Bidirectional(LSTM(n, implementation=1,
                                        return_sequences=True,
                                        dropout=0.0,
                                        activation='tanh',
                                        recurrent_dropout=self.dropout,
                                        kernel_regularizer=regularizers.l2(0.03))))

        twin.add(LSTM(self.recurrent_neurons[-1], implementation=1,
                      return_sequences=False,
                      dropout=0.0,
                      activation='tanh',
                      recurrent_dropout=self.dropout,
                      kernel_regularizer=regularizers.l2(0.03)))
        twin.add(Dense(self.dense_units, activation='linear', kernel_regularizer=regularizers.l2(0.03)))

        left_in = Input((self.loader.sentence_len,))
        left_twin = twin(left_in)
        right_in = Input((self.loader.sentence_len,))
        right_twin = twin(right_in)
        merged = Abs()([left_twin, right_twin])
        out = Dense(1, activation='relu')(merged)

        self.model = Model(inputs=(left_in, right_in), outputs=out)

    def compile(self, optimizer=RMSprop, learning_rate=0.001):
        self.model.compile(optimizer(lr=learning_rate), 'mean_squared_error',
                           metrics=['mae'])
        self._compiled = True

    def train(self, epochs=100, batch_size=30, start_from=0,
              train_key='train', test_key='test', callback=True):
        left_train = self.loader.X['train'][0]
        right_train = self.loader.X['train'][1]
        y_train = self.loader.y['train']
        left_test = self.loader.X['test'][0]
        right_test = self.loader.X['test'][1]
        y_test = self.loader.y['test']

        stopper = EarlyStopping(monitor='loss', patience=1)
        callbacks = [stopper]

        if callback:
            if keras.backend.backend() == 'tensorflow':
                board = TensorBoard(os.path.join(self.directory, 'tensorboard'),
                                    batch_size=batch_size, histogram_freq=0,
                                    write_images=True, write_grads=True)
                callbacks.append(board)
            primary = LSTMCallback(self)
            logger = CSVLogger(self.directory + '/epochs.csv')
            callbacks.extend([primary, logger])

        if not self._compiled:
            print('WARNING: Automatically compiling using default parameters.')
            self.compile()
        return self.model.fit([left_train, right_train], y_train,
                              validation_data=([left_test, right_test], y_test),
                              batch_size=batch_size, epochs=epochs,
                              callbacks=callbacks, initial_epoch=start_from)

    def distance(self, text1, text2):
        if not isinstance(text1, np.ndarray):
            l_chars, l_indices = self.loader.prepare_text(text1)
        else:
            const = self.loader.char_to_index[MASK_TOKEN]
            l_indices = np.pad(text1, (0, self.loader.sentence_len - np.size(text1, -1)), 'constant',
                               constant_values=const)
        l_indices = np.reshape(l_indices, (1, -1))

        if not isinstance(text2, np.ndarray):
            r_chars, r_indices = self.loader.prepare_text(text2)
        else:
            const = self.loader.char_to_index[MASK_TOKEN]
            r_indices = np.pad(text2, (0, self.loader.sentence_len - np.size(text2, -1)), 'constant',
                               constant_values=const)
        r_indices = np.reshape(r_indices, (1, -1))

        return self.model.predict([l_indices, r_indices])[0, 0]

    def save(self):
        self.loader.save(self.directory)

        f1 = os.path.join(self.directory, 'weights.hdf5')
        f2 = os.path.join(self.directory, 'config.pkl')

        self.model.save_weights(f1)
        config = {
            'recurrent_neurons': self.recurrent_neurons,
            'dropout': self.dropout,
            'dense_units': self.dense_units,
        }
        with open(f2, 'wb') as f:
            pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    @classmethod
    def load(cls, directory):
        loader = TwinLoader.load(directory)

        f1 = os.path.join(directory, 'weights.hdf5')
        f2 = open(os.path.join(directory, 'config.pkl'), 'rb')

        config = pickle.load(f2)
        f2.close()

        recurrent_neurons = config['recurrent_neurons']
        dense_units = config['dense_units']
        dropout = config.get('dropout', 0.0)

        lstm = LSTMSiameseNet(loader, recurrent_neurons=recurrent_neurons,
                              dropout=dropout, dense_units=dense_units)
        lstm.model.load_weights(f1)
        return lstm

    def predict(self, text, end_at, verbose=0):
        raise NotImplementedError('This method is inherited and unused. Use distance(t1, t2) instead.')