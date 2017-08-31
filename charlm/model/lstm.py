from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, Conv1D
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger
import numpy as np
import os
import pickle

from charlm.model.callbacks import LSTMCallback
from charlm.utils.data import DataLoader
from charlm import DEFAULT_TOKEN, MASK_TOKEN, SENTENCE_END_TOKEN


class LSTMLanguageModel:
    def __init__(self, loader, recurrent_neurons=(500,),
                 dropout=0.5, filters=100, kernel_size=4,
                 directory='trained/debug'):
        self.directory = directory
        self.dropout = dropout
        self.recurrent_neurons = recurrent_neurons
        self.loader = loader
        self.filters = filters
        self.kernel_size = kernel_size
        self._compiled = False

        self._create_model()

    def _create_model(self):
        self.model = Sequential()
        try:
            embed_matrix = self.loader.embed_matrix
        except AttributeError:
            embed_matrix = np.zeros((len(self.loader.index_to_char), self.loader.embed_dims))
        self.model.add(Embedding(input_dim=np.size(embed_matrix, 0),
                                 output_dim=np.size(embed_matrix, 1),
                                 input_shape=(self.loader.sentence_len,),
                                 weights=[embed_matrix],
                                 trainable=True, mask_zero=False,
                                 name='Embedding'))

        self.model.add(Conv1D(self.filters, self.kernel_size, strides=1, padding='same',
                              activation='relu'))

        for n in self.recurrent_neurons:
            self.model.add(LSTM(n, implementation=1,
                                return_sequences=True,
                                dropout=self.dropout,
                                activation='tanh',
                                recurrent_dropout=0.0))

        self.model.add(TimeDistributed(Dense(len(self.loader.index_to_char),
                                             activation='softmax')))

    def compile(self, optimizer=RMSprop, learning_rate=0.001):
        self.model.compile(optimizer(lr=learning_rate), 'categorical_crossentropy',
                           sample_weight_mode='temporal', metrics=[])
        self._compiled = True

    def train(self, epochs=100, batch_size=30, start_from=0,
              train_key='train', test_key='test'):
        logger = CSVLogger(self.directory + '/epochs.csv')
        primary = LSTMCallback(self)

        callbacks = [logger, primary]

        if not self._compiled:
            print('WARNING: Automatically compiling using default parameters.')
            self.compile()

        xtrain = self.loader.X[train_key]
        ytrain = self.loader.y[train_key]
        mtrain = self.loader.mask[train_key]
        xtest = self.loader.X[test_key]
        ytest = self.loader.y[test_key]
        mtest = self.loader.mask[test_key]

        return self.model.fit(xtrain, ytrain, sample_weight=mtrain,
                              validation_data=(xtest, ytest, mtest),
                              batch_size=batch_size, epochs=epochs,
                              callbacks=callbacks, initial_epoch=start_from)

    def train_generator(self, epochs=100, batch_size=30, start_from=0,
                        train_key='train', test_key='test'):
        logger = CSVLogger(self.directory + '/epochs.csv')
        primary = LSTMCallback(self)

        callbacks = [logger, primary]

        if not self._compiled:
            print('WARNING: Automatically compiling using default parameters.')
            self.compile()

        return self.model.fit_generator(self.loader.generate(train_key, batch_size=batch_size),
                                        steps_per_epoch=self.loader.steps_per_epoch(train_key, batch_size=batch_size),
                                        epochs=epochs, callbacks=callbacks,
                                        validation_data=self.loader.generate(test_key, batch_size=batch_size),
                                        validation_steps=self.loader.steps_per_epoch(test_key, batch_size=batch_size),
                                        max_queue_size=1, workers=1, initial_epoch=start_from)

    def predict(self, text, end_at, verbose=0):
        index_to_char = self.loader.index_to_char
        chars, indices = self.loader.prepare_text(text)

        current_len = len(chars) - 1
        last_token = chars[-1]
        predictions = []
        while last_token not in end_at:
            result = np.argmax(self.model.predict(indices, verbose=verbose)[0, current_len])
            last_token = index_to_char[result]

            chars.append(last_token)
            predictions.append(last_token)

            current_len += 1
            if current_len >= self.loader.sentence_len:
                break
            indices[0, current_len] = result

        return ''.join(chars), predictions

    def predict_word(self, text, verbose=0):
        return self.predict(text, (' ', '\n', '\t', MASK_TOKEN, SENTENCE_END_TOKEN),
                            verbose=verbose)

    def predict_sentence(self, text, verbose=0):
        return self.predict(text, (SENTENCE_END_TOKEN, MASK_TOKEN),
                            verbose=verbose)

    def save(self):
        self.loader.save(self.directory)

        f1 = os.path.join(self.directory, 'weights.hdf5')
        f2 = os.path.join(self.directory, 'config.pkl')

        self.model.save_weights(f1)
        config = {
            'neurons': self.recurrent_neurons,
            'dropout': self.dropout,
            'filters': self.filters,
            'kernel_size': self.kernel_size
        }
        pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, directory):
        loader = DataLoader.load(directory)

        f1 = os.path.join(directory, 'weights.hdf5')
        f2 = open(os.path.join(directory, 'config.pkl'), 'rb')

        config = pickle.load(f2)
        recurrent_neurons = config['neurons']
        dropout = config.get('dropout', 0.0)
        filters = config['filters']
        kernel_size = config['kernel_size']

        lstm = LSTMLanguageModel(loader, recurrent_neurons=recurrent_neurons,
                                 dropout=dropout, filters=filters, kernel_size=kernel_size)
        lstm.model.load_weights(f1)
        return lstm
