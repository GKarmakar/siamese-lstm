from sklearn.neighbors import KNeighborsClassifier

import numpy as np

from siamese.model import LSTMSiameseNet


class NeuralKNN:
    def __init__(self, model_path, k=1):
        self.model = LSTMSiameseNet.load(model_path, skip_gen=True)
        self._classifier = KNeighborsClassifier(n_neighbors=k,
                                                metric=self.model.distance,
                                                algorithm='brute')
        self._isfit = False

    def _format_data(self, key):
        self._y = self.model.loader.raw_label[key]
        self._X, self._y = self.__create_data(self.model.loader.raw[key],
                                              self.model.loader.raw_label[key])

    def __create_data(self, raw, raw_label):
        X = np.zeros((len(raw), self.model.loader.sentence_len), dtype=int)
        for i, string in enumerate(raw):
            for j, char in enumerate(list(string)):
                X[i, j] = self.model.loader.char_to_index[char]

        y = raw_label
        return X, y

    def fit(self, key='train'):
        self._format_data(key)
        self._classifier.fit(self._X, self._y)
        self._isfit = True

    def predict(self, text):
        if not self._isfit:
            self.fit()
        chars, array = self.model.loader.prepare_text(text)
        return self._classifier.predict(array)[0]

    def evaluate(self, key='test', sample_weight=None):
        if not self._isfit:
            self.fit()

        try:
            X, y = self.__create_data(self.model.loader.raw[key],
                                      self.model.loader.raw_label[key])
        except KeyError as e:
            raise ValueError('Invalid dataset keys %s' % key)
        return self._classifier.score(X, y, sample_weight)
