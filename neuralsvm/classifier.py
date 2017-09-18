from sklearn.svm import SVC

import numpy as np

from siamese.model import LSTMSiameseWord


class NeuralSVM:
    def __init__(self, model_path: str):
        self.model = LSTMSiameseWord.load(model_path, skip_gen=True)
        self._classifier = SVC(kernel='linear', verbose=True)
        self._isfit = False

    def _format_data(self, key):
        self._y = self.model.loader.raw_label[key]
        self._X, self._y = self.__create_data(self.model.loader.raw[key],
                                              self.model.loader.raw_label[key])

    def __create_data(self, raw, raw_label):
        X = np.zeros((len(raw), self.model.output_vec_len), dtype=float)
        for i, string in enumerate(raw):
            X[i] = self.model.predict_sent_vector(string)
        y = raw_label
        return X, y

    def fit(self, key='train'):
        self._format_data(key)
        self._classifier.fit(self._X, self._y)
        self._isfit = True

    def predict(self, text):
        if not self._isfit:
            self.fit()
        array = self.model.predict_sent_vector(text)
        return self._classifier.predict(array.reshape(1, -1))[0]

    def evaluate(self, key='test', sample_weight=None):
        if not self._isfit:
            self.fit()

        try:
            X, y = self.__create_data(self.model.loader.raw[key],
                                      self.model.loader.raw_label[key])
        except KeyError as e:
            raise ValueError('Invalid dataset keys %s' % key)
        return self._classifier.score(X, y, sample_weight)
