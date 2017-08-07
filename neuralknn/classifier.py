from sklearn.neighbors import KNeighborsClassifier

import numpy as np

from siamese.model import LSTMSiameseNet


class NeuralKNN:
    def __init__(self, model_path, k=1):
        self.model = LSTMSiameseNet.load(model_path)
        self._classifier = KNeighborsClassifier(n_neighbors=k, metric='pyfunc',
                                                func=self.model.distance,
                                                algorithm='brute')
        self._isfit = False

    def _format_data(self):
        raw = self.model.loader.raw['train']
        self._X = np.zeros((len(raw), self.model.loader.sentence_len), dtype=int)
        for i, string in enumerate(raw):
            for j, char in enumerate(list(string)):
                self._X[i, j] = self.model.loader.char_to_index[char]

        self._y = self.model.loader.raw_label['train']

    def fit(self):
        self._format_data()
        self._classifier.fit(self._X, self._y)
        self._isfit = True

    def predict(self, text):
        if not self._isfit:
            self.fit()
        chars, array = self.model.loader.prepare_text(text)
        return self._classifier.predict(array)[0]
