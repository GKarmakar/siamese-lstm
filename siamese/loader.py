from charlm.utils.data import DataLoader
from charlm import MASK_TOKEN, SENTENCE_END_TOKEN

import numpy as np
import itertools
import os
import pickle


class TwinLoader(DataLoader):
    def __init__(self, pos_value=1.0, neg_value=0.0, **kwargs):
        DataLoader.__init__(self, **kwargs)
        self.pos_value = pos_value
        self.neg_value = neg_value
        self.raw_label = {}
        del self.mask

    def load_data(self):
        if self.path_alias is None:
            self.path_alias = self.paths[:]

        if not hasattr(self.nlines, '__iter__'):
            self.nlines = [None] * len(self.paths)
        elif len(self.nlines) < len(self.paths):
            self.nlines += [None] * (len(self.paths) - len(self.nlines))

        for path, alias, nline in zip(self.paths, self.path_alias, self.nlines):
            f = open(path, 'rt')
            lines = f.readlines() if nline is None else f.readlines()[:nline]
            f.close()

            text, label = [], []
            for line in lines:
                l, t = line.split('\t')
                text.append(t), label.append(l)
            self.raw[alias] = [t.strip()[:self.sentence_len] for t in text]
            self.raw_label[alias] = [l.strip() for l in label]

        all_raw = [j for i in self.raw.values() for j in i]
        self._index_chars(all_raw)
        self._embed_func()
        self._create_data()

    def _create_data(self):
        for alias in self.raw.keys():
            x_values = self.raw[alias]
            y_values = self.raw_label[alias]
            combined_values = [(x, y) for x, y in zip(x_values, y_values)]
            combined_values = list(itertools.combinations(combined_values, 2))

            self.X[alias] = (np.ones((len(combined_values), self.sentence_len), dtype=int) *
                             self.char_to_index[MASK_TOKEN],
                             np.ones((len(combined_values), self.sentence_len), dtype=int) *
                             self.char_to_index[MASK_TOKEN])
            self.y[alias] = np.zeros((len(combined_values),))

            for i, tup in enumerate(combined_values):
                d1 = tup[0][0]
                d2 = tup[1][0]
                l1 = tup[0][1]
                l2 = tup[1][1]

                for j, c in enumerate(d1):
                    self.X[alias][0][i, j] = self.char_to_index[c]
                for j, c in enumerate(d2):
                    self.X[alias][1][i, j] = self.char_to_index[c]
                self.y[alias][i] = self.pos_value if l1 == l2 else self.neg_value

    def save(self, folder):
        f1 = os.path.join(folder, 'loader.pkl')

        config = {
            'paths': self.paths,
            'path_alias': self.path_alias,
            'nlines': self.nlines,
            'embed_type': 'hot_coded' if self._embed_func == self._hot_coded_embed else 'uniform',
            'embed_dims': self.embed_dims,
            'sentence_len': self.sentence_len,
            'char_to_index': self.char_to_index,
            'index_to_char': self.index_to_char,
            'raw': self.raw,
            'raw_label': self.raw_label,
            'pos_value': self.pos_value,
            'neg_value': self.neg_value
        }

        with open(f1, 'wb') as f:
            pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    @classmethod
    def load(cls, folder):
        f1 = os.path.join(folder, 'loader.pkl')

        with open(f1, 'rb') as f:
            config = pickle.load(f)
            f.close()
        paths = config.get('paths', None)
        path_alias = config.get('path_alias', None)
        nlines = config.get('nlines', None)
        embed_type = config.get('embed_type', 'hot_coded')
        embed_dims = config.get('embed_dims', None)
        sentence_len = config['sentence_len']
        pos_value = config.get('pos_value', 0.0)
        neg_value = config.get('neg_value', 1.0)

        loader = TwinLoader(pos_value=pos_value, neg_value=neg_value,
                            paths=paths, path_alias=path_alias, nlines=nlines,
                            embed_type=embed_type, embed_dims=embed_dims,
                            sentence_len=sentence_len)

        try:
            char_to_index = config['char_to_index']
            index_to_char = config['index_to_char']
            raw_label = config['raw_label']
            raw = config['raw']
            loader.char_to_index = char_to_index
            loader.index_to_char = index_to_char
            loader.raw = raw
            loader.raw_label = raw_label
            loader._create_data()
        except KeyError as e:
            # print('WARNING: Can\'t locate loader indices. Reloading might cause inaccuracies...')
            # loader.load_data()
            raise e
        return loader