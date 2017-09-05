import numpy as np
import string
import pickle
import os
import math

from charlm import SENTENCE_END_TOKEN, MASK_TOKEN, DEFAULT_TOKEN


def to_hot_coded(index, num_classes):
    v = np.zeros((num_classes,), dtype=np.float32)
    v[index] = 1
    return v


class DataLoader:
    def __init__(self, paths=None, path_alias=None, nlines=None, embed_type='hot_coded',
                 embed_dims=None, sentence_len=100):
        self.paths = paths if paths is not None else []
        self.path_alias = path_alias

        self.nlines = nlines
        self.embed_dims = embed_dims
        self.sentence_len = sentence_len

        self.__embed_types = {
            'hot_coded': self._hot_coded_embed,
            'uniform': self._uniform_embed
        }
        try:
            self._embed_func = self.__embed_types[embed_type]
        except KeyError:
            raise ValueError('Invalid embed type %s. Please choose from %s' %
                             (embed_type, self.__embed_types.keys()))

        self.X = {}
        self.y = {}
        self.mask = {}
        self.raw = {}

    def load_data(self, skip_gen=False):
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
            self.raw[alias] = [l.strip()[:self.sentence_len-1] for l in lines]

        all_raw = [j for i in self.raw.values() for j in i]
        self._index_tokens(all_raw)
        self._embed_func()
        if not skip_gen:
            self._create_data()

    def _index_tokens(self, documents):
        self.char_to_index = {}
        self.index_to_char = []

        self.char_to_index[MASK_TOKEN] = 0
        self.index_to_char.append(MASK_TOKEN)
        self.char_to_index[SENTENCE_END_TOKEN] = 1
        self.index_to_char.append(SENTENCE_END_TOKEN)

        # Index every ASCII printable
        for c in string.printable:
            # c = c.encode(encoding='utf-8')
            self.char_to_index[c] = len(self.index_to_char)
            self.index_to_char.append(c)

        # Index additional characters in documents
        for doc in documents:
            for c in doc:
                if c not in self.char_to_index.keys():
                    self.char_to_index[c] = len(self.index_to_char)
                    self.index_to_char.append(c)

    def _hot_coded_embed(self):
        if self.embed_dims is not None:
            print('WARNING: Ignoring embed_dims for hot-coded embedding.')
        self.embed_dims = len(self.index_to_char)
        self.embed_matrix = np.eye(len(self.index_to_char), dtype=np.float32)

    def _uniform_embed(self):
        if self.embed_dims is None:
            self.embed_dims = 100
            print('WARNING: Embedding dimensions set to 100 by default.')

        num_chars = len(self.index_to_char)
        self.embed_matrix = np.random.rand(num_chars, self.embed_dims)

    def _create_data(self):
        for key, value in self.raw.items():
            self.X[key] = np.ones((len(value), self.sentence_len), dtype=int) * self.char_to_index[MASK_TOKEN]
            self.y[key] = np.zeros((len(value), self.sentence_len, len(self.index_to_char)),
                                   dtype=np.float32)
            self.mask[key] = np.zeros((len(value), self.sentence_len), dtype=np.float32)

            for i, doc in enumerate(value):
                chars = list(doc)[:self.sentence_len-1]
                chars.append(SENTENCE_END_TOKEN)
                for j, c in enumerate(chars):
                    self.X[key][i, j] = self.char_to_index[c]
            for i, doc in enumerate(value[1:]):
                chars = list(doc)[:self.sentence_len-1]
                chars.append(SENTENCE_END_TOKEN)
                for j, c in enumerate(chars):
                    self.y[key][i, j] = to_hot_coded(self.char_to_index[c], len(self.index_to_char))
                    self.mask[key][i, j] = 1.0

    def prepare_text(self, text):
        chars = list(text)[:self.sentence_len]
        indices = np.ones((self.sentence_len,), dtype=np.float32) * self.char_to_index[MASK_TOKEN]
        for i, ch in enumerate(chars):
            if ch not in self.index_to_char:
                indices[i] = self.char_to_index[DEFAULT_TOKEN]
            else:
                indices[i] = self.char_to_index[ch]
        return chars, indices

    def generate(self, key, batch_size):
        while True:
            for i in range(batch_size, len(self.raw[key]), batch_size):
                x = np.ones((batch_size, self.sentence_len), dtype=int) * self.char_to_index[MASK_TOKEN]
                y = np.zeros((batch_size, self.sentence_len, len(self.index_to_char)),
                             dtype=np.float32)
                mask = np.zeros((batch_size, self.sentence_len), dtype=np.float32)

                value = self.raw[key][i-batch_size:i]

                for j, doc in enumerate(value):
                    chars = list(doc)[:self.sentence_len - 1]
                    chars.append(SENTENCE_END_TOKEN)
                    for k, c in enumerate(chars):
                        x[j, k] = self.char_to_index[c]
                for j, doc in enumerate(value[1:]):
                    chars = list(doc)[:self.sentence_len - 1]
                    chars.append(SENTENCE_END_TOKEN)
                    for k, c in enumerate(chars):
                        y[j, k] = to_hot_coded(self.char_to_index[c], len(self.index_to_char))
                        mask[j, k] = 1.0

                yield x, y, mask

    def steps_per_epoch(self, key, batch_size):
        return int(math.ceil(float(len(self.raw[key])) / float(batch_size)))

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
            'raw': self.raw
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

        loader = DataLoader(paths, path_alias, nlines,
                            embed_type, embed_dims, sentence_len)

        try:
            char_to_index = config['char_to_index']
            index_to_char = config['index_to_char']
            raw = config['raw']
            loader.char_to_index = char_to_index
            loader.index_to_char = index_to_char
            loader.raw = raw
            loader._create_data()
        except KeyError:
            print('WARNING: Can\'t locate loader indices. Reloading might cause inaccuracies...')
            loader.load_data()

        return loader

    @property
    def doc_count(self):
        total = 0
        for v in self.raw.values():
            total += len(v)
        return total

    def get_embed_type(self):
        return 'hot_coded' if self._embed_func == self._hot_coded_embed else 'uniform'
