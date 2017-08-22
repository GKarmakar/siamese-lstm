import numpy as np
import unittest

from loader import TwinLoader


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.loader = TwinLoader(nlines=(20, 20), paths=('data/amazon/train.tsv', 'data/amazon/test.tsv'),
                                 path_alias=('train', 'test'))

    def test_balance(self):
        self.loader.load_data()
        self.loader.balance()
        x = self.loader.X['train']
        y = self.loader.y['train']
        self.assertTupleEqual(np.shape(x[0]), np.shape(x[1]))
        self.assertEqual(np.size(x[0], 0), np.size(y, 0))

        pos = len([i for i in y if i == 0.0])
        neg = len([i for i in y if i != 0.0])
        self.assertGreaterEqual(pos, neg)


if __name__ == '__main__':
    unittest.main()
