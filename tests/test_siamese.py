import numpy as np
import unittest

from loader import TwinLoader
from siamese.model import LSTMSiameseNet


class SiameseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model = LSTMSiameseNet.load('trained/debug')
        print('Finished loading.')

    @unittest.skip('Time saving...')
    def test_creation(self):
        loader = TwinLoader(nlines=(5, 5), paths=('data/amazon/train.tsv', 'data/amazon/test.tsv'),
                            path_alias=('train', 'test'))
        loader.load_data()
        model = LSTMSiameseNet(loader)
        model.compile()
        model.train(epochs=1, callback=False)

    @unittest.skip('Time saving...')
    def test_save(self):
        self.test_model.save()

    def test_distance_str(self):
        r = self.test_model.distance('Tôi muốn 2 soda', 'Anh muốn mua 2 soda')
        self.assertIsInstance(r, np.float32)

    def test_distance_arr(self):
        r = self.test_model.distance(np.asarray([3, 6, 12, 11, 6]),
                                     np.asarray([2, 7, 8, 19, 21, 32, 17]))
        self.assertIsInstance(r, np.float32)


if __name__ == '__main__':
    unittest.main()
