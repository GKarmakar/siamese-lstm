import unittest

from neuralknn.classifier import NeuralKNN


class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model = NeuralKNN('trained/debug')
        print('Finished loading.')

    def test_fit(self):
        self.test_model.fit()

    def test_predict(self):
        r = self.test_model.predict('cho anh xem đơn hàng với em')
        self.assertIsInstance(r, str)

if __name__ == '__main__':
    unittest.main()