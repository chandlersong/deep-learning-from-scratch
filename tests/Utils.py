import unittest
import numpy as np

from common.util import im2col


class MyTestCase(unittest.TestCase):
    def test_im2ColDemos(self):
        x = np.random.rand(10, 1, 6, 6)
        print(x.shape)
        col1 = im2col(x, 2, 2, stride=1, pad=0)
        print(col1.shape)
        print(col1)

    def test_reshape(self):
        x = np.arange(6)
        print(x.shape)
        print(x)
        # x = x.reshape((2, 3))
        x = x.reshape(2, -1)
        print(x.shape)
        print(x)
        x = x.transpose((1, 0))
        print(x.shape)
        print(x)

    def test_pad(self):
        x = np.arange(6)
        print(x.shape)
        print(x)
        # x = x.reshape((2, 3))
        x = x.reshape(2, -1)
        print(x.shape)
        print(x)
        x1 = np.pad(x, (5, 0), 'constant')
        print(x1.shape)
        print(x1)
        x2 = np.pad(x, ((0, 0), (5, 5)), 'constant')
        print(x2.shape)
        print(x2)

        if __name__ == '__main__':
            unittest.main()
