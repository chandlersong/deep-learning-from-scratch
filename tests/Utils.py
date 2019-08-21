import unittest
import numpy as np

from common.util import im2col


class MyTestCase(unittest.TestCase):
    def test_im2ColDemos(self):
        x = np.random.rand(10, 1, 28, 28)
        print(x.shape)
        col1 = im2col(x, 5, 5, stride=1, pad=0)
        self.printData(col1)

    def test_reshape(self):
        x = np.arange(6)
        self.printData(x)
        # x = x.reshape((2, 3))
        x = x.reshape(2, -1)
        self.printData(x)
        x = x.transpose((1, 0))
        self.printData(x)

    def test_reshape_2(self):
        x = np.arange(30)
        self.printData(x)
        x = x.reshape(5, -1);
        self.printData(x)
        x = x.reshape(3, -1)
        self.printData(x)
        x = x.reshape(3, 2, -1)
        self.printData(x)

    def printData(self, x):
        print(x.shape)
        print(x)

    def test_pad(self):
        x = np.arange(6)
        self.printData(x)
        # x = x.reshape((2, 3))
        x = x.reshape(2, -1)
        self.printData(x)
        x1 = np.pad(x, (5, 0), 'constant')
        self.printData(x1)
        x2 = np.pad(x, ((0, 0), (5, 5)), 'constant')
        self.printData(x2)

        if __name__ == '__main__':
            unittest.main()
