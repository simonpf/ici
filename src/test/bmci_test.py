import unittest
import os
import numpy as np

from bmci import BMCI

class TestBMCI(unittest.TestCase):
    def setUp(self):
        # Generate random data
        self.m = 1000
        self.n = 10
        self.y = np.random.randn(self.m, self.n)
        self.x = np.abs(np.random.randn(self.m, 1))
        self.y_test = np.random.randn(1, self.n)
        self.s = 10.0 * np.ones(10)
        self.bmci = BMCI(self.y, self.x, self.s)

    def test_expectation(self):
        x_1 = self.bmci.expectation(self.y_test)
        x_2 = self.bmci.expectation_float(self.y_test)
        x_3 = self.bmci.expectation_native(self.y_test)
        self.assertTrue(np.all(np.isclose(x_1, x_2)))

    def test_pdf(self):
        _,x_1 = self.bmci.pdf(self.y_test, 10)
        _,x_2 = self.bmci.pdf_float(self.y_test, 10)
        x_3,_ = self.bmci.pdf_native(self.y_test, 10)
        self.assertTrue(np.all(np.isclose(x_1, x_2)))

if __name__ == '__main__':
    unittest.main()


