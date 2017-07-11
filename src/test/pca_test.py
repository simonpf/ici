import unittest
import os
import numpy as np

from pca import PCA

class TestPCA(unittest.TestCase):
    def setUp(self):
        # Generate random data
        m = 10
        n = 10000
        x       = np.random.randn(m, n)
        xmean   = x.mean(axis = 1)
        self.x = np.transpose(np.transpose(x) - np.transpose(xmean))
        u, s, v = np.linalg.svd(self.x, full_matrices=0)
        self.u = u
        np.save('u.npy', u)

    def test_fromCMatrix(self):
        pca = PCA.fromCMatrix(self.u)
        self.assertTrue(np.all(pca.u == np.transpose(self.u)))

    def test_fromCMatrixFile(self):
        pca = PCA.fromCMatrixFile('u.npy')
        self.assertTrue(np.all(pca.u == np.transpose(self.u)))

    def test_fromRMatrix(self):
        pca = PCA.fromRMatrix(self.u)
        self.assertTrue(np.all(pca.u == self.u))

    def test_fromRMatrixFile(self):
        pca = PCA.fromRMatrixFile('u.npy')
        self.assertTrue(np.all(pca.u == self.u))

    def test_transform(self):
        pca = PCA.fromCMatrix(self.u)
        y   = pca.apply(self.x)
        z   = pca.revert(y)
        self.assertTrue(np.allclose(z, self.x))

    def test_transform_transposed(self):
        pca = PCA.fromCMatrix(self.u)
        y   = pca.apply(np.transpose(self.x))
        z   = pca.revert(y)
        self.assertTrue(np.allclose(z, np.transpose(self.x)))

    def cleanUp(self):
        os.remove('u.py')

if __name__ == '__main__':
    unittest.main()



