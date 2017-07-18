import numpy             as np
import scipy.linalg.blas as blas

from bmcixx import bmci_expectation

class BMCI:
    def __init__(self, Y, x, s):
        # Ensure Y is in C order.
        if (Y.flags.f_contiguous):
            self.Y = np.array(Y, order='C')
        else:
            self.Y = Y

        if x.ndim == 1:
            self.x = x.reshape(-1,1)
        else:
            self.x = x
        self.s = s
        self.m = Y.shape[0]
        self.n = Y.shape[1]

    def expectation(self, Y):
        # Ensure Y is in C order.
        if (Y.flags.f_contiguous):
            return bmci_expectation(self.Y, self.x, self.s, np.array(Y, order='C'))
        else:
            return bmci_expectation(self.Y, self.x, self.s, Y)

    def expectation_native(self, Y):
        x = np.zeros((Y.shape[0],1))
        for i in range(Y.shape[0]):
            print (i,Y.shape[0])
            dY = self.Y - Y[i,:]
            p  = np.exp(-np.sum(dY * dY / self.s, axis=1, keepdims=True))
            p_sum = np.sum(p)
            x[i,0] = np.sum(np.multiply(self.x, p)) / p_sum
        return x

    def pdf(self, Y, nbins):
        pdfs  = np.zeros((Y.shape[0],nbins))
        edges = np.append(np.asarray([0.0, np.finfo(float).tiny]), np.logspace(-8, 2, nbins-1))
        for i in range(Y.shape[0]):
            dY = self.Y - Y[i,:]
            p  = np.exp(-np.sum(dY * dY / self.s, axis=1, keepdims=True))
            pdfs[i,:],es = np.histogram(self.x, bins=nbins, weights=p, density=True)
        return pdfs, edges

    def cdf(self, Y, nbins):
        pdfs, edges = self.pdf(Y, nbins)
        return np.cumsum(pdfs, axis=1), edges
