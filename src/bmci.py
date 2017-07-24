import numpy             as np
import scipy.linalg.blas as blas

from bmcixx import bmci_initialize, bmci_finalize, bmci_expectation, bmci_pdf

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
        self.hist_min = self.x[self.x > 0.0].min()
        self.hist_max = self.x.max()

        # Initialize bmci object.
        bmci_initialize(self.Y, self.x, self.s, 64)
        bmci_initialize(self.Y, self.x, self.s, 32)

    def expectation(self, Y):
        # Ensure Y is in C order.
        if (Y.flags.f_contiguous):
            return bmci_expectation(np.array(Y, order='C'))
        else:
            return bmci_expectation(Y)

    def expectation_float(self, Y):
        # Ensure Y is in C order.
        if (Y.flags.f_contiguous):
            return bmci_expectation(np.array(Y, order='C', dtype=np.float32))
        else:
            return bmci_expectation(Y)

    def expectation_native(self, Y):
        x = np.zeros((Y.shape[0],1))
        for i in range(Y.shape[0]):
            dY = self.Y - Y[i,:]
            p  = np.exp(-np.sum(dY * dY / self.s, axis=1, keepdims=True))
            p_sum = np.sum(p)
            x[i,0] = np.sum(np.multiply(self.x, p)) / p_sum
        return x

    def pdf(self, Y, nbins):
        # Ensure Y is in C order.
        if (Y.flags.f_contiguous):
            return bmci_pdf(np.array(Y, order='C'), nbins)
        else:
            return bmci_pdf(Y, nbins)

    def pdf_float(self, Y, nbins):
        return bmci_pdf(np.array(Y, order='C', dtype=np.float32), nbins)

    def pdf_native(self, Y, nbins):
        pdfs          = np.zeros((Y.shape[0],nbins))
        edges         = np.linspace(np.log(self.hist_min), np.log(self.hist_max), nbins)
        zero_inds     = self.x == 0.0
        non_zero_inds = self.x != 0.0

        for i in range(Y.shape[0]):
            dY = self.Y - Y[i,:]
            p  = np.exp(-np.sum(dY * dY / self.s, axis=1, keepdims=True))
            pdfs[i,0]    = p[zero_inds].sum()
            pdfs[i,1:], _ = np.histogram(np.log(self.x[non_zero_inds]), bins=edges, weights=p[non_zero_inds], density=True)

        return pdfs, edges

    def cdf(self, Y, nbins):
        pdfs, edges = self.pdf(Y, nbins)
        return np.cumsum(pdfs, axis=1), edges
