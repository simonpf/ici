import numpy as np
"""
PCA module representing a PCA transformation.

A PCA transformation can be constructed from a matrix containing the eigenvalues either
row-wise or column-wise. The resulting PCA object can then be used to apply the transformation
to or its inverse(revert) to a given dataset.
"""
class PCA:
    def __init__(self, mat):
        self.u = mat
        self.n  = mat.shape[0]
        assert self.u.ndim == 2
        assert self.u.shape[0] == self.u.shape[1]

    def apply(self, mat):
        """
        Apply transformation to mat. The transformation is applied by projecting either the last or
        second last dimension of mat onto the eigenvector of the PCA transformation. The exact
        axis to project onto the eigenvectors is determined by comparing the last two dimension
        with the dimension of the eigenvectors. Precedence is given to the last axis (mat.ndim - 1).
        """
        assert (mat.shape[-1] == self.n) or (mat.shape[-2] == self.n)
        if (mat.shape[-1] == self.n):
            dmat = mat
            return np.tensordot(dmat, self.u, [mat.ndim - 1, 1])

        elif (mat.shape[-2] == self.n):
            dmat = mat
            return np.tensordot(self.u, dmat, [1, mat.ndim - 2])

    def revert(self, mat):
        """
        Revert transformation applied to mat. Works as apply but transforms the data back
        into the original space.
        """
        assert (mat.shape[-1] == self.n) or (mat.shape[-2] == self.n)
        if (mat.shape[-1] == self.n):
            return np.tensordot(mat, self.u, [mat.ndim - 1, 0])
        elif (mat.shape[-2] == self.n):
            return np.tensordot(self.u, mat, [0, mat.ndim - 2])

    @classmethod
    def fromCMatrix(cls, mat):
        """
        Create PCA transformation from matrix containing the eigenvalues of the covariance matrix as
        columns.
        """
        assert mat.ndim == 2
        assert mat.shape[0] == mat.shape[1]
        return PCA(np.transpose(mat))

    @classmethod
    def fromRMatrix(cls, mat):
        """
        Create PCA transformation from matrix containing the eigenvalues of the covariance matrix as
        rows.
        """
        assert mat.ndim == 2
        assert mat.shape[0] == mat.shape[1]
        return PCA(mat)

    @classmethod
    def fromCMatrixFile(cls, filename):
        """
        Similar to fromCMatrix but load matrix from file using numpy.load.
        """
        return PCA.fromCMatrix(np.load(filename))

    @classmethod
    def fromRMatrixFile(cls, filename):
        """
        Similar to fromRMatrix but load matrix from file using numpy.load.
        """
        return PCA.fromRMatrix(np.load(filename))

