from __future__ import annotations

from numpy import ndarray
from scipy.spatial.distance import cdist, pdist, squareform

from causallearn.utils.KCI.Kernel import Kernel


class DeltaKernel(Kernel):
    def __init__(self, width=1.0):
        Kernel.__init__(self)

    def kernel(self, X: ndarray, Y: ndarray | None = None):
        """
        Parameters
        ----------
        X : array-like
            data array. X is considered as an integer array.
        Y : array-like, optional (default=None)
            data array. Y is considered as an integer array.

        Returns
        -------
        K : array-like
            kernel matrix.
        """

        if Y is None:
            K = (~squareform(pdist(X)).astype(bool)).astype(int)
        else:
            K = (~cdist(X, X).astype(bool)).astype(int)
        return K
