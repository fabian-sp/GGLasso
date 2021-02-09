'''
Author: https://github.com/rahuln/lvsglasso

'''


from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.lib.stride_tricks import as_strided

def pack_and_stack(R, S, L, out=None):
    """ vstack R, S, and L matrices at each frequency."""
    F, p, _ = R.shape
    if out is None:
        out = np.empty((F, 3*p, p), dtype=np.complex128)

    for f in range(F):
        out[f] = np.vstack((R[f], S[f], L[f]))
    return out


def partial_coher(iSDM):
    """ Partial coherence from inverse spectral density.

        See Dahlhaus reference for definition.

        iSDM : F x p x p Inverse spectral density matrices.

        Returns:
          
          Pcoh : F x p x p Partial coherence values (in [0, 1]).
    """
    iSDM = np.ascontiguousarray(iSDM)
    F, P, _ = iSDM.shape
    iSDM_dg = as_strided(iSDM, strides=(P*P*iSDM.itemsize, (P+1)*iSDM.itemsize),
                             shape=(F,P))
    Pcoh = np.empty_like(iSDM)
    for f in range(F):
        invchol = np.diag(1. / np.sqrt(iSDM_dg[f]))
        Pcoh[f] = -np.dot(invchol, np.dot(iSDM[f], invchol))
    return np.abs(Pcoh)
