'''
Author: https://github.com/rahuln/lvsglasso

'''

from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.lib.stride_tricks import as_strided


def soft_thresh_all(A, lam):
    # NOTE: This only works for real-valued data.
    """ soft-threshold all values in A by lam """
    D = A.shape[0]
    S = np.sign(A)*np.maximum(np.abs(A) - lam, 0)
    return S

def soft_thresh(A, lam):
    # NOTE: This only works for real-valued data.
    # NOTE: Leaves diagonal untouched, see `soft_thresh_all` to soft-threshold
    # entire matrix.
    """ soft-threshold all off-diagonal entries in A by lam """
    D = A.shape[0]
    A_dg = np.diagonal(A).copy()
    S = np.sign(A)*np.maximum(np.abs(A) - lam, 0)
    S[np.diag_indices(D)] = A_dg
    return S

# This is soft-thresholding for complex-valued data.
def block_thresh_entrywise(A_, kapp):
    A_norm = np.abs(A_)
    scale = np.maximum(1. - kapp/A_norm, 0)
    return A_ * scale


def block_thresh(A_, kappa):
    F, P, _ = A_.shape
    A_norm = 1./np.sqrt(F) * np.linalg.norm(A_, axis=0)
    scale = np.maximum(1. - kappa/A_norm, 0)
    res = A_ * scale
    return res

def block_thresh_2d(A, kappa):
    F, p = A.shape
    #A_norm = 1./np.sqrt(F)*np.linalg.norm(A, axis=0)
    A_norm = np.linalg.norm(A, axis=0)
    scale = np.maximum(1. - kappa/A_norm, 0)
    res = A * scale[np.newaxis,:]
    return res


def prox_nuc(S, lam):
    """ proximal operator for the nuclear norm (soft-thresholding of the singular values in S) """
    Sig = np.maximum(S - lam, 0)
    return Sig
