import math
from scipy.sparse import coo_matrix

cimport cython
cimport numpy as np
import numpy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def count_skipgrams(skipfreqs, vocabulary, idx2word, word_freq):
    cdef np.float32_t skip_total = sum(skipfreqs.values())
    alpha = 0.75
    n = len(vocabulary)

    cdef np.ndarray[np.float32_t, ndim = 1] data = numpy.zeros((len(vocabulary)),dtype=numpy.float32)
    cdef np.ndarray[np.int32_t, ndim = 1] row = numpy.zeros((len(vocabulary)),dtype=numpy.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] col = numpy.zeros((len(vocabulary)), dtype=numpy.int32)
    i = 0
    for k, v in skipfreqs.items():
        a = idx2word[k[0]]
        b = idx2word[k[1]]
        aidx = vocabulary.index(a)
        bidx = vocabulary.index(b)
        pa = word_freq[aidx]
        pb = word_freq[bidx] ** alpha
        pab = v / skip_total
        npmi = math.log2(pab / (pa * pb))
        data[i] = npmi
        row[i] = aidx
        col[i] = aidx

    M = coo_matrix((data, (row, col)), shape=(n, n))
    return M
