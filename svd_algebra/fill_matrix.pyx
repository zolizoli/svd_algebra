import math
from scipy.sparse import coo_matrix

cimport cython
from cython.parallel import prange
cimport numpy as np
import numpy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def count_skipgrams(skipfreqs, vocabulary, idx2word, word_freq):
    cdef np.float32_t skip_total = sum(skipfreqs.values())
    cdef np.float32_t alpha = 0.75
    cdef np.int32_t n = len(vocabulary)
    cdef np.int32_t m = len(skipfreqs)
    cdef dict skip = skipfreqs
    cdef np.ndarray[np.float32_t, ndim = 1] data = numpy.zeros(m, dtype=numpy.float32)
    cdef np.ndarray[np.int32_t, ndim = 1] row = numpy.zeros(m, dtype=numpy.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] col = numpy.zeros(m, dtype=numpy.int32)
    i = 0
    for k, v in skip.items():
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
        i += 1
    M = coo_matrix((data, (row, col)), shape=(n, n))
    return M
