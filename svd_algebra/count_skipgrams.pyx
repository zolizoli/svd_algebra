cimport cython
cimport numpy as np

import math

from bounter import bounter
import numpy
from scipy.sparse import coo_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def count_skipgrams(skipfreqs, vocabulary, word_freq):
    cdef np.float32_t skip_total = sum(skipfreqs.values())
    cdef np.float32_t word_total = sum(word_freq.values())
    cdef np.float32_t alpha = 0.75
    cdef np.int32_t n = len(vocabulary)
    cdef np.int32_t m = len(skipfreqs)

    cdef dict words = word_freq
    cdef dict skip = skipfreqs

    cdef np.ndarray[np.float32_t, ndim=1] data = numpy.zeros(m, dtype=numpy.float32)
    cdef np.ndarray[np.int32_t, ndim=1] row = numpy.zeros(m, dtype=numpy.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col = numpy.zeros(m, dtype=numpy.int32)
    i = 0
    for k, v in skip.items():
        k = k.split('#')
        a = k[0]
        b = k[1]
        pa = words[a] / word_total
        pb = (words[b] / word_total) ** alpha
        pab = v / skip_total
        npmi = math.log2(pab / (pa * pb))
        data[i] = npmi
        row[i] = vocabulary.index(a)
        col[i] = vocabulary.index(b)
        i += 1
    M = coo_matrix((data, (row, col)), shape=(n, n))
    return M
