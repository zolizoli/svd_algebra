import pyximport
pyximport.install()

from bounter import bounter
from nltk.util import skipgrams
import numpy as np
from scipy.sparse.linalg import svds

from svd_algebra.count_skipgrams import count_skipgrams

with open('tests/testdata/mesek.txt', 'r') as f:
    corpus = f.read().split('\n')

skip_counts = bounter(size_mb=1024)
word_counts = bounter(size_mb=1024)
for l in corpus:
    wds = l.split()
    skips = list(skipgrams(wds, 2, 5))
    skips = ['#'.join(t) for t in skips]
    if len(wds) > 0 and len(skips) > 0:
        skip_counts.update(skips)
        word_counts.update(wds)

vocabulary = list(word_counts)
shift = 1
M = count_skipgrams(skip_counts, word_counts, vocabulary, shift)
# singular value decomposition
U, _, V = svds(M, k=256)  # U, S, V
# add context to U
word_vecs = U + V.T
# normalize rows
word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs,
                                            axis=0,
                                            keepdims=True))
