import pyximport
pyximport.install()

from nltk.util import skipgrams
from bounter import bounter
from scipy.sparse.linalg import svds

from svd_algebra.count_skipgrams import count_skipgrams

with open('tests/testdata/ady.txt', 'r') as f:
    corpus = f.read().split('\n')

skip_counts = bounter(size_mb=1024)
word_counts = bounter(size_mb=1024)
words_total = 0
skip_total = 0
for l in corpus:
    wds = l.split()
    skips = list(skipgrams(wds, 2, 10))
    skips = ['#'.join(t) for t in skips]
    if len(wds) > 0 and len(skips) > 0:
        skip_counts.update(skips)
        word_counts.update(wds)
        words_total += len(wds)
        skip_total += len(skips)

skip_freqs = dict(skip_counts.items())
word_freqs = dict(word_counts.items())

vocabulary = list(word_freqs.keys())

M = count_skipgrams(skip_freqs, vocabulary, word_freqs)

# singular value decomposition
U, _, V = svds(M, k=256)  # U, S, V
# add context to U
word_vecs = U + V.T
# normalize rows
word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs,
                                            axis=0,
                                            keepdims=True))
