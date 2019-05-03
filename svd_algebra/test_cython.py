import pyximport
pyximport.install()

import heapq
import math
import pickle
from collections import Counter
from os.path import isfile, join

import numpy as np
from bounter import bounter
import psutil
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from svd_algebra.fill_matrix import *

with open('tests/testdata/ady.txt', 'r') as f:
    texts = [f.read().strip()]

tokenizer = text.Tokenizer(filters='\t\n')
tokenizer.fit_on_texts(texts)
idx2word = tokenizer.index_word
word2idx = tokenizer.word_index
vocabulary = list(word2idx.keys())
vocab_size = len(vocabulary) + 1

doc_freq = tokenizer.texts_to_matrix(texts, mode='freq')
word_freq = np.average(doc_freq, axis=0)[1:]

wids = [[word2idx[w] for w in text.text_to_word_sequence(doc, filters='\t\n')]
        for doc in texts]
# don't use negative samples and don't shuffle words!!!
skip_grams = [skipgrams(wid,
                        vocabulary_size=vocab_size,
                        window_size=10,
                        negative_samples=0.0,
                        shuffle=False)
              for wid in wids]

# collect skipgram frequencies
memory_to_use = int(psutil.virtual_memory().free / 1024 / 1024)
bounts = bounter(size_mb=memory_to_use)

for skip in skip_grams:
    raw_skip = [(p[0], p[1]) for p in skip[0]]
    bounts.update(raw_skip)
# M = count_skipgrams(skip_freqs, vocabulary, idx2word, word_freq)
#
# # singular value decomposition
# U, _, V = svds(M, k=256)  # U, S, V
# # add context to U
# word_vecs = U + V.T
# # normalize rows
# word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs,
#                                             axis=0,
#                                             keepdims=True))
