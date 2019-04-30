import heapq
import math
import pickle
from collections import Counter
from os.path import isfile, join

import numpy as np
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from concurrent.futures import ProcessPoolExecutor

with open('tests/testdata/mesek.txt', 'r') as f:
    texts = [f.read().strip()]

with open('tests/testdata/ady.txt', 'r') as f:
    texts.append(f.read().strip())


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
skip_freqs = dict()
for skip in skip_grams:
    raw_skip = [(p[0], p[1])for p in skip[0]]
    freqs = Counter(raw_skip)
    for k,v in freqs.items():
        if k not in skip_freqs:
            skip_freqs[k] = v
        else:
            skip_freqs[k] += v

skip_total = sum(skip_freqs.values())
alpha = 0.75
n = len(vocabulary)
data = []
row = []
col = []


for k, v in skip_freqs.items():
    a = idx2word[k[0]]
    b = idx2word[k[1]]
    aidx = vocabulary.index(a)
    bidx = vocabulary.index(b)
    pa = word_freq[aidx]
    pb = word_freq[bidx] ** alpha
    pab = v / skip_total
    npmi = math.log2(pab / (pa * pb))
    data.append(npmi)
    row.append(aidx)
    col.append(aidx)


M = coo_matrix((data, (row, col)), shape=(n, n))
U, _, V = svds(M, k=256) # U, S, V
word_vecs = U + V.T
word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs*word_vecs,
                                            axis=0,
                                            keepdims=True))
