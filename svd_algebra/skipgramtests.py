import heapq
import math
import pickle
from collections import Counter
from os import listdir, cpu_count
from os.path import isfile, join

import icu
import numpy as np
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from scipy.spatial.distance import cosine
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from concurrent.futures import ProcessPoolExecutor

with open('tests/testdata/ady.txt', 'r') as f:
    texts = [f.read().strip()]

unigram_freqs = {}
for e in texts:
    unigram_freqs.update(Counter(e.split()))
uni_total = sum(unigram_freqs.values())

## vocabulary -> sort it!
collator = icu.Collator.createInstance(
    icu.Locale('hu_HU.UTF-8'))  # TODO: language should be a parameter!
vocabulary = list(unigram_freqs.keys())  # sort vocabulary
vocabulary = sorted(vocabulary, key=collator.getSortKey)


## initialize skipgram from keras
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(texts)

word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}
vocab_size = len(word2id) + 1

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc
        in texts]
# don't use negative samples and don't shuffle words!!!
skip_grams = [skipgrams(wid,
                        vocabulary_size=vocab_size,
                        window_size=10,
                        negative_samples=0.0,
                        shuffle=False)
              for wid in wids]

# collect skipgram frequencies
skip_freqs = {}
for i in range(len(skip_grams)):
    pairs, _ = skip_grams[i][0], skip_grams[i][1]
    for p in pairs:
        if (p[0], p[1]) not in skip_freqs.keys():
            skip_freqs[(p[0], p[1])] = 1
        else:
            skip_freqs[(p[0], p[1])] += 1

# calculate pointwise mutual information for words
# store it in a scipy lil matrix
skip_total = sum(skip_freqs.values())
n = len(vocabulary)
ks = list(skip_freqs.keys())
data_len = len(ks)
data = [0.0] * data_len
row = [0.0] * data_len
col = [0.0] * data_len


def get_pmi(k):
    a = id2word[k[0]]
    b = id2word[k[1]]
    pa = unigram_freqs[a] / uni_total
    pb = unigram_freqs[b] / uni_total
    pab = skip_freqs[k] / skip_total
    pmi = math.log2(pab / (pa * pb))
    npmi = (pmi / math.log2(pab)) * -1.0
    i = ks.index(k)
    data[i] = npmi
    row[i] = vocabulary.index(a)
    col[i] = vocabulary.index(b)


workers = cpu_count() - 1
with ProcessPoolExecutor(max_workers=workers) as executor:
    executor.map(get_pmi, ks)

# for k, v in skip_freqs.items():
#     a = id2word[k[0]]
#     b = id2word[k[1]]
#     pa = unigram_freqs[a] / uni_total
#     pb = unigram_freqs[b] / uni_total
#     pab = v / skip_total
#     pmi = math.log2(pab / (pa * pb))
#     # normalized pmi https://pdfs.semanticscholar.org/1521/8d9c029cbb903ae7c729b2c644c24994c201.pdf
#     #pmi2 = math.log2(pab**2 / (pa * pb))
#     npmi = (pmi / math.log2(pab)) * -1.0
#     data.append(npmi)
#     row.append(vocabulary.index(a))
#     col.append(vocabulary.index(b))
M = coo_matrix((data, (row, col)), shape=(n, n))
# singular value decomposition
U, _, _ = svds(M, k=256) # U, S, V
# Unorm = U / np.sqrt(np.sum(U * U, axis=1, keepdims=True))
# Vnorm = V / np.sqrt(np.sum(V * V, axis=0, keepdims=True))
# word_vecs = U + V.T
# word_vecs_norm = word_vecs / np.sqrt(
#     np.sum(word_vecs * word_vecs, axis=1, keepdims=True))
