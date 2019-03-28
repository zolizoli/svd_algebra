# -*- coding: utf-8 -*-

"""Main module."""
import heapq
import math
import pickle
from collections import Counter
from os import listdir
from os.path import isfile, join

import scipy
import numpy as np
from nltk.util import skipgrams
from scipy.spatial.distance import cosine


class SVDAlgebra:

    def __init__(self, corpus_dir):
        fs = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]
        endings = [f.split('.')[1] for f in fs]
        if 'npy' not in endings and 'p' not in endings:
            self.corpus = self.read_corpus(self.corpus_dir)
            self.unigram_probs, self.normalized_skipgram_probs = self.generate_skipgram_probs(self.corpus)
            self.vocabulary = sorted(self.unigram_probs.keys())
            self.pmi_matrix = self.generate_pmi_matrix()
            self.U = self.decompose_pmi()
        else:
            U = [f for f in fs if f.endswith('.npy')][0]
            vocab = [f for f in fs if f.endswith('.p')][0]
            self.U = np.load(corpus_dir + '/' + U)
            self.vocabulary = pickle.load(open(corpus_dir + '/' + vocab, 'rb'))

    def read_corpus(self, corpus_dir):
        txts = [f for f in listdir(corpus_dir)
                if isfile(join(corpus_dir, f))]
        for txt in txts:
            with open(join(corpus_dir, txt), 'r') as f:
                for l in f:
                    yield l.strip().split()

    def generate_skipgram_probs(self, corpus):
        unigram_freqs = {}
        t_freqs = {}
        for text in corpus:
            unigram_freqs.update(Counter(text))
            t = skipgrams(text, 2, 10)
            for e in t:
                if e not in t_freqs.keys():
                    t_freqs[e] = 1
                else:
                    t_freqs[e] += 1

        uni_total = sum(unigram_freqs.values())
        unigram_probs = {}
        for k in unigram_freqs.keys():
            unigram_probs[k] = unigram_freqs[k] / uni_total

        skip_total = sum(t_freqs.values())
        skip_probs = {}
        for k in t_freqs.keys():
            skip_probs[k] = t_freqs[k] / skip_total

        # normalized skipgrams
        normalized_skipgram_probs = {}
        for k in skip_probs.keys():
            a = k[0]
            b = k[1]
            pa = unigram_probs[a]
            pb = unigram_probs[b]
            pab = skip_probs[k]
            nsp = math.log2(pab / pa / pb)
            normalized_skipgram_probs[k] = nsp

        return unigram_probs, normalized_skipgram_probs

    def generate_pmi_matrix(self):
        m = []
        for wd1 in self.vocabulary:
            row = []
            for wd2 in self.vocabulary:
                k = (wd1, wd2)
                if k in self.normalized_skipgram_probs.keys():
                    pmi = self.normalized_skipgram_probs[k]
                else:
                    pmi = 0.0
                row.append(pmi)
            m.append(np.asarray(row))
        return np.asarray(m)

    def decompose_pmi(self):
        U, S, V = scipy.sparse.linalg.svds(self.pmi_matrix, k=256)
        return U

    def distance(self, wd1, wd2):
        """returns the cosine distance btw wd1 and wd2"""
        try:
            wdidx1 = self.vocabulary.index(wd1)
            wdidx2 = self.vocabulary.index(wd2)
            w1_vector = self.U[wdidx1]
            w2_vector = self.U[wdidx2]
            return cosine(w1_vector, w2_vector)
        except Exception as e:
            print(e)

    def most_similar_n(self, wd, n):
        """returns the n most similar words to wd"""
        try:
            wdidx = self.vocabulary.index(wd)
            w_vector = self.U[wdidx]
            sims = list(self.U.dot(w_vector))
            most_similar_values = heapq.nlargest(n+1, sims)
            most_similar_indices = [sims.index(e) for e in list(most_similar_values)]
            most_similar_words = [self.vocabulary[e] for e in most_similar_indices]
            if wd in most_similar_words:
                most_similar_words.remove(wd)
            return most_similar_words
        except Exception as e:
            print(e)

    def save_model(self, name, dir):
        np.save(dir + '/' + name + '.npy', self.U)
        pickle.dump(self.vocabulary, open(dir + '/' + name + '.p', 'wb'))

    # def load_model(self, name, dir):
    #     self.U = np.load(dir + '/' + name + '.npy', self.U)
    #     self.vocabulary = pickle.load(open(dir + '/' + name + '.p', 'rb'))

    #TODO:
    # - take care of "private" functions
    # - serialize and load serialized model
    # - better corpus handling
    # - faster ngram/skipgram/pmi matrix generation using batches
    # - pmi matrix to scipy sparse matrix
    # - more functions

# just for testing
a = SVDAlgebra('tests/testdata')
a.save_model('test', 'tests/models')
print(a.distance('adatfelvétel', 'adat'))
print(a.distance('nép', 'népcsoport'))
print(a.most_similar_n('adat', 10))
b = SVDAlgebra('tests/models')
print(b.most_similar_n('szegregáció', 10))
print(b.distance('adél', 'zsuzsanna'))
