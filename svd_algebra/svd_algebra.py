# -*- coding: utf-8 -*-

"""Main module."""
import pyximport
pyximport.install()

import heapq
import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
from bounter import bounter
from nltk.util import skipgrams
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import svds

from svd_algebra.count_skipgrams import count_skipgrams


class SVDAlgebra:

    def __init__(self, corpus_dir):
        #TODO: we need a much nicer and more pythonic __init_function
        fs = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]
        endings = [f.split('.')[1] for f in fs]
        if 'npy' not in endings and 'p' not in endings:
            self.corpus = self.read_corpus(corpus_dir)
            self.vocabulary, self.U = self.decompose(self.corpus)
        else:
            #TODO: use load_model instead of this part of init
            U = [f for f in fs if f.endswith('.npy')][0]
            vocab = [f for f in fs if f.endswith('.p')][0]
            self.U = np.load(corpus_dir + '/' + U)
            self.vocabulary = pickle.load(open(corpus_dir + '/' + vocab, 'rb'))

    ###########################################################################
    #####                     Initialize object                           #####
    ###########################################################################
    def read_corpus(self, corpus_dir):
        #TODO: read corpus files line by line
        txts = [f for f in listdir(corpus_dir)
                if isfile(join(corpus_dir, f))]
        for txt in txts:
            with open(join(corpus_dir, txt), 'r') as f:
                for l in f:
                    yield l.strip()

    def decompose(self, corpus):

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
        shift = 1 # shift 1 does nothing since log(1) == 0.0
        M = count_skipgrams(skip_counts, word_counts, vocabulary, shift)
        #TODO: eigen something trick
        # singular value decomposition
        U, _, V = svds(M, k=256)  # U, S, V
        # add context to U
        word_vecs = U + V.T
        # normalize rows
        word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs,
                                                    axis=0,
                                                    keepdims=True))
        return vocabulary, word_vecs_norm

    ###########################################################################
    #####                      Serialize model                            #####
    ###########################################################################
    #TODO: add load model after the better __init__
    def save_model(self, name, dir):
        np.save(dir + '/' + name + '.npy', self.U)
        pickle.dump(self.vocabulary, open(dir + '/' + name + '.p', 'wb'))

    # def load_model(self, name, dir):
    #     self.U = np.load(dir + '/' + name + '.npy', self.U)
    #     self.vocabulary = pickle.load(open(dir + '/' + name + '.p', 'rb'))
    ###########################################################################
    #####                        Word Algebra                             #####
    ###########################################################################
    def distance(self, wd1, wd2):
        """returns the cosine distance btw wd1 and wd2"""
        try:
            wdidx1 = self.vocabulary.index(wd1)
            wdidx2 = self.vocabulary.index(wd2)
            w1_vector = self.U[wdidx1]
            w2_vector = self.U[wdidx2]
            return min(cosine(w1_vector, w2_vector), 1.0) # nicer distance
        except Exception as e:
            print(e)

    def most_similar_n(self, wd, n):
        """returns the n most similar words to wd"""
        try:
            wdidx = self.vocabulary.index(wd)
            w_vector = self.U[wdidx]
            sims = list(self.U.dot(w_vector))
            most_similar_values = heapq.nlargest(n+1, sims)
            most_similar_indices = [sims.index(e) for e
                                    in list(most_similar_values)]
            most_similar_words = [self.vocabulary[e] for e
                                  in most_similar_indices]
            if wd in most_similar_words:
                most_similar_words.remove(wd)
            return most_similar_words
        except Exception as e:
            print(e)

    def similar(self, positive, negative, topn=3):
        """Analogy difference"""
        #TODO: implement function
        pass

    def doesnt_match(self, lst):
        """odd-one-out"""
        #TODO: finish implementation
        word_idxs = [self.vocabulary.index(wd) for wd in lst]
        word_vectors = np.vstack(self.U[i] for i in word_idxs)
        mean = np.mean(word_vectors)
        pass

    #TODO:
    # - better save/load corpus
    # - more functions

# just for testing
a = SVDAlgebra('tests/testdata')
a.save_model('full', 'tests/models')
#TODO:
# initialize an empty object like a = SVDAlgebra()
# read in a cropus like a.read_corpus(path-to-folder)
# read in a model a.load_model(path-to-model)
