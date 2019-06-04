# -*- coding: utf-8 -*-

"""Main module."""
import pyximport

import io
import heapq
import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
from bounter import bounter
from sparsesvd import sparsesvd
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
            with io.open(join(corpus_dir, txt), 'r', encoding='utf-8') as f:
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
        #U, _, V = svds(M, k=256)  # U, S, V
        U, _, V = sparsesvd(M, 300)
        # add context to U
        word_vecs = U.T + V.T
        del U
        del V
        # normalize rows
        word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs,
                                                    axis=0,
                                                    keepdims=True))
        del word_vecs
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
    def clean_word(self, wd):
        if wd.isalpha() and len(wd) > 3:
            return True
        else:
            return False

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
            most_similar_values = heapq.nlargest(n+10, sims)
            most_similar_indices = [sims.index(e) for e
                                    in list(most_similar_values)]
            most_similar_words = [self.vocabulary[e] for e
                                  in most_similar_indices]
            most_similar_words = [e for e in most_similar_words
                                  if self.clean_word(e)]
            if wd in most_similar_words:
                most_similar_words.remove(wd)
            return most_similar_words[:n]
        except Exception as e:
            print(e)

    def similar(self, positive, negative, topn=3):
        """Analogy difference"""
        try:
            wdidx1 = self.vocabulary.index(positive[0])
        except Exception as e:
            print('Not in vocabulary', positive[0])
            return []
        try:
            wdidx2 = self.vocabulary.index(positive[1])
        except Exception as e:
            print('Not in vocabulary', positive[1])
            return []
        try:
            wdidx3 = self.vocabulary.index(negative)
        except Exception as e:
            print('Not in vocabulary', negative)
            return []
        pos1_vector = self.U[wdidx1]
        pos2_vector = self.U[wdidx2]
        neg_vector = self.U[wdidx3]
        target_vector = np.subtract(neg_vector, np.add(pos1_vector, pos2_vector))
        sims = list(self.U.dot(target_vector))
        most_similar_values = heapq.nlargest(topn+10, sims)
        most_similar_indices = [sims.index(e) for e
                                in list(most_similar_values)]
        most_similar_words = [self.vocabulary[e] for e
                              in most_similar_indices]
        most_similar_words = [e for e in most_similar_words if self.clean_word(e)]
        if positive[0] in most_similar_words:
            most_similar_words.remove(positive[0])
        if positive[1] in most_similar_words:
            most_similar_words.remove(positive[1])
        if negative in most_similar_words:
            most_similar_words.remove(negative)
        if len(most_similar_words) > 0:
            return most_similar_words[:topn]
        else:
            return []


    def doesnt_match(self, lst):
        """odd-one-out"""
        word_idxs = [self.vocabulary.index(wd) for wd in lst]
        word_vectors = np.vstack(self.U[i] for i in word_idxs)
        mean = np.mean(word_vectors)
        dists = [abs(cosine(e, mean)) for e in word_vectors]
        mdist = max(dists)
        midx = dists.index(mdist)
        return lst[midx]


# just for testing
a = SVDAlgebra('tests/testdata')
a.save_model('full', 'tests/models')
#TODO:
# initialize an empty object like a = SVDAlgebra()
# read in a cropus like a.read_corpus(path-to-folder)
# read in a model a.load_model(path-to-model)
