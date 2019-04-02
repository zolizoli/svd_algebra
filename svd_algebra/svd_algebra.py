# -*- coding: utf-8 -*-

"""Main module."""
import heapq
import icu
import math
import pickle
from collections import Counter
from os import listdir
from os.path import isfile, join

import scipy
import numpy as np
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import svds


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
        txts = [f for f in listdir(corpus_dir)
                if isfile(join(corpus_dir, f))]
        for txt in txts:
            with open(join(corpus_dir, txt), 'r') as f:
                for l in f:
                    yield l.strip()

    def decompose(self, corpus):
        ## uningram probabilities
        unigram_freqs = {}
        texts = []
        for e in corpus:
            unigram_freqs.update(Counter(e.split()))
            texts.append(e)

        uni_total = sum(unigram_freqs.values())
        unigram_probs = {}
        for k in unigram_freqs.keys():
            unigram_probs[k] = unigram_freqs[k] / uni_total
        del unigram_freqs # free up

        ## initialize skipgram from keras
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(texts)

        word2id = tokenizer.word_index
        #id2word = {v: k for k, v in word2id.items()}
        vocab_size = len(word2id) + 1

        wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc
                in corpus]
        skip_grams = [
            skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid
            in wids]

        # collect skipgram frequencies
        skip_freqs = {}
        for i in range(len(skip_grams)):
            pairs, labels = skip_grams[i][0], skip_grams[i][1]
            for j in range(len(labels)):
                if labels[j] == 1: # add only positive samples
                    if pairs[j] not in skip_freqs.keys():
                        skip_freqs[pairs[j]] = 1
                    else:
                        skip_freqs[pairs[j]] += 1

        skip_total = sum(skip_freqs.values())
        # skipgram probabilities
        skip_probs = {}
        for k in skip_freqs.keys():
            skip_probs[k] = skip_freqs[k] / skip_total
        del skip_freqs  # free up

        ## vocabulary -> sort it!
        collator = icu.Collator.createInstance(icu.Locale('hu_HU.UTF-8')) #TODO: language should be a parameter!
        vocabulary = list(unigram_probs.keys()) # sort vocabulary
        vocabulary = sorted(vocabulary, key=collator.getSortKey)

        # calculate pointwise mutual information for words
        # store it in a scipy lil matrix
        n = len(vocabulary)
        M = scipy.sparse.lil_matrix((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                a = vocabulary[i]
                b = vocabulary[j]
                k = (word2id[a], word2id[b])
                if k in skip_probs.keys():
                    pa = unigram_probs[a]
                    pb = unigram_probs[b]
                    pab = skip_probs[k]
                    pmi = math.log2(pab / pa / pb)
                else:
                    pmi = 0.0
                M[i, j] = pmi # i, j -> row, column

        # singular value decomposition
        U, _, _ = svds(M, k=256) # U, S, V
        return vocabulary, U

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
a.save_model('test', 'tests/models')
print(a.distance('adatfelvétel', 'adat'))
print(a.distance('nép', 'népcsoport'))
print(a.most_similar_n('adat', 10))
# b = SVDAlgebra('tests/models')
# print(b.most_similar_n('szegregáció', 10))
# print(b.distance('adél', 'zsuzsanna'))
# print(b.most_similar_n('cigány', 10))
# print(b.distance('oláh', 'cigány'))
# print(b.distance('beás', 'cigány'))
# print(b.distance('beás', 'oláh'))
# print(b.distance('roma', 'cigány'))
# print(b.distance('beás', 'roma'))
# print(b.distance('oláh', 'roma'))

