# -*- coding: utf-8 -*-

"""Main module."""
import heapq
import math
import pickle
from collections import Counter
from os import listdir
from os.path import isfile, join

import icu
import numpy as np
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from scipy.spatial.distance import cosine
from scipy.sparse import coo_matrix
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
        unigram_relfreqs = {}
        uni_total = sum(unigram_freqs.values())
        for k,v in unigram_freqs.items():
            relfreq = v / uni_total
            # subsampling and deleting rare words
            if relfreq < 1 - (((10**-5)/relfreq)**0.5) and v > 10:
                unigram_relfreqs[k] = relfreq
        ## vocabulary -> sort it!
        collator = icu.Collator.createInstance(
            icu.Locale('hu_HU.UTF-8'))  # TODO: language should be a parameter!
        vocabulary = list(unigram_relfreqs.keys())  # sort vocabulary
        vocabulary = sorted(vocabulary, key=collator.getSortKey)

        # use only vocabulary words for building the skipgram model
        filtered_texts = []
        for txt in texts:
            t = txt.split()
            t = [wd for wd in txt if wd in vocabulary]
            filtered_texts.append(' '.join(t))
        ## initialize skipgram from keras
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(filtered_texts)

        word2id = tokenizer.word_index
        id2word = {v: k for k, v in word2id.items()}
        vocab_size = len(word2id) + 1

        wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc
                in filtered_texts]
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

        skip_total = sum(skip_freqs.values())

        # calculate pointwise mutual information for words
        # store it in a scipy lil matrix
        n = len(vocabulary)
        data = []
        row = []
        col = []
        alpha = 0.75 # context distribution smoothing
        #TODO: we have a bottleneck here, this is way too slow
        # either we iterate over the dict, so we have enough RAM
        # or we use ProcessPoolExecutor and we don't have enough RAM
        for k, v in skip_freqs.items():
            a = id2word[k[0]]
            b = id2word[k[1]]
            pa = unigram_relfreqs[a]
            pb = unigram_relfreqs[b] ** alpha
            pab = v / skip_total
            npmi = math.log2(pab / (pa * pb))
            data.append(npmi)
            row.append(vocabulary.index(a))
            col.append(vocabulary.index(b))
        M = coo_matrix((data, (row, col)), shape=(n, n))
        # singular value decomposition
        U, _, V = svds(M, k=256) # U, S, V
        word_vecs = U + V.T
        word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs*word_vecs,
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
a.save_model('ady', 'tests/models')
#TODO:
# initialize an empty object like a = SVDAlgebra()
# read in a cropus like a.read_corpus(path-to-folder)
# read in a model a.load_model(path-to-model)
