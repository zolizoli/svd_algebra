import pickle
from gensim.corpora.dictionary import Dictionary
import pyximport
pyximport.install()

from svd_algebra.cooccur_matrix import *
from svd_algebra.create_sspmi import SPPMIFactory
from svd_algebra.sentences import SentenceIter
from svd_algebra.sspmimodel import SPPMIModel
#
# with open('tests/testdata/ady.txt', 'r') as f:
#     corpus.append(f.read().split())

corpus = SentenceIter('tests/testdata/')

dct = Dictionary(corpus)
words2ids = dct.token2id
ids2words = dct.id2token
cooccurance = get_cooccur(corpus, words2ids, 10)


a = SPPMIFactory()
a._save_word2id(dct.token2id, 'models/adywords.p')
a._save_freqs()
a.create(pathtomapping='', pathtocorpus='tests/testdata', corpusname='ady', window=10)
b = SPPMIModel(pathtomapping='adymapping.json',
               pathtovectors='ady-SPPMI-sparse-1-shift.npz',
               pathtocounts='adyfreqs.json')
