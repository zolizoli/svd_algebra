from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from os import listdir
from os.path import isfile, join

in_path = 'tests/testdata'

corpus = []
ml_files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
for ml_file in ml_files:
    with open(join(in_path, ml_file), 'r') as f:
        txt = f.read().strip()
        corpus.append(txt)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(corpus)

word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}
vocab_size = len(word2id) + 1

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in corpus]
skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid in wids]
t_freqs = {}
for i in range(len(skip_grams)):
    pairs, labels = skip_grams[i][0], skip_grams[i][1]
    for j in range(len(labels)):
        if labels[j] == 1:
            w1 = id2word[pairs[j][0]]
            w2 = id2word[pairs[j][1]]
            t = (w1, w2)
            if t not in t_freqs.keys():
                t_freqs[t] = 1
            else:
                t_freqs[t] += 1

