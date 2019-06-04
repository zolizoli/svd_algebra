from os.path import join

from svd_algebra.svd_algebra import *

a = SVDAlgebra('tests/models')
vocab = set(a.vocabulary)

with open('evaluate/data/questions-words.txt', 'r') as f:
    analogies = {}
    analogy_type = ''
    for l in f:
        l = l.strip().split()
        if l[0].startswith(':'):
            analogy_type = l[1]
            analogies[analogy_type] = []
            pass
        else:
            l = [wd.lower().strip() for wd in l]
            if set(l).intersection(vocab) == set(l):
                analogies[analogy_type].append(l)

out_path = 'evaluate/results'
for k,v in analogies.items():
    fname = k + '.txt'
    with open(join(out_path, fname), 'w') as outfile:
        total = 0
        good = 0
        for e in v:
            try:
                positive = e[:2]
                negative = e[2]
                expected = e[3]
                guess = a.similar(positive, negative, 1)[0]
                if guess == expected:
                    good += 1
                o = '\t'.join(e) + '\t' + guess + '\n'
                outfile.write(o)
                total += 1
            except Exception as e:
                print(e)
                continue
    print('Accuracy of %s is %f' % (k, good/total))
