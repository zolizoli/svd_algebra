===========
SVD Algebra
===========


.. image:: https://img.shields.io/pypi/v/svd_algebra.svg
        :target: https://pypi.python.org/pypi/svd_algebra

.. image:: https://img.shields.io/travis/zolizoli/svd_algebra.svg
        :target: https://travis-ci.org/zolizoli/svd_algebra

.. image:: https://readthedocs.org/projects/svd-algebra/badge/?version=latest
        :target: https://svd-algebra.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A package for generating and exploring old-school
word embeddings.

The package is under development and it is not suitable
for serious work right now.



* Free software: MIT license
* Documentation (link is broken): https://svd-algebra.readthedocs.io.


Pre-trained word embeddings
---------------------------
* A model trained on the English wikipedia dump along with the accompanying vocabulary can be downloaded here https://drive.google.com/open?id=1C1o53_6S4bS-Lw3wBBvaP9011tajZrq1

Evaluation
----------
Coming soon!

Why don't you use word2vec or other neural embeddings?
------------------------------------------------------
* I'd like to learn Cython and linalg, that's the main reason
* For most NLP task, a PMI matrix with some SVD is enough, read Chris Moody's Stop using word2vec post https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/
* A well-parametrised old-school embedding is as good as a neural one according to this https://rare-technologies.com/making-sense-of-word2vec/





Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
