.. image:: https://raw.githubusercontent.com/YeoLab/anchor/master/logo/v1/logo.png

===============================
anchor: Find modalities in data
===============================

.. image:: https://img.shields.io/travis/YeoLab/anchor.svg
        :target: https://travis-ci.org/YeoLab/anchor

.. image:: https://img.shields.io/pypi/v/anchor.svg
        :target: https://pypi.python.org/pypi/anchor


Anchor is a python package to find unimodal, bimodal, and multimodal features in any data that is normalized between 0 and 1, for example alternative splicing or other percent-based units.

* Free software: BSD license
* Documentation: https://YeoLab.github.io/anchor.

Features
--------

* TODO

Installation
------------

To install ``outrigger``, we recommend using the `Anaconda Python
Distribution <http://anaconda.org/>`__ and creating an environment.



Stable
~~~~~~

To install this code from the Python Package Index, you'll need to specify ``anchor-bio`` (``anchor`` was already taken - boo).

::

    pip install anchor-bio


Bleeding-edge
~~~~~~~~~~~~~

If you want the latest and greatest version, clone this github repository and use `pip` to install

::

    git clone git@github.com:YeoLab/anchor
    cd anchor
    pip install .  # The "." means "install *this*, the folder where I am now"


Usage
-----

`anchor` was structured like `scikit-learn`, where if you want the "final
answer" of your estimator, you use `fit_transform()`, but if you want to see the
intermediates, you use `fit()`.

If you want the modality assignments for your data, first make sure that you
have a `pandas.DataFrame`, here it is called `data`, in the format (samples,
features). This uses a log2 Bayes Factor cutoff of 5, and the default Beta
distribution parameterizations (shown [here]())

```python
import anchor

bm = anchor.BayesianModalities()
modalities = bm.fit_transform(data)
```

If you want to see all the intermediate Bayes factors, then you can do:

```python
import anchor

bm = anchor.BayesianModalities()
bayes_factors = bm.fit(data)
```

