![Anchor logo](https://raw.githubusercontent.com/YeoLab/anchor/master/logo/v1/logo.png)

[![](https://img.shields.io/travis/YeoLab/anchor.svg)](https://travis-ci.org/YeoLab/anchor)[![](https://img.shields.io/pypi/v/anchor.svg)](https://pypi.python.org/pypi/anchor)[![codecov](https://codecov.io/gh/YeoLab/anchor/branch/master/graph/badge.svg)](https://codecov.io/gh/YeoLab/anchor)

## What is `anchor`?

Anchor is a python package to find unimodal, bimodal, and multimodal features in any data that is normalized between 0 and 1, for example alternative splicing or other percent-based units.

* Free software: BSD license
* Documentation: https://YeoLab.github.io/anchor

## Installation

To install this code, clone this github repository and use `pip` to install

    git clone git@github.com:YeoLab/anchor
    cd anchor
    pip install .  # The "." means "install *this*, the folder where I am now"


To install ``anchor``, we recommend using the `Anaconda Python
Distribution <http://anaconda.org/>`__ and creating an environment.



### Stable (recommended)


To install this code from the Python Package Index, you'll need to specify ``anchor-bio`` (``anchor`` was already taken - boo).

```
pip install anchor-bio
```

### Bleeding-edge (for the brave)

If you want the latest and greatest version, clone this github repository and use `pip` to install

```
git clone git@github.com:YeoLab/anchor
cd anchor
pip install .  # The "." means "install *this*, the folder where I am now"
```


## Usage

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


## History

### 1.0.0 (2017-06-28)

* Updated to Python 3.5, 3.6

### 0.1.0 (2015-07-08)

* First release on PyPI.
