import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=['no_na', 'with_na'])
def event(request):
    x = np.arange(0, 1.1, .1)
    if request.param == 'no_na':
        return x
    elif request.param == 'with_na':
        x[x < 0.5] = np.nan
        return x


@pytest.fixture
def positive_control():
    """Exact, known positive controls for modality estimation"""
    size = 20
    half = int(size/2)
    psi0 = pd.Series(np.zeros(size), name='excluded')
    psi1 = pd.Series(np.ones(size), name='included')
    middle = pd.Series(0.5 * np.ones(size), name='middle')
    bimodal = pd.Series(np.concatenate([np.ones(half),
                                        np.zeros(half)]),
                        name='bimodal')
    uncategorized = pd.Series(np.linspace(0, 1, size),
                              name='uncategorized')
    df = pd.concat([psi0, psi1, middle, bimodal, uncategorized], axis=1)
    return df
