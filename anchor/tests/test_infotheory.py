import numpy as np
import pandas.util.testing as pdt
import pandas as pd
import pytest
import six


@pytest.fixture
def size():
    return 10


@pytest.fixture
def data(size):
    df = pd.DataFrame(np.tile(np.arange(size), (size, 1)))
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


@pytest.fixture
def df1(data):
    return data


@pytest.fixture
def df2(data):
    return data.T


@pytest.fixture
def p(df1, bins):
    from anchor.infotheory import binify

    return binify(df1, bins)


@pytest.fixture
def q(df2, bins):
    from anchor.infotheory import binify

    return binify(df2, bins)


@pytest.fixture
def bins(size):
    return np.linspace(0, size, num=5)


@pytest.fixture(
    params=((None, ['0-2.5', '2.5-5', '5-7.5', '7.5-10']),
            (':.2f', ['0.00-2.50', '2.50-5.00', '5.00-7.50', '7.50-10.00'])))
def fmt_true(request):
    return request.param


def test_bin_range_strings(bins, fmt_true):
    from anchor.infotheory import bin_range_strings

    fmt, true = fmt_true

    if fmt is None:
        test = bin_range_strings(bins)
    else:
        test = bin_range_strings(bins, fmt=fmt)

    assert test == true


@pytest.fixture(
    params=(pytest.mark.xfail(-np.ones(10)),
            pytest.mark.xfail(np.zeros(10)),
            pytest.mark.xfail(np.ones(10))))
def x(request):
    return request.param


def test__check_prob_dist(x):
    from anchor.infotheory import _check_prob_dist

    # All the tests should raise an error
    _check_prob_dist(x)


def test_binify(df1, bins):
    from anchor.infotheory import binify

    test = binify(df1, bins)

    s = ''',0,1,2,3,4,5,6,7,8,9
0-2.5,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2.5-5,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0
5-7.5,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0
7.5-10,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0'''
    true = pd.read_csv(six.StringIO(s), index_col=0)
    pdt.assert_frame_equal(test, true)


def test_kld(p, q):
    from anchor.infotheory import kld
    test = kld(p, q)

    s = '''0,1.7369655941662063
1,1.7369655941662063
2,1.7369655941662063
3,2.321928094887362
4,2.321928094887362
5,1.7369655941662063
6,1.7369655941662063
7,1.7369655941662063
8,2.321928094887362
9,2.321928094887362'''
    true = pd.read_csv(six.StringIO(s), index_col=0, squeeze=True, header=None)
    true.index.name = None
    true.name = None
    true.index = true.index.astype(str)

    pdt.assert_series_equal(test, true)


def test_jsd(p, q):
    from anchor.infotheory import jsd
    test = jsd(p, q)

    s = '''0,0.49342260576014463
1,0.49342260576014463
2,0.49342260576014463
3,0.6099865470109875
4,0.6099865470109875
5,0.49342260576014463
6,0.49342260576014463
7,0.49342260576014463
8,0.6099865470109875
9,0.6099865470109875'''
    true = pd.read_csv(six.StringIO(s), index_col=0, squeeze=True, header=None)
    true.index.name = None
    true.name = None
    true.index = true.index.astype(str)

    pdt.assert_series_equal(test, true)
