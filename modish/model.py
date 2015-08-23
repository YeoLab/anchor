"""
Model splicing events as beta distributions
"""

from collections import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.misc import logsumexp
import seaborn as sns


class ModalityModel(object):
    """Object to model modalities from beta distributions"""

    def __init__(self, alphas, betas, value_name='$\Psi$'):
        """Model a family of beta distributions

        Parameters
        ----------
        alphas : float or list-like
            List of values for the alpha parameter of the Beta distribution. If
            this is a single value (not a list), it will be assumed that this
            value is constant, and will be propagated through to have as many
            values as the "betas" parameter
        betas : float or list-like
            List of values for the alpha parameter of the Beta distribution. If
            this is a single value (not a list), it will be assumed that this
            value is constant, and will be propagated through to have as many
            values as the "alphas" parameter
        value_name : str, optional
            Name of the value you're estimating. Originally developed for
            alternative splicing "percent spliced in"/"Psi" scores, the default
            is the Greek letter Psi
        """
        if not isinstance(alphas, Iterable) and not isinstance(betas,
                                                               Iterable):
            alphas = [alphas]
            betas = [betas]

        self.value_name = value_name

        self.alphas = np.array(alphas) if isinstance(alphas, Iterable) \
            else np.ones(len(betas)) * alphas
        self.betas = np.array(betas) if isinstance(betas, Iterable) \
            else np.ones(len(alphas)) * betas

        self.rvs = [stats.beta(a, b) for a, b in
                    zip(self.alphas, self.betas)]
        self.scores = np.ones(self.alphas.shape).astype(float)
        self.prob_parameters = self.scores/self.scores.sum()

    def __eq__(self, other):
        """Test equality with other model"""
        return np.all(self.alphas == other.alphas) \
            and np.all(self.betas == other.betas) \
            and np.all(self.prob_parameters == other.prob_parameters)

    def __ne__(self, other):
        """Test not equality with other model"""
        return not self.__eq__(other)

    def logliks(self, x):
        """Calculate log-likelihood of a feature x for each model

        Converts all values that are exactly 1 or exactly 0 to 0.999 and 0.001
        because they are out of range of the beta distribution.

        Parameters
        ----------
        x : numpy.array-like
            A single vector to estimate the log-likelihood of the models on

        Returns
        -------
        logliks : numpy.array
            Log-likelihood of these data in each member of the model's family
        """
        x = x.copy()
        x[x == 0] = 0.001
        x[x == 1] = 0.999

        return np.array([np.log(prob) + rv.logpdf(x[np.isfinite(x)]).sum()
                         for prob, rv in
                         zip(self.prob_parameters, self.rvs)])

    def logsumexp_logliks(self, x):
        """Calculate how well this model fits these data

        Parameters
        ----------
        x : numpy.array-like
            A single vector to estimate the log-likelihood of the models on

        Returns
        -------
        logsumexp_logliks : float
            Total log-likelihood of this model given this data
        """
        return logsumexp(self.logliks(x))

    def violinplot(self, n=1000, **kwargs):
        """Plot violins of each distribution in the model family

        Parameters
        ----------
        n : int
            Number of random variables to generate
        kwargs : dict or keywords
            Any keyword arguments to seaborn.violinplot

        Returns
        -------
        ax : matplotlib.Axes object
            Axes object with violins plotted
        """
        kwargs.setdefault('bw', 0.2)
        kwargs.setdefault('inner', None)
        kwargs.setdefault('scale', 'width')
        kwargs.setdefault('palette', 'Purples')

        dfs = []

        for rv in self.rvs:
            psi = rv.rvs(n)
            df = pd.Series(psi, name=self.value_name).to_frame()
            df['parameters'] = '$\\alpha$ = {:.2f}\n$\\beta$ = {:.2f}'.format(
                *rv.args)
            dfs.append(df)
        bimodal_df = pd.concat(dfs)

        if 'ax' not in kwargs:
            fig, ax = plt.subplots(figsize=(len(self.alphas) / 1.25, 4))
        else:
            ax = kwargs.pop('ax')
        ax = sns.violinplot(x='parameters', y=self.value_name, data=bimodal_df,
                            ax=ax, **kwargs)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)
        return ax
