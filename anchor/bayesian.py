import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import logsumexp

from .names import NEAR_ZERO, NEAR_HALF, NEAR_ONE, BIMODAL, NULL_MODEL
from .model import ModalityModel
from .visualize import MODALITY_TO_CMAP, _ModelLoglikPlotter, MODALITY_ORDER

CHANGING_PARAMETERS = np.arange(2, 21, step=1)


TWO_PARAMETER_MODELS = {
    BIMODAL: {'alphas': 1. / (CHANGING_PARAMETERS + 10),
              'betas': 1./(CHANGING_PARAMETERS+10)},
    NEAR_HALF: {'alphas': CHANGING_PARAMETERS,
                'betas': CHANGING_PARAMETERS}}
ONE_PARAMETER_MODELS = {
    NEAR_ZERO: {'alphas': 1, 'betas': CHANGING_PARAMETERS},
    NEAR_ONE: {'alphas': CHANGING_PARAMETERS, 'betas': 1}
}


class BayesianModalities(object):
    """Use Bayesian methods to estimate modalities of splicing events"""

    score_name = '$\log_2 K$'

    def __init__(self, one_parameter_models=ONE_PARAMETER_MODELS,
                 two_parameter_models=TWO_PARAMETER_MODELS,
                 logbf_thresh=10):
        """Initialize an object with models to estimate splicing modality

        Parameters
        ----------
        step : float
            Distance between parameter values
        vmax : float
            Maximum parameter value
        logbf_thresh : float
            Minimum threshold at which the bayes factor difference is defined
            to be significant
        """
        self.logbf_thresh = logbf_thresh
        # self.modality_to_cmap = modality_to_cmap

        self.one_param_models = {k: ModalityModel(**v)
                                 for k, v in one_parameter_models.items()}
        self.two_param_models = {k: ModalityModel(**v)
                                 for k, v in two_parameter_models.items()}
        self.models = self.one_param_models.copy()
        self.models.update(self.two_param_models)

    def _single_feature_logliks_one_step(self, feature, models):
        """Get log-likelihood of models at each parameterization for given data

        Parameters
        ----------
        feature : pandas.Series
            Percent-based values of a single feature. May contain NAs, but only
            non-NA values are used.

        Returns
        -------
        logliks : pandas.DataFrame

        """
        x_non_na = feature[~feature.isnull()]
        if x_non_na.empty:
            return pd.DataFrame()
        else:
            dfs = []
            for name, model in models.items():
                df = model.single_feature_logliks(feature)
                df['Modality'] = name
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def assert_non_negative(x):
        """Ensure all values are greater than zero

        Parameters
        ----------
        x : array_like
            A numpy array

        Raises
        ------
        AssertionError
            If any value in ``x`` is less than 0
        """
        assert np.all(x[np.isfinite(x)] >= 0)

    @staticmethod
    def assert_less_than_or_equal_1(x):
        """Ensure all values are less than 1

        Parameters
        ----------
        x : array_like
            A numpy array

        Raises
        ------
        AssertionError
            If any value in ``x`` are greater than 1
        """
        assert np.all(x[np.isfinite(x)] <= 1)

    def fit(self, data):
        """Get the modality assignments of each splicing event in the data

        Parameters
        ----------
        data : pandas.DataFrame
            A (n_samples, n_events) dataframe of splicing events' PSI scores.
            Must be psi scores which range from 0 to 1

        Returns
        -------
        log2_bayes_factors : pandas.DataFrame
            A (n_modalities, n_events) dataframe of the estimated log2
            bayes factor for each splicing event, for each modality

        Raises
        ------
        AssertionError
            If any value in ``data`` does not fall only between 0 and 1.
        """
        self.assert_less_than_or_equal_1(data.values.flat)
        self.assert_non_negative(data.values.flat)

        if isinstance(data, pd.DataFrame):
            log2_bayes_factors = data.apply(self.single_feature_fit)
        elif isinstance(data, pd.Series):
            log2_bayes_factors = self.single_feature_fit(data)
        log2_bayes_factors.name = self.score_name
        return log2_bayes_factors

    def predict(self, log2_bayes_factors, reset_index=False):
        """Guess the most likely modality for each event

        For each event that has at least one non-NA value, if no modalilites
        have logsumexp'd logliks greater than the log Bayes factor threshold,
        then they are assigned the 'multimodal' modality, because we cannot
        reject the null hypothesis that these did not come from the uniform
        distribution.

        Parameters
        ----------
        log2_bayes_factors : pandas.DataFrame
            A (4, n_events) dataframe with bayes factors for the Psi~1, Psi~0,
            bimodal, and middle modalities. If an event has no bayes factors
            for any of those modalities, it is ignored
        reset_index : bool
            If True, remove the first level of the index from the dataframe.
            Useful if you are using this function to apply to a grouped
            dataframe where the first level is something other than the
            modality, e.g. the celltype

        Returns
        -------
        modalities : pandas.Series
            A (n_events,) series with the most likely modality for each event

        """
        if reset_index:
            x = log2_bayes_factors.reset_index(level=0, drop=True)
        else:
            x = log2_bayes_factors
        if isinstance(x, pd.DataFrame):
            not_na = (x.notnull() > 0).any()
            not_na_columns = not_na[not_na].index
            x.ix[NULL_MODEL, not_na_columns] = self.logbf_thresh
        elif isinstance(x, pd.Series):
            x[NULL_MODEL] = self.logbf_thresh
        return x.idxmax()

    def fit_predict(self, data):
        """Convenience function to assign modalities directly from data"""
        return self.predict(self.fit(data))

    def single_feature_logliks(self, feature):
        """Calculate log-likelihoods of each modality's parameterization

        Used for plotting the estimates of a single feature

        Parameters
        ----------
        featre : pandas.Series
            A single feature's values. All values must range from 0 to 1.

        Returns
        -------
        logliks : pandas.DataFrame
            The log-likelihood the data, for each model, for each
            parameterization

        Raises
        ------
        AssertionError
            If any value in ``x`` does not fall only between 0 and 1.
        """
        self.assert_less_than_or_equal_1(feature.values)
        self.assert_non_negative(feature.values)

        logliks = self._single_feature_logliks_one_step(
            feature, self.one_param_models)

        logsumexps = self.logliks_to_logsumexp(logliks)

        # If none of the one-parameter models passed, try the two-param models
        if (logsumexps <= self.logbf_thresh).all():
            logliks_two_params = self._single_feature_logliks_one_step(
                feature, self.two_param_models)
            logliks = pd.concat([logliks, logliks_two_params])
        return logliks

    @staticmethod
    def logliks_to_logsumexp(logliks):
        return logliks.groupby('Modality')[r'$\log$ Likelihood'].apply(
            logsumexp)

    def single_feature_fit(self, feature):
        """Get the log2 bayes factor of the fit for each modality"""
        if np.isfinite(feature).sum() == 0:
            series = pd.Series(index=MODALITY_ORDER)
        else:
            logbf_one_param = pd.Series(
                {k: v.logsumexp_logliks(feature) for
                 k, v in self.one_param_models.items()})

            # Check if none of the previous features fit
            if (logbf_one_param <= self.logbf_thresh).all():
                logbf_two_param = pd.Series(
                    {k: v.logsumexp_logliks(feature)
                     for k, v in self.two_param_models.items()})
                series = pd.concat([logbf_one_param, logbf_two_param])
                series[NULL_MODEL] = self.logbf_thresh
            else:
                series = logbf_one_param
        series.index.name = 'Modality'
        series.name = self.score_name
        return series

    def plot_single_feature_calculation(self, feature, renamed=''):
        if np.isfinite(feature).sum() == 0:
            raise ValueError('The feature has no finite values')
        logliks = self.single_feature_logliks(feature)
        logsumexps = self.logliks_to_logsumexp(logliks)
        logsumexps[NULL_MODEL] = self.logbf_thresh

        plotter = _ModelLoglikPlotter()
        return plotter.plot(feature, logliks, logsumexps, self.logbf_thresh,
                            renamed=renamed)

    def violinplot(self, n=1000, figsize=None, **kwargs):
        r"""Visualize all modality family members with parameters

        Use violinplots to visualize distributions of modality family members

        Parameters
        ----------
        n : int
            Number of random variables to generate
        kwargs : dict or keywords
            Any keyword arguments to seaborn.violinplot

        Returns
        -------
        fig : matplotlib.Figure object
            Figure object with violins plotted
        """
        if figsize is None:
            nrows = len(self.models)
            width = max(len(m.rvs) for name, m in self.models.items())*0.625
            height = nrows*2.5
            figsize = width, height
        fig, axes = plt.subplots(nrows=nrows, figsize=figsize)

        for ax, model_name in zip(axes, MODALITY_ORDER):
            try:
                model = self.models[model_name]
                cmap = MODALITY_TO_CMAP[model_name]
                palette = cmap(np.linspace(0, 1, len(model.rvs)))
                model.violinplot(n=n, ax=ax, palette=palette, **kwargs)
                ax.set(title=model_name, xlabel='')
            except KeyError:
                continue
        fig.tight_layout()
