import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import logsumexp
import seaborn as sns

from .infotheory import binify, bin_range_strings, jsd
from .model import ModalityModel
from .visualize import MODALITY_TO_CMAP, MODALITY_ORDER, violinplot, _ModelLoglikPlotter

CHANGING_PARAMETERS = np.arange(2, 21, step=1)

TWO_PARAMETER_MODELS = {'bimodal': {'alphas': 1./(CHANGING_PARAMETERS+10),
                                    'betas': 1./(CHANGING_PARAMETERS+10)},
                        'middle': {'alphas': CHANGING_PARAMETERS,
                                   'betas': CHANGING_PARAMETERS}}
ONE_PARAMETER_MODELS = {'~0': {'alphas': 1,
                               'betas': CHANGING_PARAMETERS},
                        '~1': {'alphas': CHANGING_PARAMETERS,
                               'betas': 1}}

class ModalityPredictor(object):

    modalities = MODALITY_ORDER

    def __init__(self, bins=(0, 0.2, 0.8, 1), jsd_thresh=0.1):
        self.bins = bins
        self.jsd_thresh = jsd_thresh

        self.bin_ranges = bin_range_strings(self.bins)
        self.desired_distributions = pd.DataFrame(
            np.array([[1, 0, 0], [0, 1, 0],
                      [0, 0, 1], [0.5, 0, 0.5], [1./3, 1./3, 1./3]]).T,
            index=self.bin_ranges, columns=self.modalities)

    def fit(self, data):
        binned = binify(data, bins=self.bins)
        if isinstance(binned, pd.DataFrame):
            fitted = binned.apply(lambda x: self.desired_distributions.apply(
                lambda y: jsd(x, y)))
            # fitted.loc['multimodal'] = self.jsd_thresh
        else:
            fitted = self.desired_distributions.apply(lambda x: jsd(x, binned))
            # fitted['multimodal'] = self.jsd_thresh
        return fitted

    def predict(self, fitted):
        if fitted.shape[0] != len(self.modalities):
            raise ValueError("This data doesn't look like it had the distance "
                             "between it and the five modalities calculated")
        return fitted.idxmin()

    def fit_predict(self, data):
        return self.predict(self.fit(data))


class ModalityEstimator(object):
    """Use Bayesian methods to estimate modalities of splicing events"""

    # palette = dict(
    # zip(['excluded', 'middle', 'included', 'bimodal', 'uniform'],
    #         sns.color_palette('deep', n_colors=5)))

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

    def assign_modalities(self, log2_bayes_factors, reset_index=False):
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
        not_na = (x.notnull() > 0).any()
        not_na_columns = not_na[not_na].index
        x.ix['multimodal', not_na_columns] = self.logbf_thresh
        return x.idxmax()

    def _fit_transform_one_step(self, data, models):
        non_na = data.count() > 0
        non_na_columns = non_na[non_na].index
        data_non_na = data[non_na_columns]
        if data_non_na.empty:
            return pd.DataFrame()
        else:
            return data_non_na.apply(lambda x: pd.Series(
                {k: v.logsumexp_logliks(x)
                 for k, v in models.items()}), axis=0)

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

    def fit_transform(self, data):
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

        log2_bayes_factors = data.apply(self.single_feature_fit_transform)
        return log2_bayes_factors

    def single_feature_logliks(self, feature):
        """Calculate log-likelihoods of each modality for a single feature

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

    def single_feature_fit_transform(self, feature):
        logbf_one_param = pd.Series({k: v.logsumexp_logliks(feature)
             for k, v in self.one_param_models.items()})

        # Check if none of the previous features fit
        if (logbf_one_param <= self.logbf_thresh).all():
            logbf_two_param = pd.Series(
                {k: v.logsumexp_logliks(feature)
                 for k, v in self.two_param_models.items()})
            series = pd.concat([logbf_one_param, logbf_two_param])
            series['multimodal'] = self.logbf_thresh
        else:
            series = logbf_one_param
        series.index.name = 'Modality'
        series.name = '$\log_2 K$'
        return series

    def plot_single_feature_calculation(self, feature, renamed=''):
        logliks = self.single_feature_logliks(feature)
        logsumexps = self.logliks_to_logsumexp(logliks)
        logsumexps['multimodal'] = self.logbf_thresh

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

        for ax, (model_name, model) in zip(axes, self.models.items()):
            cmap = MODALITY_TO_CMAP[model_name]
            palette = cmap(np.linspace(0, 1, len(model.rvs)))
            model.violinplot(n=n, ax=ax, palette=palette, **kwargs)
            ax.set(title=model_name, xlabel='')
        fig.tight_layout()

    @staticmethod
    def add_noise(data, iteration_per_noise=100,
                  noise_percentages=np.arange(0, 101, step=10), plot=True,
                  violinplot_kws=None, figure_prefix='modish_simulation'):
        data_dfs = []

        violinplot_kws = {} if violinplot_kws is None else violinplot_kws

        width = len(data.columns) * 0.75
        alpha = max(0.05, 1. / iteration_per_noise)

        for noise_percentage in noise_percentages:
            if plot:
                fig, ax = plt.subplots(figsize=(width, 3))
            for iteration in range(iteration_per_noise):
                if iteration > 0 and noise_percentage == 0:
                    continue
                noisy_data = data.copy()
                shape = (noisy_data.shape[0] * noise_percentage / 100,
                         noisy_data.shape[1])
                size = np.product(shape)
                noise_ind = np.random.choice(noisy_data.index,
                                             size=noise_percentage,
                                             replace=False)
                noisy_data.loc[noise_ind] = np.random.uniform(
                    low=0., high=1., size=size).reshape(shape)

                renamer = dict(
                    (col, '{}_noise{}_iter{}'.format(
                        col, noise_percentage, iteration))
                    for col in noisy_data.columns)

                renamed = noisy_data.rename(columns=renamer)
                data_dfs.append(renamed)
                if plot:
                    noisy_data_tidy = noisy_data.unstack()
                    noisy_data_tidy = noisy_data_tidy.reset_index()
                    noisy_data_tidy = noisy_data_tidy.rename(
                        columns={'level_0': 'Feature ID',
                                 'level_1': 'Sample ID',
                                 0: '$\Psi$'})
                    violinplot(x='Feature ID', y='$\Psi$',
                               data=noisy_data_tidy, ax=ax,
                               **violinplot_kws)

            if plot:
                if noise_percentage > 0:
                    for c in ax.collections:
                        c.set_alpha(alpha)
                ax.set(ylim=(0, 1), title='{}% Uniform Noise'.format(
                    noise_percentage), yticks=(0, 0.5, 1), ylabel='$\Psi$')
                sns.despine()
                fig.tight_layout()
                fig.savefig('{}_noise_percentage_{}.pdf'.format(figure_prefix,
                                                                noise_percentage))

        all_noisy_data = pd.concat(data_dfs, axis=1)
        return all_noisy_data
