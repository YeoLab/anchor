import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import ModalityModel

CHANGING_PARAMETERS = np.arange(2, 20)

TWO_PARAMETER_MODELS = {'bimodal': {'alphas': 1./CHANGING_PARAMETERS,
                                    'betas': 1./CHANGING_PARAMETERS},
                        'middle': {'alphas': CHANGING_PARAMETERS,
                                   'betas': CHANGING_PARAMETERS}}
ONE_PARAMETER_MODELS = {'~0': {'alphas': 1,
                               'betas': CHANGING_PARAMETERS},
                        '~1': {'alphas': CHANGING_PARAMETERS,
                               'betas': 1}}
MODEL_PALETTES = {'bimodal': 'Purples',
                  'middle': 'Greens',
                  '~0': 'Blues',
                  '~1': 'Reds'}


class ModalityEstimator(object):
    """Use Bayesian methods to estimate modalities of splicing events"""

    # palette = dict(
    # zip(['excluded', 'middle', 'included', 'bimodal', 'uniform'],
    #         sns.color_palette('deep', n_colors=5)))

    def __init__(self, one_parameter_models=ONE_PARAMETER_MODELS,
                 two_parameter_models=TWO_PARAMETER_MODELS,
                 logbf_thresh=3, model_palettes=MODEL_PALETTES):
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
        self.model_palettes = model_palettes

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
            If ``data`` does not fall only between 0 and 1.
        """
        assert np.all(data.values.flat[np.isfinite(data.values.flat)] <= 1)
        assert np.all(data.values.flat[np.isfinite(data.values.flat)] >= 0)

        # Estimate Psi~0/Psi~1 first (only one parameter change with each
        # paramterization)
        logbf_one_param = self._fit_transform_one_step(data,
                                                       self.one_param_models)

        # Take everything that was below the threshold for included/excluded
        # and estimate bimodal and middle (two parameters change in each
        # parameterization
        ind = (logbf_one_param < self.logbf_thresh).all()
        multimodal_columns = ind[ind].index
        data2 = data.ix[:, multimodal_columns]
        logbf_two_param = self._fit_transform_one_step(data2,
                                                       self.two_param_models)
        log2_bayes_factors = pd.concat([logbf_one_param, logbf_two_param],
                                       axis=0)

        # Make sure the returned dataframe has the same number of columns
        empty = data.count() == 0
        empty_columns = empty[empty].index
        empty_df = pd.DataFrame(np.nan, index=log2_bayes_factors.index,
                                columns=empty_columns)
        log2_bayes_factors = pd.concat([log2_bayes_factors, empty_df], axis=1)
        return log2_bayes_factors

    def violinplot(self, n=1000, figsize=None, **kwargs):
        r"""Visualize all modality family members with parameters

        Use violinplots to visualize distributions of modality family members

        Parameters
        ----------


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
            width = max(len(m.rvs) for name, m in self.models.items())
            height = nrows*3
            figsize = width, height
        fig, axes = plt.subplots(nrows=nrows, figsize=figsize)

        for ax, (model_name, model) in zip(axes, self.models.items()):
            palette = self.model_palettes[model_name]
            model.violinplot(n=n, ax=ax, palette=palette, **kwargs)
        return fig
