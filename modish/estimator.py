import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import logsumexp
import seaborn as sns

from .model import ModalityModel
from .visualize import MODALITY_TO_CMAP, violinplot, _ModelLoglikPlotter

CHANGING_PARAMETERS = np.arange(2, 21, step=1)

TWO_PARAMETER_MODELS = {'bimodal': {'alphas': 1./(CHANGING_PARAMETERS+10),
                                    'betas': 1./(CHANGING_PARAMETERS+10)},
                        'middle': {'alphas': CHANGING_PARAMETERS,
                                   'betas': CHANGING_PARAMETERS}}
ONE_PARAMETER_MODELS = {'~0': {'alphas': 1,
                               'betas': CHANGING_PARAMETERS},
                        '~1': {'alphas': CHANGING_PARAMETERS,
                               'betas': 1}}


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

    def single_feature_fit_transform(self, x):
        logbf_one_param = pd.Series({k: v.logsumexp_logliks(x)
             for k, v in self.one_param_models.items()})
        if (logbf_one_param <= self.logbf_thresh).all():
            logbf_two_param = pd.Series(
                {k: v.logsumexp_logliks(x)
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

    def add_noise_and_assign_modality(self, data,
                                      iteration_per_noise=100,
                                      noise_percentages=np.arange(0, 101,
                                                                  step=10),
                                      figure_prefix='modish_simulation',
                                      violinplot_kws=None):
        """Randomly replace data with uniform data and categorize to modality

        A one-line summary that does not use variable names or the
        function name.

        Several sentences providing an extended description. Refer to
        variables using back-ticks, e.g. `var`.

        Parameters
        ----------
        var1 : array_like
            Array_like means all those objects -- lists, nested lists, etc. --
            that can be converted to an array.  We can also refer to
            variables like `var1`.
        var2 : int
            The type above can either refer to an actual Python type
            (e.g. ``int``), or describe the type of the variable in more
            detail, e.g. ``(N,) ndarray`` or ``array_like``.
        Long_variable_name : {'hi', 'ho'}, optional
            Choices in brackets, default first when optional.

        Returns
        -------
        type
            Explanation of anonymous return value of type ``type``.
        describe : type
            Explanation of return value named `describe`.
        out : type
            Explanation of `out`.

        Other Parameters
        ----------------
        only_seldom_used_keywords : type
            Explanation
        common_parameters_listed_above : type
            Explanation

        Raises
        ------
        BadException
            Because you shouldn't have done that.

        See Also
        --------
        otherfunc : relationship (optional)
        newfunc : Relationship (optional), which could be fairly long, in which
                  case the line wraps here.
        thirdfunc, fourthfunc, fifthfunc

        Notes
        -----
        Notes about the implementation algorithm (if needed).

        This can have multiple paragraphs.

        You may include some math:

        .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

        And even use a greek symbol like :math:`omega` inline.

        References
        ----------
        Cite the relevant literature, e.g. [1]_.  You may also cite these
        references in the notes section above.

        .. [1] O. McNoleg, "The integration of GIS, remote sensing,
           expert systems and adaptive co-kriging for environmental habitat
           modelling of the Highland Haggis using object-oriented, fuzzy-logic
           and neural-network techniques," Computers & Geosciences, vol. 22,
           pp. 585-588, 1996.

        Examples
        --------
        These are written in doctest format, and should illustrate how to
        use the function.

        >>> a=[1,2,3]
        >>> print [x + 3 for x in a]
        [4, 5, 6]
        >>> print "a\n\nb"
        a
        b
        """
        log2bf_dfs = []
        modalities_dfs = []
        data_dfs = []

        violinplot_kws = {} if violinplot_kws is None else violinplot_kws

        width = len(data.columns)*0.75
        alpha = max(0.05, 1./iteration_per_noise)

        for noise_percentage in noise_percentages:
            fig, ax = plt.subplots(figsize=(width, 3))
            for iteration in range(iteration_per_noise):
                if iteration > 0 and noise_percentage == 0:
                    continue
                noisy_data = data.copy()
                shape = (noisy_data.shape[0]*noise_percentage/100,
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

                noisy_data_tidy = noisy_data.unstack()
                noisy_data_tidy = noisy_data_tidy.reset_index()
                noisy_data_tidy = noisy_data_tidy.rename(
                    columns={'level_0': 'Feature ID',
                             'level_1': 'Sample ID',
                             0: '$\Psi$'})
                # noisy_data_tidy.Modality = pd.Categorical(
                #     noisy_data_tidy['Modality'],
                #     categories=['~0', 'middle', '~1', 'bimodal'],
                #     ordered=True)

                violinplot(x='Feature ID', y='$\Psi$', data=noisy_data_tidy,
                           **violinplot_kws)

                log2bf = self.fit_transform(noisy_data)
                modalities = self.assign_modalities(log2bf)

                log2bf_df = log2bf.unstack().reset_index()
                log2bf_df = log2bf_df.rename(
                    columns={'level_0': 'Original Feature ID',
                             'level_1': 'Modality',
                             0: '$\log_2 K$'})
                log2bf_df['Noise Percentage'] = noise_percentage
                log2bf_df['Noise Iteration'] = iteration
                log2bf_df['Feature ID'] = log2bf_df['Original Feature ID'].map(
                    lambda x: renamer[x])
                log2bf_dfs.append(log2bf_df)

                modalities_df = modalities.reset_index()
                modalities_df = modalities_df.rename(
                    columns={'index': 'Original Feature ID',
                             0: 'Assigned Modality'})
                modalities_df['Noise Percentage'] = noise_percentage
                modalities_df['Noise Iteration'] = iteration
                modalities_df['Feature ID'] = \
                    modalities_df['Original Feature ID'].map(
                        lambda x: renamer[x])
                modalities_dfs.append(modalities_df)
            if noise_percentage > 0:
                for c in ax.collections:
                    c.set_alpha(alpha)
            ax.set(ylim=(0, 1), title='{}% Uniform Noise'.format(
                noise_percentage), yticks=(0, 0.5, 1), ylabel='$\Psi$')
            sns.despine()
            fig.tight_layout()
            fig.savefig('{}_noise_percentage_{}.pdf'.format(figure_prefix,
                                                            noise_percentage))

            modalities = pd.concat(modalities_dfs, ignore_index=True)

            log2bf = pd.concat(log2bf_dfs, ignore_index=True)
            m = log2bf.set_index(['Feature ID', 'Modality'])['$\log_2 K$']
            modalities = modalities.join(m, on=['Feature ID',
                                                'Assigned Modality'])


        simulated_data = pd.concat(data_dfs, axis=1)

        return modalities, log2bf, simulated_data
