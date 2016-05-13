
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import six

from .visualize import violinplot, MODALITY_ORDER, MODALITY_TO_COLOR, barplot


def add_noise(data, iteration_per_noise=100,
              noise_percentages=np.arange(0, 101, step=10), plot=True,
              violinplot_kws=None, figure_prefix='anchor_simulation'):

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
                noise_percentage), yticks=(0, 0.5, 1), ylabel='$\Psi$',
                   xlabel='')
            plt.setp(ax.get_xticklabels(), rotation=90)
            sns.despine()
            fig.tight_layout()
            fig.savefig('{}_noise_percentage_{}.pdf'.format(figure_prefix,
                                                            noise_percentage))

    all_noisy_data = pd.concat(data_dfs, axis=1)
    return all_noisy_data


class ModalityEvaluator(object):

    def __init__(self, estimator, data, waypoints, fitted, predicted):
        self.estimator = estimator
        self.data = data
        self.predicted = predicted
        self.fitted = fitted
        self.waypoints = waypoints


def evaluate_estimator(estimator, data, waypoints=None, figure_prefix=''):
    #
    # estimator.violinplot(n=1e3)
    # fig = plt.gcf()
    # for ax in fig.axes:
    #     ax.set(yticks=[0, 0.5, 1], xlabel='')
    # #     xticklabels =
    # #     ax.set_xticklabels(fontsize=20)
    # fig.tight_layout()
    # sns.despine()
    # fig.savefig('{}_modality_parameterization.pdf'.format(figure_prefix))

    fitted = estimator.fit(data)
    predicted = estimator.predict(fitted)
    predicted.name = 'Predicted Modality'

    fitted_tidy = fitted.stack().reset_index()
    fitted_tidy = fitted_tidy.rename(
        columns={'level_1': 'Feature ID', 'level_0': "Modality",
                 0: estimator.score_name}, copy=False)

    predicted_tidy = predicted.to_frame().reset_index()
    predicted_tidy = predicted_tidy.rename(columns={'index': 'Feature ID'})
    predicted_tidy = predicted_tidy.merge(
        fitted_tidy, left_on=['Feature ID', 'Predicted Modality'],
        right_on=['Feature ID', 'Modality'])

    # Make categorical so they are plotted in the correct order
    predicted_tidy['Predicted Modality'] = \
        pd.Categorical(predicted_tidy['Predicted Modality'],
                       categories=MODALITY_ORDER, ordered=True)
    predicted_tidy['Modality'] = \
        pd.Categorical(predicted_tidy['Modality'],
                       categories=MODALITY_ORDER, ordered=True)

    grouped = data.groupby(predicted, axis=1)

    size = 5

    fig, axes = plt.subplots(figsize=(size*0.75, 8), nrows=len(grouped))

    for ax, (modality, df) in zip(axes, grouped):
        random_ids = np.random.choice(df.columns, replace=False, size=size)
        random_df = df[random_ids]

        tidy_random = random_df.stack().reset_index()
        tidy_random = tidy_random.rename(columns={'level_0': 'sample_id',
                                                  'level_1': 'event_id',
                                                  0: '$\Psi$'})
        sns.violinplot(x='event_id', y='$\Psi$', data=tidy_random,
                       color=MODALITY_TO_COLOR[modality], ax=ax,
                       inner=None, bw=0.2, scale='width')
        ax.set(ylim=(0, 1), yticks=(0, 0.5, 1), xticks=[], xlabel='',
               title=modality)
    sns.despine()
    fig.tight_layout()
    fig.savefig('{}_random_estimated_modalities.pdf'.format(figure_prefix))

    g = barplot(predicted_tidy, hue='Modality')
    g.savefig('{}_modalities_barplot.pdf'.format(figure_prefix))

    plot_best_worst_fits(predicted_tidy, data, modality_col='Modality',
                         score=estimator.score_name)
    fig = plt.gcf()
    fig.savefig('{}_best_worst_fit_violinplots.pdf'.format(figure_prefix))

    fitted.to_csv('{}_fitted.csv'.format(figure_prefix))
    predicted.to_csv('{}_predicted.csv'.format(figure_prefix))

    result = ModalityEvaluator(estimator, data, waypoints, fitted, predicted)

    return result


def plot_best_worst_fits(assignments_df, data, modality_col='Modality',
                         score='$\log_2 K$'):
    """Violinplots of the highest and lowest scoring of each modality"""
    ncols = 2
    nrows = len(assignments_df.groupby(modality_col).groups.keys())

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(nrows*4, ncols*6))

    axes_iter = axes.flat

    fits = 'Highest', 'Lowest'

    for modality, df in assignments_df.groupby(modality_col):
        df = df.sort_values(score)

        color = MODALITY_TO_COLOR[modality]

        for fit in fits:
            if fit == 'Highest':
                ids = df['Feature ID'][-10:]
            else:
                ids = df['Feature ID'][:10]
            fit_psi = data[ids]
            tidy_fit_psi = fit_psi.stack().reset_index()
            tidy_fit_psi = tidy_fit_psi.rename(columns={'level_0': 'Sample ID',
                                                        'level_1':
                                                            'Feature ID',
                                                        0: '$\Psi$'})
            if tidy_fit_psi.empty:
                continue
            ax = six.next(axes_iter)
            violinplot(x='Feature ID', y='$\Psi$', data=tidy_fit_psi,
                       color=color, ax=ax)
            ax.set(title='{} {} {}'.format(fit, score, modality), xticks=[])
    sns.despine()
    fig.tight_layout()
