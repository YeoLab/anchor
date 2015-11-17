
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .visualize import violinplot

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
                noise_percentage), yticks=(0, 0.5, 1), ylabel='$\Psi$')
            sns.despine()
            fig.tight_layout()
            fig.savefig('{}_noise_percentage_{}.pdf'.format(figure_prefix,
                                                            noise_percentage))

    all_noisy_data = pd.concat(data_dfs, axis=1)
    return all_noisy_data
