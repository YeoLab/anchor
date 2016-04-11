# -*- coding: utf-8 -*-
"""See log bayes factors which led to modality categorization"""
import locale

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .names import NEAR_ZERO, NEAR_HALF, NEAR_ONE, BIMODAL, \
    NULL_MODEL


locale.setlocale(locale.LC_ALL, 'en_US')

darkblue, green, red, purple, yellow, lightblue = sns.color_palette('deep')
MODALITY_ORDER = [NEAR_ZERO, BIMODAL, NEAR_ONE, NEAR_HALF, NULL_MODEL]

MODALITY_TO_COLOR = {NEAR_ZERO: lightblue, NEAR_HALF: yellow, NEAR_ONE: red,
                     BIMODAL: purple, NULL_MODEL: 'lightgrey'}
MODALITY_PALETTE = [MODALITY_TO_COLOR[m] for m in MODALITY_ORDER]

MODALITY_TO_CMAP = {
    NEAR_ZERO: sns.light_palette(MODALITY_TO_COLOR[NEAR_ZERO], as_cmap=True),
    NEAR_HALF: sns.light_palette(MODALITY_TO_COLOR[NEAR_HALF], as_cmap=True),
    NEAR_ONE: sns.light_palette(MODALITY_TO_COLOR[NEAR_ONE], as_cmap=True),
    BIMODAL: sns.light_palette(MODALITY_TO_COLOR[BIMODAL], as_cmap=True),
    NULL_MODEL: mpl.cm.Greys}

MODALITY_FACTORPLOT_KWS = dict(hue_order=MODALITY_ORDER,
                               palette=MODALITY_PALETTE)


def violinplot(x=None, y=None, data=None, bw=0.2, scale='width',
               inner=None, ax=None, **kwargs):
    """Wrapper around Seaborn's Violinplot specifically for [0, 1] ranged data

    What's different:
    - bw = 0.2: Sets bandwidth to be small and the same between datasets
    - scale = 'width': Sets the width of all violinplots to be the same
    - inner = None: Don't plot a boxplot or points inside the violinplot
    """
    if ax is None:
        ax = plt.gca()

    sns.violinplot(x, y, data=data, bw=bw, scale=scale, inner=inner, ax=ax,
                   **kwargs)
    ax.set(ylim=(0, 1), yticks=(0, 0.5, 1))
    return ax


class _ModelLoglikPlotter(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(5 * 2, 4))
        self.ax_violin = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=1)
        self.ax_loglik = plt.subplot2grid((3, 5), (0, 1), rowspan=3, colspan=3)
        self.ax_bayesfactor = plt.subplot2grid((3, 5), (0, 4), rowspan=3,
                                               colspan=1)

    def plot(self, feature, logliks, logsumexps, log2bf_thresh, renamed=''):
        modality = logsumexps.idxmax()

        self.logliks = logliks
        self.logsumexps = logsumexps

        x = feature.to_frame()
        if feature.name is None:
            feature.name = 'Feature'
        x['sample_id'] = feature.name

        violinplot(x='sample_id', y=feature.name, data=x, ax=self.ax_violin,
                   color=MODALITY_TO_COLOR[modality])

        self.ax_violin.set(xticks=[], ylabel='')

        for name, loglik in logliks.groupby('Modality')[r'$\log$ Likelihood']:
            # print name,
            self.ax_loglik.plot(loglik, 'o-', label=name, alpha=0.75,
                                color=MODALITY_TO_COLOR[name])
            self.ax_loglik.legend(loc='best')
        self.ax_loglik.set(ylabel=r'$\log$ Likelihood',
                           xlabel='Parameterizations',
                           title='Assignment: {}'.format(modality))
        self.ax_loglik.set_xlabel('phantom', color='white')

        for i, (name, height) in enumerate(logsumexps.iteritems()):
            self.ax_bayesfactor.bar(i, height, label=name,
                                    color=MODALITY_TO_COLOR[name])
        xmin, xmax = self.ax_bayesfactor.get_xlim()
        self.ax_bayesfactor.hlines(log2bf_thresh, xmin, xmax,
                                   linestyle='dashed')
        self.ax_bayesfactor.set(ylabel='$\log K$', xticks=[])
        if renamed:
            text = '{} ({})'.format(feature.name, renamed)
        else:
            text = feature.name
        self.fig.text(0.5, .025, text, fontsize=10, ha='center',
                      va='bottom')
        sns.despine()
        self.fig.tight_layout()
        return self


class ModalitiesViz(object):
    """Visualize results of modality assignments"""

    modality_order = MODALITY_ORDER
    modality_to_color = MODALITY_TO_COLOR
    modality_palette = MODALITY_PALETTE

    def bar(self, counts, phenotype_to_color=None, ax=None, percentages=True):
        """Draw barplots grouped by modality of modality percentage per group

        Parameters
        ----------


        Returns
        -------


        Raises
        ------

        """
        if percentages:
            counts = 100 * (counts.T / counts.T.sum()).T

        # with sns.set(style='whitegrid'):
        if ax is None:
            ax = plt.gca()

        full_width = 0.8
        width = full_width / counts.shape[0]
        for i, (group, series) in enumerate(counts.iterrows()):
            left = np.arange(len(self.modality_order)) + i * width
            height = [series[i] if i in series else 0
                      for i in self.modality_order]
            color = phenotype_to_color[group]
            ax.bar(left, height, width=width, color=color, label=group,
                   linewidth=.5, edgecolor='k')
        ylabel = 'Percentage of events' if percentages else 'Number of events'
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(self.modality_order)) + full_width / 2)
        ax.set_xticklabels(self.modality_order)
        ax.set_xlabel('Splicing modality')
        ax.set_xlim(0, len(self.modality_order))
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        sns.despine()

    def event_estimation(self, event, logliks, logsumexps, renamed=''):
        """Show the values underlying bayesian modality estimations of an event

        Parameters
        ----------


        Returns
        -------


        Raises
        ------
        """
        plotter = _ModelLoglikPlotter()
        plotter.plot(event, logliks, logsumexps, self.modality_to_color,
                     renamed=renamed)
        return plotter


def annotate_bars(x, group_col, percentage_col, modality_col, count_col,
                  **kwargs):
    data = kwargs.pop('data')
    # print kwargs
    ax = plt.gca()
    width = 0.8/5.
    x_base = -.49 - width/2.5
    for group, group_df in data.groupby(group_col):
        i = 0
        modality_grouped = group_df.groupby(modality_col)
        for modality in MODALITY_ORDER:
            i += 1
            try:
                modality_df = modality_grouped.get_group(modality)
            except KeyError:
                continue
            x_position = x_base + width*i + width/2
            y_position = modality_df[percentage_col]
            try:
                value = modality_df[count_col].values[0]
                formatted = locale.format('%d', value, grouping=True)
                ax.annotate(formatted, (x_position, y_position),
                            textcoords='offset points', xytext=(0, 2),
                            ha='center', va='bottom', fontsize=12)
            except IndexError:
                continue
        x_base += 1


def barplot(modalities_tidy, x=None, y='Percentage of Features', order=None,
            hue='Assigned Modality', **factorplot_kws):
    factorplot_kws.setdefault('hue_order', MODALITY_ORDER)
    factorplot_kws.setdefault('palette', MODALITY_PALETTE)
    factorplot_kws.setdefault('size', 3)
    factorplot_kws.setdefault('aspect', 3)
    factorplot_kws.setdefault('linewidth', 1)

    if order is not None and x is None:
        raise ValueError('If specifying "order", "x" must also '
                         'be specified.')
    # y = 'Percentage of features'
    groupby = [hue]
    groupby_minus_hue = []
    if x is not None:
        groupby = [x] + groupby
        groupby_minus_hue.append(x)
    if 'row' in factorplot_kws:
        groupby = groupby + [factorplot_kws['row']]
        groupby_minus_hue.append(factorplot_kws['row'])
    if 'col' in factorplot_kws:
        groupby = groupby + [factorplot_kws['col']]
        groupby_minus_hue.append(factorplot_kws['col'])

    # if x is not None:
    modality_counts = modalities_tidy.groupby(
        groupby).size().reset_index()
    modality_counts = modality_counts.rename(columns={0: 'Features'})
    if groupby_minus_hue:
        modality_counts[y] = modality_counts.groupby(
            groupby_minus_hue)['Features'].apply(
            lambda x: 100 * x / x.astype(float).sum())
    else:
        modality_counts[y] = 100 * modality_counts['Features']\
            / modality_counts['Features'].sum()
    if order is not None:
        modality_counts[x] = pd.Categorical(
            modality_counts[x], categories=order,
            ordered=True)
    # else:
    #     modality_counts[y] = pd.Categorical(
    #         modality_counts[x], categories=order,
    #         ordered=True)
    # else:
    #     modality_counts = modalities_tidy.groupby(
    #         hue).size().reset_index()
    #     modality_counts = modality_counts.rename(columns={0: 'Features'})
    #     modality_counts[y] = \
    #         100 * modality_counts.n_events/modality_counts.n_events.sum()
    if x is None:
        x = ''
        modality_counts[x] = x

    g = sns.factorplot(y=y, x=x,
                       hue=hue, kind='bar', data=modality_counts,
                       legend=False, **factorplot_kws)

    # Hacky workaround to add numeric annotations to the plot
    g.map_dataframe(annotate_bars, x, group_col=x,
                    modality_col=hue, count_col='Features',
                    percentage_col=y)
    g.add_legend(label_order=MODALITY_ORDER, title='Modalities')
    for ax in g.axes.flat:
        ax.locator_params('y', nbins=5)
        if ax.is_first_col():
            ax.set(ylabel=y)
    return g
