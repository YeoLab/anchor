# -*- coding: utf-8 -*-
"""See log bayes factors which led to modality categorization"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


darkblue, green, red, purple, yellow, lightblue = sns.color_palette('deep')
MODALITY_ORDER = ['~0', 'middle', '~1', 'bimodal', 'multimodal']
MODALITY_TO_COLOR = {'~0': lightblue, 'middle': yellow, '~1': red,
                     'bimodal': purple, 'multimodal': 'lightgrey'}
MODALITY_PALETTE = [MODALITY_TO_COLOR[m] for m in MODALITY_ORDER]
MODALITY_TO_CMAP = {'~0': sns.light_palette(lightblue, as_cmap=True),
                    'middle': sns.light_palette(yellow, as_cmap=True),
                    '~1': sns.light_palette(red, as_cmap=True),
                    'bimodal': sns.light_palette(purple, as_cmap=True),
                    'multimodal': mpl.cm.Greys}


class _ModalityEstimatorPlotter(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(5 * 2, 3 * 2))
        self.ax_violin = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=1)
        self.ax_loglik = plt.subplot2grid((3, 5), (0, 1), rowspan=3, colspan=3)
        self.ax_bayesfactor = plt.subplot2grid((3, 5), (0, 4), rowspan=3,
                                               colspan=1)

    def plot(self, event, logliks, logsumexps, modality_to_color,
             renamed=''):
        modality = logsumexps.idxmax()

        sns.violinplot(event.dropna(), bw=0.2, ax=self.ax_violin,
                       color=modality_to_color[modality])

        self.ax_violin.set_ylim(0, 1)
        self.ax_violin.set_title('Guess: {}'.format(modality))
        self.ax_violin.set_xticks([])
        self.ax_violin.set_yticks([0, 0.5, 1])

        for name, loglik in logliks.iteritems():
            # print name,
            self.ax_loglik.plot(loglik, 'o-', label=name,
                                color=modality_to_color[name])
            self.ax_loglik.legend(loc='best')
        self.ax_loglik.set_title('Log likelihoods at different '
                                 'parameterizations')
        self.ax_loglik.grid()
        self.ax_loglik.set_xlabel('phantom', color='white')

        for i, (name, height) in enumerate(logsumexps.iteritems()):
            self.ax_bayesfactor.bar(i, height, label=name,
                                    color=modality_to_color[name])
        self.ax_bayesfactor.set_title('$\log$ Bayes factors')
        self.ax_bayesfactor.set_xticks([])
        self.ax_bayesfactor.grid()
        self.fig.tight_layout()
        self.fig.text(0.5, .025, '{} ({})'.format(event.name, renamed),
                      fontsize=10, ha='center', va='bottom')
        sns.despine()
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
        plotter = _ModalityEstimatorPlotter()
        plotter.plot(event, logliks, logsumexps, self.modality_to_color,
                     renamed=renamed)
        return plotter


from modish.visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE

import locale


locale.setlocale(locale.LC_ALL, 'en_US')


def annotate_bars(x, group_col, percentage_col, **kwargs):
    data = kwargs.pop('data')
    ax = plt.gca()
    width = 0.8/5.
    x_base = -.49 - width/2.5
    for group, group_df in data.groupby(group_col):
        i = 0
        for modality, modality_df in group_df.groupby('modality'):
            i += 1
            x_position = x_base + width*i + width/2
            y_position = modality_df[percentage_col]
            try:
                value = modality_df.n_events.values[0]
                formatted = locale.format('%d', value, grouping=True)
                ax.annotate(formatted, (x_position, y_position),
                            textcoords='offset points', xytext=(0, 2),
                            ha='center', va='bottom', fontsize=12)
            except IndexError:
                continue
        x_base += 1

modality_factorplot_kws = dict(hue_order=MODALITY_ORDER, palette=MODALITY_PALETTE)

def modalities_barplot(modalities_tidy, group_order=None, group_col=None, factorplot_kws=None):
    factorplot_kws = {} if factorplot_kws == None else factorplot_kws

    if group_order is not None and group_col is None:
        raise ValueError('If specifying "group_order", "group_col" must also '
                         'be specified.')

    if group_col is None:
        modality_counts = modalities_tidy.groupby(
            [group_col, 'modality']).size().reset_index()
        modality_counts = modality_counts.rename(columns={0: 'n_events'})
        modality_counts['Percentage of events'] = modality_counts.groupby(
            group_col).n_events.apply(
            lambda x: 100 * x / x.astype(float).sum())
        if group_order is not None:
            modality_counts[group_col] = pd.Categorical(
                modality_counts[group_col], categories=group_order,
                ordered=True)
        else:
            modality_counts[group_col] = pd.Categorical(
                modality_counts[group_col], categories=group_order,
                ordered=True)
    else:
        modality_counts = modalities_tidy.groupby(
            'modality').size().reset_index()
        modality_counts = modality_counts.rename(columns={0: 'n_events'})
        modality_counts['Percentage of events'] = \
            100 * modality_counts.n_events/modality_counts.n_events.sum()
        group_col = 'group'
        modality_counts[group_col] = group_col

    modality_counts.modality = pd.Categorical(
        modality_counts.modality, categories=MODALITY_ORDER, ordered=True)
    g = sns.factorplot(y='Percentage of events', x='phenotype', hue='modality',
                       kind='bar', data=modality_counts, aspect=3,
                       legend=False, linewidth=1, size=3, **factorplot_kws)

    # Hacky workaround to add numeric annotations to the
    g.map_dataframe(annotate_bars, group_col, group_col=group_col,
                    percentage_col='Percentage of events')
    g.add_legend(label_order=MODALITY_ORDER, title='Modalities')
    for ax in g.axes.flat:
        ax.locator_params('y', nbins=5)
    return g