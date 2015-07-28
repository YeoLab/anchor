# -*- coding: utf-8 -*-
"""Command line program which estimates modalities on a single file"""
import matplotlib.pyplot as plt
import seaborn as sns

MODALITY_ORDER = ['~0', 'middle', '~1', 'bimodal', 'ambiguous']
MODALITY_COLORS = dict(zip(MODALITY_ORDER, sns.color_palette('deep')))

class _ModalityEstimatorPlotter(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(5 * 2, 3 * 2))
        self.ax_violin = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=1)
        self.ax_loglik = plt.subplot2grid((3, 5), (0, 1), rowspan=3, colspan=3)
        self.ax_bayesfactor = plt.subplot2grid((3, 5), (0, 4), rowspan=3,
                                               colspan=1)

    def plot(self, event, logliks, logsumexps, modality_colors,
             renamed=''):
        modality = logsumexps.idxmax()

        sns.violinplot(event.dropna(), bw=0.2, ax=self.ax_violin,
                       color=modality_colors[modality])

        self.ax_violin.set_ylim(0, 1)
        self.ax_violin.set_title('Guess: {}'.format(modality))
        self.ax_violin.set_xticks([])
        self.ax_violin.set_yticks([0, 0.5, 1])

        for name, loglik in logliks.iteritems():
            # print name,
            self.ax_loglik.plot(loglik, 'o-', label=name,
                                color=modality_colors[name])
            self.ax_loglik.legend(loc='best')
        self.ax_loglik.set_title('Log likelihoods at different '
                                 'parameterizations')
        self.ax_loglik.grid()
        self.ax_loglik.set_xlabel('phantom', color='white')

        for i, (name, height) in enumerate(logsumexps.iteritems()):
            self.ax_bayesfactor.bar(i, height, label=name,
                                    color=modality_colors[name])
        self.ax_bayesfactor.set_title('$\log$ Bayes factors')
        self.ax_bayesfactor.set_xticks([])
        self.ax_bayesfactor.grid()
        self.fig.tight_layout()
        self.fig.text(0.5, .025, '{} ({})'.format(event.name, renamed),
                      fontsize=10, ha='center', va='bottom')
        sns.despine()
        return self