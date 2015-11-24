__author__ = 'Olga'

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import beta
import seaborn as sns

from .visualize import MODALITY_ORDER

def _assign_modality_from_estimate(mean_alpha, mean_beta):
    """
    Given estimated alpha and beta parameters from an Markov Chain Monte Carlo
    run, assign a modality.
    """
    # check if one parameter is much larger than another, and that they're
    # both larger than 1 (then it's either skewed towards 0 or 1)
#     if mean_alpha/mean_beta > 2 or mean_beta/mean_alpha > 2:
    if (mean_alpha > mean_beta) and (mean_beta > 1):
        # if alpha is greater than beta, then the probability is skewed
        # higher towards 1
        return '~1'
    elif (mean_beta > mean_alpha) and (mean_alpha > 1):
        # If alpha is less than beta, then the probability is skewed
        # higher towards 0
        return '~0'
    elif (mean_alpha < 1) and (mean_beta < 1):
        # if they're both under 1, then there's a valley in the middle,
        # and higher probability at the extremes of 0 and 1
        return 'bimodal'
    elif mean_alpha >= 1.5 and mean_beta >= 1.5:
        # if they're both fairly big, then there's a hump in the middle, and
        # low probability near the extremes of 0 and  1
        return 'middle'
    else:
        # maybe should check if both mean_alpha and mean_beta are near 1?
        return 'multimodal'
#         elif abs(mean_alpha - 1) < 0.25 and abs(mean_beta - 1) < 0.25:
#             return 'uniform'
#         else:
#             return None

def _print_and_plot(mean_alpha, mean_beta, alphas, betas, n_iter, data):
    print
    print mean_alpha, mean_beta, '  estimated modality:', \
        _assign_modality_from_estimate(mean_alpha, mean_beta)

    import numpy as np
    from scipy.stats import beta
    import matplotlib.pyplot as plt
    import prettyplotlib as ppl

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    ax = axes[0]

    ppl.plot(alphas, label='alpha', ax=ax)
    ppl.plot(betas, label='beta', ax=ax)
    ppl.legend(ax=ax)
    ax.hlines(mean_alpha, 0, n_iter)
    ax.hlines(mean_beta, 0, n_iter)

    ax.annotate('mean_alpha = {:.5f}'.format(mean_alpha),
                (0, mean_alpha), fontsize=12,
                xytext=(0, 1), textcoords='offset points')
    ax.annotate('mean_beta = {:.5f}'.format(betas.mean()),
                (0, mean_beta), fontsize=12,
                xytext=(0, 1), textcoords='offset points')
    ax.set_xlim(0, n_iter)

    ax = axes[1]
    ppl.hist(data, facecolor='grey', alpha=0.5, bins=np.arange(0, 1, 0.05),
             zorder=10, ax=ax)
    ymin, ymax = ax.get_ylim()

    one_x = np.arange(0, 1.01, 0.01)
    x = np.repeat(one_x, n_iter).reshape(len(one_x), n_iter)
    beta_distributions = np.vstack((beta(a, b).pdf(one_x)
                                    for a, b in zip(alphas, betas))).T

    ppl.plot(x, beta_distributions, color=ppl.colors.set2[0], alpha=0.1,
             linewidth=2, ax=ax, zorder=1)
    ax.set_ylim(0, ymax)

rv_included = beta(5, 1)
rv_excluded = beta(1, 5)
rv_middle = beta(5, 5)
rv_uniform = beta(1, 1)
rv_bimodal = beta(.65, .65)

models = {'included': rv_included,
          'excluded': rv_excluded,
          'middle': rv_middle,
          'uniform': rv_uniform,
          'bimodal': rv_bimodal}
# model_args = pd.DataFrame.from_dict(dict((name, np.array(model.args).astype(float)) for name, model in models.iteritems()))
# model_names = models.keys()

# # @pm.deterministic
# def modality_i(a, b, model_args=model_args):
#     # Not sure about this next line...
#     return model_args.apply(lambda x: spatial.distance.euclidean([a, b], x)).argmin()

def estimate_alpha_beta(data, n_iter=1000, plot=False):
#     data = data.dropna()
    alpha_var = pm.Exponential('alpha', .5)
    beta_var = pm.Exponential('beta', .5)

    observations = pm.Beta('observations', alpha_var, beta_var, value=data,
                           observed=True)

    model = pm.Model([alpha_var, beta_var])
    mcmc = pm.MCMC(model)
    mcmc.sample(n_iter)

    if plot:
        from pymc.Matplot import plot
        plot(mcmc)
        sns.despine()

    alphas = mcmc.trace('alpha')[:]
    betas = mcmc.trace('beta')[:]

    mean_alpha = alphas.mean()
    mean_beta = betas.mean()
    return mean_alpha, mean_beta


def estimate_modality(data, n_iter=1000, plot=False):
    alpha_var = pm.Exponential('alpha', .5)
    beta_var = pm.Exponential('beta', .5)

    # observations = pm.Beta('observations', alpha_var, beta_var, value=data,
    #                        observed=True)

    model = pm.Model([alpha_var, beta_var])
    mcmc = pm.MCMC(model)
    mcmc.sample(n_iter)

    if plot:
        from pymc.Matplot import plot
        plot(mcmc)
        sns.despine()

    alphas = mcmc.trace('alpha')[:]
    betas = mcmc.trace('beta')[:]

    counter = Counter(_assign_modality_from_estimate(a,b) for a, b in zip(alphas, betas))
    print 'estimated_modalities'

    mean_alpha = alphas.mean()
    mean_beta = betas.mean()
#     estimated_modality = _assign_modality_from_estimate(mean_alpha, mean_beta)

    if plot:
        _print_and_plot(mean_alpha, mean_beta, alphas, betas, n_iter, data)

#     return pd.Series({'mean_alpha':mean_alpha, 'mean_beta':mean_beta, 'modality':estimated_modality})
    print counter
    for modality in MODALITY_ORDER:
        if modality not in counter:
            counter[modality] = 0
    series = pd.Series(counter)
#     print series

    return series


def estimate_modality_latent(data, n_iter=1000,
                             model_params = [(2,1), (1,2), (5,5), (1,1),
                                             (.65,.65)]):
    fig, ax = plt.subplots()
    sns.distplot(data, ax=ax, bins=np.arange(0, 1, 0.05))

    assignment = pm.Categorical('assignment', [0.2]*5)

    alpha_var = pm.Lambda('alpha_var', lambda a=assignment: model_params[a][0])
    beta_var = pm.Lambda('beta_var', lambda a=assignment: model_params[a][1])

    observations = pm.Beta('observations', alpha_var, beta_var, value=data,
                           observed=True)

    model = pm.Model([assignment, observations])
    mcmc = pm.MCMC(model)
    mcmc.sample(n_iter)

    assignment_trace = mcmc.trace('assignment')[:]
    print assignment_trace

#     plot(mcmc)
#     sns.despine()
    counter = Counter(MODALITY_ORDER[i] for i in assignment_trace)
    print counter
    for modality in MODALITY_ORDER:
        if modality not in counter:
            counter[modality] = 0
    series = pd.Series(counter)
    print series

    return series
