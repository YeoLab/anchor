import numpy as np
import pandas as pd

from .infotheory import binify, bin_range_strings, jsd
from .visualize import MODALITY_ORDER


class BinnedModalities(object):

    modalities = MODALITY_ORDER
    score_name = 'Jensen-Shannon Divergence'

    def __init__(self, bins=(0, 1./3, 2./3, 1)):
        if len(bins) != 4:
            raise ValueError('Length of "bins" must be exactly 4 bin edges')
        self.bins = bins

        self.bin_ranges = bin_range_strings(self.bins)
        uniform_probabilities = [stop-start for start, stop in
                                 zip(bins, bins[1:])]

        self.desired_distributions = pd.DataFrame(
            np.array([[1, 0, 0], [0.5, 0, 0.5],
                      [0, 0, 1], [0, 1, 0], uniform_probabilities]).T,
            index=self.bin_ranges, columns=self.modalities)

    def fit(self, data):
        binned = binify(data, bins=self.bins)
        if isinstance(binned, pd.DataFrame):
            fitted = binned.apply(lambda x: self.desired_distributions.apply(
                lambda y: jsd(x, y)))
        else:
            fitted = self.desired_distributions.apply(lambda x: jsd(x, binned))
        fitted.name = self.score_name
        return fitted

    def predict(self, fitted):
        """Assign the most likely modality given the fitted data

        Parameters
        ----------
        fitted : pandas.DataFrame or pandas.Series
            Either a (n_modalities, features) DatFrame or (n_modalities,)
            Series, either of which will return the best modality for each
            feature.
        """
        if fitted.shape[0] != len(self.modalities):
            raise ValueError("This data doesn't look like it had the distance "
                             "between it and the five modalities calculated")
        return fitted.idxmin()

    def fit_predict(self, data):
        return self.predict(self.fit(data))
