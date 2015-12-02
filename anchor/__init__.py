# -*- coding: utf-8 -*-
from .model import ModalityModel
from .bayesian import BayesianModalities
from .binning import BinnedModalities
from .monte_carlo import MonteCarloModalities
from .simulate import add_noise
from .visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE,\
    MODALITY_TO_CMAP, ModalitiesViz, violinplot, barplot

__author__ = 'Olga Botvinnik'
__email__ = 'olga.botvinnik@gmail.com'
__version__ = '0.1.0'


__all__ = ['ModalityModel', 'BayesianModalities', 'MODALITY_ORDER',
           'MODALITY_PALETTE', 'MODALITY_TO_COLOR', 'ModalitiesViz',
           'violinplot', 'MODALITY_TO_CMAP', 'BinnedModalities',
           'add_noise']
