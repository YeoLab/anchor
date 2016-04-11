# -*- coding: utf-8 -*-
from .bayesian import BayesianModalities
from .binning import BinnedModalities
from .model import ModalityModel
from .names import NEAR_ZERO, NEAR_HALF, NEAR_ONE, BIMODAL, \
    NULL_MODEL
from .simulate import add_noise
from .visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE,\
    MODALITY_TO_CMAP, ModalitiesViz, violinplot, barplot

__author__ = 'Olga Botvinnik'
__email__ = 'olga.botvinnik@gmail.com'
__version__ = '0.1.0'


__all__ = ['ModalityModel', 'BayesianModalities', 'MODALITY_ORDER',
           'MODALITY_PALETTE', 'MODALITY_TO_COLOR', 'ModalitiesViz',
           'violinplot', 'MODALITY_TO_CMAP', 'BinnedModalities',
           'add_noise', 'BIMODAL', 'NEAR_HALF', 'NEAR_ONE', 'NEAR_ZERO',
           'barplot', 'NULL_MODEL']
