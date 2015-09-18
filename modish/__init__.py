# -*- coding: utf-8 -*-
from .model import ModalityModel
from .estimator import ModalityEstimator
from .visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE,\
    ModalitiesViz

__author__ = 'Olga Botvinnik'
__email__ = 'olga.botvinnik@gmail.com'
__version__ = '0.1.0'


__all__ = ['ModalityModel', 'ModalityEstimator', 'MODALITY_ORDER',
           'MODALITY_PALETTE', 'MODALITY_TO_COLOR', 'ModalitiesViz']
