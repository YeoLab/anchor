# -*- coding: utf-8 -*-
from .model import ModalityModel
from .estimator import ModalityEstimator, ModalityPredictor
from .visualize import MODALITY_TO_COLOR, MODALITY_ORDER, MODALITY_PALETTE,\
    MODALITY_TO_CMAP, ModalitiesViz, violinplot, barplot

__author__ = 'Olga Botvinnik'
__email__ = 'olga.botvinnik@gmail.com'
__version__ = '0.1.0'


__all__ = ['ModalityModel', 'ModalityEstimator', 'MODALITY_ORDER',
           'MODALITY_PALETTE', 'MODALITY_TO_COLOR', 'ModalitiesViz',
           'violinplot', 'MODALITY_TO_CMAP', 'ModalityPredictor']

class ModishTestResult(object):
    
    def __init__(self, original_data, estimator, modality_assignments, 
                 bayesian_fit, data_with_noise, waypoint_transformer, 
                 waypoints):
        self.original_data = original_data
        self.estimator = estimator
        self.modality_assignments = modality_assignments
        self.bayesian_fit = bayesian_fit
        self.data_with_noise = data_with_noise
        self.waypoint_transformer = waypoint_transformer
        self.waypoints = waypoints
                                                        
