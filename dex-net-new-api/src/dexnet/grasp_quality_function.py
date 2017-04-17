"""
Returns user-friendly functiosn for computing grasp quality metrics
Author: Jeff
"""
from abc import ABCMeta, abstractmethod

import copy
import itertools as it
import logging
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mlab
except:
    logging.warning('Failed to import mayavi')

import numpy as np
import os
import sys
import time

import scipy.stats

import gripper as gp
import grasp as gr
import graspable_object as go
import grasp_quality_config as gq
import mayavi_visualizer as mv
import obj_file
import quality as pgq
import robust_grasp_quality as rgq
import random_variables as rvs
import sdf_file
import similarity_tf as stf
import tfx
import feature_functions as ff

import discrete_adaptive_samplers as das
import models
import objectives
import termination_conditions as tc

import IPython

class GraspQualityResult:
    """ Stores the results of grasp quality computation """
    def __init__(self, mean_quality, std_quality=0.0, quality_config=None):
        self.mean_quality = mean_quality
        self.std_quality = std_quality
        self.quality_config = quality_config            

class GraspQualityFunction(object):
    """
    Abstraction for grasp quality functions to make labelling easier
    """
    __metaclass__ = ABCMeta

    def __init__(self, graspable, quality_config):
        # check valid types
        if not isinstance(graspable, go.GraspableObject):
            raise ValueError('Must provide GraspableObject')
        if not isinstance(quality_config, gq.GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')

        # set member variables
        self.graspable_ = graspable
        self.quality_config_ = quality_config

        self._setup()

    def __call__(self, grasp):
        return self.quality(grasp)

    @abstractmethod
    def _setup(self):
        """ Sets up common variables for grasp quality evaluations """
        pass

    @abstractmethod
    def quality(self, grasp):
        """ Compute the quality for a grasp object """
        pass
        
class QuasiStaticQualityFunction(GraspQualityFunction):
    def __init__(self, graspable, quality_config):
        GraspQualityFunction.__init__(self, graspable, quality_config)

    def _setup(self):
        if self.quality_config_.quality_type != 'quasi_static':
            raise ValueError('Quality configuration must be quasi static')

    def quality(self, grasp):
        if not isinstance(grasp, gr.Grasp):
            raise ValueError('Must provide Grasp object to compute quality')

        quality = pgq.PointGraspMetrics3D.grasp_quality(grasp, self.graspable_,
                                                        self.quality_config_)
        return GraspQualityResult(quality, quality_config=self.quality_config_)

class RobustQuasiStaticQualityFunction(GraspQualityFunction):
    def __init__(self, graspable, quality_config):
        GraspQualityFunction.__init__(self, graspable, quality_config)

    def _setup(self):
        if self.quality_config_.quality_type != 'robust_quasi_static':
            raise ValueError('Quality configuration must be robust quasi static')
        self.graspable_rv_ = rvs.GraspableObjectPoseGaussianRV(self.graspable_,
                                                               self.quality_config_.obj_uncertainty)
        self.params_rv_ = rvs.ParamsGaussianRV(self.quality_config_,
                                               self.quality_config_.params_uncertainty)

    def quality(self, grasp):
        if not isinstance(grasp, gr.Grasp):
            raise ValueError('Must provide Grasp object to compute quality')
        grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp,
                                                      self.quality_config_.grasp_uncertainty)
        mean_q, std_q = rgq.expected_quality(grasp_rv,
                                             self.graspable_rv_,
                                             self.params_rv_,
                                             self.quality_config_)
        return GraspQualityResult(mean_q, std_q, quality_config=self.quality_config_)

class GraspQualityFunctionFactory:
    @staticmethod
    def create_quality_function(graspable, quality_config):
        """ Creates a quality function object based on the config """
        # check valid types
        if not isinstance(graspable, go.GraspableObject):
            raise ValueError('Must provide GraspableObject')
        if not isinstance(quality_config, gq.GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')
        
        if quality_config.quality_type == 'quasi_static':
            return QuasiStaticQualityFunction(graspable, quality_config)
        elif quality_config.quality_type == 'robust_quasi_static':
            return RobustQuasiStaticQualityFunction(graspable, quality_config)
        else:
            raise ValueError('Grasp quality type %s not supported' %(quality_config.quality_type))
