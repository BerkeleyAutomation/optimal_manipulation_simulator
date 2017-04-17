"""
Configurations for grasp quality computation
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
import mayavi_visualizer as mv
import obj_file
import quality as pgq
import sdf_file
import similarity_tf as stf
import tfx
import feature_functions as ff

import discrete_adaptive_samplers as das
import models
import objectives
import termination_conditions as tc

import IPython

class GraspQualityConfig(object):
    """
    Class to create and enforce correct usage of grasp quality configurations
    """
    __metaclass__ = ABCMeta
    def __init__(self, config):
        # check valid config
        self.check_valid(config)

        # parse config
        for key, value in config.iteritems():
            setattr(self, key, value)

    def contains(self, key):
        """ Checks whether or not the key is supported """
        if key in self.__dict__.keys():
            return True
        return False

    def __getattr__(self, key):
        if self.contains(key):
            return object.__getattribute__(self, key)
        return None

    def __getitem__(self, key):
        if self.contains(key):
            return object.__getattribute__(self, key)
        raise KeyError('Key %s not found' %(key))
    
    def keys(self):
        return self.__dict__.keys()

    @abstractmethod
    def check_valid(self, config):
        """ Raise an exception if the config is missing required keys """
        pass

class QuasiStaticGraspQualityConfig(GraspQualityConfig):
    REQUIRED_KEYS = ['quality_method',
                     'friction_coef',
                     'num_cone_faces',
                     'soft_fingers',
                     'quality_type']

    def __init__(self, config):
        GraspQualityConfig.__init__(self, config)

    def __copy__(self):
        """ Makes a copy of the config """
        obj_copy = QuasiStaticGraspQualityConfig(self.__dict__)
        return obj_copy

    def check_valid(self, config):
        for key in QuasiStaticGraspQualityConfig.REQUIRED_KEYS:
            if key not in config.keys():
                raise ValueError('Invalid configuration. Key %s must be specified' %(key))

class RobustQuasiStaticGraspQualityConfig(GraspQualityConfig):
    ROBUST_REQUIRED_KEYS = ['num_quality_samples']

    def __init__(self, config):
        GraspQualityConfig.__init__(self, config)

    def __copy__(self):
        """ Makes a copy of the config """
        obj_copy = RobustQuasiStaticGraspQualityConfig(self.__dict__)
        return obj_copy
        
    def check_valid(self, config):
        required_keys = QuasiStaticGraspQualityConfig.REQUIRED_KEYS + \
            RobustQuasiStaticGraspQualityConfig.ROBUST_REQUIRED_KEYS
        for key in required_keys:
            if key not in config.keys():
                raise ValueError('Invalid configuration. Key %s must be specified' %(key))        

class GraspQualityConfigFactory:
    @staticmethod
    def create_config(config):
        if config['quality_type'] == 'quasi_static':
            return QuasiStaticGraspQualityConfig(config)
        elif config['quality_type'] == 'robust_quasi_static':
            return RobustQuasiStaticGraspQualityConfig(config)
        else:
            raise ValueError('Quality config type %s not supported' %(config['type']))
