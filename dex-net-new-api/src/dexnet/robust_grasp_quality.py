"""
Computation of deterministic and robust grasp quality metrics
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

class QuasiStaticGraspQualityRV(rvs.RandomVariable):
    """ RV class for grasp quality on an object """
    def __init__(self, grasp_rv, obj_rv, params_rv, quality_config):
        self.grasp_rv_ = grasp_rv
        self.obj_rv_ = obj_rv
        self.params_rv_ = params_rv # samples extra params for quality

        self.sample_count_ = 0
        self.quality_config_ = quality_config

        # preallocation not available
        rvs.RandomVariable.__init__(self, num_prealloc_samples=0)

    @property
    def obj(self):
        return self.graspable_rv_.obj

    @property
    def grasp(self):
        return self.grasp_rv_.grasp

    def sample(self, size=1):
        """ Samples deterministic quality metrics """
        # sample grasp
        cur_time = time.clock()
        grasp_sample = self.grasp_rv_.rvs(size=1, iteration=self.sample_count_)
        grasp_time = time.clock()

        # sample object
        obj_sample = self.obj_rv_.rvs(size=1, iteration=self.sample_count_)
        obj_time = time.clock()

        # sample params
        params_sample = None
        if self.params_rv_ is not None:
            params_sample = self.params_rv_.rvs(size=1, iteration=self.sample_count_)
            params_time = time.clock()

        # compute deterministic quality
        q = pgq.PointGraspMetrics3D.grasp_quality(grasp_sample, obj_sample,
                                                  params_sample)
        quality_time = time.clock()
        #logging.debug('Took %f sec to sample grasp' %(grasp_time - cur_time))
        #logging.debug('Took %f sec to sample object' %(obj_time - grasp_time))
        #logging.debug('Took %f sec to sample params' %(params_time - obj_time))
        #logging.debug('Took %f sec to compute quality' %(quality_time - params_time))
        
        """
        mlab.figure()
        self.obj_rv_.obj.mesh.visualize(style='surface', color=(0,0,1))
        mv.MayaviVisualizer.plot_grasp(self.grasp_rv_.grasp)

        mlab.figure()
        obj_sample.mesh.visualize(style='surface')
        mv.MayaviVisualizer.plot_grasp(grasp_sample)

        mlab.show()
        """

        self.sample_count_ = self.sample_count_ + 1
        return q

def expected_quality(grasp_rv, graspable_rv, params_rv, quality_config):
    """
    Get the expected quality wrt given random variables
    """
    # set up random variable
    q_rv = QuasiStaticGraspQualityRV(grasp_rv, graspable_rv,
                                     params_rv, quality_config)
    candidates = [q_rv]
    
    # brute force with uniform allocation
    snapshot_rate = quality_config['sampling_snapshot_rate']
    num_samples = quality_config['num_quality_samples']
    objective = objectives.RandomContinuousObjective()
    ua = das.GaussianUniformAllocationMean(objective, candidates)
    ua_result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(num_samples),
                         snapshot_rate = snapshot_rate)

    # convert to estimated prob success
    final_model = ua_result.models[-1]
    mn_q = final_model.means
    std_q = final_model.sample_vars
    return mn_q[0], std_q[0]
        
