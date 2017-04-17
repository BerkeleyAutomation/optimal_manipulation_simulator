"""
Random Variables for sampling force closure, etc
Author: Jeff Mahler
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
import time

import scipy.linalg
import scipy.stats
import sklearn.cluster

import grasp as gr
import graspable_object as go
import grasp_quality_config as gqf
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

# TODO: move this function somewhere interesting
def skew(xi):
    S = np.array([[0, -xi[2], xi[1]],
                  [xi[2], 0, -xi[0]],
                  [-xi[1], xi[0], 0]])
    return S

def deskew(S):
    x = np.zeros(3)
    x[0] = S[2,1]
    x[1] = S[0,2]
    x[2] = S[1,0]
    return x

class RandomVariable(object):
    __metaclass__ = ABCMeta

    def __init__(self, num_prealloc_samples=0):
        self.num_prealloc_samples_ = num_prealloc_samples
        if self.num_prealloc_samples_ > 0:
            self._preallocate_samples()

    def _preallocate_samples(self, size=1):
        """ Preallocate samples for faster adative sampling """
        self.prealloc_samples_ = []
        for i in range(self.num_prealloc_samples_):
            self.prealloc_samples_.append(self.sample())

    @abstractmethod
    def sample(self, size=1):
        """ Sample | size | random variables """
        pass

    def rvs(self, size=1, iteration=1):
        """ Sample |size| random variables with the option of using preallocated samples """
        if self.num_prealloc_samples_ > 0:
            samples = []
            for i in range(size):
                samples.append(self.prealloc_samples_[(iteration + i) % self.num_prealloc_samples_])
            if size == 1:
                return samples[0]
            return samples
        # generate a new sample
        return self.sample(size=size)

class ArtificialRV(RandomVariable):
    """
    A fake RV that always returns the given object
    """
    def __init__(self, obj, num_prealloc_samples=0):
        self.obj_ = obj
        super(ArtificialRV, self).__init__(num_prealloc_samples)

    def sample(self, size = 1):
        return [self.obj_] * size
        
class ArtificialSingleRV(ArtificialRV):
    def sample(self, size = None):
        return self.obj_
        
class GraspableObjectPoseGaussianRV(RandomVariable):
    def __init__(self, obj, config):
        self.obj_ = obj
        self._parse_config(config)

        translation_sigma = self.R_sample_sigma_.T.dot(obj.tf.translation)
        self.s_rv_ = scipy.stats.norm(obj.tf.scale, self.sigma_scale_**2)
        self.t_rv_ = scipy.stats.multivariate_normal(translation_sigma, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)

        RandomVariable.__init__(self, self.num_prealloc_samples_)

    def _parse_config(self, config):
        self.sigma_rot_ = 0
        self.sigma_trans_ = 0 
        self.sigma_scale_ = 0
        self.R_sample_sigma_ = np.eye(3)
        self.num_prealloc_samples_ = 0

        if config is not None:
            if 'sigma_obj_rot' in config.keys():
                self.sigma_rot_ = config['sigma_obj_rot']
            elif 'sigma_obj_rot_x' in config.keys() and \
                    'sigma_obj_rot_y' in config.keys() and \
                    'sigma_obj_rot_z' in config.keys():
                self.sigma_rot_ = np.diag([config['sigma_obj_rot_x'],
                                           config['sigma_obj_rot_y'],
                                           config['sigma_obj_rot_z']])
            if 'sigma_obj_trans' in config.keys():
                self.sigma_trans_ = config['sigma_obj_trans']
            elif 'sigma_obj_trans_x' in config.keys() and \
                    'sigma_obj_trans_y' in config.keys() and \
                    'sigma_obj_trans_z' in config.keys():
                self.sigma_trans_ = np.diag([config['sigma_obj_trans_x'],
                                           config['sigma_obj_trans_y'],
                                           config['sigma_obj_trans_z']])
            if 'sigma_obj_scale' in config.keys():
                self.sigma_scale_ = config['sigma_obj_scale']
            if 'R_sample_sigma' in config.keys():
                self.R_sample_sigma_ = config['R_sample_sigma']
            if 'num_prealloc_samples' in config.keys():
                self.num_prealloc_samples_ = config['num_prealloc_samples']

    @property
    def obj(self):
        return self.obj_

    def sample(self, size=1):
        """ Sample |size| random variables from the model """
        samples = []
        for i in range(size):
            # sample random pose
            xi = self.r_xi_rv_.rvs(size=1)
            S_xi = skew(xi)
            R = self.R_sample_sigma_.dot(scipy.linalg.expm(S_xi).dot(self.R_sample_sigma_.T.dot(self.obj_.tf.rotation)))
            s = self.s_rv_.rvs(size=1)[0]
            t = self.R_sample_sigma_.dot(self.t_rv_.rvs(size=1).T).T
            sample_tf = stf.SimilarityTransform3D(tfx.transform(R.T, t), s)

            # transform object by pose
            obj_sample = self.obj_.transform(sample_tf)
            samples.append(obj_sample)

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples

class ParallelJawGraspPoseGaussianRV(RandomVariable):
    def __init__(self, grasp, config):
        self.grasp_ = grasp
        self._parse_config(config)

        center_sigma = self.R_sample_sigma_.T.dot(grasp.center)
        self.t_rv_ = scipy.stats.multivariate_normal(center_sigma, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)

        RandomVariable.__init__(self, self.num_prealloc_samples_)

    def _parse_config(self, config):
        self.sigma_rot_ = 0
        self.sigma_trans_ = 0 
        self.R_sample_sigma_ = np.eye(3)
        self.num_prealloc_samples_ = 0

        if config is not None:
            if 'sigma_grasp_rot' in config.keys():
                self.sigma_rot_ = config['sigma_grasp_rot']
            elif 'sigma_grasp_rot_x' in config.keys() and \
                    'sigma_grasp_rot_y' in config.keys() and \
                    'sigma_grasp_rot_z' in config.keys():
                self.sigma_rot_ = np.diag([config['sigma_grasp_rot_x'],
                                           config['sigma_grasp_rot_y'],
                                           config['sigma_grasp_rot_z']])
            if 'sigma_grasp_trans' in config.keys():
                self.sigma_trans_ = config['sigma_grasp_trans']
            elif 'sigma_grasp_trans_x' in config.keys() and \
                    'sigma_grasp_trans_y' in config.keys() and \
                    'sigma_grasp_trans_z' in config.keys():
                self.sigma_trans_ = np.diag([config['sigma_grasp_trans_x'],
                                           config['sigma_grasp_trans_y'],
                                           config['sigma_grasp_trans_z']])
            if 'R_sample_sigma' in config.keys():
                self.R_sample_sigma_ = config['R_sample_sigma']
            if 'num_prealloc_samples' in config.keys():
                self.num_prealloc_samples_ = config['num_prealloc_samples']

    @property
    def grasp(self):
        return self.grasp_

    def sample(self, size=1):
        samples = []
        for i in range(size):
            # sample random pose
            xi = self.r_xi_rv_.rvs(size=1)
            S_xi = skew(xi)

            axis_sigma = self.R_sample_sigma_.T.dot(self.grasp_.axis)
            v = self.R_sample_sigma_.dot(scipy.linalg.expm(S_xi).dot(axis_sigma))
            t = self.R_sample_sigma_.dot(self.t_rv_.rvs(size=1).T).T

            # transform object by pose
            grasp_sample = gr.ParallelJawPtGrasp3D(gr.ParallelJawPtGrasp3D.configuration_from_params(t, v, self.grasp_.grasp_width))

            samples.append(grasp_sample)

        if size == 1:
            return samples[0]
        return samples

class ParamsGaussianRV(RandomVariable):
    def __init__(self, params, u_config):
        if not isinstance(params, gqf.GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')
        self.params_ = params
        self._parse_config(u_config)

        self.rvs_ = {}
        for param_name, param_rv in self.sigmas_.iteritems():
            self.rvs_[param_name] = scipy.stats.multivariate_normal(param_rv[0], param_rv[1])
        RandomVariable.__init__(self, self.num_prealloc_samples_)

    def _parse_config(self, sigma_params):
        self.sigmas_ = {}
        self.num_prealloc_samples_ = 0

        if sigma_params is not None:
            for key in sigma_params.keys():
                # only parse the "sigmas"
                ind = key.find('sigma')
                if ind == 0 and len(key) > 7 and key[6:] in self.params_.keys():
                    param_name = key[6:]
                    self.sigmas_[param_name] = (self.params_[param_name], sigma_params[key])
            if 'num_prealloc_samples' in sigma_params.keys():
                self.num_prealloc_samples_ = sigma_params['num_prealloc_samples']

    def mean(self):
        return self.params_
        
    def sample(self, size=1):
        samples = []
        for i in range(size):
            # sample random force, torque, etc
            params_sample = copy.copy(self.params_)
            for rv_name, rv in self.rvs_.iteritems():
                params_sample.rv_name = rv.rvs(size=1)
            samples.append(params_sample)

        if size == 1:
            return samples[0]
        return samples

def plot_value_vs_time_beta_bernoulli(result, candidate_true_p, true_max=None, color='blue'):
    """ Plots the number of samples for each value in for a discrete adaptive sampler"""
    best_values = [candidate_true_p[m.best_pred_ind] for m in result.models]
    plt.plot(result.iters, best_values, color=color, linewidth=2)
    if true_max is not None: # also plot best possible
        plt.plot(result.iters, true_max*np.ones(len(result.iters)), color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Success')

def test_antipodal_grasp_thompson():
    np.random.seed(100)

    # load object
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = go.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)

    config = {
        'grasp_width': 0.1,
        'friction_coef': 0.5,
        'num_cone_faces': 8,
        'grasp_samples_per_surface_point': 4,
        'dir_prior': 1.0,
        'alpha_thresh_div': 32,
        'rho_thresh': 0.75, # as pct of object max moment
        'vis_antipodal': False,
        'min_num_grasps': 20,
        'alpha_inc': 1.1,
        'rho_inc': 1.1,
        'sigma_mu': 0.1,
        'sigma_trans_grasp': 0.001,
        'sigma_rot_grasp': 0.1,
        'sigma_trans_obj': 0.001,
        'sigma_rot_obj': 0.1,
        'sigma_scale_obj': 0.1,
        'num_prealloc_obj_samples': 100,
        'num_prealloc_grasp_samples': 0,
        'min_num_collision_free_grasps': 10,
        'grasp_theta_res': 1
    }
    sampler = ags.AntipodalGraspSampler(config)

    start_time = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(graspable, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Antipodal grasp candidate generation took %f sec' %(duration))

    # convert grasps to RVs for optimization
    graspable_rv = GraspableObjectGaussianPose(graspable, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu'])
    candidates = []
    for grasp in grasps:
        grasp_rv = ParallelJawGraspGaussian(grasp, config)
        candidates.append(ForceClosureRV(grasp_rv, graspable_rv, f_rv, config))

    objective = objectives.RandomBinaryObjective()

    # run bandits
    eps = 5e-4
    ua_tc_list = [tc.MaxIterTerminationCondition(1000)]#, tc.ConfidenceTerminationCondition(eps)]
    ua = das.UniformAllocationMean(objective, candidates)
    ua_result = ua.solve(termination_condition = tc.OrTerminationCondition(ua_tc_list), snapshot_rate = 100)
    logging.info('Uniform allocation took %f sec' %(ua_result.total_time))

    ts_tc_list = [tc.MaxIterTerminationCondition(1000), tc.ConfidenceTerminationCondition(eps)]
    ts = das.ThompsonSampling(objective, candidates)
    ts_result = ts.solve(termination_condition = tc.OrTerminationCondition(ts_tc_list), snapshot_rate = 100)
    logging.info('Thompson sampling took %f sec' %(ts_result.total_time))

    true_means = models.BetaBernoulliModel.beta_mean(ua_result.models[-1].alphas, ua_result.models[-1].betas)

    # plot results
    plt.figure()
    plot_value_vs_time_beta_bernoulli(ua_result, true_means, color='red')
    plot_value_vs_time_beta_bernoulli(ts_result, true_means, color='blue')
    plt.show()

    das.plot_num_pulls_beta_bernoulli(ua_result)
    plt.title('Observations Per Variable for Uniform allocation')

    das.plot_num_pulls_beta_bernoulli(ts_result)
    plt.title('Observations Per Variable for Thompson sampling')

    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_antipodal_grasp_thompson()
