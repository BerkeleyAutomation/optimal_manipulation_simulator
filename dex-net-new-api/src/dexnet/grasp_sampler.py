"""
Classes for sampling grasps

Author: Jeff Mahler
"""

from abc import ABCMeta, abstractmethod
import copy
import IPython
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random
import sys
import time

USE_OPENRAVE = True
try:
    import openravepy as rave
except:
    logging.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False

import scipy.stats as stats

import contacts
import experiment_config as ec
import grasp
import graspable_object
from grasp import ParallelJawPtGrasp3D
import grasp_collision_checker as gcc
import gripper as gr
import obj_file
import sdf_file

class GraspSampler:
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_grasps(self, graspable):
        """
        Create a list of candidate grasps for the object specified in graspable
        """
        pass

class ExactGraspSampler(GraspSampler):
    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """Configures the grasp generator."""
        self.friction_coef = config['friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.dir_prior = config['dir_prior']
        self.target_num_grasps = config['target_num_grasps']
        if self.target_num_grasps is None:
            self.target_num_grasps = config['min_num_grasps']

        self.min_contact_dist = config['min_contact_dist']
        self.num_grasp_rots = config['coll_check_num_grasp_rots']
        if 'max_num_surface_points' in config.keys():
            self.max_num_surface_points_ = config['max_num_surface_points']
        else:
            self.max_num_surface_points_ = 100

    def generate_grasps(self, graspable, target_num_grasps=None, grasp_gen_mult=3, max_iter=3,
                        check_collisions=False, vis=False, **kwargs):
        """Generates an exact number of grasps.
        Params:
            graspable - (GraspableObject3D) the object to grasp
            target_num_grasps - (int) number of grasps to return, defualts to
                self.target_num_grasps
            grasp_gen_mult - (int) number of additional grasps to generate
            max_iter - (int) number of attempts to return an exact number of
                grasps before giving up
        """
        # setup collision checking
        if USE_OPENRAVE and check_collisions:
            rave.raveSetDebugLevel(rave.DebugLevel.Error)
            collision_checker = gcc.OpenRaveGraspChecker(self.gripper, view=False)
            collision_checker.set_object(graspable)

        # get num grasps 
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps

        grasps = []
        k = 1
        while num_grasps_remaining > 0 and k <= max_iter:
            # generate more than we need
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining
            new_grasps = self._generate_grasps(graspable, num_grasps_generate,
                                               vis, **kwargs)

            # prune grasps in collision
            coll_free_grasps = []
            if check_collisions and USE_OPENRAVE:
                logging.info('Checking collisions for %d candidates' %(len(new_grasps)))
                for grasp in new_grasps:
                    # construct a set of rotated grasps
                    for i in range(self.num_grasp_rots):
                        rotated_grasp = copy.copy(grasp)
                        rotated_grasp.set_approach_angle(2 * i * np.pi / self.num_grasp_rots)
                        if not collision_checker.in_collision(rotated_grasp):
                            coll_free_grasps.append(rotated_grasp)
                            break
            else:
                coll_free_grasps = new_grasps

            grasps += coll_free_grasps
            logging.info('%d/%d grasps found after iteration %d.',
                         len(grasps), target_num_grasps, k)

            grasp_gen_mult *= 2
            num_grasps_remaining = target_num_grasps - len(grasps)
            k += 1

        random.shuffle(grasps)
        if len(grasps) > target_num_grasps:
            logging.info('Truncating %d grasps to %d.',
                         len(grasps), target_num_grasps)
            grasps = grasps[:target_num_grasps]
        logging.info('Found %d grasps.', len(grasps))

        return grasps

class UniformGraspSampler(ExactGraspSampler):
    def _generate_grasps(self, graspable, num_grasps,
                         vis=False, max_num_samples=1000):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF
        Params:
            graspable - (GraspableObject3D) the object to grasp
            num_grasps - (int) the number of grasps to generate
        Returns:
            list of ParallelJawPtGrasp3D objects
        """
        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]
        i = 0
        grasps = []

        # get all grasps
        while len(grasps) < num_grasps and i < max_num_samples:
            # get candidate contacts
            indices = np.random.choice(num_surface, size=2, replace=False)
            c0 = surface_points[indices[0], :]
            c1 = surface_points[indices[1], :]

            if np.linalg.norm(c1 - c0) > self.gripper.min_width and np.linalg.norm(c1 - c0) < self.gripper.max_width:
                # compute centers and axes
                grasp_center = ParallelJawPtGrasp3D.grasp_center_from_endpoints(c0, c1)
                grasp_axis = ParallelJawPtGrasp3D.grasp_axis_from_endpoints(c0, c1)
                g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center,
                                                                                        grasp_axis,
                                                                                        self.gripper.max_width))
                # keep grasps if the fingers close
                success, contacts = g.close_fingers(graspable)
                if success:
                    grasps.append(g)
            i += 1

        return grasps

class GaussianGraspSampler(ExactGraspSampler):
    def _generate_grasps(self, graspable, num_grasps,
                         vis=False,
                         sigma_scale=2.5):
        """
        Returns a list of candidate grasps for graspable object by Gaussian with
        variance specified by principal dimensions
        Params:
            graspable - (GraspableObject3D) the object to grasp
            num_grasps - (int) the number of grasps to generate
            sigma_scale - (float) the number of sigmas on the tails of the
                Gaussian for each dimension
        Returns:
            list of ParallelJawPtGrasp3D objects
        """
        # get object principal axes
        center_of_mass = graspable.mesh.center_of_mass
        principal_dims = graspable.mesh.principal_dims()
        sigma_dims = principal_dims / (2 * sigma_scale)

        # sample centers
        grasp_centers = stats.multivariate_normal.rvs(
            mean=center_of_mass, cov=sigma_dims**2, size=num_grasps)

        # samples angles uniformly from sphere
        u = stats.uniform.rvs(size=num_grasps)
        v = stats.uniform.rvs(size=num_grasps)
        thetas = 2 * np.pi * u
        phis = np.arccos(2 * v - 1.0)
        grasp_dirs = np.array([np.sin(phis) * np.cos(thetas), np.sin(phis) * np.sin(thetas), np.cos(phis)])
        grasp_dirs = grasp_dirs.T

        # convert to grasp objects
        grasps = []
        for i in range(num_grasps):
            grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i,:], grasp_dirs[i,:], self.gripper.max_width))
            contacts_found, contacts = grasp.close_fingers(graspable)

            # add grasp if it has valid contacts
            if contacts_found and np.linalg.norm(contacts[0].point - contacts[1].point) > self.min_contact_dist:
                grasps.append(grasp)

        # visualize
        if vis:
            for grasp in grasps:
                plt.clf()
                h = plt.gcf()
                plt.ion()
                grasp.close_fingers(graspable, vis=vis)
                plt.show(block=False)
                time.sleep(0.5)

            grasp_centers_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_centers.T)
            grasp_centers_grid = grasp_centers_grid.T
            com_grid = graspable.sdf.transform_pt_obj_to_grid(center_of_mass)

            plt.clf()
            ax = plt.gca(projection = '3d')
            graspable.sdf.scatter()
            ax.scatter(grasp_centers_grid[:,0], grasp_centers_grid[:,1], grasp_centers_grid[:,2], s=60, c=u'm')
            ax.scatter(com_grid[0], com_grid[1], com_grid[2], s=120, c=u'y')
            ax.set_xlim3d(0, graspable.sdf.dims_[0])
            ax.set_ylim3d(0, graspable.sdf.dims_[1])
            ax.set_zlim3d(0, graspable.sdf.dims_[2])
            plt.show()

        return grasps

class AntipodalGraspSampler(ExactGraspSampler):
    def sample_from_cone(self, cone, num_samples=1):
        """
        Samples points from within the cone.
        Params:
            cone - friction cone's supports
        Returns:
            v_samples - list of self.num_samples vectors in the cone
        """
        num_faces = cone.shape[1]
        v_samples = np.empty((num_samples, 3))
        for i in range(num_samples):
            lambdas = np.random.gamma(self.dir_prior, self.dir_prior, num_faces)
            lambdas = lambdas / sum(lambdas)
            v_sample = lambdas * cone
            v_samples[i, :] = np.sum(v_sample, 1)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Returns True if a grasp will not slip due to friction.
        Params:
            cone - friction cone's supports
            n - outward pointing normal vector at c1
            v - direction vector between c1 and c2
        Returns:
            in_cone - True if alpha is within the cone
            alpha - the angle between the normal and v
        """
        if (v.dot(cone) < 0).any(): # v should point in same direction as cone
            v = -v # don't worry about sign, we don't know it anyway...
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point """
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def _generate_grasps(self, graspable, num_grasps,
                         vis=False):
        """Returns a list of candidate grasps for graspable object.
        Params:
            graspable - (GraspableObject3D) the object to grasp
            num_grasps - currently unused TODO
        Returns:
            list of ParallelJawPtGrasp3D objects
        """
        # get surface points
        grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        shuffled_surface_points = surface_points[:min(self.max_num_surface_points_, len(surface_points))]
        logging.info('Num surface: %d' %(len(surface_points)))

        for k, x_surf in enumerate(shuffled_surface_points):
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(self.num_samples):
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)

                # compute friction cone faces
                c1 = contacts.Contact3D(graspable, x1, in_direction=None)
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(cone1, num_samples=1)
                sample_time = time.clock()

                for v in v_samples:
                    if vis:
                        x1_grid = graspable.sdf.transform_pt_obj_to_grid(x1)
                        cone1_grid = graspable.sdf.transform_pt_obj_to_grid(cone1, direction=True)
                        plt.clf()
                        h = plt.gcf()
                        plt.ion()
                        ax = plt.gca(projection = '3d')
                        for i in range(cone1.shape[1]):
                            ax.scatter(x1_grid[0] - cone1_grid[0], x1_grid[1] - cone1_grid[1], x1_grid[2] - cone1_grid[2], s = 50, c = u'm')

                    # random axis flips since we don't have guarantees on surface normal directoins
                    if random.random() > 0.5:
                        v = -v

                    # start searching for contacts
                    grasp, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(graspable, x1, v, self.gripper.max_width,
                                                                                         min_grasp_width_world=self.gripper.min_width,
                                                                                         vis=vis)

                    if grasp is None or c2 is None:
                        continue

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        plt.figure()
                        ax = plt.gca(projection='3d')
                        c1_proxy = c1.plot_friction_cone(color='m')
                        c2_proxy = c2.plot_friction_cone(color='y')
                        ax.view_init(elev=5.0, azim=0)
                        plt.show(block=False)
                        time.sleep(0.5)
                        plt.close() # lol

                    # check friction cone
                    in_cone1, alpha1 = self.within_cone(cone1, n1, v_true.T)
                    in_cone2, alpha2 = self.within_cone(cone2, n2, -v_true.T)
                    within_cone_time = time.clock()

                    # add points if within friction cone
                    if in_cone1 and in_cone2:
                        grasps.append(grasp)

        # randomly sample max num grasps from total list
        random.shuffle(grasps)
        return grasps

def test_gaussian_grasp_sampling(vis=False):
    np.random.seed(100)

    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = graspable_object.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)

    config_file = 'cfg/correlated.yaml'
    config = ec.ExperimentConfig(config_file)
    sampler = GaussianGraspSampler(config)

    start_time = time.clock()
    grasps = sampler.generate_grasps(graspable, target_num_grasps=200, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Gaussian grasp candidate generation took %f sec' %(duration))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_gaussian_grasp_sampling()
