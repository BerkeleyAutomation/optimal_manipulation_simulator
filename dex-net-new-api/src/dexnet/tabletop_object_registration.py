"""
Classes for registering objects from Dex-Net to tabletops imaged with the Dex-Net sensor.
Author: Jeff Mahler
"""
import caffe
import copy
import glob
import logging
import math
import numpy as np
import scipy.ndimage.filters as skf
import scipy.ndimage.morphology as snm
import os
import sys
import time

import cv
import cv2
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dexnet.cnn_query_image_generator import DepthCNNQueryImageGenerator
from dexnet.cnn_database_indexer import CNN_Hdf5ObjectIndexer, CNN_Hdf5ObjectStablePoseIndexer
from dexnet.mesh import Mesh3D
from dexnet.rendered_image import RenderedImage

import camera_params as cp
import database as db
import stp_file

USE_ALAN = True
try:
    from alan.core import RigidTransform, Point, Direction, PointCloud, NormalCloud, PointNormalCloud, Box
    from alan.core import Visualizer as vis
    from alan.rgbd import PointToPlaneFeatureMatcher, PointToPlaneICPSolver, RegistrationResult
    from alan.rgbd import DepthImage, CameraIntrinsics
except:
    USE_ALAN = False

class DatabaseRegistrationResult:
    """ Struct to hold relevant output of registering an object from Dex-Net to real data """
    def __init__(self, T_camera_obj, nearest_images, nearest_distances, registration_results, best_index, total_runtime):
        self.T_camera_obj = T_camera_obj
        self.nearest_images = nearest_images
        self.nearest_distances = nearest_distances
        self.registration_results = registration_results
        self.best_index = best_index
        self.total_runtime = total_runtime

class TabletopRegistrationSolver:
    def __init__(self, logging_dir=None):
        self._logging_dir = logging_dir

    def log_to(self, logging_dir):
        self._logging_dir = logging_dir

    def _table_to_stp_transform(self, T_virtual_camera_table, T_actual_camera_table, config):
        """ Compute a transformation to align the table normal with the z axis """
        # load table plane from calib
        R_table_actual_camera = T_actual_camera_table.inverse().rotation
        table_normal_virtual_camera = Direction(R_table_actual_camera[:,2], frame=T_virtual_camera_table.from_frame)

        # transform the table normal from camera to table basis
        z_table = T_virtual_camera_table * table_normal_virtual_camera
        x_table, y_table = z_table.orthogonal_basis()
        t0 = x_table.data[0]
        t1 = y_table.data[0]
        xp = t0 * x_table.data + t1 * y_table.data
        xp = xp / np.linalg.norm(xp)
        yp = np.cross(z_table.data, xp)
        Rp = np.c_[xp, yp, z_table.data]

        T_table_stp = RigidTransform(rotation=Rp.T,
                                     from_frame='stp',
                                     to_frame='table')        
        return T_table_stp

    def _create_query_image(self, depth_im, camera_intr, T_camera_table, config, debug=False):
        """ Creates the query image for the database indexer """
        # setup debugging
        logging_dir = None
        if debug:
            logging_dir = self._logging_dir

        # read in calibration params
        workspace_bbox = Box(min_pt=np.array(config['workspace_min_pt']),
                             max_pt=np.array(config['workspace_max_pt']),
                             frame='world')

        # generate query image
        generator = DepthCNNQueryImageGenerator(T_camera_table, camera_intr, workspace_bbox, config)
        cnn_query_params = generator.query_image(depth_im)
        return cnn_query_params

    def _query_database(self, query_image, database_indexer, config, debug=False):
        """ Query the database for the nearest neighbors """
        # read params
        num_nearest_neighbors = config['num_nearest_neighbors']

        # look up nearest neighbors
        query_rendered_image = RenderedImage(query_image.data)
        nearest_neighbors = database_indexer.k_nearest(query_rendered_image, k=num_nearest_neighbors)
        nearest_images = nearest_neighbors[0]
        nearest_distances = nearest_neighbors[1]    
        
        # visualize nearest neighbors for debugging
        if debug:
            font_size = 15
            plt.figure()
            plt.subplot(2, num_nearest_neighbors, math.ceil(float(num_nearest_neighbors)/2))
            plt.imshow(query_rendered_image.image, cmap=plt.cm.Greys_r, interpolation='none')
            plt.title('QUERY IMAGE', fontsize=font_size)
            plt.axis('off')

            for j, (image, distance) in enumerate(zip(nearest_images, nearest_distances)):
                plt.subplot(2, num_nearest_neighbors, j+num_nearest_neighbors+1)
                plt.imshow(image.image, cmap=plt.cm.Greys_r, interpolation='none')
                plt.title('N %d (%d), DIST = %.2f' %(j, image.id, distance), fontsize=10)
                plt.axis('off')

            if self._logging_dir is None:
                plt.show()
            else:
                figname = 'query_nearest_neighbors.png'
                plt.savefig(os.path.join(self._logging_dir, figname))
        return nearest_images, nearest_distances

    def _find_best_transformation(self, cnn_query_params, candidate_rendered_images, T_camera_table, dataset, config, debug=False):
        """ Finds the best transformation from the candidate set using Point to Plane Iterated closest point """
        T_camera_table = copy.copy(T_camera_table)
        T_camera_table._to_frame = 'table' # for ease of use

        # read params
        icp_sample_size = config['icp_sample_size']
        icp_relative_point_plane_cost = config['icp_relative_point_plane_cost']
        icp_regularization_lambda = config['icp_regularization_lambda']
        feature_matcher_dist_thresh = config['feature_matcher_dist_thresh']
        feature_matcher_norm_thresh = config['feature_matcher_norm_thresh']
        num_registration_iters = config['num_registration_iters']
        compute_total_cost = config['compute_total_registration_cost']
        threshold_cost = config['threshold_cost']

        # register from nearest images
        registration_results = []
        min_cost = np.inf
        best_reg = None
        best_T_virtual_camera_obj = None        
        best_index = -1
        for i, neighbor_image in enumerate(candidate_rendered_images):
            logging.info('Registering to neighbor %d' %(i))

            # load object mesh
            database_start = time.time()
            mesh = dataset.mesh(neighbor_image.obj_key)
            database_stop = time.time()

            # form transforms
            preproc_start = time.time()
            T_virtual_camera_stp = neighbor_image.stp_to_camera_transform().inverse()
            T_virtual_camera_stp._from_frame = cnn_query_params.point_normal_cloud.frame
            T_obj_stp = neighbor_image.object_to_stp_transform()
            source_mesh_x0_obj = Point(neighbor_image.stable_pose.x0, frame='obj')
            source_mesh_x0_stp = T_obj_stp * source_mesh_x0_obj
            block1_stop = time.time()
            logging.debug('Preproc block 1 took %.2f sec' %(block1_stop-preproc_start))

            # get source object points in table basis
            logging.info('Transforming source mesh')
            mesh_stp = mesh.transform(T_obj_stp)
            z = mesh_stp.min_coords()[2]
            obj_center_table = np.array([0,0,-z])

            T_obj_table = RigidTransform(rotation=T_obj_stp.rotation,
                                         translation=obj_center_table,
                                         from_frame='obj',
                                         to_frame='table')
            mesh_table = mesh.transform(T_obj_table)
            source_points_table = PointCloud(np.array(mesh_table.vertices()).T, frame=T_obj_table.to_frame)
            block2_stop = time.time()
            logging.debug('Preproc block 2 took %.2f sec' %(block2_stop-block1_stop))
            
            # read target points and normals
            target_points_normals_virtual_camera = cnn_query_params.point_normal_cloud
            if target_points_normals_virtual_camera.num_points == 0:
                logging.info('Found zero target points, skipping')
                registration_results.append(RegistrationResult(RigidTransform(from_frame=target_points_normals_virtual_camera.frame, to_frame='obj'), np.inf))
                continue
            block3_stop = time.time()
            logging.debug('Preproc block 3 took %.2f sec' %(block3_stop-block2_stop))

            # match table normals
            logging.info('Matching table normals')
            target_points_stp = T_virtual_camera_stp * target_points_normals_virtual_camera.points
            target_normals_stp = T_virtual_camera_stp * target_points_normals_virtual_camera.normals

            T_stp_table = self._table_to_stp_transform(T_virtual_camera_stp, T_camera_table, config)
            target_points_table = T_stp_table * target_points_stp
            target_normals_table = T_stp_table * target_normals_stp
            block4_stop = time.time()
            logging.debug('Preproc block 4 took %.2f sec' %(block4_stop-block3_stop))

            # render depth image of source points
            logging.info('Rendering virtual depth')
            
            # transform mesh to virtual camera basis
            T_virtual_camera_table = T_stp_table * T_virtual_camera_stp
            T_table_virtual_camera = T_virtual_camera_table.inverse()
            mesh_virtual_camera = mesh_table.transform(T_table_virtual_camera)
            block5_stop = time.time()
            logging.debug('Preproc block 5 took %.2f sec' %(block5_stop-block4_stop))

            # render virtual depth image
            T_camera_virtual_camera = cnn_query_params.T_camera_virtual_camera
            virtual_camera_intr = cnn_query_params.cropped_ir_intrinsics
            depth_im = mesh_virtual_camera.project_depth(virtual_camera_intr)
            source_depth_im = DepthImage(depth_im, virtual_camera_intr.frame)
            block6_stop = time.time()
            logging.debug('Preproc block 6 took %.2f sec' %(block6_stop-block5_stop))

            # project points
            source_points_normals_virtual_camera = source_depth_im.point_normal_cloud(virtual_camera_intr)
            source_points_normals_virtual_camera.remove_zero_points()
            source_points_table = T_virtual_camera_table * source_points_normals_virtual_camera.points
            source_normals_table = T_virtual_camera_table * source_points_normals_virtual_camera.normals
            block7_stop = time.time()
            logging.debug('Preproc block 7 took %.2f sec' %(block7_stop-block6_stop))

            # align the lowest and closest points to the camera
            logging.info('Aligning lowest and closest')
            table_center_camera = Point(T_camera_table.inverse().translation, frame=T_camera_table.from_frame)
            table_x0_table = T_virtual_camera_table * T_camera_virtual_camera * table_center_camera
            block8_stop = time.time()
            logging.debug('Preproc block 8 took %.2f sec' %(block8_stop-block7_stop))

            # align points closest to the camera
            camera_optical_axis_table = -T_virtual_camera_table.rotation[:,2]
            source_ip = source_points_table.data.T.dot(camera_optical_axis_table)
            closest_ind = np.where(source_ip == np.max(source_ip))[0]
            source_x0_closest_table = source_points_table[closest_ind[0]]

            max_z_ind = np.where(source_points_table.z_coords == np.max(source_points_table.z_coords))[0][0]
            source_x0_highest_table = source_points_table[max_z_ind] # lowest point in table frame

            target_ip = target_points_table.data.T.dot(camera_optical_axis_table)
            closest_ind = np.where(target_ip == np.max(target_ip))[0]
            target_x0_closest_table = target_points_table[closest_ind[0]]

            max_z_ind = np.where(target_points_table.z_coords == np.max(target_points_table.z_coords))[0][0]
            target_x0_highest_table = target_points_table[max_z_ind] # lowest point in table frame

            t_table_t_table_s = source_x0_closest_table.data - target_x0_closest_table.data
            t_table_t_table_s[2] = source_x0_highest_table.data[2] - target_x0_highest_table.data[2]

            T_table_t_table_s = RigidTransform(translation=t_table_t_table_s, from_frame='table', to_frame='table')
            target_points_table = T_table_t_table_s * target_points_table
            target_normals_table = T_table_t_table_s * target_normals_table           
            T_virtual_camera_table = T_table_t_table_s * T_virtual_camera_table
            preproc_stop = time.time()
            block9_stop = time.time()
            logging.debug('Preproc block 9 took %.2f sec' %(block9_stop-block8_stop))

            # display the points relative to one another
            if debug:
                logging.info('Pre-registration alignment')
                vis.figure()
                vis.mesh(mesh_table)
                vis.points(source_points_table, color=(1,0,0))
                vis.points(target_points_table, color=(0,1,0))
                vis.points(source_x0_closest_table, color=(0,0,1), scale=0.02)
                vis.points(target_x0_closest_table, color=(0,1,1), scale=0.02)
                vis.table(dim=0.15)
                vis.show()

            # point to plane ICP solver
            icp_start = time.time()
            ppis = PointToPlaneICPSolver(sample_size=icp_sample_size, gamma=icp_relative_point_plane_cost, mu=icp_regularization_lambda)
            ppfm = PointToPlaneFeatureMatcher(dist_thresh=feature_matcher_dist_thresh, norm_thresh=feature_matcher_norm_thresh) 
            registration = ppis.register_2d(source_points_table, target_points_table, source_normals_table, target_normals_table, ppfm, num_iterations=num_registration_iters,
                                            compute_total_cost=compute_total_cost, vis=debug)

            registration_results.append(registration)
            icp_stop = time.time()

            logging.info('Neighbor %d registration cost %f' %(i, registration.cost))
            logging.info('Neighbor %d timings' %(i))
            logging.info('Database read took %.2f sec' %(database_stop-database_start))
            logging.info('Preproc took %.2f sec' %(preproc_stop-preproc_start))
            logging.info('ICP took %.2f sec' %(icp_stop-icp_start))

            if debug:
                logging.info('Post-registration alignment')
                vis.figure()
                vis.points(registration.T_source_target * source_points_table, color=(1,0,0))
                vis.points(target_points_table, color=(0,1,0))
                vis.table(dim=0.15)
                vis.show()


            if registration.cost < min_cost:
                min_cost = registration.cost
                best_reg = registration
                best_T_table_s_table_t = registration.T_source_target
                best_T_virtual_camera_obj = T_obj_table.inverse().dot(best_T_table_s_table_t.inverse()).dot(T_virtual_camera_table)
                best_index = i

            if min_cost < threshold_cost:
                logging.info('Satisfactory registration found. Terminating early.')
                break

        if debug:
            logging.info('Best alignment')
            vis.figure()
            vis.mesh(mesh)
            vis.points(best_T_virtual_camera_obj * target_points_normals_virtual_camera.points, color=(1,0,0))
            vis.show()

        # compute best transformation from object to camera basis
        return best_T_virtual_camera_obj, registration_results, best_index

    def register(self, depth_im, camera_intr, T_camera_table, dataset, database_indexer, config, debug=False):
        """
        Register an object from Dex-Net to the tabletop in the color and depth images.
        The conventions are:
           obj = object basis
        Params:
           depth_im: (DepthImage object) The depth image corresponding to the scene to register. Must be in same frame of reference as the color image
           camera_intr: (CameraIntrinsics object) The intrinsics of the depth camera
           T_camera_table: (RigidTransform object) Known transformation from camera to table basis
           dataset:  (Hdf5Dataset instance) Dataset containing the data
           database_indexer: (Hdf5DatabaseIndexer instance) Object to index Dex-Net for registration hypotheses
           config: (ExperimentConfig or python dictionary) Key-value dictionary-like object with the following params:
              - workspace_min_pt     (3-list)- the minimum point of a bounding box for the workspace in world frame
              - workspace_max_pt     (3-list)- the maximum point of a bounding box for the workspace in world frame
              - index_im_dim         (int)   - dimension of the image to use for indexing (256)
              - depth_im_median_filter_dim (int)  - window size of median filter
              - depth_im_erosion_filter_dim (int) - window size of erosion filter
              - cache_im_filename     (string) - filename to save preprocessed query image
              - num_nearest_neighbors (int)    - number of nearest neighbors to use for registration
              - icp_sample_size       (int)    - number of subsampled points to use in ICP iterations (similar to batch size in SGD)
              - icp_relative_point_plane_cost (float) - multiplicative factor for point-based ICP cost term relative to point-plane ICP cost term
              - icp_regularization_lambda     (int)   - regularization constant for ICP
              - feature_matcher_dist_thresh   (float) - point distance threshold for ICP feature matching
              - feature_matcher_norm_thresh   (float) - normal inner product threshold for ICP feature matching
              - num_registration_iters        (int)   - number of iterations to run registration for
           debug: (bool) whether or not to display additional debugging output
        Returns:
           DatabaseRegistrationResult object
        """
        if not isinstance(depth_im, DepthImage):
            raise ValueError('Input depth_im must be of type DepthImage') 
        if not isinstance(camera_intr, CameraIntrinsics):
            raise ValueError('Input camera_intr must be of type CameraIntrinsics') 
        if not isinstance(T_camera_table, RigidTransform):
            raise ValueError('Input T_camera_table must be of type RigidTransform') 
        if depth_im.frame != camera_intr.frame or depth_im.frame != T_camera_table.from_frame:
            raise ValueError('Depth image, camera intrinsics, and T_camera_table must have same frame')

        debug_or_save = debug or (self._logging_dir is not None)

        # remove points beyond table
        ip_start = time.time()
        cnn_query_params = \
            self._create_query_image(depth_im, camera_intr, T_camera_table, config, debug=debug_or_save)
        ip_end = time.time()

        # index the database for similar objects
        query_start = time.time()
        nearest_images, nearest_distances = self._query_database(cnn_query_params.query_im, database_indexer, config, debug=debug_or_save)
        query_end = time.time()

        # register to the candidates
        registration_start = time.time()
        T_virtual_camera_obj, registration_results, best_index = \
            self._find_best_transformation(cnn_query_params, nearest_images, T_camera_table, dataset, config, debug=debug)
        T_camera_obj =  T_virtual_camera_obj * cnn_query_params.T_camera_virtual_camera
        registration_end = time.time()

        # log runtime
        total_runtime = registration_end-registration_start + query_end-query_start + ip_end-ip_start
        logging.info('Image processing took %.2f sec' %(ip_end-ip_start))
        logging.info('Database query took %.2f sec' %(query_end-query_start))
        logging.info('ICP took %.2f sec' %(registration_end-registration_start))
        logging.info('Total registration time: %.2f sec' %(total_runtime))

        # display transformed mesh projected into the query image
        if debug:
            font_size = 15
            best_object = dataset.graspable(nearest_images[best_index].obj_key)
            best_object_mesh = best_object.mesh
            best_object_mesh_tf = best_object_mesh.transform(T_camera_obj.inverse())
            object_point_cloud = PointCloud(np.array(best_object_mesh_tf.vertices()).T, frame=camera_intr.frame)
            object_mesh_proj_pixels = camera_intr.project(object_point_cloud)

            plt.figure()
            plt.imshow(depth_im.data, cmap=plt.cm.Greys_r, interpolation='none')
            plt.scatter(object_mesh_proj_pixels.j_coords, object_mesh_proj_pixels.i_coords, s=5, c='r')
            plt.title('Projected object mesh pixels', fontsize=font_size)
            if self._logging_dir is None:
                plt.show()
            else:
                figname = 'projected_mesh.png'
                plt.savefig(os.path.join(self._logging_dir, figname))

        # construct and return output
        return DatabaseRegistrationResult(T_camera_obj, nearest_images, nearest_distances, registration_results, best_index, total_runtime)

class KnownObjectTabletopRegistrationSolver(TabletopRegistrationSolver):
    def __init__(self, object_key, dataset, config, logging_dir=None):
        TabletopRegistrationSolver.__init__(self, logging_dir)
        self.object_key_ = object_key
        self.dataset_ = dataset
        self.cnn_indexer_ = CNN_Hdf5ObjectIndexer(object_key, dataset, config)
        self.config_ = config

    def register(self, depth_im, camera_intr, T_camera_table, debug=False):
        """ Create a CNN object indexer for registration """
        return TabletopRegistrationSolver.register(self, depth_im, camera_intr, T_camera_table, self.dataset_, self.cnn_indexer_, self.config_['registration'], debug=debug)

class KnownObjectStablePoseTabletopRegistrationSolver(TabletopRegistrationSolver):
    def __init__(self, object_key, stable_pose_id, dataset, config, logging_dir=None):
        TabletopRegistrationSolver.__init__(self, logging_dir)
        self._object_key = object_key
        self._stable_pose_id = stable_pose_id
        self._dataset = dataset
        self._config = config

        if 'caffe' not in self._config.keys():
            raise ValueError('Must provide caffe configuration')
        if 'registration' not in self._config.keys():
            raise ValueError('Must provide registration configuration')

        self._cnn_indexer = CNN_Hdf5ObjectStablePoseIndexer(self._object_key, self._stable_pose_id, self._dataset, self._config['caffe'])

    def register(self, depth_im, camera_intr, T_camera_table, debug=False):
        """ Create a CNN object indexer for registration """
        return TabletopRegistrationSolver.register(self, depth_im, camera_intr, T_camera_table, self._dataset, self._cnn_indexer, self._config['registration'], debug=debug)
