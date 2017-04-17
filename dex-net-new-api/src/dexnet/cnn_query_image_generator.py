"""
Process kinect image in preparation for querying database for registration
Authors: Jeff Mahler, Jacky Liang
"""
import argparse
import copy
import IPython
import logging
import numpy as np
import os
import scipy.ndimage.filters as skf
import scipy.ndimage.morphology as snm
import yaml

import matplotlib.pyplot as plt

USE_ALAN = True
try:
    from alan.rgbd import Image, DepthImage, ColorImage, CameraIntrinsics
    from alan.core import RigidTransform
    from alan.core import Visualizer as vis
except:
    USE_ALAN = False

class CNNQueryImageParams:
    def __init__(self, query_im, point_normal_cloud, cropped_ir_intrinsics, T_camera_virtual_camera):
        self.query_im = query_im
        self.point_normal_cloud = point_normal_cloud
        self.cropped_ir_intrinsics = cropped_ir_intrinsics
        self.T_camera_virtual_camera = T_camera_virtual_camera

class DepthCNNQueryImageGenerator:

    def __init__(self, T_camera_world, ir_intrinsics, workspace_bbox,
                 config):

        self.T_camera_world = T_camera_world
        self.ir_intrinsics = ir_intrinsics
        self.workspace_bbox = workspace_bbox

        self.depth_im_median_filter_dim = config['depth_im_median_filter_dim']
        self.depth_im_erosion_filter_dim = config['depth_im_erosion_filter_dim']
        self.depth_im_rescale_factor = config['depth_im_rescale_factor']
        self.depth_im_grad_thresh = config['depth_im_grad_thresh']
        self.area_thresh = config['area_thresh']
        self.index_im_dim = config['index_im_dim']
        self.cache_im_filename = config['cache_im_filename']

        self.num_areas = 1
        if 'num_areas' in config.keys():
            self.num_areas = config['num_areas']
                
    def isolate_workspace(self, depth_im):
        # compute point cloud in world frame
        point_cloud_camera = self.ir_intrinsics.deproject(depth_im)
        point_cloud_world = self.T_camera_world * point_cloud_camera

        # threshold to find objects on the table
        point_cloud_world_isolated, point_cloud_world_isolated_ind = point_cloud_world.box_mask(self.workspace_bbox)
        depth_im_masked = depth_im.mask_by_linear_ind(point_cloud_world_isolated_ind)

        return depth_im_masked

    def filter_binary_image(self, binary_im):
        # keep largest connected object
        binary_im_pruned = binary_im.prune_contours(area_thresh=self.area_thresh, num_areas=self.num_areas)

        # median filter
        binary_im_median = binary_im_pruned.apply(skf.median_filter, size=self.depth_im_median_filter_dim)
        binary_im_eroded = binary_im_median.apply(snm.grey_erosion, size=(self.depth_im_erosion_filter_dim, self.depth_im_erosion_filter_dim))
        
        return binary_im_eroded

    def query_image(self, depth_im, output_path=None):
        """
        Creates and returns a query image. 
        If output path is given, intermediate images will be saved for
        debug purposes to the output path.
        """
        logging.info("Deprojecting into point clouds.")
        # isolating workspace through point clouds
        logging.info("Isolating workspace in point clouds.")
        depth_im_isolated = self.isolate_workspace(depth_im)
        depth_im_isolated_thresh = depth_im_isolated.threshold_gradients(self.depth_im_grad_thresh)

        # turn depth into binary image
        logging.info("Binarizing depth image.")
        binary_im_isolated = depth_im_isolated_thresh.to_binary()
        
        # filter binary image by pruning contours and applying median filter
        logging.info("Filtering binary image.")
        binary_im_filtered = self.filter_binary_image(binary_im_isolated)
        if binary_im_filtered is None:
            logging.warn("No sizeable object detected in scene! Returning None.")
            return None

        # centering filtered binary image
        logging.info("Centering depth image.")
        depth_im_filtered = depth_im_isolated.mask_binary(binary_im_filtered)
        depth_im_centered, diff_px = depth_im_filtered.center_nonzero(self.ir_intrinsics)

        T_camera_virtual_camera = depth_im_filtered.px_shift_to_rigid_transform(self.ir_intrinsics,
                                                                                diff_px,
                                                                                depth_im_centered.frame)

        # cropping centered binary image
        logging.info("Cropping binary image.")
        binary_im_centered = depth_im_centered.to_binary()
        binary_im_cropped = binary_im_centered.crop(self.index_im_dim, self.index_im_dim)

        # create point clouds
        logging.info("Creating point clouds.")        
        depth_im_centered_rescaled = depth_im_centered.resize(self.depth_im_rescale_factor)
        ir_centered_intrinsics = CameraIntrinsics(frame=depth_im_centered.frame,
                                                  fx=self.depth_im_rescale_factor * self.ir_intrinsics.fx,
                                                  cx=self.depth_im_rescale_factor * self.ir_intrinsics.cx,
                                                  cy=self.depth_im_rescale_factor * self.ir_intrinsics.cy,
                                                  height=depth_im_centered_rescaled.height,
                                                  width=depth_im_centered_rescaled.width)
        point_normal_cloud = depth_im_centered_rescaled.point_normal_cloud(ir_centered_intrinsics)
        point_normal_cloud.remove_zero_points()

        # save and load binary image to get distortion parameters
        logging.info("Adding JPEG distortion.")
        binary_im_cropped.to_color().save(self.cache_im_filename)
        query_im = ColorImage.open(self.cache_im_filename)

        # form cropped intrinsics
        ir_cropped_intrinsics = CameraIntrinsics(frame=depth_im_centered.frame,
                                                 fx=self.ir_intrinsics.fx,
                                                 cx=self.index_im_dim / 2,
                                                 cy=self.index_im_dim / 2,
                                                 height=query_im.height,
                                                 width=query_im.width)
 
        if output_path is not None:
            logging.info("Saving images.")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            depth_im.savefig(output_path, '0 input depth image', cmap=plt.cm.gray)
            depth_im_isolated.savefig(output_path, '1 masked depth image', cmap=plt.cm.gray)
            depth_im_isolated_thresh.savefig(output_path, '2 masked thresholded depth image', cmap=plt.cm.gray)
            binary_im_isolated.savefig(output_path, '3 masked binary image', cmap=plt.cm.gray)
            binary_im_filtered.savefig(output_path, '4 filtered binary image', cmap=plt.cm.gray)
            binary_im_centered.savefig(output_path, '5 center binary image', cmap=plt.cm.gray)
            binary_im_cropped.savefig(output_path, '6 cropped binary image', cmap=plt.cm.gray)
            query_im.savefig(output_path, '7 jpeg-distorted binary image', cmap=plt.cm.gray)

        return CNNQueryImageParams(query_im, point_normal_cloud, ir_cropped_intrinsics, T_camera_virtual_camera)
