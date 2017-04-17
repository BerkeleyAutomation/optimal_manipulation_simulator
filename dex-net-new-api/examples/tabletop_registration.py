"""
Script to test full tabletop registration of objects
Author: Jeff Mahler
"""
import copy
import IPython
import numpy as np
import os
import logging
import sys
import argparse

import matplotlib.pyplot as plt

import dexnet.database as db
from dexnet.rendered_image import RenderedImage 
import dexnet.experiment_config as ec
from dexnet.cnn_database_indexer import CNN_Hdf5ObjectStablePoseIndexer
from dexnet.cnn_query_image_generator import DepthCNNQueryImageGenerator
from dexnet.tabletop_object_registration import KnownObjectStablePoseTabletopRegistrationSolver

from alan.rgbd import Kinect2Sensor, Kinect2PacketPipelineMode, Image, DepthImage, CameraIntrinsics
from alan.core import RigidTransform, Box
from alan.core import Visualizer as vis

def load_depth_images(cfg):
    depths = []

    if cfg['prestored_data'] == 0:
        # read images from the Kinect
        packet_pipeline_mode = Kinect2PacketPipelineMode.OPENGL
        if cfg['sensor']['pipeline_mode'] == 1:
            packet_pipeline_mode = Kinect2PacketPipelineMode.CPU

        sensor = Kinect2Sensor(device_num=cfg['sensor']['device_num'], frame=cfg['sensor']['frame'], packet_pipeline_mode=packet_pipeline_mode)
        sensor.start()
        ir_intrinsics = sensor.ir_intrinsics

        # get raw images
        for _ in range(cfg['num_images']):
            _, depth, _ = sensor.frames()
            depths.append(depth)

        sensor.stop()
    else:
        # load data from dir
        ir_intrinsics = CameraIntrinsics.load(os.path.join(cfg['prestored_data_dir'], '%s_ir.intr' %(cfg['sensor']['frame'])))
        
        for i in range(cfg['num_images']):
            depth_im_filename = os.path.join(os.path.join(cfg['prestored_data_dir'], 'depth_%d.npy' %(i)))
            depth_im = DepthImage.open(depth_im_filename)
            depth_im._frame = ir_intrinsics.frame
            depths.append(depth_im)

    return depths, ir_intrinsics

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path')
    args = parser.parse_args()

    # open config
    cfg_file = 'cfg/examples/tabletop_registration.yaml'
    cfg = ec.ExperimentConfig(cfg_file)
    logging.getLogger().setLevel(logging.INFO)
    object_key = cfg['object_key']
    stp_id = cfg['stp_id']

    # open db
    db_filename = os.path.join(cfg['database_dir'], cfg['database_name'])
    database = db.Hdf5Database(db_filename, cfg)
    dataset = database.datasets[0]
    obj = dataset[object_key]
    stable_pose = dataset.stable_pose(object_key, stp_id)

    # get images
    depth_images, ir_intrinsics = load_depth_images(cfg)
    depth_im = Image.median_images(depth_images)
    point_cloud_cam = ir_intrinsics.deproject(depth_im) # for debug only
    T_camera_world = RigidTransform.load(os.path.join(cfg['calib_dir'], '%s_registration.tf' %(ir_intrinsics.frame)))
    point_cloud_world = T_camera_world * point_cloud_cam

    # create registration solver
    registration_solver = KnownObjectStablePoseTabletopRegistrationSolver(object_key, stp_id, database.datasets[0], cfg, args.output_path)
    registration_result = registration_solver.register(depth_im,
                                                       ir_intrinsics,
                                                       T_camera_world,
                                                       debug=cfg['debug'])

    # visualize setup
    T_camera_world = RigidTransform.load(os.path.join(cfg['calib_dir'], '%s_registration.tf' %(ir_intrinsics.frame)))
    T_camera_obj = registration_result.T_camera_obj
    T_obj_world = T_camera_world * T_camera_obj.inverse() 
    vis.figure()
    vis.mesh(obj.mesh, T_obj_world, color=(1,1,1))
    vis.points(point_cloud_world, color=(0,1,0), subsample=20)
    vis.table(dim=0.5)
    vis.view(focalpoint=(0,0,0))
    vis.show()
