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
import tfx
from time import sleep
import matplotlib.pyplot as plt

import dexnet.database as db
from dexnet.rendered_image import RenderedImage 
import dexnet.experiment_config as ec
from dexnet.cnn_database_indexer import CNN_Hdf5ObjectStablePoseIndexer
from dexnet.cnn_query_image_generator import DepthCNNQueryImageGenerator
from dexnet.tabletop_object_registration import KnownObjectStablePoseTabletopRegistrationSolver
from dexnet.gripper import RobotGripper

from alan.rgbd import Kinect2Sensor, Image
from alan.core import RigidTransform, Box
from alan.core import Visualizer as vis

from alan.control import YuMiRobot, YuMiConstants

def load_depth_images(cfg):
    sensor = Kinect2Sensor(device_num=cfg['sensor']['device_num'], frame=cfg['sensor']['frame'])
    sensor.start()
    ir_intrinsics = sensor.ir_intrinsics

    # get raw images
    depths = []

    for _ in range(cfg['num_images']):
        _, depth, _ = sensor.frames()
        depths.append(depth)

    sensor.stop()

    return depths, ir_intrinsics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # open configuration
    cfg_file = 'cfg/examples/execute_single_grasp.yaml'
    cfg = ec.ExperimentConfig(cfg_file)
    logging.getLogger().setLevel(logging.INFO)
    object_key = cfg['object_key']
    stp_id = cfg['stp_id']

    # open db
    db_filename = os.path.join(cfg['database_dir'], cfg['database_name'])
    database = db.Hdf5Database(db_filename, cfg)
    dataset = database.datasets[0]
    
    # read in params
    gripper = RobotGripper.load(cfg['gripper'])
    obj = dataset[object_key]
    stable_pose = dataset.stable_pose(object_key, stp_id)
    grasps = dataset.grasps(object_key, gripper=gripper.name)
    grasp_metrics = dataset.grasp_metrics(object_key, grasps, gripper=gripper.name)

    # create registration solver
    registration_solver = KnownObjectStablePoseTabletopRegistrationSolver(object_key, stp_id, dataset, cfg, args.output_path)

    # init robot
    robot = YuMiRobot()
    robot.set_v(cfg['control']['velocity'])
    robot.right.reset_home()
    robot.right.goto_pose(YuMiConstants.R_AWAY_POSE)

    # get images
    depth_images, ir_intrinsics = load_depth_images(cfg)
    depth_im = Image.median_images(depth_images)
    point_cloud_cam = ir_intrinsics.deproject(depth_im) # for debug only
    T_camera_world = RigidTransform.load(os.path.join(cfg['calib_dir'], '%s_registration.tf' %(ir_intrinsics.frame)))
    point_cloud_world = T_camera_world * point_cloud_cam

    # register
    registration_result = registration_solver.register(depth_im, ir_intrinsics, T_camera_world,debug=cfg['debug'])
    
    # parse output
    T_camera_obj = registration_result.T_camera_obj
    T_obj_world = T_camera_world * T_camera_obj.inverse() 
    T_base_world = RigidTransform(from_frame='base', to_frame='world')

    # read grasps
    candidate_grasps = []
    candidate_gripper_poses = []
    candidate_grasp_metrics = []
    for grasp in grasps:
        aligned_grasp = grasp.parallel_table(stable_pose)
        T_gripper_world = T_obj_world * aligned_grasp.gripper_pose(gripper)
        world_x_axis_angle = np.arccos(T_gripper_world.z_axis[1])
        if world_x_axis_angle < np.pi * cfg['control']['approach_angle_thresh'] and \
           T_gripper_world.translation[2] > cfg['control']['grasp_height_thresh']:
            candidate_grasps.append(aligned_grasp)
            candidate_gripper_poses.append(T_gripper_world)
            candidate_grasp_metrics.append(grasp_metrics[aligned_grasp.grasp_id]['ppc_pose_5_f_0.050000_tg_0.002500_rg_0.050000_to_0.002500_ro_0.050000'])

    # setup grasp
    candidate_grasps_and_metrics = zip(candidate_grasps,
                                       candidate_gripper_poses,
                                       candidate_grasp_metrics)
    candidate_grasps_and_metrics.sort(key=lambda x: x[2], reverse=True)
    target_grasp = candidate_grasps_and_metrics[0][0]
    target_gripper_pose = candidate_grasps_and_metrics[0][1]
    target_grasp_metrics = grasp_metrics[target_grasp.grasp_id]
    logging.info('Executing grasp %d' %(target_grasp.grasp_id))
    for metric_name, metric_val in target_grasp_metrics.iteritems():
        logging.info('Quality according to %s: %f' %(metric_name, metric_val))

    # approach pose
    t_target_approach = np.array([0,0,cfg['control']['approach_dist']])
    T_target_approach = RigidTransform(translation=t_target_approach,
                                       from_frame='gripper',
                                       to_frame='approach')
    approach_gripper_pose = target_gripper_pose * T_target_approach.inverse()

    # lift pose
    t_lift = np.array([0,0,cfg['control']['lift_height']])
    T_lift = RigidTransform(translation=t_lift,
                                       from_frame='world',
                                       to_frame='world')
    lift_gripper_pose = T_lift * target_gripper_pose

    # visualize setup
    """
    vis.figure()
    vis.mesh(obj.mesh, T_obj_world, color=(1,1,1))
    vis.points(point_cloud_world, color=(0,1,0), subsample=20)
    vis.pose(T_camera_world, alpha=0.1)
    vis.pose(T_base_world, alpha=0.1)
    vis.table(dim=0.2)
    vis.pose(target_gripper_pose, alpha=0.1)
    vis.pose(approach_gripper_pose, alpha=0.1)
    vis.pose(lift_gripper_pose, alpha=0.1)
    vis.view(focalpoint=(0,0,0))
    vis.show()
    """

    # open YuMi
    robot.right.open_gripper()
    sleep(0.5)
    robot.right.goto_pose(approach_gripper_pose, wait_for_res=True)
    robot.right.goto_pose(target_gripper_pose, wait_for_res=True)
    robot.right.close_gripper(wait_for_res=True)
    robot.right.goto_pose(lift_gripper_pose)
    sleep(5)
    robot.right.open_gripper()
    robot.stop()
