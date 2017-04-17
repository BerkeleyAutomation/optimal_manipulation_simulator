"""
Script to run grasp experiments with the YuMi
Authors: Jeff Mahler, Jacky Liang
"""
import copy
import IPython
import numpy as np
import os
import logging
import sys
import argparse

from time import sleep
import matplotlib.pyplot as plt

from dexnet import Hdf5Database, RenderedImage, ExperimentConfig, CNN_Hdf5ObjectStablePoseIndexer, \
                   DepthCNNQueryImageGenerator, KnownObjectStablePoseTabletopRegistrationSolver, \
                   RobotGripper, ExperimentLogger

from alan.rgbd import Kinect2Sensor, Image
from alan.core import RigidTransform, Box
from alan.core import Visualizer as vis
from alan.control import YuMiRobot, YuMiConstants, ResetDevice

class EvaluateGrasps:

    def __init__(self, cfg, output_path):
        self.cfg = cfg
        self.output_path = output_path
        self.elog = ExperimentLogger(output_path)

        # init robot
        self.robot = YuMiRobot()
        self.robot.set_v(cfg['control']['velocity'])
        self.robot.reset_home()
        self.robot.open_grippers()

        # useful configs
        self.object_key = cfg['object_key']
        self.stp_id = cfg['stp_id']

        # open db
        db_filename = os.path.join(self.cfg['database_dir'], self.cfg['database_name'])
        database = Hdf5Database(db_filename, self.cfg)
        self.dataset = database.datasets[0]
        self.gripper = RobotGripper.load(self.cfg['gripper'])
        self.obj = self.dataset[self.object_key]
        self.stable_pose = self.dataset.stable_pose(self.object_key, self.stp_id)

        # load computed grasps to execute
        self._set_grasps_to_execute()

        # create registration solver
        self.registration_solver = KnownObjectStablePoseTabletopRegistrationSolver(self.object_key, self.stp_id, 
                                                                            self.dataset, self.cfg, self.output_path)

    def _load_depth_images(self):
        sensor = Kinect2Sensor(device_num=self.cfg['sensor']['device_num'], frame=self.cfg['sensor']['frame'])
        sensor.start()
        ir_intrinsics = sensor.ir_intrinsics

        # get raw images
        depths = []

        for _ in range(cfg['num_images']):
            _, depth, _ = sensor.frames()
            depths.append(depth)

        sensor.stop()

        return depths, ir_intrinsics

    def _register_obj(self):

        # get images
        depth_images, ir_intrinsics = self._load_depth_images()
        depth_im = Image.median_images(depth_images)
        point_cloud_cam = ir_intrinsics.deproject(depth_im) # for debug only
        self.T_camera_world = RigidTransform.load(os.path.join(cfg['calib_dir'], '%s_registration.tf' %(ir_intrinsics.frame)))
        self.point_cloud_world = self.T_camera_world * point_cloud_cam

        # register
        registration_result = self.registration_solver.register(depth_im, ir_intrinsics, self.T_camera_world, debug=self.cfg['debug'])
        
        # parse output
        self.T_camera_obj = registration_result.T_camera_obj
        T_obj_world = self.T_camera_world * self.T_camera_obj.inverse() 

        self.T_obj_world = T_obj_world
        self.T_base_world = RigidTransform(from_frame='base', to_frame='world')
        return T_obj_world

        
        # return T_camera_obj, T_obj_world, T_base_world

    def _choose_grasp(self, T_obj_world):
        # read grasps
        candidate_grasps = []
        candidate_gripper_poses = []
        candidate_grasp_metrics = []
        for grasp in self.grasps_to_execute:
            aligned_grasp = grasp.parallel_table(self.stable_pose)
            T_gripper_world = T_obj_world * aligned_grasp.gripper_pose(self.gripper)
            world_x_axis_angle = np.arccos(T_gripper_world.z_axis[1])
            if world_x_axis_angle < np.pi * cfg['control']['approach_angle_thresh'] and \
               T_gripper_world.translation[2] > cfg['control']['grasp_height_thresh']:
                candidate_grasps.append(aligned_grasp)
                candidate_gripper_poses.append(T_gripper_world)
                candidate_grasp_metrics.append(self.grasp_metrics[aligned_grasp.grasp_id]['ppc_pose_5_f_0.050000_tg_0.002500_rg_0.050000_to_0.002500_ro_0.050000'])

        # setup grasp
        candidate_grasps_and_metrics = zip(candidate_grasps,
                                           candidate_gripper_poses,
                                           candidate_grasp_metrics)
        candidate_grasps_and_metrics.sort(key=lambda x: x[2], reverse=True)
        target_grasp = candidate_grasps_and_metrics[0][0]
        target_gripper_pose = candidate_grasps_and_metrics[0][1]
        target_grasp_metrics = self.grasp_metrics[target_grasp.grasp_id]
        logging.info('Executing grasp %d' %(target_grasp.grasp_id))
        for metric_name, metric_val in target_grasp_metrics.iteritems():
            logging.info('Quality according to %s: %f' %(metric_name, metric_val))

        return target_grasp, target_gripper_pose

    def _perform_grasp(self, target_gripper_pose):
        # approach pose
        t_target_approach = np.array([0,0,cfg['control']['approach_dist']])
        T_target_approach = RigidTransform(translation=t_target_approach,
                                           from_frame='gripper',
                                           to_frame='approach')
        approach_gripper_pose = target_gripper_pose * T_target_approach.inverse()

        # lift pose
        t_lift = np.array([0,0,cfg['control']['lift_height']])
        T_lift = RigidTransform(translation=t_lift,
                                   from_frame='world', to_frame='world')
        lift_gripper_pose = T_lift * target_gripper_pose

        # visualize setup
        vis.figure()
        vis.mesh(self.obj.mesh, self.T_obj_world, color=(1,1,1))
        vis.points(self.point_cloud_world, color=(0,1,0), subsample=20)
        vis.pose(self.T_camera_world, alpha=0.1)
        vis.pose(self.T_base_world, alpha=0.1)
        vis.table(dim=0.2)
        vis.pose(target_gripper_pose, alpha=0.1)
        vis.pose(approach_gripper_pose, alpha=0.1)
        vis.pose(lift_gripper_pose, alpha=0.1)
        vis.view(focalpoint=(0,0,0))
        vis.show()
        exit(0)

        # open YuMi
        self.robot.right.goto_pose(approach_gripper_pose)
        self.robot.right.goto_pose(target_gripper_pose)
        self.robot.right.close_gripper(wait_for_res=True)
        self.robot.right.goto_pose(lift_gripper_pose)
        sleep(5)

        # TODO: Take picture and save

        self.robot.right.open_gripper()

    def _try_single_grasp(self):
        self.robot.right.open_gripper(wait_for_res=False)

        # reset object
        logging.info("Resetting object...")
        ResetDevice.reset()
        logging.info("Finished Reset")

        # register object
        logging.info("Registering obejct...")
        T_obj_world = self._register_obj()
        logging.info("Finished registration")

        # choose grasp
        logging.info("Choosing target grasp...")
        target_grasp, target_gripper_pose =self. _choose_grasp(T_obj_world)
        logging.info("Finished choosing grasp")

        # perform grasp
        logging.info("Performing grasp...")
        self._perform_grasp(target_gripper_pose)        

    def run_evaluate_grasps(self):
        ''' Evaluate a list of candidate grasps via physical experiment '''

        for trial_i in range(self.cfg['num_grasp_trials']):
            self.robot.reset_home()
            self.robot.right.goto_pose(YuMiConstants.R_AWAY_POSE)
            self._try_single_grasp()

        self.robot.reset_home()
        self.robot.stop()

    def _set_grasps_to_execute(self):
        ''' Load the list of candidate grasps to experiment '''

        # read in params
        self.grasps_to_execute = self.dataset.grasps(self.object_key, gripper=self.gripper.name)
        self.grasp_metrics = self.dataset.grasp_metrics(self.object_key, self.grasps_to_execute, gripper=self.gripper.name)

        # TODO: Remove after grasps are actually generated
        return

        # load the list of grasps to execute
        candidate_grasp_dir = os.path.join(self.cfg['grasp_candidate_dir'], ds.name)
        grasp_id_filename = os.path.join(candidate_grasp_dir,
                                        '{0}_{1}_{2}_grasp_ids.npy'.format(object_name, stable_pose.id, cfg['gripper']))
        grasp_ids = np.load(grasp_id_filename)
        grasps_to_execute = []
        for grasp in grasps:
            if grasp.grasp_id in grasp_ids and grasp.grasp_id >= config['start_grasp_id']:
                grasps_to_execute.append(grasp.grasp_aligned_with_stable_pose(stable_pose))
                if len(grasps_to_execute) >= 3:
                    break

        self.grasps_to_execute = grasps_to_execute
        self.grasp_metrics = dataset.grasp_metrics(object_key, grasps_to_execute, gripper=gripper.name)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path')
    args = parser.parse_args()

    # open configuration
    cfg_file = 'cfg/scripts/evaluate_grasps.yaml'
    cfg = ExperimentConfig(cfg_file)
    logging.getLogger().setLevel(logging.INFO)

    eval_grasps = EvaluateGrasps(cfg, args.output_path)
    eval_grasps.run_evaluate_grasps()