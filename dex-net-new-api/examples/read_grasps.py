"""
Example grasp loading from hdf5
Author: Jeff Mahler
"""
import copy
import logging
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numpy as np
import os
import random
import sys

sys.path.append("./src/")
sys.path.append("/home/cloudminds/catkin_ws/src/yumi_cloudminds/yumi_driver/src/")
import dexnet.database as db
import dexnet.experiment_config as ec
import dexnet.gripper as gr
import dexnet.obj_file as objf
import dexnet.similarity_tf as stf
import dexnet.stable_pose_class as stp
import tfx

from YuMiRobot import YuMiRobot
class Pose:
    @property
    def position(self):
        return self._position
    
    @property
    def rotation(self):
        return self._rotation

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z
    
if __name__ == '__main__':
    config_filename = 'cfg/examples/read_grasps.yaml'
    logging.getLogger().setLevel(logging.INFO)

    # read config file and open database
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, access_level=db.READ_ONLY_ACCESS)

    # read gripper
    gripper = gr.RobotGripper.load(config['gripper_name'])

    # loop through available datasets
    dataset_name = config['dataset']
    dataset = database.dataset(dataset_name)

    # loop through objects in dataset
    object_key = config['object_name']
    obj = dataset[object_key]
    logging.info('Reading grasps for object {}'.format(object_key))


    dataset.obj_mesh_filename("WizRook", output_dir="data/meshes/chess_pieces/")

    # read grasps and grasp metadata
    grasps = dataset.grasps(object_key, gripper=gripper.name)
    grasp_features = dataset.grasp_features(object_key, grasps, gripper=gripper.name)
    grasp_metrics = dataset.grasp_metrics(object_key, grasps, gripper=gripper.name)

    # read in object stable poses
    stable_pose = dataset.stable_pose(object_key, config['stable_pose_id'])

    # grasp = grasps[65]




    # T_obj_world = stf.SimilarityTransform3D(from_frame='world', to_frame='obj')
    # T_obj_world.translation = (.3, 0, .3)
    # T_obj_world.rotation = (0, 0, 0.70710678118654757, 0.70710678118654746)

    # T_gripper_obj = grasp.gripper_pose(gripper).inverse()
    # T_mesh_obj = gripper.T_mesh_gripper.dot(T_gripper_obj)
    # T_mesh_world = T_mesh_obj.dot(T_obj_world)
    # yumi = YuMiRobot(ip="192.168.125.1")
    # pose = Pose()
    # pose.position = T_mesh_world.translation
    # pose.rotation = T_mesh_world.rotation
    # yumi.left.goto_pose(pose)


    # get gripper pose matrix parallel with the table for each stable pose
    for grasp in grasps:
        # align grasp with table
        grasp_parallel_table = grasp.parallel_table(stable_pose)

        # convert to a similarity transform pose object
        gripper_pose = grasp_parallel_table.gripper_pose(gripper)
        gripper_t = gripper_pose.translation
        gripper_q = gripper_pose.quaternion

        print 'GRASP', grasp.grasp_id
        print 'translation', gripper_t
        print 'rotation', gripper_q
        print ''

        # TODO: Publish grasp pose here!

