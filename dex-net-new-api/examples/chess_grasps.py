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

print sys.path

import dexnet.database as db
import dexnet.experiment_config as ec
import dexnet.gripper as gr
import dexnet.obj_file as objf
import dexnet.similarity_tf as stf
import dexnet.stable_pose_class as stp
import tfx

if __name__ == '__main__':
    config_filename = 'cfg/examples/chess_grasps.yaml'
    logging.getLogger().setLevel(logging.INFO)

    # read config file and open database
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, access_level=db.READ_ONLY_ACCESS)

    # read gripper
#    gripper = gr.RobotGripper.load(config['gripper_name'])
    gripper = gr.RobotGripper.load(config['gripper_name'])

    # loop through available datasets
    dataset_name = config['dataset']
    dataset = database.dataset(dataset_name)

    # loop through objects in dataset
    object_key = config['object_name']
    obj = dataset[object_key]
    logging.info('Reading grasps for object {}'.format(object_key))

    # read grasps and grasp metadata
    grasps = dataset.grasps(object_key, gripper=gripper.name)
   # grasp_features = dataset.grasp_features(object_key, grasps, gripper=gripper.name)
   # grasp_metrics = dataset.grasp_metrics(object_key, grasps, gripper=gripper.name)

    grasp_features = dataset.grasp_features(object_key, grasps, gripper='cloud_mind')
    grasp_metrics = dataset.grasp_metrics(object_key, grasps, gripper='cloud_mind')
    
    # read in object stable poses
    stable_pose = dataset.stable_pose(object_key, config['stable_pose_id'])

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

