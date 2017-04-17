"""
+X is front, +Y is left, +Z is up
"""
import copy
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
import numpy as np

USE_OPENRAVE = True
try:
    import openravepy as rave
except:
    logging.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False

import database as db
import grasp as g
import graspable_object
import gripper as gr
import obj_file
import sdf_file
import similarity_tf as stf
import tfx

import IPython

class OpenRaveGraspChecker(object):
    # global environment vars
    env_ = None

    def __init__(self, gripper, env = None, view = True, win_height = 1200, 
                 win_width = 1200, cam_dist = 0.5):
        """ Defaults to using the Zeke gripper """
        if not USE_OPENRAVE:
            raise ValueError('Cannot instantiate OpenRave collision checker')

        if env is None and OpenRaveGraspChecker.env_ is None:
            OpenRaveGraspChecker._setup_rave_env()        

        self.object_ = None
        self.view_ = view
        self.gripper_ = gripper
        self.gripper_obj_ = self._load_obj_from_file(gripper.mesh_filename)
        if view:
            self._init_viewer(win_height, win_width, cam_dist)

    def __del__(self):
        self.env.Remove(self.gripper_obj_)
        if self.obj_ is not None:
            self.env.Remove(self.obj_)

    @property
    def env(self):
        if OpenRaveGraspChecker.env_ is None:
            OpenRaveGraspChecker._setup_rave_env()
        return OpenRaveGraspChecker.env_

    @staticmethod
    def _setup_rave_env():
        """ OpenRave environment """
        OpenRaveGraspChecker.env_ = rave.Environment()
     
    def _init_viewer(self, height, width, cam_dist):
        """ Initialize the OpenRave viewer """
        # set OR viewer
        OpenRaveGraspChecker.env_.SetViewer("qtcoin")
        viewer = self.env.GetViewer()
        viewer.SetSize(width, height)

        T_cam_obj = np.eye(4)
        R_cam_obj = np.array([[0,  0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])
        T_cam_obj[:3,:3] = R_cam_obj
        T_cam_obj[0,3] = -cam_dist
        self.T_cam_obj_ = T_cam_obj

        # set view based on object
        self.T_obj_world_ = np.eye(4)
        self.T_cam_world_ = self.T_obj_world_.dot(self.T_cam_obj_)
        viewer.SetCamera(self.T_cam_world_, cam_dist)

    def _load_obj_from_file(self, filename):
        if filename is None or filename == '':
            raise ValueError('Object to load must have a valid mesh filename to use OpenRave grasp checking')
        self.env.Load(filename)
        obj = self.env.GetBodies()[-1]
        return obj

    def _load_object(self, graspable_object):
        """ Load the object model into OpenRave """ 
        if graspable_object.model_name is None:
            raise ValueError('Graspable object model file name must be specified!')

        # load object model
        object_mesh_filename = graspable_object.model_name 
        return self._load_obj_from_file(object_mesh_filename)

    def move_to_pregrasp(self, T_gripper_obj, eps=1e-2):
        """ Move the gripper to the pregrasp pose given by the grasp object """
        # get grasp pose in rave
        T_mesh_obj = self.gripper_.T_mesh_gripper.dot(T_gripper_obj)

        gripper_position = T_mesh_obj.inverse().pose.position
        gripper_orientation = T_mesh_obj.inverse().pose.orientation
        gripper_pose = np.array([gripper_orientation.w, gripper_orientation.x, gripper_orientation.y, gripper_orientation.z, gripper_position.x, gripper_position.y, gripper_position.z])
        T_mesh_obj = rave.matrixFromPose(gripper_pose)    
        self.gripper_obj_.SetTransform(T_mesh_obj)

        return T_mesh_obj
        
    def view_grasps(self, graspable, grasps, auto_step=False, delay = 1):
        """ Display all of the grasps """
        if self.env.GetViewer() is None and self.view_:
            self.env.SetViewer('qtcoin')
        ind = 0
        logging.debug("Total grasps: %d"%len(grasps))
        self.set_object(graspable)

        for grasp in grasps:
            logging.debug('Visualizing grasp %d' %(ind))

            self.move_to_pregrasp(grasp.gripper_transform(self.gripper_))
            in_collision = self.in_collision(grasp)
            logging.debug("Colliding? {0}".format(in_collision))

            if auto_step:
                time.sleep(delay)
            else:
                user_input = "x"
                while user_input != '':
                    user_input = raw_input()
            ind += 1
        self.env.Remove(self.obj_)

    def set_view(self, view):
        self.view_ = view

    def set_object(self, graspable):
        """ Temporarily set object for speed ups """
        self.obj_ = self._load_object(graspable)
        
    def in_collision(self, grasp):
        """ Check collision of grasp with current objects """
        if isinstance(grasp, g.ParallelJawPtGrasp3D):
            grasp = grasp.gripper_transform(self.gripper_)

        if self.obj_ is None:
            logging.warning('Cannot use fast collision check without preloaded object')
            return False

        if self.env.GetViewer() is None and self.view_:
            self.env.SetViewer('qtcoin')

        self.move_to_pregrasp(grasp)
        return self.env.CheckCollision(self.gripper_obj_, self.obj_)

    def collision_between(self, graspable, grasp):
        """ Returns true if the gripper collides with grraspable in the test_grasp, false otherwise"""
        if self.env.GetViewer() is None and self.view_:
            self.env.SetViewer('qtcoin')

        self.obj_ = self._load_object(graspable)

        self.move_to_pregrasp(grasp.gripper_transform(self.gripper_))
        in_collision = self.env.CheckCollision(self.gripper_obj_, self.obj_)

        self.env.Remove(self.obj_)
        return in_collision
