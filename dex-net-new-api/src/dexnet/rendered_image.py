"""
Wrapper for images rendered in Maya
Author: Jeff Mahler
"""
import IPython
import logging
import numpy as np

USE_ALAN=True
try:
    from alan.core import RigidTransform
except:
    logging.warning('Failed to import ALAN')
    USE_ALAN=False

class RenderedImage:
    """ Class to encapculate data from rendered images from maya """

    def __init__(self, image, cam_pos=np.zeros(3), cam_rot=np.zeros(3),
                 cam_interest_pt=np.zeros(3), image_id=-1, stable_pose=None,
                 obj_key=None):
        self.image = image
        self.cam_pos = cam_pos # camera pos relative to world
        self.cam_rot = cam_rot # camera rotation relative to world in... euler angles :(
        self.cam_interest_pt = cam_interest_pt # point that the camera is fixated on (for determining optical axis)
        self.id = image_id
        self.stable_pose = stable_pose
        self.obj_key = obj_key
        self.descriptors = {}

    def object_to_stp_transform(self):
        """ The rotation from the object frame to the stable pose """
        R_obj_stp = self.stable_pose.r
        return RigidTransform(rotation=R_obj_stp,
                              from_frame='obj', to_frame='stp')

    def stp_to_camera_transform(self):
        """ Returns the transformation from the object stable pose to the virtual camera frame where the image was rendered """
        # setup variables
        camera_xyz_w = self.cam_pos
        camera_rot_w = self.cam_rot
        camera_int_pt_w = self.cam_interest_pt
        camera_xyz_obj_p = camera_xyz_w - camera_int_pt_w
        
        # get the distance from the camera to the world
        camera_dist_xy = np.linalg.norm(camera_xyz_w[:2])
        z = [0,0,np.linalg.norm(camera_xyz_w[:3])]

        # form the rotations about the x and z axis for the object on the tabletop
        theta = camera_rot_w[0] * np.pi / 180.0
        phi = -camera_rot_w[2] * np.pi / 180.0 + np.pi / 2.0
        camera_rot_obj_p_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                                       [np.sin(phi), np.cos(phi), 0],
                                       [0, 0, 1]])

        camera_rot_obj_p_x = np.array([[1, 0, 0],
                                       [0, np.cos(theta), -np.sin(theta)],
                                       [0, np.sin(theta), np.cos(theta)]])
        
        # form the full rotation matrix, swapping axes to match maya
        camera_md = np.array([[0,  1,  0],
                              [1,  0,  0],
                              [0,  0, -1]])
        camera_rot_obj_p = camera_md.dot(camera_rot_obj_p_z.dot(camera_rot_obj_p_x))
        camera_rot_obj_p = camera_rot_obj_p.T
            
        # form the full object to camera transform
        R_stp_camera = camera_rot_obj_p
        t_stp_camera = np.array(z)
        return RigidTransform(rotation=R_stp_camera,
                              translation=t_stp_camera,
                              from_frame='stp', to_frame='camera')
    
    def camera_to_object_transform(self):
        """ Returns the transformation from camera to object when the object is in the given stable pose """
        # form the full object to camera transform
        T_stp_camera = self.stp_to_camera_transform()
        T_obj_stp = self.object_to_stp_transform()
        T_obj_camera = T_stp_camera.dot(T_obj_stp)
        return T_obj_camera

