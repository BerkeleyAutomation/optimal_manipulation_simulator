'''
Constants for YuMi interface and control
Author: Jacky
'''
import logging
from YuMiState import YuMiState
from collections import namedtuple
from tfx import pose

class YuMiConstants:

    IP = '192.168.125.1'
    PORT_L = 5000
    PORT_R = 5001
    BUFSIZE = 4096
    TIMEOUT = 10

    # used to rate limit real-time YuMi controls
    COMM_PERIOD = 0.01
    
    DEBUG = False
    LOGGING_LEVEL = logging.INFO
    
    CMD_CODES = {
        'ping': 0,
        'goto_pose':1,
        'goto_joints': 2,
        'get_pose': 3,
        'get_joints':4,
        'set_tool':6,
        'set_speed':8,
        'set_zone':9,
        
        'goto_pose_sync':11,
        'goto_joints_sync':12,
        'goto_pose_delta':13,
        
        'close_gripper': 20,
        'open_gripper': 21,
        'calibrate_gripper': 22,
        'set_gripper_max_speed': 23,
        'set_gripper_force': 24,
        'move_gripper': 25,
        
        'set_circ_point':35,
        'move_by_circ_point':36,
        'buffer_add': 30,
        'buffer_clear': 31,
        'buffer_size': 32,
        'buffer_move': 33,
        'add_to_trajectory_buffer': 40,
        'follow_trajectory_buffer': 41,
        
        'close_connection': 99,
        
        'reset_home': 100,
    }
    
    RES_CODES = {
        'failure': 0,
        'success': 1
    }
    
    TCP_DEFAULT_GRIPER = pose((0,0,136), (0,0,0,1))
    
    L_HOME_STATE = YuMiState([0, -130, 30, 0, 40, 0, 135])
    L_HOME_POSE = pose([-7.3, 183.71, 199.51], [0.07395, 0.82584, -0.10624, 0.54885])
    
    R_HOME_STATE = YuMiState([0, -130, 30, 0, 40, 0, -135])
    R_HOME_POSE = pose([-10.1, -181.6, 197.75], [0.06481, -0.84133, -0.11456, -0.52426])
    
    R_FORWARD_STATE = YuMiState([9.66, -133.36, 34.69, -13.19, 28.85, 28.81, -110.18])
    R_FORWARD_POSE = pose([70.58, -265.19, 197.75], [0.0648, -0.84133, -0.11454, -0.52425])
    
    L_RAISED_STATE = YuMiState([5.5, -99.46, 20.52, -21.03, 67, -22.31, 110.11])
    L_RAISED_POSE = pose([-7.3, 399.02, 318.28], [0.07398, 0.82585, -0.10630, 0.54882])
    
    L_FORWARD_STATE = YuMiState([-21.71, -142.45, 43.82, 31.14, 10.43, -40.67, 106.63])
    L_FORWARD_POSE = pose([138.85, 215.43, 197.74], [0.07064, 0.82274, -0.1009, 0.55491])
