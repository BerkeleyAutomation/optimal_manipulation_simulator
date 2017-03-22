#! /usr/bin/env python
import rospy
import numpy as np
import sys
dexnet_path = "/home/cloudminds/chomp_ws/src/dex-net-new-api"
sys.path.append(dexnet_path + "/src/")
from YuMiRobot import YuMiRobot
from YuMiState import YuMiState
from Scene import Scene
from DexnetChess import DexnetChess

HOME_JOINTS = [0,-130,-135,30,0,40,0,7]
RIGHT_JOINTS = [26.73, -106.19, 23.41, 17.67, 33.93, 114.98, -89.57]

if __name__ == "__main__":
    rospy.init_node("test_dexnet")
    if len(sys.argv) == 2:
        add_to_scene = sys.argv[1] == "True"
    else:
        add_to_scene = True
    purturbate = True

    #first one is for real robot, second is for simulation
    # yumi = YuMiRobot(ip="192.168.125.1")
    yumi = YuMiRobot(ip="127.0.0.1")
    scene = Scene(yumi, dexnet_path)
    dexnet_chess = DexnetChess(dexnet_path)

    position = [.52, 0, .02]
    rook_quaternion = [0,.061,0,.998] # [0,1,0,0]
    position, rook_quaternion = scene.update_scene(position, rook_quaternion, yumi, add_to_scene)

    # home position
    yumi.right.goto_state(YuMiState(vals=HOME_JOINTS))

    # right arm position
    yumi.right.goto_state(YuMiState(vals=RIGHT_JOINTS))

    yumi.right.open_gripper()

    T_world_mesh, T_world_tip = dexnet_chess.get_rook_grasp(position, rook_quaternion)
    # add_gripper(yumi, T_world_mesh)
    
    # dexnet puts the w term last
    q = T_world_tip.quaternion
    new_quaternion = [q[3], q[0], q[1], q[2]] 
    path = yumi.right.collision_free_move(T_world_tip.translation, new_quaternion)
