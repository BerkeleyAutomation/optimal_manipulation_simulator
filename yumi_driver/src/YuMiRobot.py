#! /usr/bin/env python
'''
Abstraction for the YuMi Robot
Authors: Jacky, Chris, Billy
'''
import logging
from YuMiEthernet import YuMiEthernet
from YuMiConstants import YuMiConstants as YMC
from YuMiListener import YuMiStatePublisher
import time
import rospy
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import actionlib, control_msgs.msg, sensor_msgs.msg
from YuMiState import YuMiState
import numpy as np

class YuMiRobot:

    def __init__(self, ip=YMC.IP, port_l = YMC.PORT_L, port_r = YMC.PORT_R, tcp = YMC.TCP_DEFAULT_GRIPER):
        '''Initializes a YuMiRobot
        
        Args: 
            ip: IP address of YuMi
            port_l: Port of left arm server
            port_r: Port of right arm server
        '''
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self._ysp = YuMiStatePublisher(ip, 6000, 6001)
        self._ysp.start()

        # this sleep is necessary if you run yumi.launch for opening rviz, but not if you run real_demo and the test script separately (and don't cancel real_demo)
        # time.sleep(10)


        # Instantiate a MoveGroupCommander object.  This object is an interface to one group of joints.  In this case 
        # the group is the joints in the left arm.  This interface can be used to plan and execute motions on the left arm.
        self.left_group = moveit_commander.MoveGroupCommander("left_arm")
        self.left_group.set_planning_time(7)

        self.right_group = moveit_commander.MoveGroupCommander("right_arm")
        self.right_group.set_planning_time(7)
        # self.right_group.set_max_velocity_scaling_factor(.001)
        # print "params: ", self.right_group.get_planner_params()
        self.left = YuMiEthernet(self.left_group, ip=ip, port=port_l)
        self.right = YuMiEthernet(self.right_group, ip=ip, port=port_r)
        self.tcp = tcp
        self.left.set_tool(self.tcp)
        self.right.set_tool(self.tcp)


    def stop(self):
        self.left.stop()
        self.right.stop()

    def goto_state_sync(self, left_state, right_state):
        '''Commands both arms to go to assigned states in sync. Sync means both
        motions will end at the same time.
        
        Args:
            left_state: target state for left arm
            right_state: target state for right arm
        '''
        self.left.goto_state_sync(left_state)
        self.right.goto_state_sync(right_state)
    
    def goto_pose_sync(self, left_pose, right_pose):
        '''Commands both arms to go to assigned poses in sync. Sync means both
        motions will end at the same time.
        
        Args:
            left_pose: target pose for left arm
            right_pose: target pose for right arm
        '''
        self.left.goto_pose_sync(left_pose)
        self.right.goto_pose_sync(right_pose)

    def move_delta_sync(self, left_delta, right_delta):
        '''Commands both arms to go to assigned delta poses in sync. Sync means 
        both motions will end at the same time.
        
        Args:
            left_delta: target delta for left arm
            right_delta: target delta for right arm
        '''
        self.left.move_delta(left_delta, sync=True)
        self.right.move_delta(right_delta, sync=True)
        
    def transform_sync(self, left_transform, right_transform):
        '''Commands both arms to go to assigned delta pose transformations in sync. 
        Sync means both motions will end at the same time.
        
        Args:
            left_transform: target transform for left arm
            right_transform: target transform for right arm
        '''
        self.left.transform(left_transform, sync=True)
        self.right.transform(right_transform, sync=True)
        
    def set_v(self, n):
        '''Sets speed for both arms using n as the speed number.
        
        Args:
            n: speed number. If n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
        '''
        v = YuMiRobot.get_v(n)
        self.left.set_speed(v)
        self.right.set_speed(v)
        
    def set_z(self, name):
        '''Sets zoning settings for both arms according to name.
        
        Args:
            name: Name of zone setting. ie: z10, z200, fine
        '''
        z = YuMiRobot.ZONES[name]
        point_motion = name == 'fine'
        self.left.set_zone(z, point_motion=point_motion)
        self.right.set_zone(z, point_motion=point_motion)
        
    def set_tool(self, pose):
        '''Sets TCP (Tool Center Point) for both arms using given pose as offset
        
        Args:
            pose: Pose of new TCP as offset from the default TCP
        '''
        self.left.set_tool(pose)
        self.right.set_tool(pose)
        
    def reset_home(self):
        '''Moves both arms to home position
        '''
        self.left.goto_state(YMC.L_HOME_STATE, wait_for_res=False)
        self.right.goto_state(YMC.R_HOME_STATE, wait_for_res=True)
    
    @staticmethod
    def construct_speed_data(tra, rot):
        '''Constructs a speed data tuple that's in the same format as ones used in RAPID.
        
        Args:
            tra: translational speed (milimeters per second)
            rot: rotational speed (degrees per second)
            
        Returns:
            A tuple of correctly formatted speed data: (tra, rot, tra, rot)
        '''
        return (tra, rot, tra, rot)
        
    @staticmethod
    def get_v(n):
        '''Gets the corresponding speed data for n as the speed number. 
        
        Args:
            n: speed number. If n = 100, will return the same speed data as v100 in RAPID
            
        Returns:
            Corresponding speed data using n as speed number
        '''
        return YuMiRobot.construct_speed_data(n, 500)
        
    @staticmethod
    def construct_zone_data(pzone_tcp, pzone_ori, zone_ori):
        '''Constructs tuple for zone data
        
        Args:
            pzone_tcp: path zone size for TCP
            pzone_ori: path zone size for orientation
            zone_ori: zone size for orientation
            
        Returns:
            A tuple of correctly formatted zone data: (pzone_tcp, pzone_ori, zone_ori)
        '''
        return (pzone_tcp, pzone_ori, zone_ori)

    ZONES = {
        'fine' : (0,0,0),#these values actually don't matter for fine
        'z0'  : (.3,.3,.03), 
        'z1'  : (1,1,.1), 
        'z5'  : (5,8,.8), 
        'z10' : (10,15,1.5), 
        'z15' : (15,23,2.3), 
        'z20' : (20,30,3), 
        'z30' : (30,45,4.5), 
        'z50' : (50,75,7.5), 
        'z100': (100,150,15), 
        'z200': (200,300,30)
    }
    
if __name__ == '__main__':
    logging.getLogger().setLevel(YMC.LOGGING_LEVEL)
    rospy.init_node("YuMiRobot")
    yumi = YuMiRobot(ip="192.168.125.1")
