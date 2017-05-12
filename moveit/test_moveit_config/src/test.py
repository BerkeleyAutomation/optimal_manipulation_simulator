#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory, RobotState, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import ExecuteKnownTrajectory
import moveit_commander
import sys
import tf
import numpy as np


def add_obstacle(height, operation):
    planning_scene_publisher = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)

    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    rospy.sleep(2)
    pose = PoseStamped()
    pose.header.frame_id = robot.get_planning_frame()
    pose.pose.position.x = .6
    pose.pose.position.y = 0
    pose.pose.position.z = .6
    pose.pose.orientation.w = 0.707
    pose.pose.orientation.x = 0
    pose.pose.orientation.y = 0.707
    pose.pose.orientation.z = 0
    
    co = CollisionObject()
    co.operation = operation
    co.id = "obstacle"
    co.header = pose.header
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = (0.1, 0.5, 0.5)
    co.primitives = [box]
    co.primitive_poses = [pose.pose]

    planning_scene_publisher.publish(co)

def collision_free_move():
    # self.group.set_start_state_to_current_state()
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander('group')
    group.set_planning_time(5)
    group.set_joint_value_target([.1])
    plan = group.plan()
    # while len(plan.joint_trajectory.points) == 0:
    #     plan = self.group.plan()
    #     print "retrying planning"
    # points = plan.joint_trajectory.points

    return plan


if __name__ == '__main__':
    rospy.init_node('test_controller')
    add_obstacle(-.25, CollisionObject.ADD)
    collision_free_move()
