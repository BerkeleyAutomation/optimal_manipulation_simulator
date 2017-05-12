#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
# from moveit_msgs.srv import ExecuteKnownTrajectory
from chomp_msgs.srv import SetInitialTrajectory
import moveit_commander
import sys
import tf
import numpy as np
from math import sin
from random import random
from rosgraph_msgs.msg import Log

class RobotController(object):
    def __init__(self):
        rospy.loginfo("To stop project CTRL + C")
        rospy.on_shutdown(self.shutdown)
        
        moveit_commander.roscpp_initialize(sys.argv)
        self.group = moveit_commander.MoveGroupCommander('manipulator')
        self.group.set_planning_time(5)
        self.display_planned_path_publisher = rospy.Publisher('manipulator/display_planned_path', DisplayTrajectory, queue_size=10)
        self.initial_trajectory_proxy = rospy.ServiceProxy('manipulator/set_initial_trajectory', SetInitialTrajectory)
        # self.ik_proxy = rospy.ServiceProxy('compute_ik', GetPositionIK)
        # listener = tf.TransformListener()
        # from_frame = 'odom'
        # to_frame = 'base_link'
        # listener.waitForTransform(from_frame, to_frame, rospy.Time(), rospy.Duration(5.0))
        self.joint_state_publisher = rospy.Publisher("joint_states", JointState, queue_size=10)
        rate = rospy.Rate(10);

    def shutdown(self):
        rospy.loginfo("Stopping project")
        rospy.sleep(1)

    def collision_free_plan(self, joint_start, joint_target, initial_trajectory=None, start_value=0):
        '''
        uses Moveit and OMPL to plan a path and generate the trajectory.  The 
        trajectory is sent point by point to the robot.  A final message is sent
        to signify the end of the trajectory and to trigger the motion.  
        '''

        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.group.get_joints()[:-1]
        joint_state.position = joint_start
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        self.group.set_start_state(robot_state)
        self.group.set_joint_value_target(joint_target)

        if initial_trajectory is not None:
            initial_trajectory.joint_trajectory.points = initial_trajectory.joint_trajectory.points[start_value:]

            self.initial_trajectory_proxy(initial_trajectory.joint_trajectory, 1, len(initial_trajectory.joint_trajectory.points)-2)
        else:
            self.initial_trajectory_proxy(JointTrajectory(), -1, -1)

        self.group.set_workspace([-3, -3, -3, 3, 3, 3])
        plan = self.group.plan()
        return plan

    def visualize_plan(self, plan):
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = plan.points[0]
        display_trajectory.trajectory.extend(plan.points)
        self.display_planned_path_publisher.publish(display_trajectory)

    def publish_joints(self, joints):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.group.get_joints()[:-1]
        joint_state.position = joints
        self.joint_state_publisher.publish(joint_state)

    # def execute(self, plan):

    #     trajectory = plan.joint_trajectory
    #     for i in range(len(trajectory.points)):


def add_obstacle(height, operation):
    planning_scene_publisher = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)

    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    rospy.sleep(2)
    pose = PoseStamped()
    pose.header.frame_id = robot.get_planning_frame()
    pose.pose.position.x = 1
    pose.pose.position.y = height - .2
    pose.pose.position.z = 1
    # pose.pose.orientation.w = 0.707
    # pose.pose.orientation.x = 0
    # pose.pose.orientation.y = 0.707
    # pose.pose.orientation.z = 0
    pose.pose.orientation.w = 1
    pose.pose.orientation.x = 0
    pose.pose.orientation.y = 0
    pose.pose.orientation.z = 0

    co = CollisionObject()
    co.operation = operation
    co.id = "obstacle"
    co.header = pose.header
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = (0.5, .7, 0.1)
    co.primitives = [box]
    co.primitive_poses = [pose.pose]
    planning_scene_publisher.publish(co)



    pose.pose.position.x = .7
    pose.pose.position.y = height + .7
    pose.pose.position.z = .9
    # pose.pose.orientation.w = 1
    # pose.pose.orientation.x = 0rostime
    # pose.pose.orientation.y = 0
    # pose.pose.orientation.z = 0
    pose.pose.orientation.w = 0.924
    pose.pose.orientation.x = 0
    pose.pose.orientation.y = 0
    pose.pose.orientation.z = 0.383
    co = CollisionObject()
    co.operation = operation
    co.id = "obstacle2"
    co.header = pose.header
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = (1, .1, 1)
    co.primitives = [box]
    co.primitive_poses = [pose.pose]

    planning_scene_publisher.publish(co)
    raw_input("press enter(1)")

def make_pose(position, orientation, frame):
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.pose.position.x = position[0]
    pose.pose.position.y = position[1]
    pose.pose.position.z = position[2]
    pose.pose.orientation.w = orientation[0]
    pose.pose.orientation.x = orientation[1]
    pose.pose.orientation.y = orientation[2]
    pose.pose.orientation.z = orientation[3]
    return pose

def update_collision_box(id, position, orientation=(1, 0, 0, 0), dimensions=(1, 1, 1), operation=CollisionObject.ADD):
    pose = make_pose(position, orientation, robot.get_planning_frame())
    co = CollisionObject()
    co.operation = operation
    co.id = id
    co.header = pose.header
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = dimensions
    co.primitives = [box]
    co.primitive_poses = [pose.pose]
    planning_scene_publisher.publish(co)

def remove_collision_box(id):
    co = CollisionObject()
    co.operation = CollisionObject.REMOVE
    co.id = id
    # co.header = pose.header
    # box = SolidPrimitive()
    # box.type = SolidPrimitive.BOX
    # box.dimensions = dimensions
    # co.primitives = [box]
    # co.primitive_poses = [pose.pose]
    planning_scene_publisher.publish(co)

def truncate_plan(plan, current_joints):
    trajectory = plan.joint_trajectory
    differences = []
    for point in trajectory.points:
        differences.append(np.linalg.norm(np.array(point.positions) - np.array(current_joints)))
    min_index = np.argmin(differences)

    trajectory.points = trajectory.points[min_index+2:] # hack
    time_from_start = trajectory.points[0].time_from_start

    for point in trajectory.points:
        point.time_from_start -= time_from_start

    return plan

def log_subscriber(logfile):

    def log_iterations(log):
        if log.level == 2:
            if log.msg.startswith("Terminated after"):
                logfile.write(log.msg)
                logfile.write('\n')
        elif log.level == 4:
            if log.msg.startswith("Fail: ABORTED"):
                logfile.write(log.msg)
                logfile.write('\n')

    rospy.Subscriber("rosout_agg", Log, log_iterations)

if __name__ == '__main__':
    rospy.init_node('robot_controller')
    planning_scene_publisher = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    robot_controller = RobotController()
    group = robot_controller.group
    rospy.sleep(1)
    log_subscriber(sys.stdout)

    joints1 = [0, 0, 0, 0, 0, 0]
    joints2 = [-0.0974195, 1.3523, 0.682611, 0.156142, 0.675658, -0.122225]

    # plan = robot_controller.collision_free_plan(joints1, joints2)
    # raw_input('Using 1st Plan')
    # group.execute(plan)

    # plan = robot_controller.collision_free_plan(joints2, joints1)
    # raw_input('Using 2nd Plan')
    # group.execute(plan)


    # rate = rospy.Rate(2)

    # group.set_joint_value_target(group.get_current_joint_values())

    # while not rospy.is_shutdown():
    #     pos = (rospy.get_rostime().secs / 5.0) % 1.0
    #     pos = 2.0 * pos - 1.0
    #     update_collision_box('obstacle', (0.7, 0 + pos, 0.9), (1, 0, 0, 0), (0.1, 0.4, 0.1))
    #     rate.sleep()

    #     if np.allclose(group.get_current_joint_values(), group.get_joint_value_target(), atol=0.01):
    #         robot_controller.collision_free_plan((0.5, 0.0, 0.15), (1, 0, 0, 0))
    # robot_controller.publish_joints(joints1)
    




    # getting to the initial state
    # remove_collision_box('obstacle')
    # plan = robot_controller.collision_free_plan(group.get_current_joint_values(), joints1)
    # group.execute(plan)

    # update_collision_box('obstacle', (0.7, 0, 0.9), (1, 0, 0, 0), (0.2, 0.4, 0.1))
    
    # raw_input('Press enter to start the demo')
    # plan = robot_controller.collision_free_plan(joints1, joints2)
    # print "Original planning time: ", group.get_planning_time()
    # group.execute(plan, wait=False)
    
    # time_to_wait = 0.5*random() + 0.5
    # time_to_wait = 0.75
    # rospy.sleep(time_to_wait)
    # group.stop()
    # # print "adding obstacle now"
    # update_collision_box('obstacle', (0.7, -.2, 0.9), (1, 0, 0, 0), (0.2, 0.8, 0.1))
    # # raw_input('Press enter to replan')

    # # plan.joint_trajectory.points = plan.joint_trajectory.points[find_current_index(plan, group.get_current_joint_values()):]
    # plan = truncate_plan(plan, group.get_current_joint_values())

    # plan = robot_controller.collision_free_plan(group.get_current_joint_values(), joints2, plan)
    # print "Replanning time: ", group.get_planning_time()
    # plan.joint_trajectory.points[0].positions = group.get_current_joint_values()
    # group.execute(plan)



    remove_collision_box('obstacle')
    plan = robot_controller.collision_free_plan(group.get_current_joint_values(), joints1)
    group.execute(plan)

    raw_input('Press enter to start the demo')
    print "\n"
    position = 0
    update_collision_box('obstacle', (0.7, 0, 0.9), (1, 0, 0, 0), (0.2, 0.4, 0.1))
    plan = robot_controller.collision_free_plan(group.get_current_joint_values(), joints2)

    iterations = 0
    while True:
        plan = robot_controller.collision_free_plan(group.get_current_joint_values(), joints2, plan)
        plan = truncate_plan(plan, group.get_current_joint_values())
        plan.joint_trajectory.points[0].positions = group.get_current_joint_values()
        group.execute(plan, wait=False)
        rospy.sleep(.2)
        # group.stop()
        
        position += .01
        update_collision_box('obstacle', (0.7, -position, 0.9), (1, 0, 0, 0), (0.2, 0.4+position*2, 0.1))
        iterations += 1
        print "iterations: ", iterations
    print "\n\n\n\nEnd of Project\n\n\n\n"
    
