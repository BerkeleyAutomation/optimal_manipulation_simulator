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

class YumiController(object):
    def __init__(self):
        rospy.loginfo("To stop project CTRL + C")
        rospy.on_shutdown(self.shutdown)
        
        moveit_commander.roscpp_initialize(sys.argv)
        self.group = moveit_commander.MoveGroupCommander('right_arm')
        self.group.set_planning_time(5)
        self.display_planned_path_publisher = rospy.Publisher('right_arm/display_planned_path', DisplayTrajectory, queue_size=10)
        self.initial_trajectory_proxy = rospy.ServiceProxy('right_arm/set_initial_trajectory', ExecuteKnownTrajectory)
        # self.ik_proxy = rospy.ServiceProxy('compute_ik', GetPositionIK)
        # listener = tf.TransformListener()
        # from_frame = 'odom'
        # to_frame = 'base_link'
        # listener.waitForTransform(from_frame, to_frame, rospy.Time(), rospy.Duration(5.0))

        rate = rospy.Rate(10);

    def shutdown(self):
        rospy.loginfo("Stopping project")
        rospy.sleep(1)

    def collision_free_move(self, position, qn, initial_trajectory=None):
        '''
        uses Moveit and OMPL to plan a path and generate the trajectory.  The 
        trajectory is sent point by point to the yumi.  A final message is sent
        to signify the end of the trajectory and to trigger the motion.  
        '''

        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose.position.x = position[0]
        target_pose.pose.position.y = position[1]
        target_pose.pose.position.z = position[2]
        # target_pose.pose.orientation.w = qn[0]
        # target_pose.pose.orientation.x = qn[1]
        # target_pose.pose.orientation.y = qn[2]
        # target_pose.pose.orientation.z = qn[3]

        print target_pose
        # self.group.set_start_state_to_current_state()

        current_joint_values = self.group.get_current_joint_values()
        start_state = RobotState()
        start_state.joint_state.name = self.group.get_joints()[:7]
        start_state.joint_state.position = current_joint_values
        # self.group.set_start_state(start_state)
        # self.group.set_pose_target(target_pose)
        broadcaster.sendTransform(position,
                [0,0,0,1],
                rospy.Time.now(),
                "target",
                "base_link")
        # print current_joint_values
        # current_joint_values[5] += 0.1
        # self.group.set_joint_value_target([0.548433, 0.554694,-1.91801,0.255809, -4.85266, 0.631296, 0.235168])
        # self.group.set_joint_value_target([-1.14074, 0.0477415, 0.323553, 0.338494, 2.37486, -0.126762, -2.58356])


        # THESE ARE THE GOOD JOINTS
        # self.group.set_joint_value_target([2.37814, -0.989366, -0.512554, -1.06674, -2.74682, 2.3268, 2.81229])
        self.group.set_joint_value_target([1.75572, -0.273738, -2.18553, -0.840665, -0.13439, 0.614535, -3.83207])


        if initial_trajectory is not None:
            self.initial_trajectory_proxy(initial_trajectory, False)
        else:
            self.initial_trajectory_proxy(RobotTrajectory(), False)
        plan = self.group.plan()
        while len(plan.joint_trajectory.points) == 0:
            plan = self.group.plan()
            print "retrying planning"
        # points = plan.joint_trajectory.points

        return plan

    def visualize_plan(self, plan):
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = plan.points[0]
        display_trajectory.trajectory.extend(plan.points)
        self.display_planned_path_publisher.publish(display_trajectory)

# def add_obstacle(height):
#     scene = moveit_commander.PlanningSceneInterface()
#     robot = moveit_commander.RobotCommander()

#     rospy.sleep(2)
#     pose = PoseStamped()
#     pose.header.frame_id = robot.get_planning_frame()
#     pose.pose.position.x = .25
#     pose.pose.position.y = height
#     pose.pose.position.z = .05
#     pose.pose.orientation.w = 1
#     pose.pose.orientation.x = 0
#     pose.pose.orientation.y = 0
#     pose.pose.orientation.z = 0
#     # scene.add_mesh("obstacle", pose, "/home/cc/ee106b/sp17/class/ee106b-aag/chomp_ws/src/project/src/box.obj", [1,1,1])
#     scene.add_box("obstacle", pose, (0.1, 0.5, 0.5))
#     rospy.sleep(2)
#     print "known names", scene.get_known_object_names()
#     scene.attach_mesh("base_link", "obstacle")
#     raw_input("press enter(1)")

def perturb(plan):
    for i in range(1, len(plan.joint_trajectory.points)-1):

        plan.joint_trajectory.points[i].positions = np.array(plan.joint_trajectory.points[i].positions) + np.random.normal(scale = .07, size=7)
    return plan

def add_obstacle(height, operation):
    planning_scene_publisher = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)

    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    rospy.sleep(2)
    pose = PoseStamped()
    pose.header.frame_id = robot.get_planning_frame()
    pose.pose.position.x = .25
    pose.pose.position.y = height
    pose.pose.position.z = .05
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
    box.dimensions = (0.1, 0.5, 0.5)
    co.primitives = [box]
    co.primitive_poses = [pose.pose]

    planning_scene_publisher.publish(co)
    raw_input("press enter(1)")


if __name__ == '__main__':
    rospy.init_node('yumi_controller')
    # publish_planning_scene_world()
    add_obstacle(-.25, CollisionObject.ADD)
    yumi_controller = YumiController()
    broadcaster = tf.TransformBroadcaster()
    # print yumi_controller.group.get_current_joint_values()
    plan = yumi_controller.collision_free_move([0.5, 0.0, .15], [1, 0, 0, 0])
    raw_input('Using 1st Plan')
    # plan = perturb(plan)
    # add_obstacle(-.35, CollisionObject.MOVE)
    plan2 = yumi_controller.collision_free_move([0.5, 0.0, 1.0], [1, 0, 0, 0], plan)
    # raw_input('Without Initialization')
    # plan3 = yumi_controller.collision_free_move([0.5, 0.0, .15], [1, 0, 0, 0])
    # raw_input('Using 1st Plan Again')
    # plan4 = yumi_controller.collision_free_move([0.5, 0.0, 1.0], [1, 0, 0, 0], plan)
    
    # yumi_controller.visualize_plan(plan.joint_trajectory)
    # print plan.joint_trajectory.points[0], plan.joint_trajectory.points[-1]
    # print plan.joint_trajectory.points
    rospy.spin()
