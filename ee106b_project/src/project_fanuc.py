#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
# from moveit_msgs.srv import ExecuteKnownTrajectory
from chomp_msgs.srv import SetInitialTrajectory
import moveit_commander
import sys
import tf
import numpy as np

class YumiController(object):
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

        rate = rospy.Rate(10);

    def shutdown(self):
        rospy.loginfo("Stopping project")
        rospy.sleep(1)

    def collision_free_move(self, position, qn, initial_trajectory=None, start_value=0):
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
        self.group.set_start_state_to_current_state()

        # print current_joint_values
        # current_joint_values[5] += 0.1
        # self.group.set_joint_value_target([0.548433, 0.554694,-1.91801,0.255809, -4.85266, 0.631296, 0.235168])
        # self.group.set_joint_value_target([-1.14074, 0.0477415, 0.323553, 0.338494, 2.37486, -0.126762, -2.58356])


        # THESE ARE THE GOOD JOINTS
        # self.group.set_joint_value_target([2.37814, -0.989366, -0.512554, -1.06674, -2.74682, 2.3268, 2.81229])
        # self.group.set_joint_value_target([1.75572, -0.273738, -2.18553, -0.840665, -0.13439, 0.614535, -3.83207])
        self.group.set_joint_value_target([-0.0974195, 1.3523, 0.682611, 0.156142, 0.675658, -0.122225])


        if initial_trajectory is not None:
            # points = initial_trajectory.joint_trajectory.points
            # points.time_from_start *= 1 - start_value / float(len(points.positions));
            # points.positions = points.positions[start_value:]
            # points.velocities = points.velocities[start_value:]
            # points.accelerations = points.accelerations[start_value:]

            # initial_trajectory.joint_trajectory.time_from_start = 1 - start_value / float(len(initial_trajectory.joint_trajectory.points))
            initial_trajectory.joint_trajectory.points = initial_trajectory.joint_trajectory.points[start_value:]

            self.initial_trajectory_proxy(initial_trajectory.joint_trajectory, 1, len(initial_trajectory.joint_trajectory.points)-2)
        else:
            self.initial_trajectory_proxy(JointTrajectory(), 1, 98)
        self.group.set_workspace([-3, -3, -3, 3, 3, 3])
        plan = self.group.plan()
        # while len(plan.joint_trajectory.points) == 0:
        #     plan = self.group.plan()
        #     print "retrying planning"
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

        plan.joint_trajectory.points[i].positions = np.array(plan.joint_trajectory.points[i].positions) + np.random.normal(scale = .05, size=6)
    return plan

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
    # pose.pose.orientation.x = 0
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

if __name__ == '__main__':
    rospy.init_node('fanuc_controller')
    add_obstacle(0, CollisionObject.ADD)
    yumi_controller = YumiController()
    broadcaster = tf.TransformBroadcaster()
    plan = yumi_controller.collision_free_move([0.5, 0.0, .15], [1, 0, 0, 0])
    raw_input('Using 1st Plan')

    yumi_controller.group.execute(plan)
    add_obstacle(.2, CollisionObject.ADD)
    # plan2 = yumi_controller.collision_free_move([0.5, 0.0, 1.0], [1, 0, 0, 0], plan, 80)
    plan2 = yumi_controller.collision_free_move([0.5, 0.0, 1.0], [1, 0, 0, 0], plan)
    yumi_controller.group.execute(plan2)
    # print plan.joint_trajectory.points[:80]
    # print plan2.joint_trajectory.points[:80]
    print "\n\n\n\nEnd of Project\n\n\n"
