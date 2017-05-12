#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from moveit_msgs.srv import ExecuteKnownTrajectory
import moveit_commander
import sys

class YumiController(object):
    def __init__(self):
        rospy.init_node('yumi_controller', anonymous=False)

        rospy.loginfo("To stop project CTRL + C")
        rospy.on_shutdown(self.shutdown)
        
        moveit_commander.roscpp_initialize(sys.argv)
        self.group = moveit_commander.MoveGroupCommander('right_arm')
        self.group.set_planning_time(5)
        self.display_planned_path_publisher = rospy.Publisher('right_arm/display_planned_path', DisplayTrajectory, queue_size=10)
        self.initial_trajectory_proxy = rospy.ServiceProxy('right_arm/set_initial_trajectory', ExecuteKnownTrajectory)
        # listener = tf.TransformListener()
        # from_frame = 'odom'
        # to_frame = 'base_link'
        # listener.waitForTransform(from_frame, to_frame, rospy.Time(), rospy.Duration(5.0))
        # broadcaster = tf.TransformBroadcaster()

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
        target_pose.pose.orientation.w = qn[0]
        target_pose.pose.orientation.x = qn[1]
        target_pose.pose.orientation.y = qn[2]
        target_pose.pose.orientation.z = qn[3]

        print target_pose
        # self.group.set_start_state_to_current_state()
        # self.group.set_pose_target(target_pose)
        v = self.group.get_current_joint_values()
        v[5] -= 0.1
        self.group.set_joint_value_target(v)
        # if initial_trajectory is not None:
        #     self.initial_trajectory_proxy(initial_trajectory, False)
        # else:
        #     self.initial_trajectory_proxy(RobotTrajectory(), False)
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

if __name__ == '__main__':
    yumi_controller = YumiController()
    print yumi_controller.group.get_current_joint_values()
    plan = yumi_controller.collision_free_move([0.25, 0, 0.25], [1, 0, 0, 0])
    # plan2 = yumi_controller.collision_free_move([0.25, 0, 0.25], [1, 0, 0, 0], plan)
    # yumi_controller.group.execute(plan)
    # yumi_controller.visualize_plan(plan.joint_trajectory)
    print plan.joint_trajectory.points[0], plan.joint_trajectory.points[-1]
    # print plan.joint_trajectory.points
    rospy.spin()
