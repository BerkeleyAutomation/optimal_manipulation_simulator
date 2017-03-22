#! /usr/bin/env python
import rospy, actionlib, control_msgs.msg, sensor_msgs.msg
from multiprocessing import Process
from threading import Thread
import numpy as np
import time
from YuMiRobot import YuMiRobot
from YuMiState import YuMiState

class _YuMiController():
    def __init__(self, name, yumi_ethernet):
        self._action_name = name
        self.yumi_ethernet = yumi_ethernet
        self._as = actionlib.SimpleActionServer(self._action_name, control_msgs.msg.FollowJointTrajectoryAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        print "\n\n\n\n\nconnected to AS"
        
    # def publish_feedback(self, point, num_joints):
    #     feedback = control_msgs.msg.FollowJointTrajectoryFeedback()
    #     feedback.desired = point
    #     feedback.actual = point
    #     feedback.error.positions = [0 for _ in range(num_joints)]
    #     feedback.error.velocities = [0 for _ in range(num_joints)]
    #     feedback.error.accelerations = [0 for _ in range(num_joints)]
    #     feedback.error.effort = [0 for _ in range(num_joints)]
    #     feedback.error.time_from_start = point.time_from_start
    #     self._as.publish_feedback(feedback)

    def publish_result(self, success):
        result = control_msgs.msg.FollowJointTrajectoryResult()
        if success:
            result.error_code = 0
            result.error_string = "All is well, everyone is happy"
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(result)
        else:
            result.error_code = -1
            result.error_string = "All is not well, nobody is happy"
            rospy.loginfo('%s: Failed' % self._action_name)
            self._as.set_succeeded(result)

    def execute_cb(self, goal):
        print "\n\n\n\n\n\n\n\n\n\nGOT TRAJECTORY\n\n\n\n\n\n\n\n\n\n"
        success = True
        num_joints = len(goal.trajectory.joint_names)

        # move gripper to final position
        if num_joints == 1:
            if goal.trajectory.points[0] == 0:
                self.yumi_ethernet.close_gripper()
            else:
                self.yumi_ethernet.open_gripper()
        else:
            # positions = [point.positions for point in goal.trajectory.points]
            # self.yumi_ethernet.follow_trajectory(positions)
            for point in goal.trajectory.points:
                state = YuMiState(map(lambda x: x * 180/np.pi, point.positions))
                # self.yumi_ethernet.goto_state(state)
                self.yumi_ethernet.add_to_trajectory_buffer(state)
            self.yumi_ethernet.follow_trajectory_buffer()

        self.publish_result(success)

class YuMiController(Thread):
    def __init__(self, name, yumi_ethernet):
        Thread.__init__(self)
        self.name = name
        self.yumi_ethernet = yumi_ethernet

    def run(self):
        print "\n\n\n\n\n\n\n\n\n\nSTARTING CONTROLLER\n\n\n\n\n\n\n\n\n\n"
        _YuMiController(self.name, self.yumi_ethernet)
        rospy.spin()

    def _stop(self):
        pass

if __name__ == '__main__':
    print "\n\n\n\n\nDid it work?\n\n\n\n\n"
    rospy.init_node('YuMiController')
    yumi_robot = YuMiRobot()
    _YuMiController("yumi_left_controller/follow_left_joint_trajectory", yumi_robot.left)
    # yumi_controller("yumi_right_controller/follow_right_joint_trajectory", yumi_robot.right)
    rospy.spin()