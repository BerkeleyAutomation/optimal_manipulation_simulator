#! /usr/bin/env python
import rospy
from multiprocessing import Process, Queue
from threading import Thread
import sensor_msgs.msg
import socket
import numpy as np
# import yumi_control
# from yumi_control.YuMiConstants import YuMiConstants as YMC

class _YuMiListener(Thread):
    def __init__(self, ip, port, queue):
        Thread.__init__(self)
        self.socket = None
        self.end_loop = False
        self.string_queue = ""
        self.queue = queue
        self.ip = ip
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))

        try:
            while not self.end_loop:
                tmp = self.socket.recv(77)
                self.string_queue += tmp
                index = self.string_queue.find("#")

                while index != -1:
                    point = self.string_queue[0:index-1]
                    self.queue.put(point)

                    self.string_queue = self.string_queue[index+1:]
                    index = self.string_queue.find("#")
        except KeyboardInterrupt:
            sys.exit(0)
    def _stop(self):
        self.end_loop = True
        self.socket.close()

class YuMiStatePublisher(Thread):
    def __init__(self, ip, left_port, right_port):
        Thread.__init__(self)
        self.end_loop = False
        self.left_queue = Queue()
        self.right_queue = Queue()

        self.left_listener = _YuMiListener(ip, left_port, self.left_queue)
        self.right_listener = _YuMiListener(ip, right_port, self.right_queue)
        self.publisher = rospy.Publisher("joint_states", sensor_msgs.msg.JointState, queue_size=20)
        self.publish_initial_state()

    def publish_joints(self, left_joint_state, left_gripper, right_joint_state, right_gripper):
        joint_state = sensor_msgs.msg.JointState()
        joint_state.name = ["yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l", "yumi_joint_6_l", "gripper_l_joint", 
                            "yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_7_r", "yumi_joint_3_r", "yumi_joint_4_r", "yumi_joint_5_r", "yumi_joint_6_r", "gripper_r_joint"]
        joint_state.position = left_joint_state
        joint_state.position.append(left_gripper)
        joint_state.position.extend(right_joint_state)
        joint_state.position.append(right_gripper)
        # raw_input("before")
        self.publisher.publish(joint_state)
        # raw_input("stopipng.....")

    def publish_initial_state(self):
        self.publish_joints([0,-130,135,30,0,40,0], 7, [0,-130,-135,30,0,40,0], 7)

    def run(self):
        self.left_listener.start()
        self.right_listener.start()
        try:
            while not self.end_loop:
                if not self.left_queue.empty() and not self.right_queue.empty():
                    raw_left_joint_values = self.left_queue.get().split(" ")
                    raw_right_joint_values = self.right_queue.get().split(" ")

                    left_joint_state = map(lambda x: float(x)*np.pi/180, raw_left_joint_values[:-1])
                    right_joint_state = map(lambda x: float(x)*np.pi/180, raw_right_joint_values[:-1])
                    
                    self.publish_joints(left_joint_state,  float(raw_left_joint_values[-1])/1000, 
                                        right_joint_state, float(raw_right_joint_values[-1])/1000)
        except KeyboardInterrupt:
            sys.exit(0)

    def _stop(self):
        self.end_loop = True

if __name__ == "__main__":
    rospy.init_node("yumi_state_publisher")
    ysp = YuMiStatePublisher("192.168.125.1", 6000, 6001)
    ysp.start()
    rospy.spin()