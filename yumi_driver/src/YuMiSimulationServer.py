#! /usr/bin/env python

import rospy
import threading
import SocketServer
from time import sleep
import sys
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
from yumi_kinematics.yumi_kinematics import Yumi_Kinematics, Joints, Pose
from YuMiState import YuMiState
import numpy as np
from socket import error as socket_error

# Generic TCP Server that also takes in a state during init
class MyServer(SocketServer.TCPServer):
    def __init__(self, server_address, RequestHandlerClass, state, solver):
        SocketServer.TCPServer.allow_reuse_address = True
        SocketServer.TCPServer.__init__(self, server_address, RequestHandlerClass)
        # self.new_server = SocketServer.TCPServer(server_address, RequestHandlerClass)
        self.state = state
        self.solver = solver
        # self.allow_reuse_address = True

    def handle_error(self, request, client_address):
        print "\n\n\n150"
    #     self.wrapper.reset()
    #     # self.shutdown()

# Handler class for the Motion Servers of Yumi
class MotionServerHandler(SocketServer.BaseRequestHandler):

    def ParseMsg(self, parsed_message):
        parsed_message = parsed_message.split(' ')
        return parsed_message[:-1]

    def handleCommand(self, instructionCode, params):
        ok = 1

        if instructionCode == 1:
            if len(params) == 7:
                # tmp = self.cart_to_jnt(params)
                # Dear future colleage: This time, it's not our fault! Please don't try to "fix" this
                # tmp.insert(2,tmp.pop(-1))
                self.server.state.raw_state = self.cart_to_jnt(params)
                ok = 1
            else:
                ok = 0
        elif instructionCode == 2:
            if len(params) == 7:
                tmp = params
                tmp.insert(2,tmp.pop(-1))
                self.server.state.raw_state = tmp
                ok = 1
            else:
                ok = 0
        elif instructionCode == 13:
            if len(params) == 3:
                current_cart = self.get_cart()
                params = [x/1000.0 for x in params]
                current_cart[0:3] = map(sum, zip(current_cart[0:3], params))
                self.server.state.raw_state = self.cart_to_jnt(current_cart)
                ok = 1
            else:
                ok = 0
        elif instructionCode == 20:
            while self.server.state.gripper_value > 5:
                self.server.state.gripper_value -= 1
                rospy.sleep(0.1)
            ok = 1
        elif instructionCode == 21:
            while self.server.state.gripper_value < 24:
                self.server.state.gripper_value += 1
                rospy.sleep(0.1)
            ok = 1
        elif instructionCode == 40:
            if len(params) == 7:
                self.traj_buffer.append(params)
                ok = 1
            else:
                ok = 0
        elif instructionCode == 41:
            if len(params) == 0:
                for point in self.traj_buffer:
                    self.server.state.raw_state = map(float, point)
                    sleep(0.1)
                self.traj_buffer = []
                ok = 1
            else:
                ok = 0
        else:
            pass

        reply = str(int(instructionCode)) + ' ' + str(ok)
        self.request.send(reply)
        print reply + '\thas been sent'

    def cart_to_jnt(self, raw_pose):
        pose = Pose()
        pose.x = raw_pose[0]
        pose.y = raw_pose[1]
        pose.z = raw_pose[2]
        pose.qw = raw_pose[3]
        pose.qx = raw_pose[4]
        pose.qy = raw_pose[5]
        pose.qz = raw_pose[6]
        joints = Joints()
        joints[:] = [0,0,0,0,0,0,0]# self.server.state.raw_state
        new_joints = [0,0,0,0,0,0,0]
        while sum(new_joints[:]) == 0:
            new_joints = self.server.solver.CartToJnt(joints, pose)
            # rospy.sleep(3)

        # radians to degrees
        new_joints = [joint*180/np.pi for joint in new_joints]
        print "\n\n\n\n\nnew joints: ", new_joints
        # new_joints = [98.95516720929857, -101.18796208290313, -103.01868549898323, 35.687040801175826, -76.33011317188694, -74.868322449107, -96.26835813166971]
        # new_joints = [119.13280263700567, -82.30289772551768, -142.99667141576506, -12.550531837333764, 96.14119290033192, 129.04122122735598, 39.606190753174786]
        return new_joints

    def get_cart(self):
        joints = Joints()
        joints[:] = [x*np.pi/180 for x in self.server.state.raw_state]
        eef = self.server.solver.JntToCart(joints)
        return [eef.x, eef.y, eef.z, eef.qw, eef.qx, eef.qy, eef.qz]

    def handle(self):
        self.traj_buffer = []
        buffered_msg = ""
        while True:
            # try:
            data = self.request.recv(1024)
            if not data:
                print "no data"
                break
            # except socket_error, ex:
            #     print "\n\n\ncaught  motion exception"
            parsed_data = data.split('#')
            
            if len(parsed_data) <= 1:
                if len(parsed_data[0]) == 0:
                    continue
                else:
                    buffered_msg.append(parsed_data)
            else:
                buffered_msg = buffered_msg + parsed_data[0]
                parsed_commands = [buffered_msg]
                buffered_msg = ""
                for i in range(1, len(parsed_data)):
                    if not parsed_data[i] == "":
                        parsed_commands.append(parsed_data[i])
                for i in range(len(parsed_commands)):
                    parsed_message = self.ParseMsg(parsed_commands[i])
                    print str(parsed_message)
                    parsed_message = map(float, parsed_message)
                    self.handleCommand(parsed_message[0], parsed_message[1:])
                    
                parsed_command = []
        print "please exit"
        # self.shutdown()

# Handler class for the Logger Servers of Yumi
class LoggerServerHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        while True:
            reply = " ".join(map(lambda x: str(float(x)), self.server.state.raw_state) + [str(self.server.state.gripper_value), "#"])
            try:
                self.request.send(reply)
            except socket_error, ex:
                print "\n\n\ncaught  logging exception"
                break
            sleep(0.1)
        print "please exit the logger"
        # self.shutdown()

# Wrapper class that runs a server in a separate thread
class ServerWrapper(threading.Thread):
    def __init__(self, host, port, server, state, solver):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.server = server
        self.state = state
        self.solver = solver

    def run(self):
        self.myserver = MyServer((self.host, self.port), self.server, self.state, self.solver)
        # try:
        # print "server", self.myserver
        self.myserver.serve_forever()
        # except socket_error, ex:
        print "\n\n\nam i here?"

# Represents an Yumi Arm. Starts up a Motion and Logger server for this arm
# and maintains a shared state
class YumiArm():
    def __init__(self, ip, port_motion, port_logger, joint_vals, solver):
        self.state = YuMiState(vals = joint_vals[:7], gripper_value = joint_vals[7])
        self.solver = solver

        self.logger = ServerWrapper(ip, port_logger, LoggerServerHandler, self.state, self.solver)
        self.motion = ServerWrapper(ip, port_motion, MotionServerHandler, self.state, self.solver)
        self.logger.start()
        self.motion.start()

    def reset(self):
        self.logger.shutdown()
        self.motion.shutdown()


if __name__ == "__main__":
    rospy.init_node("simulation_server")
    roscpp_init('yumi_kinematics', [])
    ip = "localhost"

    SocketServer.TCPServer.allow_reuse_address = True
    yk = Yumi_Kinematics()
    YumiArm(ip, 5000, 6000, [0,-130,135,30,0,40,0,7], yk.left)
    YumiArm(ip, 5001, 6001, [0,-130,-135,30,0,40,0,7], yk.right)

    # try:
    #     while True:
    #         sleep(0.1)
    # except KeyboardInterrupt:
    #     print "Server Shutting Down"
    #     sys.exit()