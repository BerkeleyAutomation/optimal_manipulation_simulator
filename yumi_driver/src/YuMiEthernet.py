'''
Interface of robotic control over ethernet. Built for the YuMi
Author: Jacky
'''

from multiprocessing import Process, Queue
import logging
import time
import socket
import sys

from collections import namedtuple

from YuMiConstants import YuMiConstants as YMC
import tfx
from YuMiState import YuMiState
from numpy import pi
import moveit_commander
import geometry_msgs.msg

_RAW_RES = namedtuple('_RAW_RES', 'mirror_code res_code message')
_RES = namedtuple('_RES', 'raw_res, data')

class _YuMiEthernet(Process):

    def __init__(self, req_q, res_q, ops_q, ip, port, bufsize, timeout):
        Process.__init__(self)
        
        self._ip = ip
        self._port = port
        self._timeout = timeout
        self._bufsize = bufsize
        self._socket = None
        
        self._req_q = req_q
        self._res_q = res_q
        self._ops_q = ops_q
        
        self._current_state = None
        self._end_run = False
        
    def run(self):
        logging.getLogger().setLevel(logging.INFO)
        
        #create socket
        if YMC.DEBUG:
            logging.info("In DEBUG mode. Messages will NOT be sent over socket.")
        else:
            logging.info('Opening socket on {0}:{1}'.format(self._ip, self._port))
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self._timeout)
            self._socket.connect((self._ip, self._port))
            logging.info('Socket successfully opened!')        
                
        try:
            while not self._end_run:
                if not self._ops_q.empty():
                    op_name = self._ops_q.get()
                    attr = '_{0}'.format(op_name)
                    if hasattr(self, attr):
                        getattr(self, attr)()
                    else:
                        logging.error("Unknown op {0}. Skipping".format(op_name))
                        
                if not self._req_q.empty():
                    req = self._req_q.get()
                    self._res_q.put(self._send_request(req))
        except KeyboardInterrupt:
            logging.info("Shutting down yumi ethernet interface")
            sys.exit(0)
            
    def _stop(self):
        if not YMC.DEBUG:
            self._socket.close()
        self._end_run = True
            
    def _send_request(self, req):
        logging.info("Sending: {0}".format(req))
        if YMC.DEBUG:
            raw_res = '1 1 MOCK RESPONSE for {0}'.format(req)
        else:
            self._socket.send(req)
            raw_res = self._socket.recv(self._bufsize)
        
        logging.info("Received: {0}".format(raw_res))
        
        if raw_res is None or len(raw_res) == 0:
            return None
        
        tokens = raw_res.split()
        
        res = _RAW_RES(int(tokens[0]), int(tokens[1]), ' '.join(tokens[2:]))
        if res.res_code != YMC.RES_CODES['success']:
            logging.error('Request failed! {0}'.format(res.message))
            return None
        
        return res
                        
class YuMiEthernet:
    
    def __init__(self, group, ip=YMC.IP, port=YMC.PORT_L, bufsize=YMC.BUFSIZE, timeout=YMC.TIMEOUT):
        '''Initializes a YuMiEthernet interface. This interface will communicate with one arm (port) on the YuMi Robot.
        This uses a subprocess to handle non-blocking socket communication with the RAPID server.
        
        Args:
            ip: IP of YuMi Robot
            port: Port of target arm's server
            bufsize: Buffer size for ethernet responses
            timeout: Timeout for ethernet communication
        '''
        self._timeout = timeout
        self._ip = ip
        self._port = port
        self._bufsize = bufsize
        self._stopping = False
        
        self._req_q = Queue()
        self._res_q = Queue()
        self._ops_q = Queue()
        
        self._yumi_ethernet = _YuMiEthernet(self._req_q, self._res_q, self._ops_q, self._ip, self._port, self._bufsize, self._timeout)
        self._yumi_ethernet.start()
        
        self.group = group
    
    # def collision_free_move(self, joints):
    #     # '''
    #     # uses Moveit and OMPL to plan a path and generate the trajectory.  The 
    #     # trajectory is sent point by point to the yumi.  A final message is sent
    #     # to signify the end of the trajectory and to trigger the motion.  
    #     # '''
    #     self.group.set_joint_value_target(joints)
    #     plan = self.group.plan()
    #     while len(plan.joint_trajectory.points) == 0:
    #         plan = self.group.plan()
    #         print "retrying planning"
    #     points = plan.joint_trajectory.points

    #     for point in points:
    #         point.positions = list(point.positions)
    #         state = YuMiState(map(lambda x: x * 180/pi, point.positions))
    #         self.add_to_trajectory_buffer(state)
    #     self.follow_trajectory_buffer()

    def collision_free_move(self, position, qn):
        '''
        uses Moveit and OMPL to plan a path and generate the trajectory.  The 
        trajectory is sent point by point to the yumi.  A final message is sent
        to signify the end of the trajectory and to trigger the motion.  
        '''

        target_pose = geometry_msgs.msg.PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose.position.x = position[0]
        target_pose.pose.position.y = position[1]
        target_pose.pose.position.z = position[2]
        target_pose.pose.orientation.w = qn[0]
        target_pose.pose.orientation.x = qn[1]
        target_pose.pose.orientation.y = qn[2]
        target_pose.pose.orientation.z = qn[3]

        self.group.set_pose_target(target_pose)
        plan = self.group.plan()
        while len(plan.joint_trajectory.points) == 0:
            plan = self.group.plan()
            print "retrying planning"
        points = plan.joint_trajectory.points

        for point in points:
            point.positions = list(point.positions)
            state = YuMiState(map(lambda x: x * 180/pi, point.positions))
            self.add_to_trajectory_buffer(state)
        self.follow_trajectory_buffer()

        return points

    def follow_trajectory(self, trajectory):
        for point in trajectory:
            state = YuMiState(vals=map(lambda x: x * 180/pi, point.positions))
            self.add_to_trajectory_buffer(state)
        self.follow_trajectory_buffer()

    def add_to_trajectory_buffer(self, state, wait_for_res = True):
        body = YuMiEthernet._iter_to_str('{:.2f}', state.raw_state)
        req = YuMiEthernet._construct_req('add_to_trajectory_buffer', body)
        return self._request(req, wait_for_res)

    def follow_trajectory_buffer(self, wait_for_res = True):
        req = YuMiEthernet._construct_req('follow_trajectory_buffer')
        return self._request(req, wait_for_res)

    def stop(self):
        '''Stops subprocess for ethernet communication
        '''
        req = YuMiEthernet._construct_req('close_connection')
        self._request(req, True)
        self._stopping = True
        self._ops_q.put("stop")

    def __del__(self):
        self.stop()
        
    def _request(self, req, wait_for_res, flush_res_q=True):
        if flush_res_q:
            while not self._res_q.empty():
                self._res_q.get_nowait()
       
        # print 'req: {0}'.format(req)
        self._req_q.put(req)
        
        if wait_for_res:
            started = time.time()
            while self._res_q.empty():
                if self._stopping:
                    return
                if time.time() - started > self._timeout:
                    logging.error("Request {0} timed out".format(req))
                    return None
            res = self._res_q.get()
            # print 'res: {0}'.format(res)
            return res

    @staticmethod
    def _construct_req(code_name, body=''):
        req = '{0:d} {1}#'.format(YMC.CMD_CODES[code_name], body)
        return req
        
    @staticmethod
    def _iter_to_str(template, iterable):
        result = ''
        for val in iterable:
            result += template.format(val).rstrip('0').rstrip('.') + ' '
        return result

    @staticmethod
    def _get_pose_body(pose):
        body = '{0}{1}'.format(YuMiEthernet._iter_to_str('{:.1f}', pose.position), 
                                            YuMiEthernet._iter_to_str('{:.5f}', pose.rotation))
        return body
            
    def ping(self, wait_for_res=True):
        req = YuMiEthernet._construct_req('ping')
        return self._request(req, wait_for_res)
        
    def get_state(self, raw_res=False):
        req = YuMiEthernet._construct_req('get_joints')
        res = self._request(req, True)
        
        if res is not None:
            tokens = res.message.split()
            try:
                if len(tokens) != YuMiState.NUM_STATES:
                    raise Exception("Invalid format for states!")
                state_vals = [float(token) for token in tokens]
                state = YuMiState(state_vals)
                if raw_res:
                    return _RES(res, state)
                else:
                    return state
            except Exception, e:
                logging.error(e)
        
    def get_pose(self, raw_res=False):
        req = YuMiEthernet._construct_req('get_pose')
        res = self._request(req, True)
        
        if res is not None:
            tokens = res.message.split()
            try:
                if len(tokens) != 7:
                    raise Exception("Invalid format for pose!")
                pose_vals = [float(token) for token in tokens]
                pose = tfx.pose(pose_vals[:3], pose_vals[3:])
                if raw_res:
                    return _RES(res, pose)
                else:
                    return pose
            except Exception, e:
                logging.error(e)
        
    def goto_state(self, state, wait_for_res=True):
        body = YuMiEthernet._iter_to_str('{:.2f}', state.raw_state)
        req = YuMiEthernet._construct_req('goto_joints', body)
        return self._request(req, wait_for_res)
    
    def goto_state_sync(self, state, wait_for_res=True):
        body = YuMiEthernet._iter_to_str('{:.2f}', state.raw_state)
        req = YuMiEthernet._construct_req('goto_joints_sync', body)
        return self._request(req, wait_for_res)
    
    def goto_pose(self, pose, wait_for_res=True):
        body = YuMiEthernet._get_pose_body(pose)
        req = YuMiEthernet._construct_req('goto_pose', body)
        return self._request(req, wait_for_res)
    
    def goto_pose_sync(self, pose, wait_for_res=True):
        body = YuMiEthernet._get_pose_body(pose)
        req = YuMiEthernet._construct_req('goto_pose_sync', body)
        return self._request(req, wait_for_res)
        
    def goto_pose_delta(self, pos_delta, rot_delta=None, wait_for_res=True):
        pos_delta_str = YuMiEthernet._iter_to_str('{:.1f}', pos_delta)
        rot_delta_str = ''
        if rot_delta is not None:
            rot_delta_str = YuMiEthernet._iter_to_str('{:.5f}', rot_delta)
            
        body = pos_delta_str + rot_delta_str
        req = YuMiEthernet._construct_req('goto_pose_delta', body)
        return self._request(req, wait_for_res)
    
    def set_tool(self, pose, wait_for_res=True):
        body = YuMiEthernet._get_pose_body(pose)
        req = YuMiEthernet._construct_req('set_tool', body)
        return self._request(req, wait_for_res)
        
    def set_speed(self, speed_data, wait_for_res=True):
        body = YuMiEthernet._iter_to_str('{:.2f}', speed_data)
        req = YuMiEthernet._construct_req('set_speed', body)
        return self._request(req, wait_for_res)
    
    def set_zone(self, zone_data, point_motion=False, wait_for_res=True):
        pm = 1 if point_motion else 0
        data = (pm,) + zone_data
        body = YuMiEthernet._iter_to_str('{:2f}', data)
        req = YuMiEthernet._construct_req('set_zone', body)
        return self._request(req, wait_for_res)
        
    def move_circular(self, circ_point, target_point, wait_for_res=True):
        #Points are poses
        body_set_circ_point = YuMiEthernet._get_pose_body(circ_point)
        body_move_by_circ_point = YuMiEthernet._get_pose_body(target_point)

        req_set_circ_point = YuMiEthernet._construct_req('set_circ_point', body_set_circ_point)
        req_move_by_circ_point = YuMiEthernet._construct_req('move_by_circ_point', body_move_by_circ_point)

        res_set_circ_point = self._request(req_set_circ_point, True)
        if res_set_circ_point is None:
            logging.error("Set circular point failed. Skipping move circular!")
            return None
        else:
            return self._request(req_move_by_circ_point, wait_for_res)
    
    def buffer_add_single(self, pose, wait_for_res=True):
        body = YuMiEthernet._get_pose_body(pose)
        req = YuMiEthernet._construct_req('buffer_add', body)
        return self._request(req, wait_for_res)
        
    def buffer_add_all(self, pose_list, wait_for_res=True):
        ress = [self.buffer_add_single(pose, wait_for_res) for pose in pose_list]
        return ress
    
    def buffer_clear(self, wait_for_res=True):
        req = YuMiEthernet._construct_req('buffer_clear')
        return self._request(req, wait_for_res)
        
    def buffer_size(self, raw_res=False):
        req = YuMiEthernet._construct_req('buffer_size')
        res = self._request(req, True)
        
        if res is not None:
            try:
                size = int(res.message)
                if raw_res:
                    return _RES(res, size)
                else:
                    return size
            except Exception, e:
                logging.error(e)
        
    def buffer_move(self, wait_for_res=True):
        req = YuMiEthernet._construct_req('buffer_move')
        return self._request(req, wait_for_res)
                
    def open_gripper(self, target_pos=None, no_wait=False, wait_for_res=True):     
        if target_pos is None:
            body = ''
        else:
            body = YuMiEthernet._iter_to_str('{0:.1f}', lst)
        req = YuMiEthernet._construct_req('open_gripper', body)
        return self._request(req, wait_for_res)
        
    def close_gripper(self, force=None, target_pos=None, no_wait=False, wait_for_res=True):
        if None in (force, target_pos):
            body = ''
        else:
            body = YuMiEthernet._iter_to_str('{0:.1f}', [force, target_pos])
        req = YuMiEthernet._construct_req('close_gripper', body)
        return self._request(req, wait_for_res)       
        
    def move_gripper(self, pos, no_wait=False, wait_for_res=True):
        lst = [pos]
        if no_wait:
            lst.append(0)
        body = YuMiEthernet._iter_to_str('{0:.1f}', lst)
        req = YuMiEthernet._construct_req('move_gripper', body)
        return self._request(req, wait_for_res)
        
    def calibrate_gripper(self, max_speed=None, hold_force=None, phys_limit=None, wait_for_res=True):
        if None in (max_speed, hold_force, phys_limit):
            body = ''
        else:
            body = self._iter_to_str('{:.1f}', [data['max_speed'], data['hold_force'], data['phys_limit']])            
        req = YuMiEthernet._construct_req('calibrate_gripper', body)
        return self._request(req, wait_for_res)
        
    def set_gripper_force(self, force, wait_for_res=True):
        body = self._iter_to_str('{:.1f}', [force])
        req = YuMiEthernet._construct_req('set_gripper_force', body)
        return self._request(req, wait_for_res)
        
    def set_gripper_max_speed(self, max_speed, wait_for_res=True):
        body = self._iter_to_str('{:1f}', [max_speed])
        req = YuMiEthernet._construct_req('set_gripper_max_speed', body)
        return self._request(req, wait_for_res)
    
    def reset_home(self, wait_for_res=True):
        req = YuMiEthernet._construct_req('reset_home')
        return self._request(req, wait_for_res)

    # def follow_trajectory(self, trajectory, wait_for_res=True):
    #     if trajectory != None:
    #         string_trajectory = ""
    #         for point in trajectory:
    #             string_trajectory += " ".join(map(lambda x: str(int(x * 180/pi)),point)) + " "
    #         req = YuMiEthernet._construct_req('follow_trajectory', string_trajectory)
    #         return self._request(req, wait_for_res)
    
        
if __name__ == '__main__':
     logging.getLogger().setLevel(YMC.LOGGING_LEVEL)
