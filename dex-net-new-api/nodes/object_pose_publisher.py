#!/usr/bin/env python  
import roslib
roslib.load_manifest('lab3')
import rospy
import math
import numpy as np
import tf
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('object_pose_publisher')

    broadcaster = tf.TransformBroadcaster()
 
    R_ar_obj = np.eye(3)
    t_ar_obj = np.array([-0.094, -0.074, 0.106])
    T_ar_obj = np.eye(4)
    T_ar_obj[:3,:3] = R_ar_obj
    T_ar_obj[:3, 3] = t_ar_obj
    q_ar_obj = tf.transformations.quaternion_from_matrix(T_ar_obj)
    
    print 'Publishing object pose'
    
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        try:
            broadcaster.sendTransform(t_ar_obj, q_ar_obj, rospy.Time.now(), '/graspable_object', '/ar_marker_8')
        except:
            continue
        rate.sleep()
