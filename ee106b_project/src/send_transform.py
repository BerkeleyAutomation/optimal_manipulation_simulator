import tf
import rospy

rospy.init_node("send_transform")
print "\n\n\n\n\n\n\nhello\n\n\n\n"
br = tf.TransformBroadcaster()
br.sendTransform((-1, 0, .5),
                 (1,0,0,0),
                 rospy.Time(0),
                 "base_link",
                 "camera_link")