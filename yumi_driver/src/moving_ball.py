#! /usr/bin/env python
import rospy
from moveit_msgs.msg import CollisionObject
import geometry_msgs.msg
from shape_msgs.msg import SolidPrimitive
import moveit_commander
print moveit_commander.__file__
# from moveit_python import *

if __name__ == "__main__":
    rospy.init_node("move_box")
    # collision_object_pub = rospy.Publisher('/collision_object', CollisionObject, queue_size=100)
    # robot = moveit_commander.RobotCommander()

    # box = CollisionObject()
    # box.operation = CollisionObject.ADD

    # obj_pose = geometry_msgs.msg.PoseStamped()
    # obj_pose.header.frame_id = robot.get_planning_frame()
    # obj_pose.pose.position.x = 0
    # obj_pose.pose.position.y = 1
    # obj_pose.pose.position.z = 0
    # obj_pose.pose.orientation.w = 1.0

    # box.id = "box"
    # box.header = obj_pose.header
    # shape = shape_msgs.msg.SolidPrimitive()
    # shape.type = shape_msgs.msg.SolidPrimitive.BOX
    # shape.dimensions = [.2, .2, .2]
    # box.primitives = [shape]

    # box.primitive_poses = [obj_pose.pose]
    # collision_object_pub.publish(box)


    # pose2 = geometry_msgs.msg.Pose()
    # pose2.position.x = 0
    # pose2.position.y = 1
    # pose2.position.z = 0
    # pose2.orientation.w = 1.0
    # scene.add_collision_objects(box)
    # print "hello"

    # # while True:
    # #     box.primitive_poses.append(pose1)

    # #     box.primitive_poses.append(pose2)








    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    rospy.sleep(2)
    
    obj_pose = geometry_msgs.msg.PoseStamped()
    obj_pose.header.frame_id = robot.get_planning_frame()
    obj_pose.pose.position.x =  .52
    obj_pose.pose.position.y =  0
    obj_pose.pose.position.z = -.022
    obj_pose.pose.orientation.w = 1
    scene.add_box("box", obj_pose, (.2,.2,.01))
    print "done adding"

    raw_input("hit enter")






    # co = CollisionObject()
    # co.operation = CollisionObject.ADD
    # co.id = "box"

    # obj_pose = geometry_msgs.msg.PoseStamped()
    # obj_pose.header.frame_id = robot.get_planning_frame()
    # co.header = obj_pose.header
    # box = SolidPrimitive()
    # box.type = SolidPrimitive.BOX
    # box.dimensions = list(size)
    # co.primitives = [box]
    # co.primitive_poses = [pose.pose]

    # collision_object_pub = rospy.Publisher('/collision_object', CollisionObject, queue_size=100)






    move_object = CollisionObject()
    move_object.id = "box"
    move_object.header.frame_id = robot.get_planning_frame()
    obj_pose = geometry_msgs.msg.PoseStamped()
    obj_pose.header.frame_id = robot.get_planning_frame()
    obj_pose.pose.position.x =  1
    obj_pose.pose.position.y =  0
    obj_pose.pose.position.z = -.022
    obj_pose.pose.orientation.w = 1
    move_object.primitive_poses = [obj_pose.pose]

    move_object.operation = move_object.MOVE
    scene._pub_co.publish(move_object)




    # co = CollisionObject()
    # co.operation = CollisionObject.ADD
    # co.id = "box2"
    # co.header = obj_pose.header
    # box = SolidPrimitive()
    # box.type = SolidPrimitive.BOX
    # box.dimensions = [1,1,1]
    # co.primitives = [box]
    # co.primitive_poses = [obj_pose.pose]

    # scene._pub_co.publish(co)