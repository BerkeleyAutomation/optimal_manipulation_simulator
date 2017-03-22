import rospy
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import random

def quat(vector, rotation):
    vector = vector / np.linalg.norm(vector);
    tmp = np.sin(0.5*rotation)*vector;
    return [np.cos(0.5*rotation), tmp[0], tmp[1], tmp[2]]

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def perturbate(position, quaternion):
    position += np.diag([1,1,0]).dot(np.random.normal(0, 0.01, 3))
    # angle = np.random.normal(0,0.0872665)
    angle = np.random.normal(0,0.2)
    q2 = quat([0,0,1], angle)
    # quaternion = q_mult(q2, quaternion)
    return position, quaternion

class Scene:
    def __init__(self, yumi, dexnet_path):
        self.scene = yumi.scene
        self.dexnet_path = dexnet_path

    def add_rook(self, position, quaternion, num, yumi):
        obj_pose = geometry_msgs.msg.PoseStamped()
        obj_pose.header.frame_id = yumi.robot.get_planning_frame()
        obj_pose.pose.position.x =  position[0]
        obj_pose.pose.position.y =  position[1]
        obj_pose.pose.position.z =  position[2]
        obj_pose.pose.orientation.w = quaternion[0]
        obj_pose.pose.orientation.x = quaternion[1]
        obj_pose.pose.orientation.y = quaternion[2]
        obj_pose.pose.orientation.z = quaternion[3]
        name = "WizRook" + str(num)
        self.scene.add_mesh(name, obj_pose, self.dexnet_path + "/data/meshes/chess_pieces/WizRook.obj", [1,1,1])

    
    def move_object(self, name, pose, yumi):
        # not working yet
        collision_object_pub = rospy.Publisher('/collision_object', moveit_msgs.msg.CollisionObject, queue_size=20)
        object_msg = moveit_msgs.msg.CollisionObject()
        object_msg.operation = object_msg.MOVE

        object_msg.id = name
        object_msg.mesh_poses = pose
        object_msg.header.stamp = rospy.Time.now()
        object_msg.header.frame_id = yumi.robot.get_planning_frame()
        collision_object_pub.publish(object_msg)

    def get_pose(position, quaternion):
        obj_pose = geometry_msgs.msg.PoseStamped()
        obj_pose.header.frame_id = yumi.robot.get_planning_frame()
        obj_pose.pose.position.x =  position[0]
        obj_pose.pose.position.y =  position[1]
        obj_pose.pose.position.z =  position[2]
        obj_pose.pose.orientation.w = quaternion[0]
        obj_pose.pose.orientation.x = quaternion[1]
        obj_pose.pose.orientation.y = quaternion[2]
        obj_pose.pose.orientation.z = quaternion[3]
        return obj_pose.pose

    def update_scene(self, position, quaternion, yumi, add_to_scene):
        if add_to_scene:
            obj_pose = geometry_msgs.msg.PoseStamped()
            obj_pose.header.frame_id = yumi.robot.get_planning_frame()
            obj_pose.pose.position.x =  .52
            obj_pose.pose.position.y =  0
            obj_pose.pose.position.z = -.022
            obj_pose.pose.orientation.w = 1
            obj_pose.pose.orientation.x = 0
            obj_pose.pose.orientation.y = 0
            obj_pose.pose.orientation.z = 0
            self.scene.add_box("board", obj_pose, (.2,.2,.01))

            new_position, new_quaternion = perturbate(position, quaternion)
            self.add_rook(new_position, new_quaternion, "middle", yumi)

            squares = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            squares = [(-1,1), (0,1), (1,-1), (1,0), (1,1)]
            for i, square in enumerate(squares):
                name = "WizRook" + str(i)
                yumi.scene.remove_world_object(name=name)
                if random.random() < 0.4:
                    second_position = [position[0] + (.06 * square[0]), position[1] + (.06 * square[1]), position[2]]
                    second_new_position, second_new_quaternion = perturbate(second_position, quaternion)
                    self.add_rook(second_new_position, second_new_quaternion, i, yumi)
        else:
            print "objects", yumi.scene.get_known_object_names()
            new_position, new_quaternion = perturbate(position, quaternion)
            pose = [get_pose(new_position, new_quaternion)]
            self.move_object("WizRook1", pose, yumi)

            second_position = [position[0] + .06, position[1] - .06, position[2]]
            second_new_position, second_new_quaternion = perturbate(second_position, quaternion)
            pose = get_pose(second_new_position, second_new_quaternion)        
            self.move_object("WizRook2", pose, yumi)
            print "done moving"
        return new_position, new_quaternion

    def add_gripper(self, yumi, T_world_mesh):
        obj_pose = geometry_msgs.msg.PoseStamped()
        obj_pose.header.frame_id = yumi.robot.get_planning_frame()
        obj_pose.pose.position.x =  T_world_mesh.translation[0]
        obj_pose.pose.position.y =  T_world_mesh.translation[1]
        obj_pose.pose.position.z =  T_world_mesh.translation[2]
        obj_pose.pose.orientation.w = T_world_mesh.quaternion[3]
        obj_pose.pose.orientation.x = T_world_mesh.quaternion[0]
        obj_pose.pose.orientation.y = T_world_mesh.quaternion[1]
        obj_pose.pose.orientation.z = T_world_mesh.quaternion[2]
        self.scene.add_mesh("gripper", obj_pose, self.dexnet_path + "/data/grippers/yumi/gripper.obj", [1,1,1])