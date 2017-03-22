import os
import dexnet.database as db
import dexnet.experiment_config as ec
import dexnet.gripper as gr
import dexnet.obj_file as objf
import dexnet.similarity_tf as stf
import dexnet.stable_pose_class as stp
import tfx
import logging
import numpy as np

def quat_to_rot_mat(q):
    return np.array([[1-2*(q[2]**2)-2*(q[3]**2),   2*(q[1]*q[2]+q[0]*q[3]),     2*(q[1]*q[3]-q[0]*q[2])],
                     [2*(q[1]*q[2]+q[0]*q[3]),     1-2*(q[1]**2)-2*(q[3]**2),   2*(q[2]*q[3]+q[0]*q[1])],
                     [2*(q[1]*q[3]-q[0]*q[2]),    2*(q[2]*q[3]-q[0]*q[1]),   1-2*(q[1]**2)-2*(q[2]**2)]])
class DexnetChess():
    def __init__(self, dexnet_path):
        self.dexnet_path = dexnet_path

    def get_rook_grasp(self, position, quaternion):
        config_filename = self.dexnet_path + '/cfg/examples/read_grasps.yaml'
        logging.getLogger().setLevel(logging.INFO)

        # read config file and open database
        config = ec.ExperimentConfig(config_filename)
        database_filename = os.path.join(config['database_dir'], config['database_name'])
        database = db.Hdf5Database(database_filename, access_level=db.READ_ONLY_ACCESS)
        # read gripper
        gripper = gr.RobotGripper.load(config['gripper_name'], self.dexnet_path + "/data/grippers")
        # loop through available datasets
        dataset_name = config['dataset']
        dataset = database.dataset(dataset_name)
        # loop through objects in dataset
        object_key = config['object_name']
        obj = dataset[object_key]
        logging.info('Reading grasps for object {}'.format(object_key))


        # dataset.obj_mesh_filename("WizRook", output_dir="data/meshes/chess_pieces/")

        # read grasps and grasp metadata
        grasps = dataset.grasps(object_key, gripper=gripper.name)
        grasp_features = dataset.grasp_features(object_key, grasps, gripper=gripper.name)
        grasp_metrics = dataset.grasp_metrics(object_key, grasps, gripper=gripper.name)

        # read in object stable poses
        stable_pose = dataset.stable_pose(object_key, config['stable_pose_id'])

        #65
        # 78
        grasp = grasps[78]

        rot = quat_to_rot_mat(quaternion)
        arr = np.array([[-1,0,0,position[0]],
                        [0,-1,0,position[1]],
                        [0,0,1,position[2]],
                        [0,0,0,1]])
        for i in range(3):
            for j in range(3):
                arr[i,j] = rot[i,j]

        T_world_obj = stf.SimilarityTransform3D(from_frame='obj', to_frame='world', pose=tfx.pose(arr))

        # for displaying the fake gripper CAD file in the scene (for visualization purposes)
        T_mesh_gripper = stf.SimilarityTransform3D.load(self.dexnet_path + '/data/grippers/yumi/T_mesh_gripper.stf')
        T_obj_gripper = grasp.gripper_pose(gripper)
        T_gripper_mesh = T_mesh_gripper.inverse()
        T_obj_mesh = T_obj_gripper.dot(T_gripper_mesh)
        T_world_mesh = T_world_obj.dot(T_obj_mesh)

        # for actual planning
        # this file is only for use with the yumi_kinematics ik solver
        T_tip_gripper = stf.SimilarityTransform3D.load(self.dexnet_path + '/data/grippers/yumi/T_tip_gripper.stf')
        T_gripper_tip = T_tip_gripper.inverse()
        T_obj_tip = T_obj_gripper.dot(T_gripper_tip)
        T_world_tip = T_world_obj.dot(T_obj_tip)

        return T_world_mesh, T_world_tip