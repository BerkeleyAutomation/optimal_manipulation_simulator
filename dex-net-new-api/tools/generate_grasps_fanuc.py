"""
Script to generate grasps for the FANUC robot
"""
import csv
import copy
import logging
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numpy as np
import os
import random
import sys
sys.path.append('src/dexnet')

import database as db
import experiment_config as ec
import gripper as gr
import obj_file as objf
import similarity_tf as stf
import stable_pose_class as stp
import tfx

"""
NAMING CONVENTION:
T_{a}_{b}
pose of a frame wrt b frame
"""

# TODO: move to a custom Dex-Net visualizer (pyplot version)
def plot_pose(T, alpha=1.0, line_width=2.0, ax=None):
    """ Provide rotation R and translation t of frame wrt world """
    T_inv = T.inverse()
    R = T_inv.rotation
    #print R
    t = T_inv.translation

    x_axis_3d_line = np.array([t, t + alpha * R[:,0]])
    y_axis_3d_line = np.array([t, t + alpha * R[:,1]])
    z_axis_3d_line = np.array([t, t + alpha * R[:,2]])

    if ax is None:
        ax = plt.gca(projection = '3d')
    ax.plot(x_axis_3d_line[:,0], x_axis_3d_line[:,1], x_axis_3d_line[:,2], c='r', linewidth=line_width)
    ax.plot(y_axis_3d_line[:,0], y_axis_3d_line[:,1], y_axis_3d_line[:,2], c='g', linewidth=line_width)
    ax.plot(z_axis_3d_line[:,0], z_axis_3d_line[:,1], z_axis_3d_line[:,2], c='b', linewidth=line_width)

    ax.scatter(t[0], t[1], t[2], c='k', s=150)
    ax.text(t[0], t[1], t[2], T.to_frame.upper())

def plot_mesh(mesh, T, color='m', size=20, fig=None):
    """ Plots a mesh object in pose T """
    if fig is None:
        ax = plt.gca(projection = '3d')
    else:
        ax = Axes3D(fig)
    mesh_tf = mesh.transform(T.inverse())
    mesh_tf_vertices = np.array(mesh_tf.vertices())
    ax.scatter(mesh_tf_vertices[:,0], mesh_tf_vertices[:,1], mesh_tf_vertices[:,2], c=color, s=size)

def plot_grasp(grasp, T, color='c', size=1, n=25, fig=None):
    """ Plots a grasp object in pose T """
    if fig is None:
        ax = plt.gca(projection = '3d')
    else:
        ax = Axes3D(fig)
    g1, g2 = grasp.endpoints()
    grasp_points = np.c_[np.array(g1), np.array(g2)]
    grasp_points_tf = T.inverse().apply(grasp_points)
    grasp_points_tf = grasp_points_tf.T
    g1_tf = grasp_points_tf[0,:]
    g2_tf = grasp_points_tf[1,:]

    interp_grasp_points = []
    for m in range(1,n-1):
        t = m * 1.0 / n
        interp_grasp_points.append(t * g1_tf + (1 - t) * g2_tf)
    interp_grasp_points = np.array(interp_grasp_points)

    ax.scatter(g1_tf[0], g1_tf[1], g1_tf[2], c=color, s=size*125)
    ax.scatter(g2_tf[0], g2_tf[1], g2_tf[2], c=color, s=size*125)
    ax.scatter(interp_grasp_points[:,0], interp_grasp_points[:,1], interp_grasp_points[:,2], c=color, s=size*25)    

    T_grasp_obj = grasp.gripper_pose()
    approach_line = np.c_[grasp.center, grasp.center + 0.1*T_grasp_obj.rotation[:,0]]
    approach_line_tf = T.inverse().apply(approach_line)
    approach_line_tf = approach_line_tf.T
    ax.plot(approach_line_tf[:,0], approach_line_tf[:,1], approach_line_tf[:,2], c='y', linewidth=10)

def prune_grasps_in_collision(grasps, stable_pose, gripper, clearance=0.0):
    """ Prune grasps in collision """
    coll_free_grasps = []
    for grasp in grasps:
        grasp_aligned = grasp.parallel_table(stable_pose)
        if not gripper.collides_with_table(grasp_aligned, stable_pose, clearance):
            coll_free_grasps.append(grasp)
    return coll_free_grasps

def generate_candidate_grasps(object_key, dataset, config):
    metrics = ['force_closure']
    thresholds = [config['force_closure_thresh']]

    grasps = dataset.grasps(object_key, gripper=config['gripper'])
    grasp_metrics = dataset.grasp_metrics(object_key, grasps, gripper=config['gripper'])

    # only keep high quality grasps
    high_quality_grasps = []
    for metric, tau in zip(metrics, thresholds):
        for grasp in grasps:
            if grasp_metrics[grasp.grasp_id][metric] >= tau:
                high_quality_grasps.append(grasp)
    grasps = high_quality_grasps
    random.shuffle(grasps)
    return grasps

def read_grasp_metrics(grasp_metrics, config):
    available_metrics = grasp_metrics.keys()
    metrics = {}

    for metric in config['deterministic_grasp_metrics']:
        metrics[metric] = grasp_metrics[metric]

    for metric in config['robust_grasp_metrics']:
        robust_metric_names = []
        search_tag = None
        if metric == 'force_closure':
            search_tag = 'pfc'
        elif metric == 'ferrari_canny_L1':
            search_tag = 'efc'
        elif metric == 'partial_closure':
            search_tag = 'ppc'
        else:
            raise ValueError('Robust metric %s not recognized' %(search_tag))

        for available_metric in available_metrics:
            if available_metric.find(search_tag) == 0:
                robust_metric_names.append(available_metric)

        robust_metric_names.sort()
        if config['uncertainty_level'] == 'low':
            metrics['robust_%s' %(metric)] = grasp_metrics[robust_metric_names[0]]
        elif config['uncertainty_level'] == 'medium':
            metrics['robust_%s' %(metric)] = grasp_metrics[robust_metric_names[1]]
        elif config['uncertainty_level'] == 'high':
            metrics['robust_%s' %(metric)] = grasp_metrics[robust_metric_names[2]]

    return metrics

"""
TODO:
  - save grasp to tabletop matrices for the spray bottle
  - edit script to optionally load grasp matrices and update with transform from template tabletop frame to the chessboard frame
         i) delta y
        ii) rotation between frames
  - add options for wrist to gripper frame offset
"""
if __name__ == '__main__':
    random.seed(100)
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]

    # open the database
    config = ec.ExperimentConfig(config_filename)
    database_name = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_name, config)
    dataset = database.dataset(config['datasets'].keys()[0])

    # read config
    stp_id = config['stp_id']
    delta_x = config['delta_x'] + config['cb_platform_x']
    delta_y = config['delta_y'] + config['cb_platform_y']
    delta_z = config['delta_z'] + config['cb_platform_z']
    delta_pregrasp = config['delta_pregrasp']
    delta_lift = config['delta_lift']
    table_clearance = config['table_clearance']
    output_dir = config['output_dir']

    object_key = config['object_key']
    line_width = config['line_width']
    alpha = config['alpha']
    dim = config['plot_dim']
    num_grasps = config['num_grasps']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # read in stable pose and rotation
    stable_pose = dataset.stable_pose(object_key, stp_id)

    if np.abs(np.linalg.det(stable_pose.r) + 1) < 0.01:
        stable_pose.r[1,:] = -stable_pose.r[1,:]

    R_stp_obj = stable_pose.r
    n = stable_pose.r[2,:]

    # read in mesh
    obj_filename = dataset.obj_mesh_filename(object_key)
    of = objf.ObjFile(obj_filename)
    mesh = of.read()
    gripper = gr.RobotGripper.load(config['gripper'])
    
    # form transformation matrices
    R_stp_cb = np.array([[0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    t_stp_cb = np.array([delta_x, delta_y, delta_z])
    T_stp_cb = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_cb, t_stp_cb), from_frame='cb', to_frame='stp')
    T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_obj.T, np.zeros(3)), from_frame='stp', to_frame='obj')
    T_obj_cb = T_obj_stp.dot(T_stp_cb)
    T_cb_obj = T_obj_cb.inverse()

    R_cb_world = np.eye(3)
    t_cb_world = np.zeros(3)
    T_cb_world = stf.SimilarityTransform3D(pose=tfx.pose(R_cb_world, t_cb_world), to_frame='cb')

    # transformations from FANUC to Dex-Net conventions
    R_fg_dg = np.array([[0, 0, 1],
                        [0, -1, 0],
                        [1, 0, 0]])
    R_fcb_dcb = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])
    
    T_fpg_fg = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(3), np.array([0, 0, delta_pregrasp])), from_frame='gripper', to_frame='pregrasp')
    T_lcb_fcb = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(3), np.array([0, 0, -delta_lift])), from_frame='fanuc_cb', to_frame='fanuc_cb')
    T_fcb_cb = stf.SimilarityTransform3D(pose=tfx.pose(R_fcb_dcb, np.zeros(3)), from_frame='cb', to_frame='fanuc_cb')
    T_fcb_world = T_fcb_cb.dot(T_cb_world)
    T_obj_world = T_obj_cb.dot(T_cb_world)
    T_obj_fcb = T_obj_world.dot(T_fcb_world.inverse())

    filename = 'object_%s_pose.csv' %(object_key)
    T_obj_fcb.inverse().save_pose_csv(os.path.join(output_dir, filename))

    # get the best grasp
    grasps_saved = 0
    grasps = generate_candidate_grasps(object_key, dataset, config)
    grasp_metrics = dataset.grasp_metrics(object_key, grasps, gripper=config['gripper'])
    metrics_to_write = []

    for i, grasp in enumerate(grasps):
        if grasps_saved >= num_grasps:
            print 'Saved %d grasps. Exiting' %(grasps_saved)
            break

        print 'Processing grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps))

        # form the grasp set to attempt
        grasp_parallel_table = grasp.parallel_table(stable_pose)
        T_par_grasp_obj = grasp_parallel_table.gripper_pose(gripper=gripper)
        T_par_grasp_cb = T_par_grasp_obj.dot(T_obj_cb)
        T_par_fg_fcb = T_par_grasp_cb.dot(T_fcb_cb.inverse())

        if T_par_fg_fcb.rotation[2,0] < 0: # align approach with chessboard X axis
            print 'flipping Z'
            grasp_parallel_table.approach_angle_ = grasp_parallel_table.approach_angle_ + np.pi
            T_par_grasp_obj = grasp_parallel_table.gripper_pose(gripper=gripper)
            T_par_grasp_cb = T_par_grasp_obj.dot(T_obj_cb)
            T_par_fg_fcb = T_par_grasp_cb.dot(T_fcb_cb.inverse())

        if np.abs(np.arccos(T_par_fg_fcb.rotation[2,0])) > config['robot_base_angle_thresh']:
            print 'Grasp %d makes too high of an angle with the base. Skipping' %(grasp.grasp_id)
            continue

        if T_par_fg_fcb.rotation[0,2] > 0: # align wrist orientation with table
            print 'flipping X'
            grasp_parallel_table.axis_ = -grasp_parallel_table.axis_

        grasp_lifted_table = copy.copy(grasp_parallel_table)
        grasp_lifted_table.approach_angle_ = grasp_parallel_table.approach_angle_ - config['angle_from_table'] * np.pi

        if gripper.collides_with_table(grasp_lifted_table, stable_pose, table_clearance):
            print 'Collides with table. Skipping...'
            continue

        # save grasps as matrices
        g = grasp_lifted_table
        T_grasp_obj = g.gripper_pose(gripper=gripper)
        x_axis_obj = T_grasp_obj.inverse().rotation[:,0]
        table_normal_obj = T_obj_stp.rotation[:,2]

        T_grasp_cb = T_grasp_obj.dot(T_obj_cb)
        T_fg_fcb = T_grasp_cb.dot(T_fcb_cb.inverse())
        T_fg_world = T_fg_fcb.dot(T_fcb_world)
        T_fpg_fcb = T_fpg_fg.dot(T_fg_fcb)
        T_fpg_world = T_fpg_fcb.dot(T_fcb_world)

        T_flg_fcb = T_fg_fcb.dot(T_lcb_fcb.inverse())
        T_flg_world = T_flg_fcb.dot(T_fcb_world)
        T_flg_fcb.to_frame = 'lifted_gripper'
        T_flg_world.to_frame = 'lifted_gripper'

        # get available metrics
        metrics = read_grasp_metrics(grasp_metrics[grasp.grasp_id], config)
        metrics['grasp_id'] = grasp.grasp_id
        metrics_to_write.append(metrics)

        # save poses
        print 'Saving grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps))
        filename = 'grasp_%d_grasp_pose.csv' %(grasp.grasp_id)
        T_fg_fcb.inverse().save_pose_csv(os.path.join(output_dir, filename))
        filename = 'grasp_%d_mid_pose.csv' %(grasp.grasp_id)
        T_fpg_fcb.inverse().save_pose_csv(os.path.join(output_dir, filename))
        filename = 'grasp_%d_lift_pose.csv' %(grasp.grasp_id)
        T_flg_fcb.inverse().save_pose_csv(os.path.join(output_dir, filename))
        grasps_saved += 1

        # plot grasp
        if config['debug']:
            fig = plt.figure()
            ax = Axes3D(fig)

            plot_pose(T_fg_world, alpha=alpha, ax=ax)
            plot_pose(T_fcb_world, alpha=alpha, ax=ax)
            plot_pose(T_fg_world, alpha=alpha, ax=ax)
            plot_pose(T_fpg_world, alpha=alpha, ax=ax)
            plot_pose(T_flg_world, alpha=alpha, ax=ax)
            plot_mesh(mesh, T_obj_world)
            
            ax.set_xlim3d(-dim, dim)
            ax.set_ylim3d(-dim, dim)
            ax.set_zlim3d(-dim, dim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

    if grasps_saved > 0:
        metric_csv_filename = os.path.join(output_dir, 'grasp_metrics.csv')
        f = open(metric_csv_filename, 'w')
        csv_writer = csv.DictWriter(f, fieldnames=metrics_to_write[0].keys())
        csv_writer.writeheader()
        for metrics in metrics_to_write:
            csv_writer.writerow(metrics)
        f.close()


