import copy
import json
import IPython
import logging
import numpy as np
try:
    import mayavi.mlab as mv
    import mayavi.mlab as mlab
except:
    logging.info('Failed to import mayavi')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import obj_file as objf
import similarity_tf as stf
import tfx
import gripper as gr
import os
import scipy.spatial.distance as ssd
import mesh as m
ZEKE_GRIPPER = gr.RobotGripper.load('zeke')
FANUC_GRIPPER = gr.RobotGripper.load('fanuc_lehf')

class MayaviVisualizer:

    # MAYAVI VISUALIZER
    @staticmethod
    def plot_table(T_table_world, d=0.5):
        """ Plots a table in pose T """
        table_vertices = np.array([[d, d, 0],
                                   [d, -d, 0],
                                   [-d, d, 0],
                                   [-d, -d, 0]])
        table_vertices_tf = T_table_world.apply(table_vertices.T).T
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        mv.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2], table_tris, representation='surface', color=(0,0,0))

    @staticmethod
    def plot_pose(T_frame_world, alpha=0.5, tube_radius=0.005, center_scale=0.025):
        T_world_frame = T_frame_world.inverse()
        R = T_world_frame.rotation
        t = T_world_frame.translation

        x_axis_tf = np.array([t, t + alpha * R[:,0]])
        y_axis_tf = np.array([t, t + alpha * R[:,1]])
        z_axis_tf = np.array([t, t + alpha * R[:,2]])
            
        mv.points3d(t[0], t[1], t[2], color=(1,1,1), scale_factor=center_scale)
        
        mv.plot3d(x_axis_tf[:,0], x_axis_tf[:,1], x_axis_tf[:,2], color=(1,0,0), tube_radius=tube_radius)
        mv.plot3d(y_axis_tf[:,0], y_axis_tf[:,1], y_axis_tf[:,2], color=(0,1,0), tube_radius=tube_radius)
        mv.plot3d(z_axis_tf[:,0], z_axis_tf[:,1], z_axis_tf[:,2], color=(0,0,1), tube_radius=tube_radius)

        mv.text3d(t[0], t[1], t[2], ' %s' %T_frame_world.to_frame.upper(), scale=0.01)

    @staticmethod
    def plot_mesh(mesh, T_mesh_world=stf.SimilarityTransform3D(from_frame='world', to_frame='mesh'),
                  style='wireframe', color=(0.5,0.5,0.5), opacity=1.0):
        mesh_tf = mesh.transform(T_mesh_world.inverse())
        mesh_tf.visualize(style=style, color=color, opacity=opacity)

    @staticmethod
    def plot_mesh_sdf(graspable):
        """ For debugging sdf vs mesh transform """
        mv.figure()
        graspable.mesh.visualize(style='wireframe')
        sdf_points, _ = graspable.sdf.surface_points(grid_basis=False)
        mv.points3d(sdf_points[:,0], sdf_points[:,1], sdf_points[:,2], color=(1,0,0), scale_factor=0.001)
        mv.show()

    @staticmethod
    def plot_stable_pose(mesh, stable_pose, T_table_world=stf.SimilarityTransform3D(from_frame='world', to_frame='table'), d=0.5, style='wireframe', color=(0.5,0.5,0.5)):
        T_mesh_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r))
        mesh_tf = mesh.transform(T_mesh_stp)
        mn, mx = mesh_tf.bounding_box()
        z = mn[2]
        x0 = np.array([0,0,-z])
        T_table_obj = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r, x0),
                                                 from_frame='obj', to_frame='table')
        T_world_obj = T_table_world.inverse().dot(T_table_obj)
        MayaviVisualizer.plot_mesh(mesh, T_world_obj.inverse(), style=style, color=color)
        MayaviVisualizer.plot_table(T_table_world, d=d)
        return T_world_obj.inverse()

    @staticmethod
    def plot_point_cloud(point_cloud, T_points_world, color=(0,1,0), scale=0.01):
        point_cloud_tf = T_points_world.apply(point_cloud).T
        mv.points3d(point_cloud_tf[:,0], point_cloud_tf[:,1], point_cloud_tf[:,2], color=color, scale_factor=scale)

    @staticmethod
    def plot_grasp(grasp, T_obj_world=stf.SimilarityTransform3D(from_frame='world', to_frame='obj'),
                   plot_approach=False, alpha=0.5, tube_radius=0.002, endpoint_color=(0,1,0),
                   endpoint_scale=0.004, grasp_axis_color=(0,1,0), palm_axis_color=(0,0,1),
                   stp=None):
        g1, g2 = grasp.endpoints()
        center = grasp.center
        g1_tf = T_obj_world.inverse().apply(g1)
        g2_tf = T_obj_world.inverse().apply(g2)
        center_tf = T_obj_world.inverse().apply(center)
        grasp_axis_tf = np.array([g1_tf, g2_tf])

        T_gripper_obj = grasp.gripper_pose(gripper=ZEKE_GRIPPER).inverse()
        palm_axis = T_gripper_obj.inverse().rotation[:,1]

        axis_tf = np.array([g1_tf, g2_tf])
        palm_axis_tf = T_obj_world.inverse().apply(palm_axis, direction=True)
        palm_axis_tf = np.array([center_tf, center_tf + alpha * palm_axis_tf])

        mv.points3d(g1_tf[0], g1_tf[1], g1_tf[2], color=endpoint_color, scale_factor=endpoint_scale)
        mv.points3d(g2_tf[0], g2_tf[1], g2_tf[2], color=endpoint_color, scale_factor=endpoint_scale)

        mv.plot3d(grasp_axis_tf[:,0], grasp_axis_tf[:,1], grasp_axis_tf[:,2], color=grasp_axis_color, tube_radius=tube_radius)
        if plot_approach:
            mv.plot3d(palm_axis_tf[:,0], palm_axis_tf[:,1], palm_axis_tf[:,2], color=palm_axis_color, tube_radius=tube_radius)

    @staticmethod
    def plot_gripper(grasp, T_obj_world=stf.SimilarityTransform3D(from_frame='world', to_frame='obj'), gripper=None, color=(0.5,0.5,0.5)):
        if gripper is None:
            gripper = FANUC_GRIPPER
        T_gripper_obj = grasp.gripper_pose(gripper).inverse()
        T_mesh_obj = gripper.T_mesh_gripper.dot(T_gripper_obj)
        T_mesh_world = T_mesh_obj.dot(T_obj_world)
        MayaviVisualizer.plot_mesh(gripper.mesh, T_mesh_world, style='surface', color=color)

    @staticmethod
    def plot_colorbar(min_q, max_q, max_val=0.35, num_interp=100, width=25):
        """ Plot a colorbar """
        vals = np.linspace(0, max_val, num=num_interp)
        vals = vals[:,np.newaxis]
        image = np.tile(vals, [1, width])
        mv.imshow(image, colormap='hsv')

    @staticmethod
    def _get_contact_points(T_c_world, w, res):
        height, width = w.proj_win_2d.shape
        xs, ys, zs = [], [], []
        for i in range(height):
            for j in range(width):
                x_contact = (j - width/2) * res
                y_contact = (i - height/2) * res
                pt_contact = np.array([x_contact, y_contact, 0])
                pt_world = T_c_world.inverse().apply(pt_contact)

                int_contact = pt_contact = np.array([x_contact, y_contact, w.proj_win_2d[i,j]])
                int_world = T_c_world.inverse().apply(int_contact)
                
                xs.append(int_world[0])
                ys.append(int_world[1])
                zs.append(int_world[2])
        return xs, ys, zs

    @staticmethod
    def plot_patches_contacts(T_c1_world, T_c2_world, w1, w2, res, scale):
        xs1, ys1, zs1 = MayaviVisualizer._get_contact_points(T_c1_world, w1, res)
        xs2, ys2, zs2 = MayaviVisualizer._get_contact_points(T_c2_world, w2, res)
       
        points1 = mlab.points3d(xs1, ys1, zs1, color=(1,0,1), scale_factor=scale)
        points2 = mlab.points3d(xs2, ys2, zs2, color=(0,1,1), scale_factor=scale)
        
        return points1, points2
        
    @staticmethod
    def plot_patch(window, T_patch_obj, window_dim_obj=(1.0,1.0), T_obj_world=stf.SimilarityTransform3D(from_frame='world', to_frame='obj'),
                   patch_color=(1,1,1), contact_color=(1,1,0), contact_scale=0.005,
                   delta_z=0.001, edge_len_thresh=3.0,
                   dist_thresh=0.05, grad_thresh=0.025):
        """ Plot a patch defined by a window and a contact """
        # extract dimensions
        proj_window = window.proj_win_2d
        grad_x_window = window.grad_x_2d
        grad_y_window = window.grad_y_2d
        win_height, win_width = proj_window.shape
        res_x = window_dim_obj[1] / win_width
        res_y = window_dim_obj[0] / win_height
        offset = 0

        t_patch_render = np.array([0, 0, -delta_z])
        T_patch_render = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(3), t_patch_render),
                                                   from_frame='render',
                                                   to_frame='contact')

        # x,y lists for triangulation
        x_list = []
        y_list = []

        # convert patch into 3d points
        points_3d = np.zeros([0,3])    
        for y in range(win_height):
            for x in range(win_width):
                # convert to 3d points
                x_3d = (x -  win_width/2) * res_x
                y_3d = (y - win_height/2) * res_y

                # add the point if the distance and gradients are appropriate
                if np.abs(proj_window[y, x]-offset) < dist_thresh and \
                        np.abs(grad_x_window[y, x]) < grad_thresh and \
                        np.abs(grad_y_window[y, x]) < grad_thresh:
                    x_list.append(x)
                    y_list.append(y)
                    points_3d = np.r_[points_3d,
                                      np.array([[x_3d, y_3d, proj_window[y, x] - offset]])]
            
        # abort if too few points
        if len(x_list) <= 3 and len(y_list) <= 3:
            logging.warning('Too few points for triangulation')

        # triangulate and prune large tris
        tri = mtri.Triangulation(x_list, y_list)
        points_2d = np.array([x_list, y_list]).T
        triangles = []
        for t in tri.triangles.tolist():
            v = points_2d[t,:]
            largest_dist = np.max(ssd.pdist(v))
            if largest_dist < edge_len_thresh:
                triangles.append(t)

        # transform into world reference frame
        points_3d_obj = T_patch_obj.inverse().dot(T_patch_render).apply(points_3d.T)
        points_3d_world = T_obj_world.inverse().apply(points_3d_obj).T
        contact_world = T_obj_world.inverse().dot(T_patch_obj.inverse()).dot(T_patch_render).apply(np.zeros(3))

        # plot
        mesh_background = mlab.triangular_mesh(points_3d_world[:,0], points_3d_world[:,1], points_3d_world[:,2], triangles,
                                               representation='surface', color=patch_color)
        mesh_tris = mlab.triangular_mesh(points_3d_world[:,0], points_3d_world[:,1], points_3d_world[:,2], triangles,
                                         representation='wireframe', color=(0,0,0))
        points = mlab.points3d(contact_world[0], contact_world[1], contact_world[2],
                               color=contact_color, scale_factor=contact_scale)      
        return mesh_background, mesh_tris, points

def test_yumi_3d_printed_gripper():
    # build new mesh
    finger_filename = '/home/jmahler/jeff_working/GPIS/data/grippers/yumi_3d_printed/finger.obj'
    base_filename = '/home/jmahler/jeff_working/GPIS/data/grippers/yumi_3d_printed/base.obj'
    of = objf.ObjFile(finger_filename)
    finger1_mesh = of.read()
    finger2_mesh = copy.copy(finger1_mesh)
    of = objf.ObjFile(base_filename)
    base_mesh = of.read()

    R_finger1_world = np.array([[1,0,0],
                                [0,0,-1],
                                [0,1,0]])
    t_finger1_world = np.array([-0.019, 0.028, 0.07])
    T_finger1_world = stf.SimilarityTransform3D(pose=tfx.pose(R_finger1_world,
                                                              t_finger1_world),
                                                from_frame='mesh', to_frame='world')
    R_finger2_world = np.array([[-1,0,0],
                                [0,0,1],
                                [0,1,0]])
    t_finger2_world = np.array([0.017, -0.028, 0.07])
    T_finger2_world = stf.SimilarityTransform3D(pose=tfx.pose(R_finger2_world,
                                                              t_finger2_world),
                                                from_frame='mesh', to_frame='world')
    T_mesh_world = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)), from_frame='world', to_frame='mesh')

    finger1_mesh = finger1_mesh.transform(T_finger1_world)
    finger2_mesh = finger2_mesh.transform(T_finger2_world)

    offset = 0
    vertices = []
    triangles = []
    meshes = [base_mesh, finger1_mesh, finger2_mesh]
    for i, mesh in enumerate(meshes):
        vertices.extend(mesh.vertices())
        offset_tris = [[t[0]+offset,t[1]+offset,t[2]+offset] for t in mesh.triangles()]
        triangles.extend(offset_tris)
        offset += len(mesh.vertices())
    gripper_mesh = m.Mesh3D(vertices, triangles)
    gripper_mesh.center_vertices_bb()

    of = objf.ObjFile('/home/jmahler/jeff_working/GPIS/data/grippers/yumi_3d_printed/gripper.obj')
    of.write(gripper_mesh)

    # frames of reference
    R_grasp_gripper = np.array([[0, 0, -1],
                                [0, 1, 0],
                                [1, 0, 0]])
    R_mesh_gripper = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    t_mesh_gripper = np.array([0.0, 0.0, 0.075])
    T_mesh_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_mesh_gripper, t_mesh_gripper),
                                               from_frame='gripper', to_frame='mesh')
    T_gripper_world = T_mesh_gripper.inverse().dot(T_mesh_world)
    T_grasp_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_grasp_gripper), from_frame='gripper', to_frame='grasp')

    T_mesh_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/yumi_3d_printed/T_mesh_gripper.stf')
    T_grasp_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/yumi_3d_printed/T_grasp_gripper.stf')

    gripper_params = {}
    gripper_params['min_width'] = 0.0
    gripper_params['max_width'] = 0.044
    f = open('/home/jmahler/jeff_working/GPIS/data/grippers/yumi_3d_printed/params.json', 'w')
    json.dump(gripper_params, f)    

    MayaviVisualizer.plot_mesh(gripper_mesh, T_mesh_world, style='surface', color=(0,0,1))
    MayaviVisualizer.plot_pose(T_gripper_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    mv.axes()
    mv.show()

def test_yumi_gripper():
    # build new mesh
    finger_filename = '/home/jmahler/jeff_working/GPIS/data/grippers/yumi/finger.obj'
    base_filename = '/home/jmahler/jeff_working/GPIS/data/grippers/yumi/base.obj'
    of = objf.ObjFile(finger_filename)
    finger1_mesh = of.read()
    finger2_mesh = copy.copy(finger1_mesh)
    of = objf.ObjFile(base_filename)
    base_mesh = of.read()

    R_finger1_world = np.eye(3)
    t_finger1_world = np.array([-0.025, -0.005, 0.082])
    T_finger1_world = stf.SimilarityTransform3D(pose=tfx.pose(R_finger1_world,
                                                              t_finger1_world),
                                                from_frame='mesh', to_frame='world')
    R_finger2_world = np.array([[-1,0,0],
                                [0,-1,0],
                                [0,0,1]])
    t_finger2_world = np.array([0.025, 0.005, 0.082])
    T_finger2_world = stf.SimilarityTransform3D(pose=tfx.pose(R_finger2_world,
                                                              t_finger2_world),
                                                from_frame='mesh', to_frame='world')
    T_mesh_world = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)), from_frame='world', to_frame='mesh')

    finger1_mesh = finger1_mesh.transform(T_finger1_world)
    finger2_mesh = finger2_mesh.transform(T_finger2_world)

    offset = 0
    vertices = []
    triangles = []
    meshes = [base_mesh, finger1_mesh, finger2_mesh]
    for i, mesh in enumerate(meshes):
        vertices.extend(mesh.vertices())
        offset_tris = [[t[0]+offset,t[1]+offset,t[2]+offset] for t in mesh.triangles()]
        triangles.extend(offset_tris)
        offset += len(mesh.vertices())
    gripper_mesh = m.Mesh3D(vertices, triangles)
    gripper_mesh.center_vertices_bb()

    of = objf.ObjFile('/home/jmahler/jeff_working/GPIS/data/grippers/yumi/gripper.obj')
    #of.write(gripper_mesh)

    # frames of reference
    R_grasp_gripper = np.array([[0, 0, -1],
                                [0, 1, 0],
                                [1, 0, 0]])
    R_mesh_gripper = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    t_mesh_gripper = np.array([0.0, -0.002, 0.058])
    T_mesh_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_mesh_gripper, t_mesh_gripper),
                                               from_frame='gripper', to_frame='mesh')
    T_gripper_world = T_mesh_gripper.inverse().dot(T_mesh_world)
    T_grasp_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_grasp_gripper), from_frame='gripper', to_frame='grasp')

    #T_mesh_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/yumi/T_mesh_gripper.stf')
    #T_grasp_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/yumi/T_grasp_gripper.stf')

    gripper_params = {}
    gripper_params['min_width'] = 0.0
    gripper_params['max_width'] = 0.05
    #f = open('/home/jmahler/jeff_working/GPIS/data/grippers/yumi/params.json', 'w')
    #json.dump(gripper_params, f)    

    MayaviVisualizer.plot_mesh(gripper_mesh, T_mesh_world, style='surface', color=(0,0,1))
    MayaviVisualizer.plot_pose(T_gripper_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    mv.axes()
    mv.show()

def test_zeke_gripper():
    mesh_filename = '/home/jmahler/jeff_working/GPIS/data/grippers/zeke/gripper.obj'
    of = objf.ObjFile(mesh_filename)
    gripper_mesh = of.read()

    gripper_mesh.center_vertices_bb()
    #oof = objf.ObjFile(mesh_filename)
    #oof.write(gripper_mesh)
    
    T_mesh_world = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)), from_frame='world', to_frame='mesh')
    R_grasp_gripper = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
    R_mesh_gripper = np.array([[0, -1, 0],
                               [0, 0, 1],
                               [-1, 0, 0]])
    t_mesh_gripper = np.array([0.09, 0.011, 0.0])
    T_mesh_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_mesh_gripper, t_mesh_gripper),
                                               from_frame='gripper', to_frame='mesh')
    T_gripper_world = T_mesh_gripper.inverse().dot(T_mesh_world)
    T_grasp_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_grasp_gripper), from_frame='gripper', to_frame='grasp')

    #T_mesh_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/zeke_new/T_mesh_gripper.stf')
    #T_grasp_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/zeke_new/T_grasp_gripper.stf')

    gripper_params = {}
    gripper_params['min_width'] = 0.0
    gripper_params['max_width'] = 0.066
    #f = open('/home/jmahler/jeff_working/GPIS/data/grippers/zeke_new/params.json', 'w')
    #json.dump(gripper_params, f)

    MayaviVisualizer.plot_pose(T_mesh_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    MayaviVisualizer.plot_pose(T_gripper_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    MayaviVisualizer.plot_mesh(gripper_mesh, T_mesh_world, style='surface', color=(1,1,1))
    mv.axes()
    mv.show()

def test_fanuc_gripper():
    mesh_filename = '/home/jmahler/jeff_working/GPIS/data/grippers/fanuc_lehf/gripper.obj'
    of = objf.ObjFile(mesh_filename)
    gripper_mesh = of.read()

    gripper_mesh.center_vertices_bb()
    oof = objf.ObjFile(mesh_filename)
    oof.write(gripper_mesh)
    
    T_mesh_world = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)), from_frame='world', to_frame='mesh')
    R_grasp_gripper = np.array([[0, 0, 1],
                                [0, -1, 0],
                                [1, 0, 0]])
    R_mesh_gripper = np.array([[0, 1, 0],
                               [-1, 0, 0],
                               [0, 0, 1]])
    t_mesh_gripper = np.array([0.0, 0.0, 0.065])
    T_mesh_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_mesh_gripper, t_mesh_gripper),
                                               from_frame='gripper', to_frame='mesh')
    T_gripper_world = T_mesh_gripper.inverse().dot(T_mesh_world)
    T_grasp_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_grasp_gripper), from_frame='gripper', to_frame='grasp')
    T_grasp_world = T_grasp_gripper.dot(T_gripper_world)

    #T_mesh_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/fanuc_lehf/T_mesh_gripper.stf')
    #T_grasp_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/fanuc_lehf/T_grasp_gripper.stf')

    gripper_params = {}
    gripper_params['min_width'] = 0.015
    gripper_params['max_width'] = 0.048
    #f = open('/home/jmahler/jeff_working/GPIS/data/grippers/fanuc_lehf/params.json', 'w')
    #json.dump(gripper_params, f)

    MayaviVisualizer.plot_pose(T_mesh_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    #MayaviVisualizer.plot_pose(T_gripper_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    MayaviVisualizer.plot_pose(T_grasp_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    MayaviVisualizer.plot_mesh(gripper_mesh, T_mesh_world, style='surface', color=(1,1,1))
    mv.axes()
    mv.show()    

def test_baxter_gripper():
    mesh_filename = '/home/jmahler/jeff_working/GPIS/data/grippers/baxter/gripper.obj'
    of = objf.ObjFile(mesh_filename)
    gripper_mesh = of.read()

    gripper_mesh.center_vertices_bb()
    #gripper_mesh.rescale(0.9) # to make fingertips at the wide 0.67 distance 
    oof = objf.ObjFile(mesh_filename)
    oof.write(gripper_mesh)
    
    T_mesh_world = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)), from_frame='world', to_frame='mesh')
    R_grasp_gripper = np.array([[0, 0, -1],
                                [0, 1, 0],
                                [1, 0, 0]])
    R_mesh_gripper = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    t_mesh_gripper = np.array([0.005, 0.0, 0.055])
    T_mesh_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_mesh_gripper, t_mesh_gripper),
                                               from_frame='gripper', to_frame='mesh')
    T_gripper_world = T_mesh_gripper.inverse().dot(T_mesh_world)
    T_grasp_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_grasp_gripper), from_frame='gripper', to_frame='grasp')

    T_mesh_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/baxter/T_mesh_gripper.stf')
    T_grasp_gripper.save('/home/jmahler/jeff_working/GPIS/data/grippers/baxter/T_grasp_gripper.stf')

    gripper_params = {}
    gripper_params['min_width'] = 0.026
    gripper_params['max_width'] = 0.060
    f = open('/home/jmahler/jeff_working/GPIS/data/grippers/baxter/params.json', 'w')
    json.dump(gripper_params, f)

    MayaviVisualizer.plot_pose(T_mesh_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    MayaviVisualizer.plot_pose(T_gripper_world, alpha=0.05, tube_radius=0.0025, center_scale=0.005)
    MayaviVisualizer.plot_mesh(gripper_mesh, T_mesh_world, style='surface', color=(1,1,1))
    mv.axes()
    mv.show()    

if __name__ == '__main__':
    #test_yumi_gripper()
    test_yumi_3d_printed_gripper()
    #test_baxter_gripper()
    #test_zeke_gripper()
    #test_fanuc_gripper()
