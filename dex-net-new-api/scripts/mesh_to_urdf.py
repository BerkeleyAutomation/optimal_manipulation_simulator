"""
Script to generate a urdf for a mesh with a convex decomposition to preserve the geometry
Author: Jeff
"""
import argparse
import glob
import IPython
import logging
import numpy as np
import os
import sys

import xml.etree.cElementTree as et

from dexnet.experiment_config import ExperimentConfig
from dexnet.mesh import Mesh3D
from dexnet.obj_file import ObjFile

OBJ_EXT = '.obj'
OFF_EXT = '.off'
STL_EXT = '.stl'
SUPPORTED_EXTENSIONS = [OBJ_EXT]

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # read args
    parser = argparse.ArgumentParser(description='Convert a mesh to a URDF')
    parser.add_argument('mesh_filename', type=str, help='OBJ filename of the mesh to convert')
    parser.add_argument('--config', type=str, default='cfg/convex_decomposition.yaml',
                        help='OBJ filename of the mesh to convert')
    args = parser.parse_args()

    # open config
    config_filename = args.config
    config = ExperimentConfig(config_filename)

    # setup temp directory for hacd
    cache_dir = config['cache_dir']
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    # check valid mesh filename
    mesh_filename = args.mesh_filename
    mesh_root, mesh_ext = os.path.splitext(mesh_filename)
    if mesh_ext.lower() not in SUPPORTED_EXTENSIONS:
        logging.error('Extension %s not supported' %(mesh_ext))
        exit(0)

    # open mesh
    of = ObjFile(mesh_filename)
    mesh = of.read()

    # convert the mesh to off for HACD
    mesh_dir, mesh_root = os.path.split(mesh_root)
    off_filename = os.path.join(cache_dir, mesh_root + OFF_EXT)
    meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(mesh_filename, off_filename) 
    os.system(meshlabserver_cmd)
    logging.info('MeshlabServer OFF Conversion Command: %s' %(meshlabserver_cmd))

    if not os.path.exists(off_filename):
        logging.error('Meshlab conversion failed for %s' %(off_filename))
        exit(0)

    # create output dir for urdf
    mesh_name = mesh_root
    urdf_dir = os.path.join(mesh_dir, mesh_name)
    if not os.path.exists(urdf_dir):
        os.mkdir(urdf_dir)

    # call HACD
    cvx_decomp_command = config['hacd_cmd_template'] %(off_filename,
                                                       config['min_num_clusters'],
                                                       config['max_concavity'],
                                                       config['invert_input_faces'],
                                                       config['extra_dist_points'],
                                                       config['add_faces_points'],
                                                       config['connected_components_dist'],
                                                       config['target_num_triangles'])
    logging.info('CV Decomp Command: %s' %(cvx_decomp_command))
    os.system(cvx_decomp_command)        

    # convert each wrl to an obj and an stl
    convex_piece_files = glob.glob('%s_hacd_*.wrl' %(os.path.join(cache_dir, mesh_root)))
    convex_piece_meshes = []
    convex_piece_filenames = []
    convex_pieces_volume = 0.0
    
    for convex_piece_filename in convex_piece_files:
        # convert to stl and obj
        file_root, file_ext = os.path.splitext(convex_piece_filename)
        obj_filename = file_root + OBJ_EXT
        file_path, file_root = os.path.split(file_root)        
        stl_filename = os.path.join(urdf_dir, file_root + STL_EXT)

        meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(convex_piece_filename, obj_filename) 
        os.system(meshlabserver_cmd)
        if not os.path.exists(obj_filename):
            logging.error('Meshlab conversion failed for %s' %(obj_filename))
            exit(0)

        meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(convex_piece_filename, stl_filename) 
        os.system(meshlabserver_cmd)
        if not os.path.exists(stl_filename):
            logging.error('Meshlab conversion failed for %s' %(stl_filename))
            exit(0)

        # read in meshes
        stl_file_path, stl_file_root = os.path.split(stl_filename)
        of = ObjFile(obj_filename)
        convex_piece = of.read()
        convex_pieces_volume += convex_piece.get_total_volume()
        convex_piece_meshes.append(of.read())
        convex_piece_filenames.append(stl_file_root)

    # open an XML tree
    root = et.Element('robot', name="test")

    # get the masses and moments of inertia
    effective_density = mesh.get_total_volume() / convex_pieces_volume
    prev_piece_name = None
    for convex_piece, filename in zip(convex_piece_meshes, convex_piece_filenames):
        # set the mass properties
        convex_piece.set_center_of_mass(np.zeros(3)) # center of mass at origin of mesh
        convex_piece.set_density(config['object_density'] * effective_density)
            
        _, file_root = os.path.split(filename)
        file_root, _ = os.path.splitext(file_root)
        stl_filename = 'package://%s/%s' %(mesh_name, filename)

        # write to xml
        piece_name = 'link_%s'%(file_root)
        I = convex_piece.inertia
        link = et.SubElement(root, 'link', name=piece_name)

        inertial = et.SubElement(link, 'inertial')
        origin = et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
        mass = et.SubElement(inertial, 'mass', value='%f'%convex_piece.mass)
        inertia = et.SubElement(inertial, 'inertia', ixx='%f'%I[0,0], ixy='%f'%I[0,1], ixz='%f'%I[0,2],
                                iyy='%f'%I[1,1], iyz='%f'%I[1,2], izz='%f'%I[2,2])
        
        visual = et.SubElement(link, 'visual')
        origin = et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(visual, 'geometry')
        mesh = et.SubElement(geometry, 'mesh', filename=stl_filename)
        material = et.SubElement(visual, 'material', name='')
        color = et.SubElement(material, 'color', rgba="0.75 0.75 0.75 1")

        collision = et.SubElement(link, 'collision')
        origin = et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")            
        geometry = et.SubElement(collision, 'geometry')
        mesh = et.SubElement(geometry, 'mesh', filename=stl_filename)

        if prev_piece_name is not None:
            joint = et.SubElement(root, 'joint', name='%s_joint'%(piece_name), type='fixed')
            origin = et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
            parent = et.SubElement(joint, 'parent', link=prev_piece_name)
            child = et.SubElement(joint, 'child', link=piece_name)

        prev_piece_name = piece_name

    # write URDF file
    mesh_root, mesh_ext = os.path.splitext(mesh_filename)
    mesh_path, mesh_root = os.path.split(mesh_root)
    urdf_name = '%s.URDF' %(mesh_root) 
    tree = et.ElementTree(root)
    tree.write(os.path.join(urdf_dir, urdf_name))

    # write config file
    root = et.Element('model')
    model = et.SubElement(root, 'name')
    model.text = mesh_root
    version = et.SubElement(root, 'version')
    version.text = '1.0'
    sdf = et.SubElement(root, 'sdf', version='1.4')
    sdf.text = urdf_name

    author = et.SubElement(root, 'author')    
    et.SubElement(author, 'name').text = 'Jeff Mahler'
    et.SubElement(author, 'email').text = 'jmahler@berkeley.edu'

    description = et.SubElement(root, 'description')        
    description.text = 'My awesome %s' %(mesh_root)

    tree = et.ElementTree(root)
    tree.write(os.path.join(urdf_dir, 'model.config'))
