"""
Rescale a set of meshes
Author: Jeff Mahler
"""
import argparse
import numpy as np
import os
import sys

import dexnet.obj_file as obj_file

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Rescale a set of meshes')
    parser.add_argument('data_dir', type=str, help='directory of mesh data')
    parser.add_argument('output_dir', type=str, help='directory to save meshes')
    parser.add_argument('scale_factor', type=float, default=1.0, help='scale factor for meshes')
    parser.add_argument('--convert_stl', type=bool, default=True, help='convert output to STL')
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    scale_factor = args.scale_factor

    # walk through meshes
    for filename in os.listdir(data_dir):
        file_root, file_ext = os.path.splitext(filename)
        if file_ext != '.obj':
            continue

        # read in mesh file
        in_fullpath = os.path.join(data_dir, filename)
        of = obj_file.ObjFile(in_fullpath)
        mesh = of.read()
        
        # rescale the mesh
        mesh.rescale(scale_factor)
        out_fullpath = os.path.join(output_dir, filename)
        of = obj_file.ObjFile(out_fullpath)
        of.write(mesh)

        # convert stl
        if args.convert_stl:
            out_file_root, out_file_ext = os.path.splitext(out_fullpath)
            out_stl_fullpath = out_file_root + '.stl'
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(out_fullpath, out_stl_fullpath) 
            os.system(meshlabserver_cmd)

        
