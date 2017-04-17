"""
Rescale a mesh
Author: Jeff Mahler
"""
import dexnet.mesh as mesh
import dexnet.obj_file as of

import os
import sys

if __name__ == '__main__':
    argc = len(sys.argv)
    filename = sys.argv[1]
    scale_factor = float(sys.argv[2])

    print 'Rescaling', filename 
    obj_f = of.ObjFile(filename)
    m = obj_f.read()
    m.rescale(scale_factor)

    obj_out_f = of.ObjFile(filename)
    obj_out_f.write(m)
