"""
Script to test CNN database indexing with the new classes
Authors: Jeff Mahler and Jacky Liang
"""
import copy
import IPython
import numpy as np
import os
import logging
import sys
import argparse

import matplotlib.pyplot as plt

import dexnet.database as db
from dexnet.rendered_image import RenderedImage 
import dexnet.experiment_config as ec
from dexnet.cnn_database_indexer import CNN_Hdf5ObjectStablePoseIndexer
from dexnet.cnn_query_image_generator import DepthCNNQueryImageGenerator

from alan.rgbd import Kinect2Sensor, Image
from alan.core import RigidTransform, Box
from alan.core import Visualizer as vis

def load_depth_images(cfg):
    sensor = Kinect2Sensor(device_num=cfg['sensor']['device_num'], frame=cfg['sensor']['frame'])
    sensor.start()
    ir_intrinsics = sensor.ir_intrinsics

    # get raw images
    depths = []

    for _ in range(cfg['num_images']):
        _, depth, _ = sensor.frames()
        depths.append(depth)

    sensor.stop()

    return depths, ir_intrinsics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path')
    args = parser.parse_args()
    
    cfg_file = 'cfg/examples/cnn_database_indexing.yaml'
    cfg = ec.ExperimentConfig(cfg_file)
    logging.getLogger().setLevel(logging.INFO)
    
    # open db
    db_filename = os.path.join(cfg['database_dir'], cfg['database_name'])
    database = db.Hdf5Database(db_filename, cfg)
    
    # get images
    depth_images, ir_intrinsics = load_depth_images(cfg)
    depth_image = Image.median_images(depth_images)
    
    # get query image
    T_camera_world = RigidTransform.load(os.path.join(cfg['calib_dir'], '%s_registration.tf' %(cfg['sensor']['frame'])))
    
    #creates query image
    workspace_bbox = Box(min_pt=np.array(cfg['workspace']['min_pt']),
                         max_pt=np.array(cfg['workspace']['max_pt']),
                         frame='world')
    generator = DepthCNNQueryImageGenerator(T_camera_world, ir_intrinsics, workspace_bbox, cfg['registration'])
    cnn_query_params = generator.query_image(depth_image, args.output_path)

    # db indexer
    db_indexer = CNN_Hdf5ObjectStablePoseIndexer(cfg['object_key'], cfg['stp_id'], database.datasets[0], cfg['caffe'])
    
    query_rendered_image = RenderedImage(cnn_query_params.query_im.data)
    
    # get top 5 imgs from indexer
    nearest_images, dists = db_indexer.k_nearest(query_rendered_image, 5)

    # saving indexed results
    fig, axarr = plt.subplots(2, 5)
    axarr[0, 2].imshow(cnn_query_params.query_im.data)
    axarr[0, 2].set_title("Query Image")
    for i in range(5):
        axarr[1, i].imshow(nearest_images[i].image)
        axarr[1, i].set_title('{:.3f}'.format(dists[i]))
        
    for i in range(2):
        for j in range(5):
            axarr[i, j].axis('off')

    fig.suptitle("Query Image and Indexed Images")
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    fig.savefig(os.path.join(args.output_path, 'queried_images.pdf'), dpi=400, format='pdf')

    # display filtered point cloud with normals
    if cfg['vis_point_cloud']:
        vis.figure()
        vis.points(cnn_query_params.point_normal_cloud.points,
                   subsample=2)
        vis.normals(cnn_query_params.point_normal_cloud.normals,
                    cnn_query_params.point_normal_cloud.points,
                    subsample=2,
                    color=(0,0,1))
        vis.show()
    
