from abc import ABCMeta, abstractmethod

import datetime as dt
import h5py
import json_serialization as jsons
import logging
import numbers
import numpy as np
import os
import sys
import time

import experiment_config as ec
import hdf5_factory as hfact
import grasp
import graspable_object as go
import obj_file
import sdf_file
import feature_file
import stp_file

import IPython

INDEX_FILE = 'index.db'
HDF5_EXT = '.hdf5'
OBJ_EXT = '.obj'
STL_EXT = '.stl'
SDF_EXT = '.sdf'

READ_ONLY_ACCESS = 'READ_ONLY'
READ_WRITE_ACCESS = 'READ_WRITE'
WRITE_ACCESS = 'WRITE'

# Keys for easy lookups in HDF5 databases
METRICS_KEY = 'metrics'
OBJECTS_KEY = 'objects'
MESH_KEY = 'mesh'
SDF_KEY = 'sdf'
GRASPS_KEY = 'grasps'
GRIPPERS_KEY = 'grippers'
NUM_GRASPS_KEY = 'num_grasps'
LOCAL_FEATURES_KEY = 'local_features'
GLOBAL_FEATURES_KEY = 'global_features'
SHOT_FEATURES_KEY = 'shot'
RENDERED_IMAGES_KEY = 'rendered_images'
SENSOR_DATA_KEY = 'sensor_data'
STP_KEY = 'stable_poses'
CATEGORY_KEY = 'category'
MASS_KEY = 'mass'

CREATION_KEY = 'time_created'
DATASETS_KEY = 'datasets'
DATASET_KEY = 'dataset'

def generate_metric_tag(root, config):
    tag = '%s_f_%f_tg_%f_rg_%f_to_%f_ro_%f' %(root, config['sigma_mu'], config['sigma_trans_grasp'], config['sigma_rot_grasp'],
                                              config['sigma_trans_obj'], config['sigma_rot_obj'])
    return tag

class Database(object):
    """ Abstract class for databases. Main purpose is to wrap individual datasets """
    __metaclass__ = ABCMeta

    def __init__(self, access_level=READ_ONLY_ACCESS):
        self.access_level_ = access_level

    @property
    def access_level(self):
        return self.access_level_


class Hdf5Database(Database):
    def __init__(self, database_filename, access_level=READ_ONLY_ACCESS,
                 cache_dir='.dexnet'):
        Database.__init__(self, access_level)
        self.database_filename_ = database_filename
        if not self.database_filename_.endswith(HDF5_EXT):
            raise ValueError('Must provide HDF5 database')

        self.database_cache_dir_ = cache_dir
        if not os.path.exists(self.database_cache_dir_):
            os.mkdir(self.database_cache_dir_)

        self._load_database()
        self._load_datasets()

    def _create_new_db(self):
        """ Creates a new database """
        self.data_ = h5py.File(self.database_filename_, 'w')

        dt_now = dt.datetime.now()
        creation_stamp = '%s-%s-%s-%sh-%sm-%ss' %(dt_now.month, dt_now.day, dt_now.year, dt_now.hour, dt_now.minute, dt_now.second) 
        self.data_.attrs[CREATION_KEY] = creation_stamp
        self.data_.create_group(DATASETS_KEY)

    def _load_database(self):
        """ Loads in the HDF5 file """
        if self.access_level == READ_ONLY_ACCESS:
            self.data_ = h5py.File(self.database_filename_, 'r')
        elif self.access_level == READ_WRITE_ACCESS:
            if os.path.exists(self.database_filename_):
                self.data_ = h5py.File(self.database_filename_, 'r+')
            else:
                self._create_new_db()
        elif self.access_level == WRITE_ACCESS:
            self._create_new_db()
        self.dataset_names_ = self.data_[DATASETS_KEY].keys()

    def _load_datasets(self):
        """ Load in the datasets """
        self.datasets_ = []
        for dataset_name in self.dataset_names_:
            if dataset_name not in self.data_[DATASETS_KEY].keys():
                logging.warning('Dataset %s not in database' %(dataset_name))
            else:
                dataset_cache_dir = os.path.join(self.database_cache_dir_, dataset_name)
                self.datasets_.append(Hdf5Dataset(dataset_name, self.data_[DATASETS_KEY][dataset_name],
                                                  cache_dir=dataset_cache_dir))

    @property
    def datasets(self):
        return self.datasets_

    def dataset(self, dataset_name):
        """ Returns handles to individual datasets """
        if self.datasets is None or dataset_name not in self.dataset_names_:
            return None
        for dataset in self.datasets_:
            if dataset.name == dataset_name:
                return dataset

    def close(self):
        """ Close the HDF5 file """
        self.data_.close()

    def __getitem__(self, dataset_name):
        """ Dataset name indexing """
        return self.dataset(dataset_name)
        
    # New dataset creation / modification functions
    def create_dataset(self, dataset_name, obj_keys=[]):
        """ Create dataset with obj keys"""
        if dataset_name in self.data_[DATASETS_KEY].keys():
            logging.warning('Dataset %s already exists. Cannot overwrite' %(dataset_name))
            return self.datasets_[self.data_[DATASETS_KEY].keys().index(dataset_name)]
        self.data_[DATASETS_KEY].create_group(dataset_name)
        self.data_[DATASETS_KEY][dataset_name].create_group(OBJECTS_KEY)
        self.data_[DATASETS_KEY][dataset_name].create_group(METRICS_KEY)
        for obj_key in obj_keys:
            self.data_[DATASETS_KEY][dataset_name][OBJECTS_KEY].create_group(obj_key)

        dataset_cache_dir = os.path.join(self.database_cache_dir_, dataset_name)
        self.dataset_names_.append(dataset_name)
        self.datasets_.append(Hdf5Dataset(dataset_name, self.data_[DATASETS_KEY][dataset_name],
                                          cache_dir=dataset_cache_dir))
        return self.datasets_[-1] # return the dataset
        
    def create_linked_dataset(self, dataset_name, graspable_list, nearest_neighbors):
        """ Creates a new dataset that links to objects physically stored as part of another dataset """
        raise NotImplementedError()

class Dataset(object):
    pass

class Hdf5Dataset(Dataset):
    def __init__(self, dataset_name, data, cache_dir=None,
                 start_index=0, end_index=None):
        self.dataset_name_ = dataset_name
        self.data_ = data
        self.object_keys_ = None
        self.start_index_ = 0
        self.end_index_ = end_index
        if self.end_index_ is None:
            self.end_index_ = len(self.objects.keys())

        self.cache_dir_ = cache_dir
        if self.cache_dir_ is None:
            self.cache_dir_ = os.path.join('.dexnet', self.dataset_name_)
        if not os.path.exists(self.cache_dir_):
            os.mkdir(self.cache_dir_)

    @property
    def name(self):
        return self.dataset_name_

    @property
    def dataset_root_dir(self):
        return self.dataset_root_dir_

    @property
    def objects(self):
        return self.data_[OBJECTS_KEY]

    @property
    def object_keys(self):
        if not self.object_keys_:
            self.object_keys_ = self.objects.keys()[self.start_index_:self.end_index_]
        return self.object_keys_

    @property
    def metrics(self):
        if METRICS_KEY in self.data_.keys():
            return self.data_[METRICS_KEY]
        return None

    # easy data accessors
    def object(self, key):
        return self.objects[key]

    def sdf_data(self, key):
        return self.objects[key][SDF_KEY]

    def mesh_data(self, key):
        return self.objects[key][MESH_KEY]

    def grasp_data(self, key, gripper=None):
        if gripper:
            return self.objects[key][GRASPS_KEY][gripper]
        return self.objects[key][GRASPS_KEY]

    def local_feature_data(self, key):
        return self.objects[key][LOCAL_FEATURES_KEY]

    def shot_feature_data(self, key):
        return self.local_feature_data(key)[SHOT_FEATURES_KEY]

    def stable_pose_data(self, key, stable_pose_id=None):
        if stable_pose_id is not None:
            self.objects[key][STP_KEY][stable_pose_id]
        return self.objects[key][STP_KEY]

    def category(self, key):
        return self.objects[key].attrs[CATEGORY_KEY]

    def rendered_image_data(self, key, stable_pose_id=None, image_type=None):
        if stable_pose_id is not None and image_type is not None:
            return self.stable_pose_data(key)[stable_pose_id][RENDERED_IMAGES_KEY][image_type]
        elif stable_pose_id is not None:
            return self.stable_pose_data(key)[stable_pose_id][RENDERED_IMAGES_KEY]
        elif image_type is not None:
            return self.object(key)[RENDERED_IMAGES_KEY][image_type]
        return self.object(key)[RENDERED_IMAGES_KEY]

    def metric_data(self, metric):
        if metric in self.metrics.keys():
            return self.metrics[metric]
        return None

    # iterators
    def __getitem__(self, index):
        """ Index a particular object in the dataset """
        if isinstance(index, numbers.Number):
            if index < 0 or index >= len(self.object_keys):
                raise ValueError('Index out of bounds. Dataset contains %d objects' %(len(self.object_keys)))
            obj = self.graspable(self.object_keys[index])
            return obj
        elif isinstance(index, (str, unicode)):
            obj = self.graspable(index)
            return obj

    def __iter__(self):
        """ Generate iterator """
        self.iter_count_ = self.start_index_ # NOT THREAD SAFE!
        return self

    def subset(self, start_index, end_index):
        """ Returns a subset of the dataset (should be used for iterating only) """
        return Hdf5Dataset(self.dataset_name_, self.data_, self.cache_dir_,
                           start_index, end_index) 
    
    def next(self):
        """ Read the next object file in the list """
        if self.iter_count_ >= len(self.object_keys) or self.iter_count_ >= self.end_index_:
            raise StopIteration
        else:
            logging.info('Returning datum %s' %(self.object_keys[self.iter_count_]))
            try:
                 obj = self.graspable(self.object_keys[self.iter_count_])    
            except:
                logging.warning('Error reading %s. Skipping' %(self.object_keys[self.iter_count_]))
                self.iter_count_ = self.iter_count_ + 1
                return self.next()

            self.iter_count_ = self.iter_count_ + 1
            return obj

    # direct reading / writing
    def graspable(self, key):
        """Read in the GraspableObject3D corresponding to given key."""
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))

        # read in data (need new interfaces for this....
        sdf = hfact.Hdf5ObjectFactory.sdf_3d(self.sdf_data(key))
        mesh = hfact.Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))
        mass = self.object(key).attrs[MASS_KEY]
        return go.GraspableObject3D(sdf, mesh=mesh, key=key,
                                    model_name=self.obj_mesh_filename(key),
                                    mass=mass)

    def create_graspable(self, key, mesh=None, sdf=None, shot_features=None, stable_poses=None, category='', mass=1.0):
        """ Creates a graspable object """
        # create object tree
        self.objects.create_group(key)
        self.object(key).create_group(MESH_KEY)
        self.object(key).create_group(SDF_KEY)
        self.object(key).create_group(STP_KEY)
        self.object(key).create_group(LOCAL_FEATURES_KEY)
        self.object(key).create_group(GLOBAL_FEATURES_KEY)
        self.object(key).create_group(RENDERED_IMAGES_KEY)
        self.object(key).create_group(SENSOR_DATA_KEY)
        self.object(key).create_group(GRASPS_KEY)

        # add the different pieces if provided
        if sdf:
            hfact.Hdf5ObjectFactory.write_sdf_3d(sdf, self.sdf_data(key))
        if mesh:
            hfact.Hdf5ObjectFactory.write_mesh_3d(mesh, self.mesh_data(key))
        if shot_features:
            hfact.Hdf5ObjectFactory.write_shot_features(shot_features, self.local_feature_data(key))
        if stable_poses:
            hfact.Hdf5ObjectFactory.write_stable_poses(stable_poses, self.stable_pose_data(key))

        # add the attributes
        self.object(key).attrs.create(CATEGORY_KEY, category)
        self.object(key).attrs.create(MASS_KEY, mass)

        # force re-read of keys
        self.object_keys_ = None
        self.end_index_ = len(self.objects.keys())

    def obj_mesh_filename(self, key, scale=1.0, output_dir=None):
        """ Writes an obj file in the database "cache"  directory and returns the path to the file """
        mesh = hfact.Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))
        mesh.rescale(scale)
        if output_dir is None:
            output_dir = self.cache_dir_
        obj_filename = os.path.join(output_dir, key + OBJ_EXT)
        of = obj_file.ObjFile(obj_filename)
        of.write(mesh)
        return obj_filename

    def stl_mesh_filename(self, key, scale=1.0, output_dir=None):
        """ Writes an stl file in the database "cache"  directory and returns the path to the file """
        obj_filename = self.obj_mesh_filename(key, scale=scale, output_dir=output_dir)
        if output_dir is None:
            output_dir = self.cache_dir_
        stl_filename = os.path.join(output_dir, key + STL_EXT)
        meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(obj_filename, stl_filename)
        os.system(meshlabserver_cmd)
        return stl_filename

    # mesh data
    def mesh(self, key):
        """ Read the mesh for the given key """
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))
        return hfact.Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))

    # metric data
    def create_metric(self, metric_name, metric_config):
        """ Creates a metric with the given name for easier access """
        # create metric data if nonexistent
        if self.metrics is None:
            self.data_.create_group(METRICS_KEY)
                        
        if metric_name in self.metrics.keys():
            logging.warning('Metric %s already exists. Aborting...' %(metric_name))
            return False
        self.metrics.create_group(metric_name)

        # add configuration
        metric_group = self.metric_data(metric_name)
        for key in metric_config.keys():
            if isinstance(metric_config[key], dict):
                for k, v in metric_config[key].iteritems():
                    metric_group.attrs.create(key+'_'+k, metric_config[key][k])
            else:
                metric_group.attrs.create(key, metric_config[key])
        return True

    def has_metric(self, metric_name):
        """ Checks if a metric already exists """
        if metric_name in self.metrics.keys():
            return True
        return False

    def delete_metric(self, metric_name):
        """ Deletes a metric from the database """
        if metric_name in self.metric_names:
            del self.metrics[metric_name]

    def available_metrics(self, key, gripper='pr2'):
        """ Returns a list of available metric names """
        grasps = self.grasps(key, gripper=gripper)
        gm = self.grasp_metrics(key, grasps, gripper=gripper)
        metrics = set()
        for grasp in grasps:
            metrics.update(gm[grasp.grasp_id].keys())
        return list(metrics)

    # grasp data
    # TODO: implement handling of stable poses and tasks
    def grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Returns the list of grasps for the given graspable, optionally associated with the given stable pose """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return []
        return hfact.Hdf5ObjectFactory.grasps(self.grasp_data(key, gripper))

    def sorted_grasps(self, key, metric, gripper='pr2', stable_pose_id=None):
        """ Returns the list of grasps for the given graspable sorted by decreasing quality according to the given metric """
        grasps = self.grasps(key, gripper=gripper, stable_pose_id=stable_pose_id)
        if len(grasps) == 0:
            return [], []
        
        grasp_metrics = self.grasp_metrics(key, grasps, gripper=gripper, stable_pose_id=stable_pose_id)
        if metric not in grasp_metrics[grasp_metrics.keys()[0]].keys():
            raise ValueError('Metric %s not recognized' %(metric))

        grasps_and_metrics = [(g, grasp_metrics[g.grasp_id][metric]) for g in grasps]
        grasps_and_metrics.sort(key=lambda x: x[1], reverse=True)
        sorted_grasps = [g[0] for g in grasps_and_metrics]
        sorted_metrics = [g[1] for g in grasps_and_metrics]
        return sorted_grasps, sorted_metrics

    def has_grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Checks if grasps exist for a given object """
        if gripper not in self.grasp_data(key).keys():
            return False
        return True

    def delete_grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Deletes a set of grasps associated with the given gripper """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Nothing to delete' %(gripper))
            return False
        del self.grasp_data(key)[gripper]
        return True

    def store_grasps(self, key, grasps, gripper='pr2', stable_pose_id=None, force_overwrite=False):
        """ Associates grasps in list |grasps| with the given object. Optionally associates the grasps with a single stable pose """
        # create group for gripper if necessary
        if gripper not in self.grasp_data(key).keys():
            self.grasp_data(key).create_group(gripper)
            self.grasp_data(key, gripper).attrs.create(NUM_GRASPS_KEY, 0)

        # store each grasp in the database
        return hfact.Hdf5ObjectFactory.write_grasps(grasps, self.grasp_data(key, gripper), force_overwrite)

    def grasp_metrics(self, key, grasps, gripper='pr2', stable_pose_id=None, task_id=None):
        """ Returns a list of grasp metric dictionaries fot the list grasps provided to the database """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return {}
        return hfact.Hdf5ObjectFactory.grasp_metrics(grasps, self.grasp_data(key, gripper))

    def store_grasp_metrics(self, key, grasp_metric_dict, gripper='pr2', stable_pose_id=None, task_id=None, force_overwrite=False):
        """ Add grasp metrics in list |metrics| to the data associated with |grasps| """
        return hfact.Hdf5ObjectFactory.write_grasp_metrics(grasp_metric_dict, self.grasp_data(key, gripper), force_overwrite)

    def grasp_features(self, key, grasps, gripper='pr2', stable_pose_id=None, task_id=None):
        """ Returns the list of grasps for the given graspable, optionally associated with the given stable pose """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return {}
        return hfact.Hdf5ObjectFactory.grasp_features(grasps, self.grasp_data(key, gripper))        

    def store_grasp_features(self, key, grasp_feature_dict, gripper='pr2', stable_pose_id=None, task_id=None, force_overwrite=False):
        """ Add grasp metrics in list |metrics| to the data associated with |grasps| """
        return hfact.Hdf5ObjectFactory.write_grasp_features(grasp_feature_dict, self.grasp_data(key, gripper), force_overwrite)

    # stable pose data
    def stable_poses(self, key, min_p=0.0):
        """ Stable poses for object key """
        stps = hfact.Hdf5ObjectFactory.stable_poses(self.stable_pose_data(key))

        # prune low probability stable poses
        stp_list = []
        for stp in stps:
            if stp.p > min_p:
                stp_list.append(stp)
        return stp_list

    def stable_pose(self, key, stable_pose_id):
        """ Stable pose of stable pose id for object key """
        if stable_pose_id not in self.stable_pose_data(key).keys():
            raise ValueError('Stable pose id %s unknown' %(stable_pose_id))
        return hfact.Hdf5ObjectFactory.stable_pose(self.stable_pose_data(key), stable_pose_id)

    # rendered image data
    def rendered_images(self, key, stable_pose_id=None, image_type="depth"):
        if stable_pose_id is not None and stable_pose_id not in self.stable_pose_data(key).keys():
            logging.warning('Stable pose id %s unknown' %(stable_pose_id))
            return[]
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in self.stable_pose_data(key)[stable_pose_id].keys():
            logging.warning('No rendered images for stable pose %s' %(stable_pose_id))
            return []
        if stable_pose_id is not None and image_type not in self.rendered_image_data(key, stable_pose_id).keys():
            logging.warning('No rendered images of type %s for stable pose %s' %(image_type, stable_pose_id))
            return []
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in self.object(key).keys():
            logging.warning('No rendered images for object')
            return []
        if stable_pose_id is None and image_type not in self.rendered_image_data(key).keys():
            logging.warning('No rendered images of type %s for object' %(image_type))
            return []

        rendered_images = hfact.Hdf5ObjectFactory.rendered_images(self.rendered_image_data(key, stable_pose_id, image_type))
        for rendered_image in rendered_images:
            rendered_image.obj_key = key
        if stable_pose_id is not None:
            stable_pose = self.stable_pose(key, stable_pose_id)
            for rendered_image in rendered_images:
                rendered_image.stable_pose = stable_pose
        return rendered_images

    def has_rendered_images(self, key, stable_pose_id=None, image_type="depth"):
        """ Checks whether or not a graspable has rendered images for the given stable pose and image type """
        if stable_pose_id is not None and stable_pose_id not in self.stable_pose_data(key).keys():
            return False
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in self.stable_pose_data(key)[stable_pose_id].keys():
            return False
        if stable_pose_id is not None and image_type not in self.rendered_image_data(key, stable_pose_id).keys():
            return False
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in self.object(key).keys():
            return False
        if stable_pose_id is None and image_type not in self.rendered_image_data(key).keys():
            return False
        return True

    def delete_rendered_images(self, key, stable_pose_id=None, image_type="depth"):
        """ Delete previously rendered images """
        if self.has_rendered_images(key, stable_pose_id, image_type):
            del self.rendered_image_data(key, stable_pose_id)[image_type]
            return True
        return False


    def store_rendered_images(self, key, rendered_images, stable_pose_id=None, image_type="depth", force_overwrite=False):
        """ Store rendered images of the object for a given stable pose """
        if stable_pose_id is not None and stable_pose_id not in self.stable_pose_data(key).keys():
            raise ValueError('Stable pose id %s unknown' %(stable_pose_id))
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in self.stable_pose_data(key)[stable_pose_id].keys():
            self.stable_pose_data(key)[stable_pose_id].create_group(RENDERED_IMAGES_KEY)
        if stable_pose_id is not None and image_type not in self.rendered_image_data(key, stable_pose_id).keys():
            self.rendered_image_data(key, stable_pose_id).create_group(image_type)
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in self.object(key).keys():
            self.object(key).create_group(RENDERED_IMAGES_KEY)
        if stable_pose_id is None and image_type not in self.rendered_image_data(key).keys():
            self.rendered_image_data(key).create_group(image_type)

        return hfact.Hdf5ObjectFactory.write_rendered_images(rendered_images, self.rendered_image_data(key, stable_pose_id, image_type),
                                                             force_overwrite)

