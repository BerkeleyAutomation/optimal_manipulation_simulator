"""
CNN-based database indexers
Author: Jeff Mahler
"""
import database as db
import database_indexer as di
import kernels
import rendered_image as ri

USE_ALAN = False
try:
    from alan.rgbd import CNNBatchFeatureExtractor
except:
    USE_ALAN = True

class CNN_Hdf5DatabaseIndexer(di.Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, config):
        self.feature_extractor_ = CNNBatchFeatureExtractor(config)
        matcher = kernels.KDTree(phi = lambda x: x.descriptor)
        self._parse_config(config)
        di.Hdf5DatabaseIndexer.__init__(self, matcher)

    def _parse_config(self, config):
        """ Parse the config to read in key parameters """
        self.use_stable_poses_ = config['use_stable_poses']
        self.image_type_ = config['image_type']

    def _featurize(self, datapoints):
        """ Converts an image x to a CNN feature vector """
        images = [x.image for x in datapoints]
        descriptors = self.feature_extractor_.extract(images)
        for x, descriptor in zip(datapoints, descriptors):
            x.descriptor = descriptor
        return datapoints

class CNN_Hdf5DatasetIndexer(CNN_Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, dataset, config):
        if not isinstance(dataset, db.Hdf5Dataset):
            raise ValueError('Must provide an Hdf5 dataset object to index')
        self.dataset_ = dataset # handle to hdf5 data
        CNN_Hdf5DatabaseIndexer.__init__(self, config)

    def _retrieve_objects(self):
        """ Retrieves objects from the provided dataset. """
        rendered_image_pool = []
        for obj_key in dataset.object_keys:
            if self.use_stable_poses_:
                stable_poses = self.dataset_.stable_poses(obj_key)
                for stable_pose in stable_poses:
                    rendered_image_pool.extend(self.dataset_.rendered_images(obj_key, stable_pose_id=stable_pose.id, image_type=self.image_type_))
            else:
                rendered_image_pool.extend(self.dataset_.rendered_images(obj_key, image_type=self.image_type_))
        return rendered_image_pool

class CNN_Hdf5ObjectIndexer(CNN_Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, obj_key, dataset, config):
        if not isinstance(dataset, db.Hdf5Dataset):
            raise ValueError('Must provide an Hdf5 dataset object to index')
        if obj_key not in dataset.object_keys:
            raise ValueError('Object key %s not in datset' %(obj_key))            
        self.obj_key_ = obj_key
        self.dataset_ = dataset # handle to hdf5 data
        CNN_Hdf5DatabaseIndexer.__init__(self, config)

    def _retrieve_objects(self):
        """ Retrieves objects from the provided dataset. """
        rendered_image_pool = []
        if self.use_stable_poses_:
            stable_poses = self.dataset_.stable_poses(self.obj_key_)
            for stable_pose in stable_poses:
                rendered_image_pool = self.dataset_.rendered_images(self.obj_key_, stable_pose_id=stable_pose.id, image_type=self.image_type_)
        else:
            rendered_image_pool = self.dataset_.rendered_images(self.obj_key_, image_type=self.image_type_)
        return rendered_image_pool

class CNN_Hdf5ObjectStablePoseIndexer(CNN_Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, obj_key, stp_id, dataset, config):
        if not isinstance(dataset, db.Hdf5Dataset):
            raise ValueError('Must provide an Hdf5 dataset object to index')
        if obj_key not in dataset.object_keys:
            raise ValueError('Object key %s not in datset' %(obj_key))            
        self.obj_key_ = obj_key
        self.stp_id_ = stp_id
        self.dataset_ = dataset # handle to hdf5 data
        CNN_Hdf5DatabaseIndexer.__init__(self, config)

    def _retrieve_objects(self):
        """ Retrieves objects from the provided dataset. """
        rendered_image_pool = []
        stable_pose = self.dataset_.stable_pose(self.obj_key_, self.stp_id_)
        rendered_image_pool = self.dataset_.rendered_images(self.obj_key_, stable_pose_id=stable_pose.id, image_type=self.image_type_)
        return rendered_image_pool
