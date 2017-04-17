"""
Classes for indexing the database
Author: Jeff
"""
from abc import ABCMeta, abstractmethod

import IPython
import numpy as np
from PIL import Image
import os
import sys

import matplotlib.pyplot as plt

import database as db
import experiment_config as ec
import graspable_object as go
import kernels

class Hdf5DatabaseIndexer:
    """
    Abstract class for database indexing. Main purpose is to wrap individual datasets.
    Basically wraps the kernel nearest neighbor classes to automatically use HDF5 data and specific featurizations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, matcher):
        if not isinstance(matcher, kernels.NearestNeighbor):
            raise ValueError('Must provide a nearest neighbor object for indexing')
        self.matcher_ = matcher # nearest neighbor object
        self._create_table()

    @abstractmethod
    def _retrieve_objects(self):
        """ Private method to retrieve objects from an HDF5 database """
        pass

    def _create_table(self):
        """ Creates the master indexing table """
        object_list = self._retrieve_objects()
        featurized_objects = self._featurize(object_list)
        self.matcher_.train(featurized_objects)

    def _featurize(self, datapoints):
        """ Featurizes the datapoints """
        return datapoints

    def nearest(self, query, return_indices=False):
        """ Featurizes a datapoint x from the database """
        return self.k_nearest(query, 1, return_indices)

    def k_nearest(self, query, k, return_indices=False):
        """ Featurizes a datapoint x from the database """
        featurized_query = self._featurize([query])[0]
        return self.matcher_.nearest_neighbors(featurized_query, k, return_indices)

    def within_distance(self, query, dist=0.5, return_indices=False):
        """ Featurizes a datapoint x from the database """
        featurized_query = self._featurize([query])[0]
        return self.matcher_.within_distance(featurized_query, dist, return_indices)
