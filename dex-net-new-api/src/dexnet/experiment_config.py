"""
YAML Configuration Parser - basically reads everything into a dictionary
Author : Jeff Mahler
"""
import argparse
import logging
import math
import numpy as np
import os
import scipy
import shutil
import sys
import time

import yaml, re
from collections import OrderedDict
import os.path as osp

import IPython

#TODO: change to use alan version of config file

class ExperimentConfig(object):
    """
    Class to load a configuration file and parse into a dictionary
    """
    def __init__(self, filename):
        self.config = None # initialize empty config
        self.load_config(filename)

    def load_config(self, filename):
        """
        Loads a yaml configuration file from the given filename
        """
        # read entire file for metadata
        fh = open(filename, 'r')
        self.file_contents = fh.read()

        # replace !include directives with content
        config_dir = os.path.split(filename)[0]
        include_re = re.compile('^!include\s+(.*)$', re.MULTILINE)
        def include_repl(matchobj):
            fname = os.path.join(config_dir, matchobj.group(1))
            with open(fname) as f:
                return f.read()
        while re.search(include_re, self.file_contents): # for recursive !include
            self.file_contents = re.sub(include_re, include_repl, self.file_contents)

        # read in dictionary
        self.config = self.__ordered_load(self.file_contents)

        # convert functions of other params to true expressions
        for k in self.config.keys():
            self.config[k] = ExperimentConfig.__convert_key(self.config[k])

        # load core configuration
        return self.config

    def keys(self):
        return self.config.keys()

    def __contains__(self, key):
        """ Override 'in' operator """
        return key in self.config.keys()

    def __getitem__(self, key):
        """ Overrides the key access operator [] """
        return self.config[key]

    def __setitem__(self, key, val):
        self.config[key] = val

    def iteritems(self):
        return self.config.iteritems()

    @staticmethod
    def __convert_key(expression):
        """ Converts keys in YAML that reference other keys """
        if type(expression) is str and len(expression) > 2 and expression[1] == '!':
            expression = eval(expression[2:-1])
        return expression

    def __ordered_load(self, stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
        """
        Load an ordered dictionary from a yaml file. Borrowed from John Schulman

        See:
        http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts/21048064#21048064"
        """
        class OrderedLoader(Loader):
            pass
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            lambda loader, node: object_pairs_hook(loader.construct_pairs(node)))
        return yaml.load(stream, OrderedLoader)

