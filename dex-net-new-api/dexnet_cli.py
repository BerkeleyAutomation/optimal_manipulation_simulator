"""
Command line tool for using database
Author: Jeff Mahler
"""

import argparse
import copy
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import readline
import signal
import sys

import dexnet.database as db
import dexnet.experiment_config as ec
import dexnet.grasp_quality_config as gqc
import dexnet.grasp_quality_function as gqf
import dexnet.grasp_sampler as gs
import dexnet.gripper as gr
import dexnet.mesh_processor as mp

USE_ALAN = True
try:
    from alan.core import Visualizer as vis
except:
    logging.warning('Failed to import ALAN')
    from dexnet.mayavi_visualizer import MayaviVisualizer as vis
    import mayavi.mlab as mlab
    USE_ALAN = False

DEFAULT_CONFIG = 'cfg/defaults.yaml'
SUPPORTED_MESH_FORMATS = ['.obj', '.off', '.wrl', '.stl']
RE_SPACE = re.compile('.*\s+$', re.M)

class Completer(object):
    """
    Tab completion class for Dex-Net CLI.
    Adapted from http://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input
    """
    def __init__(self, commands=[]):
        """ Provide a list of commands """
        self.commands = commands

    def _listdir(self, root):
        "List directory 'root' appending the path separator to subdirs."
        res = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def _complete_path(self, path=None):
        "Perform completion of filesystem path."
        if path is None or path == '':
            return self._listdir('./')
        dirname, rest = os.path.split(path)
        tmp = dirname if dirname else '.'
        res = [os.path.join(dirname, p)
                for p in self._listdir(tmp) if p.startswith(rest)]
        # more than one match, or single match which does not exist (typo)
        if len(res) > 1 or not os.path.exists(path):
            return res
        # resolved to a single directory, so return list of files below it
        if os.path.isdir(path):
            return [os.path.join(path, p) for p in self._listdir(path)]
        # exact file match terminates this completion
        return [path + ' ']

    def complete_extra(self, args):
        "Completions for the 'extra' command."
        # treat the last arg as a path and complete it
        if len(args) == 0:
            return self._listdir('./')            
        return self._complete_path(args[-1])

    def complete(self, text, state):
        "Generic readline completion entry point."
        buffer = readline.get_line_buffer()
        line = readline.get_line_buffer().split()
        # account for last argument ending in a space
        if RE_SPACE.match(buffer):
            line.append('')

        return (self.complete_extra(line) + [None])[state]

class DexNet(object):
    API = {0: ('Open a database', 'open_database'),
           1: ('Open a dataset', 'open_dataset'),
           2: ('Add object(s) to the dataset', 'add_objects'),
           3: ('Compute grasps', 'compute_grasps'),
           4: ('Display object', 'display_object'),
           5: ('Display stable poses for object', 'display_stable_poses'),
           6: ('Display grasps for object', 'display_grasps'),
           7: ('Set config (advanced)', 'set_config'),
           8: ('Quit', 'close')
           }

    def __init__(self):
        # init core members
        self.database = None
        self.dataset = None

        # setup command line parsing
        self.comp = Completer()
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self.comp.complete)

        # open default config
        self.config = ec.ExperimentConfig(DEFAULT_CONFIG)
        self.config['obj_rescaling_type'] = mp.MeshProcessor.RescalingTypeRelative

        # display welcome message
        self.display_welcome()

    def display_welcome(self):
        print '####################################################'
        print 'DEX-NET 0.1 Command Line Interface'
        print 'Brought to you by AutoLab, UC Berkeley'
        print '####################################################'
        print

    def display_menu(self):
        print 'AVAILABLE COMMANDS:'
        for command_id, command_desc in DexNet.API.iteritems():
            print '%d) %s' %(command_id, command_desc[0])
        print

    def run_user_command(self):
        command = raw_input('Enter a numeric command: ')
        try:
            try:
                command_id = int(command)
                if command_id not in DexNet.API.keys():
                    raise RuntimeError()

                command_fn = getattr(self, DexNet.API[command_id][1])
            except:
                raise RuntimeError()
            return command_fn()
        except RuntimeError:
            print 'Command %s not recognized, please try again' %(command)
        return True

    # commands below
    def open_database(self):
        """ Open a database """
        if self.database is not None:
            self.database.close()

        # get user input
        invalid_db = True
        while invalid_db:
            database_name = raw_input('Enter database name: ')
            tokens = database_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single database name'

            if database_name.lower() == 'q':
                return True
            
            # create new db
            database_root, database_ext = os.path.splitext(database_name)
            if database_ext == '':
                database_name = database_name + db.HDF5_EXT
                database_root, database_ext = os.path.splitext(database_name)
            if database_ext != db.HDF5_EXT:
                print 'Database must have extension %s' %(db.HDF5_EXT) 

            # create new db if asked
            if not os.path.exists(database_name):
                print 'Database %s does not exist' %(database_name)
                create_new = raw_input('Create new db? [y/n] ')
                while create_new.lower() != 'n' and create_new.lower() != 'y':
                    print 'Did not understand input. Please answer \'y\' or \'n\''
                    create_new = raw_input('Create new db? [y/n] ')
                
                if create_new.lower() == 'n':
                    print 'Aborting database creation'
                    return True

            # open db
            self.database = db.Hdf5Database(database_name,
                                            access_level=db.READ_WRITE_ACCESS)                

            print 'Opened database %s' %(database_name)
            print
            invalid_db = False
        
        return True

    def open_dataset(self):
        """ Open a dataset """
        if self.database is None:
            print 'You must open a database first'
            print
            return True
        
        # show existing datasets
        invalid_ds = True
        existing_datasets = [d.name for d in self.database.datasets]
        print 'Existing datasets:'
        for dataset_name in existing_datasets:
            print dataset_name
        print

        # get user input
        while invalid_ds:
            # get dataset name
            dataset_name = raw_input('Enter dataset name: ')
            tokens = dataset_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single dataset name'

            if dataset_name.lower() == 'q':
                return True
            
            # create new ds
            if dataset_name not in existing_datasets:
                print 'Dataset %s does not exist' %(dataset_name)
                create_new = raw_input('Create new dataset? [y/n] ')
                while create_new.lower() != 'n' and create_new.lower() != 'y':
                    print 'Did not understand input. Please answer \'y\' or \'n\''
                    create_new = raw_input('Create new dataset? [y/n] ')
                
                if create_new.lower() == 'n':
                    print 'Aborting dataset creation'
                else:
                    print 'Creating new dataset %s' %(dataset_name)
                    self.database.create_dataset(dataset_name)

            # open db
            self.dataset = self.database.dataset(dataset_name)
            print 'Opened dataset %s' %(dataset_name)
            print
            invalid_ds = False

        return True

    def _create_graspable(self, filename):
        """ Add graspable to dataset """
        if self.database is None:
            raise ValueError('You must open a database first')
        if self.dataset is None:
            raise ValueError('You must open a database first')

        # open mesh preprocessor
        path, root = os.path.split(filename)
        key, _ = os.path.splitext(root)
        if key in self.dataset.object_keys:
            print 'An object with key %s already exists. Overwrites not supported at this time' %(key)
            return False

        mesh_processor = mp.MeshProcessor(filename, self.config['cache_dir'])
        mesh_processor.generate_graspable(self.config)

        # mesh mass
        mass = self.config['default_mass']
        if not self.config['use_default_mass']:
            invalid_mass = True
            while invalid_mass:
                # get mass string
                print
                mass_str = raw_input('What is the mass of the object in kilograms? [ENTER for default value]: ')
                tokens = mass_str.split()
                if len(tokens) > 1:
                    print 'Please provide only a single input'

                # compute mass
                try:
                    mass = float(mass_str)
                except:
                    if mass_str != '':
                        print 'Please provide a float input'
                        continue
                invalid_mass = False
                
        # write to database
        logging.info('Creating graspable')
        self.dataset.create_graspable(key, mesh_processor.mesh, mesh_processor.sdf,
                                      mesh_processor.shot_features,
                                      mesh_processor.stable_poses,
                                      category='unknown', mass=mass)
        return True

    def add_objects(self):
        """ Add objects """
        if self.database is None:
            print 'You must open a database first'
            print
            return True

        if self.dataset is None:
            print 'You must open a dataset first'
            print
            return True

        # get user input
        invalid_obj = True
        while invalid_obj:
            # get object name
            obj_name = raw_input('Enter path to a mesh file or directory: ')
            tokens = obj_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single input'

            if obj_name.lower() == 'q':
                return True
            
            # get list of valid objects
            obj_filenames = []
            obj_root, obj_ext = os.path.splitext(obj_name)
            if obj_ext == '':
                if not os.path.exists(obj_name):
                    print 'Directory %s does not exist' %(obj_name)
                    continue
                for filename in os.listdir(obj_name):
                    file_root, file_ext = os.path.splitext(filename)
                    if file_ext in SUPPORTED_MESH_FORMATS:
                        obj_filenames.append(os.path.join(obj_name, filename))
            elif obj_ext in SUPPORTED_MESH_FORMATS:
                obj_filenames.append(obj_name)

            if len(obj_filenames) == 0:
                print 'No valid meshes provided'
                continue

            # generate new graspables
            for obj_filename in obj_filenames:
                print 'Creating graspable for', obj_filename
                self._create_graspable(obj_filename)

            invalid_obj = False
        
        return True

    def _gravity_wrench(self, obj, stable_pose):
        """ Compute the wrench exerted by gravity """
        gravity_magnitude = obj.mass * self.config['gravity_accel']
        stable_pose_normal = stable_pose.r[2]
        gravity_force = -gravity_magnitude * stable_pose_normal
        gravity_resist_wrench = -np.append(gravity_force, [0,0,0])
        return gravity_resist_wrench

    def _compute_grasps(self, obj, gripper, stable_pose_id=None):
        """ Sample grasps and compute metrics for the object and gripper """
        if self.database is None:
            raise ValueError('You must open a database first')
        if self.dataset is None:
            raise ValueError('You must open a database first')

        # sample grasps
        print 'Sampling grasps'
        if self.config['grasp_sampler'] == 'antipodal':
            sampler = gs.AntipodalGraspSampler(gripper, self.config)
        elif self.config['grasp_sampler'] == 'gaussian':
            sampler = gs.GaussianGraspSampler(gripper, self.config)
        elif self.config['grasp_sampler'] == 'uniform':
            sampler = gs.UniformGraspSampler(gripper, self.config)

        grasps = sampler.generate_grasps(
            obj, check_collisions=self.config['check_collisions'],
            max_iter=self.config['max_grasp_sampling_iters'])

        # store and reload to get database ids
        self.dataset.store_grasps(obj.key, grasps, gripper=gripper.name)
        grasps = self.dataset.grasps(obj.key, gripper=gripper.name)

        print 'Sampled %d grasps' %(len(grasps))

        # load stable poses
        if stable_pose_id is None:
            stable_poses = self.dataset.stable_poses(obj.key)
        else:
            stable_poses = [self.dataset.stable_pose(obj.key, stable_pose_id)]

        # compute grasp metrics
        print 'Computing metrics'
        grasp_metrics = {}
        for metric_name, metric_spec in self.config['metrics'].iteritems():
            # create metric
            metric_config = gqc.GraspQualityConfigFactory.create_config(metric_spec)

            # create multiple configs in the case of gravity
            metric_names = [metric_name]
            metric_configs = [metric_config]
            if metric_config.quality_method == 'partial_closure':
                # setup new configs
                metric_configs = []
                metric_names = []
                gravity_metric_config = copy.copy(metric_config)

                # add gravity wrenches
                for stable_pose in stable_poses:
                    gravity_metric_config.target_wrench = self._gravity_wrench(obj, stable_pose)
                    gravity_metric_config.obj_uncertainty['R_sample_sigma'] = stable_pose.r.T
                    gravity_metric_config.grasp_uncertainty['R_sample_sigma'] = stable_pose.r.T
                    metric_names.append(metric_name + '_' + stable_pose.id)
                    metric_configs.append(gravity_metric_config)

            # compute metrics for each config
            for metric_name, metric_config in zip(metric_names, metric_configs):
                print 'Computing metric %s' %(metric_name)

                # add to database
                if not self.dataset.has_metric(metric_name):
                    self.dataset.create_metric(metric_name, metric_config)
                    
                # add params from gripper (right now we don't want the gripper involved in quality computation)
                setattr(metric_config, 'force_limits', gripper.force_limit)
                setattr(metric_config, 'finger_radius', gripper.finger_radius)
            
                # create quality function
                quality_fn = gqf.GraspQualityFunctionFactory.create_quality_function(obj, metric_config)
                
                # compute quality for each grasp
                for k, grasp in enumerate(grasps):
                    if k % self.config['metric_display_rate'] == 0:
                        print 'Computing metric for grasp %d of %d' %(k+1, len(grasps))

                    # init grasp metric dict if necessary
                    if grasp.grasp_id not in grasp_metrics.keys():
                        grasp_metrics[grasp.grasp_id] = {}
                    
                    # compute grasp quality
                    q = quality_fn(grasp)
                    grasp_metrics[grasp.grasp_id][metric_name] = q.mean_quality

        # store the grasp metrics
        self.dataset.store_grasp_metrics(obj.key, grasp_metrics, gripper=gripper.name,
                                         force_overwrite=True)

    def compute_grasps(self):
        """ Compute grasps for an object or the entire dataset """
        if self.database is None:
            print 'You must open a database first'
            print
            return True

        if self.dataset is None:
            print 'You must open a dataset first'
            print
            return True

        # list grippers 
        print
        print 'Available grippers:'
        grippers = os.listdir(self.config['gripper_dir'])
        for gripper_name in grippers:
            print gripper_name
        print

        # set gripper
        invalid_gr = True
        while invalid_gr:
            # get object name
            gripper_name = raw_input('Enter gripper name: ')
            tokens = gripper_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single input'

            if gripper_name.lower() == 'q':
                return True

            if gripper_name not in grippers:
                print 'Gripper %s not recognized' %(gripper_name)
                continue

            gripper = gr.RobotGripper.load(gripper_name)
            print 'Loaded gripper', gripper.name
            invalid_gr = False

        # list objects
        print
        print 'Available objects:'
        for key in self.dataset.object_keys:
            print key
        print

        # set objects
        invalid_obj = True
        while invalid_obj:
            # get object name
            obj_name = raw_input('Enter object key [ENTER for entire dataset]: ')
            tokens = obj_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single input'

            if obj_name.lower() == 'q':
                return True

            obj_names = []
            if obj_name == '':
                obj_names = self.dataset.object_keys
            else:
                obj_names.append(obj_name)                
            

            illegal_obj_names = False
            for obj_name in obj_names:
                if obj_name not in self.dataset.object_keys:
                    print 'Key %s not in database' %(obj_name)
                    illegal_obj_names = True
            if illegal_obj_names:
                continue

            # compute grasps for objects
            for obj_name in obj_names:
                stable_pose_id = None

                # check for grasps, overwrite if specified
                if self.dataset.has_grasps(obj_name, gripper=gripper.name):
                    print 'Grasps already exist for gripper %s on object %s' %(gripper.name, obj_name)
                    
                    overwrite = raw_input('Overwrite grasps? [y/n] ')
                    while overwrite.lower() != 'n' and overwrite.lower() != 'y':
                        print 'Did not understand input. Please answer \'y\' or \'n\''
                        overwrite = raw_input('Create new dataset? [y/n] ')
                
                    if overwrite.lower() == 'n':
                        print 'Aborting grasp computation for object %s' %(obj_name)
                        continue
                    else:
                        print 'Overwriting grasps for object %s' %(obj_name)
                        self.dataset.delete_grasps(obj_name, gripper=gripper.name)

                # get user input for stable pose
                if self.config['custom_stable_poses']:
                    # show available stable poses
                    stable_poses = self.dataset.stable_poses(obj_name)
                    stable_pose_ids = [s.id for s in stable_poses]

                    # list available stps
                    print
                    print 'Available stable poses:'
                    for s_id in stable_pose_ids:
                        print s_id
                    print

                    # get id from user
                    invalid_stp = True
                    while invalid_stp:
                        stp_name = raw_input('Enter stable pose id: ')
                        tokens = stp_name.split()
                        if len(tokens) > 1:
                            print 'Please provide only a single input'
                            
                        if stp_name not in stable_pose_ids:
                            print 'Stable pose id %s not recognized' %(stp_name)
                            continue
                        
                        stable_pose_id = stp_name
                        invalid_stp = False

                print 'Computing grasps for object %s' %(obj_name)
                obj = self.dataset[obj_name]
                self._compute_grasps(obj, gripper, stable_pose_id=stable_pose_id)

            invalid_obj = False

        return True

    def display_object(self):
        """ Display an object """
        if self.database is None:
            print 'You must open a database first'
            print
            return True

        if self.dataset is None:
            print 'You must open a dataset first'
            print
            return True

        print 'Available objects:'
        for key in self.dataset.object_keys:
            print key
        print

        invalid_obj = True
        while invalid_obj:
            # get object name
            obj_name = raw_input('Enter object key: ')
            tokens = obj_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single input'

            if obj_name.lower() == 'q':
                return True

            if obj_name not in self.dataset.object_keys:
                print 'Key %s not in database' %(obj_name)
                continue

            print 'Displaying', obj_name
            obj = self.dataset[obj_name]

            if USE_ALAN:
                vis.figure()
                vis.mesh(obj.mesh, color=(0.5, 0.5, 0.5), style='surface')
                vis.show()
            else:
                mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
                vis.plot_mesh(obj.mesh, color=(0.5, 0.5, 0.5), style='surface')
                mlab.show()

            invalid_obj = False

        return True

    def display_stable_poses(self):
        """ Display stable poses """
        if self.database is None:
            print 'You must open a database first'
            print
            return True

        if self.dataset is None:
            print 'You must open a dataset first'
            print
            return True

        print 'Available objects:'
        for key in self.dataset.object_keys:
            print key
        print

        invalid_obj = True
        while invalid_obj:
            # get object name
            obj_name = raw_input('Enter object key: ')
            tokens = obj_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single input'

            if obj_name.lower() == 'q':
                return True

            if obj_name not in self.dataset.object_keys:
                print 'Key %s not in database' %(obj_name)
                continue

            print 'Displaying', obj_name
            obj = self.dataset[obj_name]
            stable_poses = self.dataset.stable_poses(obj_name)

            if USE_ALAN:
                for stable_pose in stable_poses:
                    print 'Stable pose %s with p=%.3f' %(stable_pose.id, stable_pose.p)
                    vis.figure()
                    vis.mesh_stable_pose(obj.mesh, stable_pose,
                                         color=(0.5, 0.5, 0.5), style='surface')
                    vis.show()
            else:
                for stable_pose in stable_poses:
                    print 'Stable pose %s with p=%.3f' %(stable_pose.id, stable_pose.p)
                    mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
                    vis.plot_stable_pose(obj.mesh, stable_pose,
                                         color=(0.5, 0.5, 0.5), style='surface',
                                         d=self.config['table_extent'])
                    mlab.show()

            invalid_obj = False

        return True

    def display_grasps(self):
        """ Display grasps for an object """
        if self.database is None:
            print 'You must open a database first'
            print
            return True

        if self.dataset is None:
            print 'You must open a dataset first'
            print
            return True

        # list grippers 
        print
        print 'Available grippers:'
        grippers = os.listdir(self.config['gripper_dir'])
        for gripper_name in grippers:
            print gripper_name
        print

        # set gripper
        invalid_gr = True
        while invalid_gr:
            # get object name
            gripper_name = raw_input('Enter gripper name: ')
            tokens = gripper_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single input'

            if gripper_name.lower() == 'q':
                return True

            if gripper_name not in grippers:
                print 'Gripper %s not recognized' %(gripper_name)
                continue

            gripper = gr.RobotGripper.load(gripper_name)
            print 'Loaded gripper', gripper.name
            invalid_gr = False

        # list objects
        print 'Available objects:'
        for key in self.dataset.object_keys:
            print key
        print

        invalid_obj = True
        while invalid_obj:
            # get object name
            obj_name = raw_input('Enter object key: ')
            tokens = obj_name.split()
            if len(tokens) > 1:
                print 'Please provide only a single input'

            if obj_name.lower() == 'q':
                return True

            if obj_name not in self.dataset.object_keys:
                print 'Key %s not in database' %(obj_name)
                continue

            # list metrics
            print
            print 'Available metrics:'
            metrics = self.dataset.available_metrics(obj_name, gripper=gripper.name)
            for metric_name in metrics:
                print metric_name
            print

            # set gripper
            invalid_mt = True
            while invalid_mt:
                # get object name
                metric_name = raw_input('Enter metric name: ')
                tokens = metric_name.split()
                if len(tokens) > 1:
                    print 'Please provide only a single input'
                    
                if metric_name.lower() == 'q':
                    return True

                if metric_name not in metrics:
                    print 'Metric %s not recognized' %(metric_name)
                    continue
                
                print 'Using metric %s' %(metric_name)
                invalid_mt = False

            print 'Displaying grasps for gripper %s on object %s' %(gripper.name, obj_name)
            obj = self.dataset[obj_name] 
            grasps, metrics = self.dataset.sorted_grasps(obj_name, metric_name,
                                                         gripper=gripper.name)
                 
            if len(grasps) == 0:
                print 'No grasps for gripper %s on object %s' %(gripper.name, obj_name)
                return True
                         
            low = np.min(metrics)
            high = np.max(metrics)
            q_to_c = lambda quality: self.config['quality_scale'] * (quality - low) / (high - low)
      
            if USE_ALAN:
                raise ValueError('ALAN does not yet support grasp display')
            else:
                if self.config['show_gripper']:
                    i = 0
                    for grasp, metric in zip(grasps, metrics):
                        print 'Grasp %d %s=%.5f' %(grasp.grasp_id, metric_name, metric)
                        mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
                        vis.plot_mesh(obj.mesh, color=(0.5, 0.5, 0.5), style='surface')
                        color = plt.get_cmap('hsv')(q_to_c(metric))[:-1]
                        vis.plot_gripper(grasp, gripper=gripper,
                                         color=color)
                        vis.plot_grasp(grasp, grasp_axis_color=color,
                                       endpoint_color=color)
                        mlab.show()
                        i += 1
                        if i >= self.config['max_plot_gripper']:
                            break
                else:
                    mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
                    vis.plot_mesh(obj.mesh, color=(0.5, 0.5, 0.5), style='surface')
                    for grasp, metric in zip(grasps, metrics):
                        print 'Grasp %d %s=%.5f' %(grasp.grasp_id, metric_name, metric)
                        color = plt.get_cmap('hsv')(q_to_c(metric))[:-1]
                        vis.plot_grasp(grasp, grasp_axis_color=color,
                                       endpoint_color=color)
                    mlab.show()
            invalid_obj = False

        return True

    def set_config(self):
        """ Set a new confguration file """
        # get user input
        invalid_cfg = True
        while invalid_cfg:
            config_filename = raw_input('Enter YAML config filename: ')
            tokens = config_filename.split()
            if len(tokens) > 1:
                print 'Please provide only a single config filename'

            if config_filename.lower() == 'q':
                return True
            
            if not os.path.exists(config_filename):
                print 'Config %s does not exist' %(config_filename)
                continue
            
            self.config = ec.ExperimentConfig(config_filename)
            print 'Using config %s' %(config_filename)
            invalid_cfg = False
        return True
        
    def close(self):
        print 'Closing Dex-Net. Goodbye!'
        if self.database:
            self.database.close()
        return False

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.WARNING)

    # parse args
    parser = argparse.ArgumentParser(description='Command line interface to Dex-Net')
    args = parser.parse_args()

    # open dex-net handle
    dexnet_cli = DexNet()

    # setup graceful exit
    continue_dexnet = True
    def close_dexnet(signal=0, frame=None):
        dexnet_cli.close()
        continue_dexnet = False
        exit(0)
    signal.signal(signal.SIGINT, close_dexnet)

    # main loop
    while continue_dexnet:
        # display menu
        dexnet_cli.display_menu()

        # get user input
        continue_dexnet = dexnet_cli.run_user_command()

