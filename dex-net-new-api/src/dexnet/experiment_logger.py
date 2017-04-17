'''
Class to handle experiment logging.
Authors: Jeff, Jacky
'''

import os
import csv

import numpy as np

class ExperimentLogger:

	_MASTER_RECORD_FILENAME = 'results.csv'

	def __init__(self, root_path):
		self.root_path = root_path

		self.id = ExperimentLogger.gen_experiment_id()

		ExperimentLogger.ensure_path_exists(self.root_path)
		self.master_record_filepath = os.path.join(self.root_path, ExperimentLogger._MASTER_RECORD_FILENAME)

		self.update_or_create_master_record()

	def update_or_create_master_record(self):
		experiment_id_dict = {'experiment_id': self.id, 'use': 0}

		if os.path.isfile(self.master_record_filepath):
			with open(self.master_record_filepath, 'a') as file:
				csv_writer = csv.DictWriter(file, experiment_id_dict.keys())
				csv_writer.writerow(experiment_id_dict)
		else:
			with open(self.master_record_filepath, 'w') as file:
				csv_writer = csv.DictWriter(file, experiment_id_dict.keys())
				csv_writer.writeheader()
				csv_writer.writerow(experiment_id_dict)


	@staticmethod
	def gen_experiment_id(n=10):
	    """ Random string for naming """
	    chrs = 'abcdefghijklmnopqrstuvwxyz'
	    inds = np.random.randint(0, len(chrs), size=n)
	    return ''.join([chrs[i] for i in inds])

	@staticmethod
	def ensure_path_exists(path):
		'''Ensures a path exists'''
		if not os.path.exists(path):
			os.makedirs(path)