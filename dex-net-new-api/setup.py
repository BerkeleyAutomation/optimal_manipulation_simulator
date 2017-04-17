"""
Setup of Dex-Net python codebase
Author: Jeff Mahler
"""
from setuptools import setup

setup(name='dex-net',
      version='0.1.dev0',
      description='Dex-Net project code',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['dexnet'],
     )
