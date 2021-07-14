from setuptools import setup, find_packages

import os

with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
name='med_segmentation',
version='1.0.0',
author='MIDAS and kSpace Astronauts',
author_email='thomas.kuestner@med.uni-tuebingen.de',
description='Medical Image Segmentation',
long_description=open(os.path.join(os.path.dirname(file), 'README.md')).read(),
package_dir={'med_seg': 'med_seg'},
packages=['med_seg'],
license='public',
keywords='None',
classifiers=[
'Natural Language :: English',
'Programming Language :: Python :: 3 :: Only'
],
install_requires==requirements,
)

