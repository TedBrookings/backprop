#!/usr/bin/env python


from setuptools import setup
from setuptools import find_packages
from glob import glob
import os


setup(
    name='backprop',
    version='0.1.0',
    description='Sandbox for learning neural net backprop algorithm',
    url='https://github.com/TedBrookings/backprop',
    author='Ted Brookings',
    author_email='ted.brookings@googlemail.com',
    license='BSD 3-clause',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    py_modules=[os.path.splitext(os.path.basename(path))[0]
                for path in glob('src/*.py')],
    install_requires=[
        'pandas',
        'scipy',
        'numpy',
        'sklearn',
        'pytest',
    ],
    scripts=['src/backprop/backprop_demo.py'],
    include_package_data=True,
    zip_safe=False
)

# need to find right way to say gsutil and tabix should be installed
