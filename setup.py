#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2023 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling under supervision of prof. Veronique
# Van Speybroeck. Usage of this package should be authorized by prof. Van
# Van Speybroeck.


from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='ThermoLIB',
    version='1.7.2',
    description='Python library with various sub packages related to thermodynamic and free energy profiles.',
    author='Louis Vanduyfhuys',
    author_email='Louis.Vanduyfhuys@UGent.be',
    #url='https://github.com/molmod/XXX',
    package_dir = {'thermolib': 'thermolib'},
    packages=['thermolib', 'thermolib.thermodynamics', 'thermolib.kinetics'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Science/Engineering :: Molecular Science'
    ],
    install_requires=['cython>=0.29.23',
                      'numpy>=1.0',
                      'scikit-learn>=0.24.2',
                      'matplotlib',
                      'molmod',
                      ],
    ext_modules = cythonize("thermolib/ext.pyx"),
    include_dirs=[numpy.get_include()]
)
