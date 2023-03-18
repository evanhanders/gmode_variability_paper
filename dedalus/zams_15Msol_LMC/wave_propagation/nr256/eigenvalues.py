"""
d3 script for eigenvalue problem of anelastic convection / waves in a massive star.

"""
import gc
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

from d3_stars.simulations.evp_functions import StellarEVP

if __name__ == '__main__':
    eigenvalues = StellarEVP()
    for ell in eigenvalues.ells:
        logger.info('solving EVP for ell = {}'.format(ell))
        eigenvalues.solve(ell)
        eigenvalues.check_eigen()
        eigenvalues.output()
        eigenvalues.get_duals(max_cond=1e10, ell=ell)
        logger.info('solved EVP for ell = {}'.format(ell))
