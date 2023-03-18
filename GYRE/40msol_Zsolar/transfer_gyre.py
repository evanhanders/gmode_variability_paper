"""
This file reads in gyre eigenfunctions, calculates the velocity and velocity dual basis, and outputs in a clean format so that it's ready to be fed into the transfer function calculation.
"""
import os
import numpy as np
import pygyre as pg

from compstar.tools.mesa import find_core_cz_radius
from compstar.waves.clean_gyre_eig import GyreMSGPostProcessor, solar_z

plot = True
use_delta_L = False
Lmin = 1
Lmax = 15
ell_list = np.arange(Lmin, Lmax+1)
folder = 'gyre_output'
for ell in ell_list:
    om_list = np.logspace(-8, -2, 1000) #Hz * 2pi

    mesa_LOG   = '../../MESA/40msol_Zsolar/LOGS/profile53.data'
    pulse_file = '{}.GYRE'.format(mesa_LOG)
    mode_base = './gyre_output/mode_id{:05d}_ell{:03d}_m+00_n{:06d}.h5'
    files = []

    max_cond = 1e12
    summary_file='gyre_output/summary_ell{:02d}.txt'.format(ell)
    summary = pg.read_output(summary_file)

    #sort eigenvalues by 1/freq
    sorting = np.argsort(summary['freq'].real**(-1))
    summary = summary[sorting]

    good_freqs = []
    for row in summary:
        this_ell = row['l']
        this_id = row['id']
        n_pg = row['n_pg']
        if complex(row['freq']).real < 0:
            continue
        if n_pg >= 0: continue
        if this_ell != ell: continue
        files.append(mode_base.format(this_id, ell, n_pg))
        good_freqs.append(complex(row['freq']))

    post = GyreMSGPostProcessor(ell, summary_file, files, pulse_file, mesa_LOG,
                  filters=['Red',], initial_z=solar_z, specgrid='OSTAR2002',
                  MSG_DIR = os.environ['MSG_DIR'],
                  GRID_DIR=os.path.join('..','..','data','MSG','specgrid'),
                  PASS_DIR=os.path.join('..','..','data','MSG','passbands'))
    post.sort_eigenfunctions()
    data_dicts = post.evaluate_magnitudes()
    data_dict = post.calculate_duals(max_cond=max_cond)
    post.calculate_transfer(plot=plot, use_delta_L=use_delta_L, N_om=3000)
