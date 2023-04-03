#!/usr/bin/python3
import os
import shutil

dirs = ['star', 'eigenvalues', 'evp_matrices', '__pycache__']
ivp_dirs = ['profiles', 'scalars', 'slices', 'checkpoint', 'final_checkpoint', 'shells', 'wave_shells']
post_dirs = ['snapshots_equatorial', 'snapshots_meridional', 'traces', 'fluxes']
files = ['stratification.png', '.DS_Store']

for d in dirs + ivp_dirs + post_dirs:
    path = './{:s}/'.format(d)
    if os.path.exists(path):
        print('removing {}'.format(path))
        shutil.rmtree(path)

for f in files:
    if os.path.exists(f):
        print('removing {}'.format(f))
        os.remove(f)
