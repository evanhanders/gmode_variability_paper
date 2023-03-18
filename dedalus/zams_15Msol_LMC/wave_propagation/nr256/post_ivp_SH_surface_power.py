"""
This script computes the wave flux in a d3 spherical simulation

Usage:
    post_ivp_SH_wave_flux.py [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_wave_shells]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 40]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --no_ft                             Do the base fourier transforms

    --no_minf                           If flagged, do one FT rather than STFT
"""
import re
import gc
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
from scipy import sparse
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor

import logging
logger = logging.getLogger(__name__)

from d3_stars.defaults import config
from d3_stars.post.power_spectrum_functions import HarmonicTimeToFreq 
from d3_stars.simulations.parser import name_star

args = docopt(__doc__)
res = re.compile('(.*),r=(.*)')

# Read in master output directory
root_dir    = './'
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

out_dir, star_file = name_star()
with h5py.File(star_file, 'r') as f:
    rB = f['r_B'][()]
    rS1 = f['r_S1'][()]
    rS2 = f['r_S2'][()]
    rhoB = np.exp(f['ln_rho0_B'][()])
    rhoS1 = np.exp(f['ln_rho0_S1'][()])
    rhoS2 = np.exp(f['ln_rho0_S2'][()])
    r = np.concatenate((rB.flatten(), rS1.flatten(), rS2.flatten()))
    rho = np.concatenate((rhoB.flatten(), rhoS1.flatten(), rhoS2.flatten()))
    rho_func = interp1d(r,rho)
    tau = f['tau_nd'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L_nd'][()]
    #Entropy units are erg/K/g
    s_c = f['s_nd'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2
    N2plateau_simunit = N2plateau * tau**2

transformer = HarmonicTimeToFreq(root_dir, data_dir, start_file=start_file, n_files=n_files)
if not args['--no_ft']:
    if args['--no_minf']:
        transformer.write_transforms()
    else:
        transformer.write_transforms(min_freq=np.sqrt(N2plateau_simunit)/(2*np.pi)/200)


radii = []
for f in transformer.fields:
    if res.match(f):
        radius_str = f.split('r=')[-1].split(')')[0]
        if radius_str not in radii:
            radii.append(radius_str)

fields = ['shell(enthalpy_fluc_S2,r=R)','shell(s1_S2,r=R)']


full_out_dir = 'FT_SH_transform_wave_shells'
with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r') as wf:
    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'w') as pf:
        if 'freqs_chunks' in wf.keys():
            freqs = wf['freqs_chunks'][()]
        else:
            freqs = wf['freqs'][()]
        ells = wf['ells'][()]
        for j, f in enumerate(fields):

            if len(freqs.shape) == 2:
                for i in range(freqs.shape[0]): #loop over STFT short transforms, get power of each.
                    logger.info('getting power of field {} on write {}/{}'.format(f, i+1, freqs.shape[0]))
                    
                    raw_freqs = freqs[i,:]
                    pos_freqs = raw_freqs[raw_freqs >= 0]
                    if i == 0 and j == 0:
                        pf['freqs'] = raw_freqs[raw_freqs >= 0]
                        pf['ells'] = ells[:,:,:,0] 
                        power = np.zeros((freqs.shape[0], pos_freqs.size, ells.size), dtype=np.float64)

                    transform = wf['{}_cft_chunks'.format(f)][i,:]
                    this_pow = (np.conj(transform)*transform).real
                    for fq in raw_freqs:
                        #collapse negative frequency power onto positive freq power
                        if fq < 0:
                            this_pow[raw_freqs == -fq] += this_pow[raw_freqs == fq]
                    this_pow = this_pow[raw_freqs >= 0,:]
                    power[i,:,:] = np.sum(this_pow, axis=-1) #sum over m
            else: 
                    logger.info('getting power of field {}'.format(f))
                    
                    raw_freqs = freqs
                    pos_freqs = raw_freqs[raw_freqs >= 0]
                    if j == 0:
                        pf['freqs'] = raw_freqs[raw_freqs >= 0]
                        pf['ells'] = ells[:,:,0] 
                        power = np.zeros((1, pos_freqs.size, ells.size), dtype=np.float64)

                    transform = wf['{}_cft'.format(f)][()]
                    this_pow = (np.conj(transform)*transform).real
                    for fq in raw_freqs:
                        #collapse negative frequency power onto positive freq power
                        if fq < 0:
                            this_pow[raw_freqs == -fq] += this_pow[raw_freqs == fq]
                    this_pow = this_pow[raw_freqs >= 0,:]
                    power[0,:,:] = np.sum(this_pow, axis=-1) #sum over m
            pf['{}'.format(f)] = power


#min and max power values to plot
pmin, pmax = 1e-30, 1e-10

#min and max freq values to plot
fmin, fmax = 2e-3, 1.5 

fig = plt.figure()
for ell in range(11):
    if ell == 0: continue
    print('plotting ell = {}'.format(ell))
    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as rf:

        freqs = rf['freqs'][()]
        power = rf['shell(s1_S2,r=R)'][:,:,ell]
        max_time_plots = 10
        skip = power.shape[0] // max_time_plots
        if skip == 0: skip = 1
        power = power[::skip, :]

        norm = mcolor.Normalize(vmin=0, vmax=power.shape[0]-1)
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)

        for i in range(power.shape[0]):
            plt.loglog(freqs, power[i], c=sm.to_rgba(i))

    cbar = plt.colorbar(sm)
    cbar.set_label('time index')

    plt.title('ell={}'.format(ell))
    plt.xlabel('freqs (sim units)')
    plt.ylabel(r'|enthalpy surface power|')

    plt.axvline(np.sqrt(N2plateau_simunit)/(2*np.pi))
    plt.ylim(pmin, pmax)
    fig.savefig('{}/freq_spectrum_ell{}.png'.format(full_out_dir, ell), dpi=300, bbox_inches='tight')
    plt.clf()

    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as rf:
        freqs = rf['freqs'][()]
        power = rf['shell(s1_S2,r=R)'][:,:,ell]
        time = np.arange(power.shape[0]) #TODO; improve time output.
    
    ff, tt = np.meshgrid(freqs.squeeze(), time.squeeze())
    cmesh = plt.pcolormesh(tt, ff, np.log10(power), rasterized=True, vmin=np.log10(pmin), vmax=np.log10(pmax))
    for t in time[::skip]:
        plt.axvline(t, c='k', lw=0.5)
    cbar = fig.colorbar(cmesh)
    cbar.set_label('log10 power')
    plt.title('ell={}'.format(ell))
    plt.yscale('log')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.ylim(fmin, fmax)
    fig.savefig('{}/evolution_freq_spectrum_ell{}.png'.format(full_out_dir, ell), dpi=300, bbox_inches='tight')
    plt.clf()


    
    

freqs_for_dfdell = [3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as rf:
    freqs = rf['freqs'][()]
    ells = rf['ells'][()].flatten()
    for f in freqs_for_dfdell:
        print('plotting f = {}'.format(f))
        f_ind = np.argmin(np.abs(freqs - f))

        power = rf['shell(s1_S2,r=R)'][:,f_ind,:]
        max_time_plots = 10
        skip = power.shape[0] // max_time_plots
        if skip == 0: skip = 1
        power = power[::skip, :]

        norm = mcolor.Normalize(vmin=0, vmax=power.shape[0]-1)
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)

        for i in range(power.shape[0]):
            plt.loglog(ells, power[i], c=sm.to_rgba(i))

        cbar = plt.colorbar(sm)
        cbar.set_label('time index')


        plt.title('f = {} 1/day'.format(f))
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'|enthalpy surface power|')
        plt.ylim(pmin, pmax)
        plt.xlim(1, ells.max())
        fig.savefig('{}/ell_spectrum_freq{}.png'.format(full_out_dir, f), dpi=300, bbox_inches='tight')
        plt.clf()
    
 

