"""
This script computes the wave flux in a d3 spherical simulation

Usage:
    post_ivp_SH_wave_flux.py [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_wave_shells]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --no_ft                             Do the base fourier transforms

    --no_minf                           If flagged, do one FT rather than STFT
"""
import re
from collections import OrderedDict

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
from scipy import sparse
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from compstar.defaults import config
from compstar.tools.power_spectrum_functions import HarmonicTimeToFreq 
from compstar.dedalus.parser import name_star

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
    rhoB = np.exp(f['ln_rho0_B'][()])
    rhoS1 = np.exp(f['ln_rho0_S1'][()])
    r = np.concatenate((rB.flatten(), rS1.flatten()))
    rho = np.concatenate((rhoB.flatten(), rhoS1.flatten()))
    rho_func = interp1d(r,rho)
    tau = f['tau_nd'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L_nd'][()]
    #Entropy units are erg/K/g
    s_c = f['s_nd'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2
    N2plateau_simunit = N2plateau * tau**2
    #P units: P = R * rho * T. R has units of entropy = [L^2 / tau^2 / T]. rho = [m / L^3]. T = [T].
    # so P = [L^2 / tau^2 / T] * [m / L^3] * [T] = [m / tau^2 / L] (matches wikipedia - good)
    #wave luminosity is u * enthalpy ~ r^2 * u * P ~ [L^2] * [L / tau] *  [m / tau^2 / L]
    # wave luminosity = [L^2 * m / tau^3]
    wave_luminosity_cgs_units = f['L_nd'][()]**2 * f['m_nd'][()] / f['tau_nd'][()]**3
    frequency_cgs_units = 1/f['tau_nd'][()]
    velocity_cgs_units = f['L_nd'][()] / f['tau_nd'][()]

# Create Plotter object, tell it which fields to plot
transformer = HarmonicTimeToFreq(root_dir, data_dir, start_file=start_file, n_files=n_files)
if not args['--no_ft']:
    if args['--no_minf']:
        transformer.write_transforms()
    else:
        transformer.write_transforms(min_freq=np.sqrt(N2plateau_simunit)/(2*np.pi)/200)


print('saving figures to {}'.format(transformer.out_dir))

radii = []
for f in transformer.fields:
    if res.match(f):
        radius_str = f.split('r=')[-1].split(')')[0]
        if radius_str not in radii:
            radii.append(radius_str)


###Calculate wave luminosities
with h5py.File('FT_SH_transform_wave_shells/transforms.h5', 'r') as FT_file:
    with h5py.File('FT_SH_transform_wave_shells/wave_luminosities.h5', 'w') as lum_file:

        for i, radius_str in enumerate(radii):
            if 'freqs_chunks' in FT_file.keys():
                freqs = FT_file['freqs_chunks'][()]
            else:
                freqs = FT_file['freqs'][()]

            if 'R' in radius_str:
                radius = float(radius_str.replace('R', ''))*r_outer
            else:
                radius = float(radius_str)
            print('using radius {}'.format(radius))

            for k in FT_file.keys():
                if 'r={}'.format(radius_str) in k:
                    if 'u_' in k:
                        print('ur key, {}, r={}'.format(k, radius_str))
                        if len(freqs.shape) == 1:
                            ur = FT_file[k][:,1,:] #freq, vec ind, ell, m
                        else:
                            ur = FT_file[k][:,:,1,:] #num FT, freq, vec ind, ell, m
                    if 'enthalpy_fluc_' in k:
                        print('enthalpy key, {}, r={}'.format(k, radius_str))
                        p = FT_file[k][()] #has rho in it. Is (Cp/R) * P.
            wave_luminosity_chunks = 4*np.pi*radius**2*(ur*np.conj(p)).real
            velocity_power_chunks = (ur*np.conj(ur)).real
            if len(freqs.shape) == 1:
                freqs = np.expand_dims(freqs, 0)
                wave_luminosity_chunks = np.expand_dims(wave_luminosity_chunks, 0)
                velocity_power_chunks = np.expand_dims(velocity_power_chunks, 0)
            true_shape = list(wave_luminosity_chunks.shape)
            true_shape[1] = np.sum(freqs[0,:] >= 0)
            freq_shape = freqs.shape
            true_wl_chunks = np.zeros(true_shape)
            true_upow_chunks = np.zeros(true_shape)
            # Collapse negative frequencies
            for j in range(freq_shape[0]):
                for f in np.unique(freqs[j]):
                    if f < 0:
                        wave_luminosity_chunks[j,freqs[j] == -f] += wave_luminosity_chunks[j,freqs[j] == f]
                        velocity_power_chunks[j,freqs[j] == -f] += velocity_power_chunks[j,freqs[j] == f]

                # Sum over m's.
                true_wl_chunks[j,:] = wave_luminosity_chunks[j,freqs[j] >= 0]
                true_upow_chunks[j,:] = velocity_power_chunks[j,freqs[j] >= 0]
            wl_chunks = np.sum(true_wl_chunks, axis=-1)
            upow_chunks = np.sum(true_upow_chunks, axis=-1)
            print('saving wave luminosity at r = {}'.format(radius_str))
            lum_file['wave_luminosity(r={})'.format(radius_str)] = wl_chunks
            lum_file['cgs_wave_luminosity(r={})'.format(radius_str)] = wave_luminosity_cgs_units*wl_chunks
            lum_file['vel_power(r={})'.format(radius_str)] = upow_chunks
            lum_file['cgs_vel_power(r={})'.format(radius_str)] = (velocity_cgs_units**2)*upow_chunks
            if i == 0:
                lum_file['freqs'] = freqs[0,freqs[0,:]>=0]
                lum_file['ells'] = FT_file['ells'][()]
                lum_file['cgs_freqs'] = frequency_cgs_units * freqs[0,freqs[0,:]>=0]
       
#Fit A f ^ alpha ell ^ beta
fit_freq_range = (3e-2, 1e-1)
fit_ell_range = (1, 4)
radius_str = radii[1]
fig = plt.figure()
possible_alphas = [-13/2,]
possible_betas = [3, 4]
fit_A = []
fit_alpha = []
fit_beta  = []
with h5py.File('FT_SH_transform_wave_shells/wave_luminosities.h5', 'r') as lum_file:
    freqs = lum_file['freqs'][()]
    good_freqs = (freqs >= fit_freq_range[0])*(freqs <= fit_freq_range[1])
    ells = lum_file['ells'][()].ravel()
    good_ells = (ells >= fit_ell_range[0])*(ells <= fit_ell_range[1])
    for i in range(lum_file['wave_luminosity(r={})'.format(radius_str)][()].shape[0]):
        wave_luminosity = np.abs(lum_file['wave_luminosity(r={})'.format(radius_str)][i,:,:])
        info = []
        error = []
        for j, alpha in enumerate(possible_alphas):
            for k, beta in enumerate(possible_betas):
                A = np.mean((wave_luminosity / freqs[:,None]**(alpha) / ells[None,:]**(beta))[good_freqs[:,None]*good_ells[None,:]])
                fit = A * freqs[:,None]**alpha * ells[None,:]**beta
                error.append(np.mean( np.abs(1 - (np.log10(fit) / np.log10(wave_luminosity))[good_freqs[:,None]*good_ells[None,:]])))
                info.append((A, alpha, beta))
        print(info, error)
        A, alpha, beta = info[np.argmin(error)]

        fit_A.append(A)
        fit_alpha.append(alpha)
        fit_beta.append(beta)
wave_luminosity_power = lambda f, ell: fit_A[-1]*f**(fit_alpha[-1])*ell**(fit_beta[-1])
wave_luminosity_str = r'{:.2e}'.format(fit_A[-1]) + r'$f^{'+'{:.1f}'.format(fit_alpha[-1])+'}\ell^{' + '{:.1f}'.format(fit_beta[-1]) + '}$'

print('fit_A', fit_A)
print('fit_A frac', np.array(fit_A[1:])/np.array(fit_A[:-1]))
print('fit_alpha', fit_alpha)
print('fit_beta', fit_beta)

#the nondim fit: A * (f/f_nd)**alpha * ell**beta
#becomes the dimensional fit: A * (wave_cgs/wave_nd) * (f_cgs/f_nd)**alpha * (f/f_cgs)**alpha * ell**beta
cgs_fit_A = wave_luminosity_cgs_units * (frequency_cgs_units)**(-fit_alpha[-1]) * fit_A[-1]
print('cgs fit A:', cgs_fit_A)

cgs_wave_luminosity_power = lambda f, ell: cgs_fit_A*f**(fit_alpha[-1])*ell**(fit_beta[-1])
cgs_wave_luminosity_str = r'{:.2e}'.format(cgs_fit_A) + r'$f^{'+'{:.1f}'.format(fit_alpha[-1])+'}\ell^{' + '{:.1f}'.format(fit_beta[-1]) + '}$'


#DIMENSIONLESS PLOTS
#plot vs f at given ell
for ell in range(11):
    if ell == 0: continue
    print('plotting ell = {}'.format(ell))
    with h5py.File('FT_SH_transform_wave_shells/wave_luminosities.h5', 'r') as lum_file:
        freqs = lum_file['freqs'][()]
        for i, radius_str in enumerate(radii):
            if not args['--no_minf']:
                wave_luminosity = np.abs(lum_file['wave_luminosity(r={})'.format(radius_str)][-1,:,ell])
            else:
                wave_luminosity = np.abs(lum_file['wave_luminosity(r={})'.format(radius_str)][0,:,ell])
            plt.loglog(freqs, wave_luminosity, label='r={}'.format(radius_str))
    plt.loglog(freqs, wave_luminosity_power(freqs, ell), c='k', label=wave_luminosity_str)
    plt.legend(loc='best')
    plt.title('ell={}'.format(ell))
    plt.xlabel('freqs (sim units)')
    plt.ylabel(r'|wave luminosity|')

    plt.axvline(np.sqrt(N2plateau_simunit)/(2*np.pi))
    plt.ylim(1e-33, 1e-12)
    fig.savefig('{}/freq_spectrum_ell{}.png'.format(transformer.out_dir, ell), dpi=300, bbox_inches='tight')
    plt.clf()
    
    
#plot vs ell at given f
freqs_for_dfdell = [3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
for f in freqs_for_dfdell:
    print('plotting f = {}'.format(f))
    with h5py.File('FT_SH_transform_wave_shells/wave_luminosities.h5', 'r') as lum_file:
        freqs = lum_file['freqs'][()]
        ells = lum_file['ells'][()].ravel()
        f_ind = np.argmin(np.abs(freqs - f))
        for i, radius_str in enumerate(radii):
            if not args['--no_minf']:
                wave_luminosity = np.abs(lum_file['wave_luminosity(r={})'.format(radius_str)][-1,f_ind,:])
            else:
                wave_luminosity = np.abs(lum_file['wave_luminosity(r={})'.format(radius_str)][0,f_ind,:])
            plt.loglog(ells, wave_luminosity, label='r={}'.format(radius_str))
        plt.loglog(ells, wave_luminosity_power(f, ells), c='k', label=wave_luminosity_str)
        plt.legend(loc='best')
        plt.title('f = {} (sim units)'.format(f))
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'|wave luminosity|')
        plt.ylim(1e-33, 1e-12)
        plt.xlim(1, ells.max())
        fig.savefig('{}/ell_spectrum_freq{}.png'.format(transformer.out_dir, f), dpi=300, bbox_inches='tight')
        plt.clf()

#CGS PLOTS
#plot vs f at given ell
for ell in range(11):
    if ell == 0: continue
    print('plotting ell = {}'.format(ell))
    with h5py.File('FT_SH_transform_wave_shells/wave_luminosities.h5', 'r') as lum_file:
        freqs = lum_file['cgs_freqs'][()]
        for i, radius_str in enumerate(radii):
            if not args['--no_minf']:
                wave_luminosity = np.abs(lum_file['cgs_wave_luminosity(r={})'.format(radius_str)][-1,:,ell])
            else:
                wave_luminosity = np.abs(lum_file['cgs_wave_luminosity(r={})'.format(radius_str)][0,:,ell])
            plt.loglog(freqs, wave_luminosity, label='r={}'.format(radius_str))
    plt.loglog(freqs, cgs_wave_luminosity_power(freqs, ell), c='k', label=cgs_wave_luminosity_str)
    plt.legend(loc='best')
    plt.title('ell={}'.format(ell))
    plt.xlabel('freqs (cgs)')
    plt.ylabel(r'|wave luminosity| (cgs)')

    plt.axvline(np.sqrt(N2plateau_simunit)/(2*np.pi)*frequency_cgs_units)
    plt.ylim(1e-33*wave_luminosity_cgs_units, 1e-12*wave_luminosity_cgs_units)
    fig.savefig('{}/cgs_freq_spectrum_ell{}.png'.format(transformer.out_dir, ell), dpi=300, bbox_inches='tight')
    plt.clf()
    
    
#plot vs ell at given f
cgs_freqs_for_dfdell = frequency_cgs_units*np.array(freqs_for_dfdell)
for f in cgs_freqs_for_dfdell:
    print('plotting f = {}'.format(f))
    with h5py.File('FT_SH_transform_wave_shells/wave_luminosities.h5', 'r') as lum_file:
        cgs_freqs = lum_file['cgs_freqs'][()]
        ells = lum_file['ells'][()].ravel()
        f_ind = np.argmin(np.abs(cgs_freqs - f))
        for i, radius_str in enumerate(radii):
            if not args['--no_minf']:
                wave_luminosity = np.abs(lum_file['cgs_wave_luminosity(r={})'.format(radius_str)][-1,f_ind,:])
            else:
                wave_luminosity = np.abs(lum_file['cgs_wave_luminosity(r={})'.format(radius_str)][0,f_ind,:])
            plt.loglog(ells, wave_luminosity, label='r={}'.format(radius_str))
        plt.loglog(ells, cgs_wave_luminosity_power(f, ells), c='k', label=cgs_wave_luminosity_str)
        plt.legend(loc='best')
        plt.title('f = {} (cgs)'.format(f))
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'|wave luminosity| (cgs)')
        plt.ylim(1e-33*wave_luminosity_cgs_units, 1e-12*wave_luminosity_cgs_units)
        plt.xlim(1, ells.max())
        fig.savefig('{}/cgs_ell_spectrum_freq{:.2e}.png'.format(transformer.out_dir, f), dpi=300, bbox_inches='tight')
        plt.clf()
    
    
