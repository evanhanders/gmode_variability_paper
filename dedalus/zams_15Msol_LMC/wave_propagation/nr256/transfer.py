"""
Calculate transfer function to get surface response of convective forcing.
Outputs a function which, when multiplied by sqrt(wave flux), gives you the surface response.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser

import compstar
from compstar.defaults import config
from compstar.dedalus.parser import name_star
from compstar.dedalus.star_builder import find_core_cz_radius
from compstar.dedalus.evp_functions import SBDF2_gamma_eff
from compstar.waves.transfer import calculate_refined_transfer

plot = False

#Grab relevant information about the simulation stratification.
out_dir, out_file = name_star()
with h5py.File(out_file, 'r') as f:
    L_nd = f['L_nd'][()]
    rs = []
    rhos = []
    chi_rads = []
    N2s = []
    gamma = f['gamma1'][()]
    for bk in ['B', 'S1', 'S2']:
        rs.append(f['r_{}'.format(bk)][()])
        rhos.append(np.exp(f['ln_rho0_{}'.format(bk)][()]))
        chi_rads.append(f['chi_rad_{}'.format(bk)][()])
        #N^2 = -g[2] * grad_s[2] / cp
        N2s.append(-f['g_{}'.format(bk)][2,:]*f['grad_s0_{}'.format(bk)][2,:]/f['Cp'][()])
    rs = np.concatenate(rs, axis=-1)
    rhos = np.concatenate(rhos, axis=-1)
    chi_rads = np.concatenate(chi_rads, axis=-1)
    N2s = np.concatenate(N2s, axis=-1)
rho = interpolate.interp1d(rs.flatten(), rhos.flatten())
chi_rad = interpolate.interp1d(rs.flatten(), chi_rads.flatten())
N2 = interpolate.interp1d(rs.flatten(), N2s.flatten())

if __name__ == '__main__':

    timestep = 0.073
    #timestep = 0.098


    # Generalized logic for getting forcing radius.
    package_path = Path(compstar.__file__).resolve().parent
    stock_path = package_path.joinpath('stock_models')
    if os.path.exists(config.star['path']):
        mesa_file_path = config.star['path']
    else:
        stock_file_path = stock_path.joinpath(config.star['path'])
        if os.path.exists(stock_file_path):
            mesa_file_path = str(stock_file_path)
        else:
            raise ValueError("Cannot find MESA profile file in {} or {}".format(config.star['path'], stock_file_path))
    core_cz_radius = find_core_cz_radius(mesa_file_path)
    min_forcing_radius = 0.97 * core_cz_radius / L_nd
    max_forcing_radius = 1.03 * core_cz_radius / L_nd
    N2_force_max = N2(max_forcing_radius)
    N2_max = N2s.max()
    N2_adjust = np.sqrt(N2_max/N2_force_max)
#    min_forcing_radius = 0.5 * core_cz_radius / L_nd
#    max_forcing_radius = 1.04 * core_cz_radius / L_nd


    #Calculate transfer functions
    Lmax = config.eigenvalue['Lmax']
    ell_list = np.arange(1, Lmax+1)
    eig_dir = 'eigenvalues'
    for ell in ell_list:
        print("ell = %i" % ell)

        #Read in eigenfunction values.
        #Require: eigenvalues, horizontal duals, transfer surface (s1), optical depths

        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
            velocity_duals = f['velocity_duals'][()]
            values = raw_values = f['good_evalues'][()]
            s1_amplitudes = f['s1_amplitudes'][()].squeeze()
            depths = f['depths'][()]
            discard = int(f['discard'][()]) #number of modes to discard
            tau_nd = f['tau_nd'][()]

            eff_evalues = []
            for ev in values:
                gamma_eff, omega_eff = SBDF2_gamma_eff(-ev.imag, np.abs(ev.real), timestep)
    #            gamma_eff, omega_eff = -ev.imag, np.abs(ev.real)
                if ev.real < 0:
                    eff_evalues.append(-omega_eff - 1j*gamma_eff)
                else:
                    eff_evalues.append(omega_eff - 1j*gamma_eff)
            values = np.array(eff_evalues)
            print('effective modes in Hz: {}'.format(values/tau_nd))


            rs = []
            for bk in ['B', 'S1', 'S2']:
                rs.append(f['r_{}'.format(bk)][()].flatten())
            r = np.concatenate(rs)
            smooth_oms = f['smooth_oms'][()]
            smooth_depths = f['smooth_depths'][()]
            depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')
        print(depths)

        #Construct frequency grid for evaluation
        om0 = values.real[-1]#np.min(np.abs(values.real))*1
        if discard == 0:
            om0 = values.real[-1]#np.min(np.abs(values.real))*1
        else:
            om0 = values.real[-discard]#np.min(np.abs(values.real))*1
#        discard = values.size-np.sum(depths < 1e-2)
        discard = 0#values.size-np.sum(depths < 1e-2)
        om1 = np.max(values.real)*1.25
        print(om0, om1)

        om = np.logspace(np.log10(om0), np.log10(om1), num=3000, endpoint=True)

        #Get forcing radius and dual basis evaluated there.
        r_range = np.linspace(min_forcing_radius, max_forcing_radius, num=200, endpoint=True)
#        r_range = np.array([min_forcing_radius, 1.0001*min_forcing_radius])
        dual_index = 1
        uh_dual_interp = interpolate.interp1d(r, velocity_duals[:,dual_index,:], axis=-1)(r_range) #m == 1 solve; recall there is some power in utheta, too

        #Calculate and store transfer function
        good_om, good_T = calculate_refined_transfer(om, values, uh_dual_interp, s1_amplitudes, r_range, ell, rho, chi_rad, N2_max, gamma, discard_num=discard, plot=plot)
#        good_T = good_T[depthfunc(good_om) <= 3] #filter only low-depth modes!
#        good_om = good_om[depthfunc(good_om) <= 3] #filter only low-depth modes!


        raw_om, raw_T = calculate_refined_transfer(om, raw_values, uh_dual_interp, s1_amplitudes, r_range, ell, rho, chi_rad, N2_max, gamma, discard_num=discard)
#        raw_T = raw_T[depthfunc(raw_om) <= 3] #filter only low-depth modes!
#        raw_om = raw_om[depthfunc(raw_om) <= 3] #filter only low-depth modes!

        print('N2 adjust:', N2_adjust)
        good_T *= np.sqrt(N2_adjust)
        raw_T  *= np.sqrt(N2_adjust)

        if plot:
            plt.figure()
            for om in raw_values:
                plt.axvline(om.real/(2*np.pi), lw=0.33)
            plt.loglog(good_om/(2*np.pi), good_T, c='k')
            plt.axvline(om0/2/np.pi)
            plt.ylim(1e-2, 1e4)
            plt.title('ell = {}'.format(ell))
            plt.show()


        with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'w') as f:
            f['om'] = good_om
            f['transfer_root_lum'] = good_T 
            f['raw_om'] = raw_om
            f['raw_transfer_root_lum'] = raw_T 



