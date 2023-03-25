import os
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d

import mesa_reader as mr
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

wave_lum = lambda f, ell: (2.33e-11)*f**(-6.5)*np.sqrt(ell*(ell+1))**4 #f in Hz.
fudge = 1

root_dir = '../../data/'
output_file = '{}/dedalus/surface_signals/wave_propagation_power_spectra.h5'.format(root_dir)
with h5py.File('../../dedalus/zams_15Msol_LMC/wave_generation/re10000/star/star_512+192_bounds0-2L_Re3.00e+04_de1.5_cutoff1.0e-10.h5', 'r') as lumf:
    s_nd = lumf['s_nd'][()]

with h5py.File('../../dedalus/zams_15Msol_LMC/wave_propagation/nr256/star/star_256+192+64_bounds0-0.93R_Re1.00e+04_de1.5_cutoff1.0e-10.h5', 'r') as starf:
    s_nd = starf['s_nd'][()]
    L_nd = starf['L_nd'][()]
    m_nd = starf['m_nd'][()]
    tau_nd = starf['tau_nd'][()]
    energy_nd = L_nd**2 * m_nd / (tau_nd**2)
    lum_nd = energy_nd/tau_nd

wave_lums_data = dict()
with h5py.File('../../data/dedalus/wave_fluxes/zams_15Msol_LMC/re03200/wave_luminosities.h5', 'r') as lum_file:
    radius_str = '1.25'
    wave_lums_data['freqs'] = lum_file['cgs_freqs'][()]
    wave_lums_data['ells'] = lum_file['ells'][()].ravel()
    wave_lums_data['lum'] = lum_file['cgs_wave_luminosity(r={})'.format(radius_str)][0,:]


for use_fit in [False, True]:
    with h5py.File(output_file, 'r') as out_f:
        freqs_hz = out_f['freqs'][()] / tau_nd  #Hz
        freqs = freqs_hz * (24 * 60 * 60) #invday
        ells = out_f['ells'][()].ravel()
        s1 = np.sqrt(out_f['shell(s1_S2,r=R)'][0,:,:])*s_nd

        fig = plt.figure(figsize=(7.5, 4)) 
        ax1 = fig.add_axes([0.04, 0.50, 0.32, 0.45])
        ax2 = fig.add_axes([0.36, 0.50, 0.32, 0.45])
        ax3 = fig.add_axes([0.68, 0.50, 0.32, 0.45])
        ax4 = fig.add_axes([0.04, 0.05, 0.32, 0.45])
        ax5 = fig.add_axes([0.36, 0.05, 0.32, 0.45])
        ax6 = fig.add_axes([0.68, 0.05, 0.32, 0.45])
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        cmap = mpl.cm.plasma
        Lmax = 6

        for j in range(1,Lmax+1):
            axs[j-1].loglog(freqs, s1[:,ells==j], color='k', lw=1, label='simulation')

            with h5py.File('../../data/dedalus/transfer/transfer_ell{:03d}_eigenvalues.h5'.format(j), 'r') as ef:
                #transfer dimensionality is (in simulation units) entropy / sqrt(wave luminosity).
                transfer_func_root_lum = ef['transfer_root_lum'][()] * s_nd/np.sqrt(lum_nd)
                transfer_freq_hz = ef['om'][()]/(2*np.pi) / tau_nd #Hz
                transfer_freq = transfer_freq_hz * (24 * 60 * 60) #invday
                transfer_interp = lambda f: 10**interp1d(np.log10(transfer_freq_hz), np.log10(transfer_func_root_lum), bounds_error=False, fill_value=-1e99)(np.log10(f))

            if use_fit:
                surface_s1_amplitude = fudge*np.abs(transfer_interp(transfer_freq_hz))*np.sqrt(wave_lum(transfer_freq_hz, j))
                axs[j-1].loglog(transfer_freq, surface_s1_amplitude, c='orange', label='transfer solution (Eqn. 74 $L_w$)', lw=0.5)
            else:
                log_wave_lum = interp1d(np.log10(wave_lums_data['freqs']), np.log10(wave_lums_data['lum'][:,j == wave_lums_data['ells']].ravel()))
                data_wave_lum = lambda f: 10**(log_wave_lum(np.log10(f)))
                surface_s1_amplitude = fudge*np.abs(transfer_interp(transfer_freq_hz))*np.sqrt(data_wave_lum(transfer_freq_hz))
                axs[j-1].loglog(transfer_freq, surface_s1_amplitude, c='orange', label='transfer solution (measured $L_w$)', lw=0.5)
            axs[j-1].text(0.96, 0.92, r'$\ell =$ ' + '{}'.format(j), transform=axs[j-1].transAxes, ha='right', va='center')

        for ax in axs:
            ax.set_xlim(7e-2,9e0)
            ax.set_ylim(1e-6, 1e2)
            ax.set_yticks((1e-5, 1e-3, 1e-1, 1e1))
        for ax in [ax4, ax5, ax6]:
            ax.set_xlabel('frequency (d$^{-1}$)')
        for ax in [ax1, ax4]:
            ax.set_ylabel('$|s_1|$ (erg g$^{-1}$ K$^{-1}$)')
        for ax in [ax1, ax2, ax3]:
            #        ax.set_ylim(2e-5,3e-1)
            ax.set_xticklabels(())
        for ax in [ax2, ax3, ax5, ax6]:
            ax.set_yticklabels(())
            ax.tick_params(axis="y",direction="in", which='both')

        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis="x", direction="in", which='both')


        ax1.legend(loc='lower left', frameon=False, borderaxespad=0.1, handletextpad=0.2, fontsize=8)
        if use_fit:
            plt.savefig('fig07_wavepropagation_transferVerification_fit.png', bbox_inches='tight', dpi=300)
            plt.savefig('fig07_wavepropagation_transferVerification_fit.pdf', bbox_inches='tight', dpi=300)
        else:
            plt.savefig('fig06_wavepropagation_transferVerification_raw.png', bbox_inches='tight', dpi=300)
            plt.savefig('fig06_wavepropagation_transferVerification_raw.pdf', bbox_inches='tight', dpi=300)
        plt.clf()
