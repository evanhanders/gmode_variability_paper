"""
Calculate transfer function to get surface response of convective forcing.
Outputs a function which, when multiplied by sqrt(wave flux), gives you the surface response.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


#Calculate transfer functions
output_file = '../../data/dedalus/predictions/magnitude_spectra.h5'
star_dirs = ['03msol_Zsolar', '15msol_ZLMC', '40msol_Zsolar']
Lmax = [3,3,3]
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()] * 24 * 60 * 60 #invday

fig = plt.figure(figsize=(7.5, 5))
ax1_1 = fig.add_axes([0.05, 0.66, 0.28, 0.33])
ax2_1 = fig.add_axes([0.385, 0.66, 0.28, 0.33])
ax3_1 = fig.add_axes([0.72, 0.66, 0.28, 0.33])
ax1_2 = fig.add_axes([0.05, 0.33, 0.28, 0.33])
ax2_2 = fig.add_axes([0.385, 0.33, 0.28, 0.33])
ax3_2 = fig.add_axes([0.72, 0.33, 0.28, 0.33])
ax1_3 = fig.add_axes([0.05, 0.00, 0.28, 0.33])
ax2_3 = fig.add_axes([0.385, 0.00, 0.28, 0.33])
ax3_3 = fig.add_axes([0.72, 0.00, 0.28, 0.33])
#axs = [ax1, ax2, ax3, ax4, ax5, ax6]
ax_rows = [[ax1_1, ax2_1, ax3_1], [ax1_2, ax2_2, ax3_2], [ax1_3, ax2_3, ax3_3]]


for i, sdir in enumerate(star_dirs):
    transfer = out_f['{}_transfer_cube'.format(sdir)][()]

    for j in range(Lmax[i]):
        ax = ax_rows[j][i]
        ax.loglog(freqs, transfer[j,:], label='star', c='k', lw=0.75)
        ax.text(0.03, 0.93, r'$M = $ ' + '{}'.format(int(sdir.split('msol')[0])) + r'$M_{\odot}$', transform=ax.transAxes, ha='left', va='center', c='k')
        ax.text(0.03, 0.85, r'$\ell = $ ' + '{}'.format(j+1), transform=ax.transAxes, ha='left', va='center', c='k')
        ax.set_xlim(5e-2, 1e1)
        ax.set_ylim(1e-21, 1e-9)
        ax.set_yticks((1e-20, 1e-17, 1e-14, 1e-11))

        if i == 0:
            ax.set_ylabel('Transfer')

#Plot dedalus transfer
with h5py.File('../../dedalus/zams_15Msol_LMC/wave_propagation/nr256/star/star_256+192+64_bounds0-0.93R_Re1.00e+04_de1.5_cutoff1.0e-10.h5', 'r') as starf:
    s_nd = starf['s_nd'][()]
    Cp = starf['Cp'][()]*s_nd
    L_nd = starf['L_nd'][()]
    m_nd = starf['m_nd'][()]
    tau_nd = starf['tau_nd'][()]
    energy_nd = L_nd**2 * m_nd / (tau_nd**2)
    lum_nd = energy_nd/tau_nd

for ell in range(3):
    with h5py.File('../../data/dedalus/transfer/transfer_ell{:03d}_eigenvalues.h5'.format(ell+1), 'r') as ef:
        #transfer dimensionality is (in simulation units) entropy / sqrt(wave luminosity).
        transfer_func_root_lum = 1e6*ef['transfer_root_lum'][()] * s_nd/np.sqrt(lum_nd) / Cp #turns sqrt(L) -> 1e6 * s/cp
        transfer_freq_hz = ef['om'][()]/(2*np.pi) / tau_nd #Hz
        transfer_freq = transfer_freq_hz * (24 * 60 * 60) #invday
    ax_rows[ell][1].loglog(transfer_freq, transfer_func_root_lum, label='WP simulation', c='orange', lw=0.75)
    ax_rows[ell][1].set_yticks((1e-20, 1e-17, 1e-14, 1e-11))


for row in ax_rows[:2]:
    for ax in row:
        ax.set_xticklabels(())
        ax.tick_params(axis='x', direction='in', which='both')
for ax in ax_rows[-1]:
    ax.set_xlabel('$f$ (d$^{-1}$)')

ax2_3.text(0.25, 0.1, 'star', ha='left',   va='center', transform=ax2_3.transAxes)
ax2_3.text(0.25, 0.6, 'WP sim', ha='left', va='center',  c='orange', transform=ax2_3.transAxes)


plt.savefig('fig08_transfer_functions.png', bbox_inches='tight', dpi=300)
plt.savefig('fig08_transfer_functions.pdf', bbox_inches='tight', dpi=300)

out_f.close()
