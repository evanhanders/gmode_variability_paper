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


star_dirs = ['3msol', '15msol', '40msol']
Lmax = [15, 15, 15]
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()] * 24 * 60 * 60

fig = plt.figure(figsize=(7.5, 7))
ax1 = fig.add_axes([0.00 , 0.60, 0.45, 0.30])#fig.add_subplot(1,2,1)
ax2 = fig.add_axes([0.575, 0.60, 0.45, 0.30])#fig.add_subplot(1,2,2)
ax3 = fig.add_axes([0.00 , 0.30, 0.45, 0.30])#fig.add_subplot(1,2,1)
ax4 = fig.add_axes([0.575, 0.30, 0.45, 0.30])#fig.add_subplot(1,2,2)
ax5 = fig.add_axes([0.00 , 0.00, 0.45, 0.30])#fig.add_subplot(1,2,1)
ax6 = fig.add_axes([0.575, 0.00, 0.45, 0.30])#fig.add_subplot(1,2,2)
cax1 = fig.add_axes([0.650, 0.975, 0.30, 0.025])
cax2 = fig.add_axes([0.075, 0.975, 0.30, 0.025])
axs = [ax1, ax2, ax3, ax4, ax5, ax6]
ax_pairs = [[ax1, ax2], [ax3, ax4], [ax5, ax6]]

even_cmap = mpl.cm.Purples_r
odds_cmap = mpl.cm.Greens_r
even_norm = mpl.colors.Normalize(vmin=2, vmax=24)
odds_norm = mpl.colors.Normalize(vmin=1, vmax=23)
even_sm = mpl.cm.ScalarMappable(norm=even_norm, cmap=even_cmap)
odds_sm = mpl.cm.ScalarMappable(norm=odds_norm, cmap=odds_cmap)
cb1 = plt.colorbar(even_sm, cax=cax1, orientation='horizontal', boundaries=[1, 3, 5, 7, 9, 11, 13, 15], ticks=[2, 4, 6, 8, 10, 12, 14, 16])
cb2 = plt.colorbar(odds_sm, cax=cax2, orientation='horizontal', ticks=[1, 3, 5, 7, 9, 11, 13, 15], boundaries=[0, 2, 4, 6, 8, 10, 12, 14, 16])
cb1.set_label(r'$\ell$ (even)')
cb2.set_label(r'$\ell$ (odd)')




for i, sdir in enumerate(star_dirs):
    magnitude_cube = out_f['{}_magnitude_cube'.format(sdir)][()]
    ell_list = np.arange(1, Lmax[i]+1)
    axl, axr = ax_pairs[i]

    for j in range(Lmax[i]):
        sum_mag = np.sqrt(np.sum(magnitude_cube[:Lmax[i]-j,:]**2, axis=0))
        if (j+1) % 2 == 0: 
            axl.loglog(freqs, magnitude_cube[j,:], color=even_sm.to_rgba(j+1), lw=0.5, zorder=2)
            axr.loglog(freqs, sum_mag, color=even_sm.to_rgba(Lmax[i]-j), lw=0.5, zorder=2)
        else:
            axl.loglog(freqs, magnitude_cube[j,:], color=odds_sm.to_rgba(j+1), lw=0.5, zorder=2)
            axr.loglog(freqs, sum_mag, color=odds_sm.to_rgba(Lmax[i]-j), lw=0.5, zorder=2)
    axr.loglog(freqs, np.sqrt(np.sum(magnitude_cube[:Lmax[i],:]**2, axis=0)), c='k', lw=0.25, zorder=2)
    axl.text(0.04, 0.93, r'$M = $ ' + '{}'.format(int(sdir.split('msol')[0])) + r'$M_{\odot}$', transform=axl.transAxes, ha='left', va='center')
    axr.text(0.96, 0.93, r'$M = $ ' + '{}'.format(int(sdir.split('msol')[0])) + r'$M_{\odot}$', transform=axr.transAxes, ha='right', va='center')

for ax in axs:
    ax.set_xlim(3e-2, 3e1)

for ax in [ax1, ax3, ax5]:
    ax.set_ylabel(r'$\Delta m_{\ell}\,(\mu\rm{mag})$')
for ax in [ax2, ax4, ax6]:
    ax.set_ylabel(r'$\sqrt{\sum_{i}^{\ell}\Delta m_{i}^2}\,(\mu\rm{mag})$')

for ax in [ax1, ax2]:
    ax.set_ylim(1e-7, 1e0)
    ax.set_xticklabels(())
    ax.tick_params(axis="x", direction="in", which='both', zorder=5, top=True, bottom=False)
for ax in [ax3, ax4]:
    ax.set_ylim(1e-6, 3e-1)
    ax.set_xticklabels(())
    ax.tick_params(axis="x", direction="in", which='both', zorder=5, top=True, bottom=False)
for ax in [ax5, ax6]:
    ax.set_ylim(3e-6, 3e0)
    ax.set_xlabel('frequency (d$^{-1}$)')
plt.savefig('fig09_ellsums.png', bbox_inches='tight', dpi=300)
plt.savefig('fig09_ellsums.pdf', bbox_inches='tight', dpi=300)
#    plt.clf()

out_f.close()
