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

def red_noise(nu, alpha0, nu_char, gamma=2):
    return alpha0/(1 + (nu/nu_char)**gamma)


star_dirs = ['03msol_Zsolar', '15msol_ZLMC', '40msol_Zsolar']
Lmax = [15, 15, 15]
alpha0 = [5.5e-3, 6e-2, 1.6e-1]
alpha_latex = [ r'$5.5 \times 10^{-3}$ $\mu$mag',\
                r'$6 \times 10^{-2}$ $\mu$mag',\
                r'$0.16$ $\mu$mag']
nu_char = [3e-1, 0.22, 0.13]
gamma  = [4.5, 3.9, 4.3]
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()]*24*60*60


fig = plt.figure(figsize=(7.5, 2.5))
ax1 = fig.add_axes([0.050, 0.025, 0.275, 0.95])
ax2 = fig.add_axes([0.375, 0.025, 0.275, 0.95])
ax3 = fig.add_axes([0.700, 0.025, 0.275, 0.95])
ax1.text(0.98, 0.98, '3 $M_{\odot}$', ha='right', va='top', transform=ax1.transAxes)
ax2.text(0.98, 0.98, '15 $M_{\odot}$', ha='right', va='top', transform=ax2.transAxes)
ax3.text(0.98, 0.98, '40 $M_{\odot}$', ha='right', va='top', transform=ax3.transAxes)
axs = [ax1, ax2, ax3]

for i, sdir in enumerate(star_dirs):
    plt.axes(axs[i])
    magnitude_sum = out_f['{}_magnitude_sum'.format(sdir)][()]

    alpha_lbl = r'$\alpha_0 = $' + alpha_latex[i]
    nu_lbl    = r'$\nu_{\rm char} = $' + '{:.2f}'.format(nu_char[i]) + ' d$^{-1}$'
    gamma_lbl = r'$\gamma = $' + '{:.1f}'.format(gamma[i])
    label     = '{}\n{}\n{}'.format(alpha_lbl, nu_lbl, gamma_lbl)
    plt.loglog(freqs, magnitude_sum, label=sdir, c='k')
    plt.loglog(freqs, red_noise(freqs, alpha0[i], nu_char[i], gamma=gamma[i]), label=label, c='orange')
    if i == 0: #3
        plt.ylim(1e-5, 8e-2)
        plt.xlim(5e-2, 5e0)
    if i == 1: #15
        plt.ylim(1e-4, 1)
        plt.xlim(3e-2, 2e0)
    if i == 2: #40
        plt.ylim(1e-3, 2e0)
        plt.xlim(2e-2, 2e0)
    axs[i].text(0.01, 0.99, label, ha='left', va='top', transform=axs[i].transAxes)

#    plt.legend()
    plt.xlabel('frequency (d$^{-1}$)')
    if i == 0:
        plt.ylabel(r'$\Delta m$ ($\mu$mag)')
fig.savefig('fig10_rednoise_fit.png', bbox_inches='tight', dpi=300)
fig.savefig('fig10_rednoise_fit.pdf', bbox_inches='tight')

out_f.close()
