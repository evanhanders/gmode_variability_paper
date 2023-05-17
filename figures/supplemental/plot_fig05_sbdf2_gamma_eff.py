import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

from compstar.dedalus.evp_functions import SBDF2_gamma_eff as eff
from dedalus.tools.general import natural_sort


a_corr = 0.4

files = natural_sort(glob.glob('../../data/dedalus/eigenvalues/dual*ell*.h5'))
with h5py.File('../../dedalus/zams_15Msol_LMC/wave_propagation/nr256/star/star_256+192+64_bounds0-0.93R_Re1.00e+04_de1.5_cutoff1.0e-10.h5', 'r') as starf:
    tau_nd = starf['tau_nd'][()] #s / sim_time
    tau_nd /= (24 * 60 * 60) #convert tau_nd to d / sim_time
    s_nd = starf['s_nd'][()]

timestep = 0.0422 * tau_nd
print('timestep {}'.format(timestep))

fig = plt.figure(figsize=(7.5,4))
ax1 = fig.add_axes([0.04, 0.50, 0.32, 0.45])
ax2 = fig.add_axes([0.04, 0.05, 0.32, 0.45])
ax3 = fig.add_axes([0.36, 0.50, 0.32, 0.45])
ax4 = fig.add_axes([0.36, 0.05, 0.32, 0.45])
ax5 = fig.add_axes([0.68, 0.50, 0.32, 0.45])
ax6 = fig.add_axes([0.68, 0.05, 0.32, 0.45])
axs = [ax1, ax2, ax3, ax4, ax5, ax6]

elltags = ['ell001', 'ell002', 'ell003']
ax_pairs = [[ax1, ax2], [ax3, ax4], [ax5, ax6]]
for file in files:
    for i, t in enumerate(elltags):
        if t not in file:
            continue
        axtop, axbot = ax_pairs[i]
        print('plotting from {}'.format(file))
        out_dir = file.split('.h5')[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with h5py.File(file, 'r') as f:
            r = f['r'][()].ravel().real
            evalues = f['good_evalues'][()] / tau_nd

        with h5py.File('../../data/dedalus/transfer/transfer_{}_eigenvalues.h5'.format(t), 'r') as f:
            transfer = f['transfer_root_lum'][()]
            transfer_om = f['om'][()] / tau_nd
            raw_transfer = f['raw_transfer_root_lum'][()]
            raw_transfer_om = f['raw_om'][()] / tau_nd


        eff_evalues = []
        for ev in evalues:
            gamma_eff, omega_eff = eff(-ev.imag, np.abs(ev.real), timestep)
            eff_evalues.append(omega_eff - 1j*gamma_eff)
        eff_evalues = np.array(eff_evalues)

        plt.axes(axtop)
        plt.scatter(np.abs(evalues.real)/(2*np.pi), np.abs(evalues.imag), label='eigenvalues', c='orange')
        plt.scatter(np.abs(eff_evalues.real)/(2*np.pi), np.abs(eff_evalues.imag), label='SBDF2 eigenvalues', marker='x', c='black')
        plt.xlabel(r'frequency (d$^{-1}$)')
        plt.yscale('log')
        plt.xscale('log')

        plt.axes(axbot)
        plt.loglog(transfer_om/(2*np.pi), a_corr*transfer, lw=2, c='k')
        plt.loglog(raw_transfer_om/(2*np.pi), a_corr*raw_transfer, lw=1, c='orange')
        plt.xlabel(r'frequency (d$^{-1}$)')

        for ax in [axtop, axbot]:
            ax.set_xlim(7e-2,9e0) 
        ellv = int(t.split('ell')[-1])
        axtop.text(0.96, 0.93, r'$\ell = $' + '{}'.format(ellv), ha='right', va='center', transform=axtop.transAxes)
        axbot.text(0.04, 0.93, r'$\ell = $' + '{}'.format(ellv), ha='left', va='center', transform=axbot.transAxes)
    #        ax.set_xlim(0.08, 2)
    #    axtop.set_ylim(1e-3, 1e-2)
    #    axtop.set_xlim(3e-2, 3e-1)

ax1.set_ylabel(r'$\gamma$ (d$^{-1}$)')
ax2.set_ylabel(r'Transfer')
ax1.legend(loc='lower left', frameon=False, borderaxespad=0.1, handletextpad=0.2)
for ax in [ax1, ax3, ax5]:
    ax.set_ylim(2e-5,3e-1)
    ax.set_xticklabels(())
    ax.tick_params(axis="x",direction="in", which='both')
for ax in [ax2, ax4, ax6]:
    ax.set_ylim(1e0, 1e5)
for ax in axs[2:]:
    ax.set_yticklabels(())
    ax.tick_params(axis="y",direction="in", which='both')
plt.savefig('fig05_timestepper_gamma_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig05_timestepper_gamma_plot.png', dpi=300, bbox_inches='tight')


