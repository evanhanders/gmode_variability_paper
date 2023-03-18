import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mesa_reader as mr

from palettable.colorbrewer.sequential import RdPu_7
cmap = RdPu_7.mpl_colors[1:]

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

mesa_LOG = '../../MESA/15msol_ZLMC/LOGS/profile47.data'
p = mr.MesaData(mesa_LOG)
Lstar = p.header_data['photosphere_L']*p.header_data['lsun']
print('stellar lum: {:.2e}'.format(Lstar))

rv = '1.25'
hz_to_invday = 24*60*60

#col, row
fig = plt.figure(figsize=(7.5, 5))
ax1_1 = fig.add_axes([0, 0.71, 0.45, 0.22])
ax1_2 = fig.add_axes([0, 0.49, 0.45, 0.22])
ax1_3 = fig.add_axes([0, 0.27, 0.45, 0.22])
ax1_4 = fig.add_axes([0, 0.05, 0.45, 0.22])
ax2_1 = fig.add_axes([0.55, 0.71, 0.45, 0.22])
ax2_2 = fig.add_axes([0.55, 0.49, 0.45, 0.22])
ax2_3 = fig.add_axes([0.55, 0.27, 0.45, 0.22])
ax2_4 = fig.add_axes([0.55, 0.05, 0.45, 0.22])
cax = fig.add_axes([0.25, 0.97, 0.50, 0.03])

bounds = [100, 200, 400, 800, 2000, 4000, 6000]
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=6)
listed_cmap = mpl.colors.ListedColormap(cmap)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=listed_cmap)

data_dir='../../data/dedalus/wave_fluxes/zams_15Msol_LMC'
subdirs = ['re00300', 're00600', 're01200', 're03200', 're06600', 're10000']
re_vals = [263, 574, 1230, 3250, 6640, 9970]

for i, sdir, re in zip(range(len(re_vals)), subdirs, re_vals):
    with h5py.File('{}/{}/wave_luminosities.h5'.format(data_dir, sdir), 'r') as f:
        freqs = f['cgs_freqs'][()] * hz_to_invday
        ells  = f['ells'][()].ravel()
        lum = f['cgs_wave_luminosity(r={})'.format(rv)][0,:] / Lstar

    for ell, ax in zip([1, 2, 3, 4], [ax1_1, ax1_2, ax1_3, ax1_4]):
        ax.loglog(freqs,      np.abs(lum[:, ells == ell]),           color=cmap[i], zorder=i+1, lw=0.5)
        if i == len(re_vals) - 1:
            kh = np.sqrt(ell*(ell+1))
            ax.loglog(freqs, (1/Lstar)*2.33e-11*(freqs/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$', zorder=7)
            ax.set_ylim(1e-29, 1e-5)
            ax.set_xlim(1e-2, 2e1)
            if ell == 1:
                ax.text(0.3, 0.92, r'$f^{-6.5}$', rotation=0, transform=ax.transAxes, ha='left', va='center')
            ax.text(0.99, 0.97, r'$\ell = {{{}}}$'.format(ell), ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel(r'$f$ (d$^{-1}$)')
            ax.set_ylabel(r'$|L_{w}|/L_{*}$')

    for freq, ax in zip([0.2, 0.4, 0.8, 1.6], [ax2_1, ax2_2, ax2_3, ax2_4]):
        ax.loglog(ells, np.abs(lum[freqs > freq, :][0,:]),                 color=cmap[i], zorder=i+1, lw=1)
        if i == len(re_vals) - 1:
            kh = np.sqrt(ells*(ells+1))
            ax.loglog(ells, (1/Lstar)*2.33e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$', zorder=7)
            ax.set_xlim(1, 100)
            ax.set_ylim(1e-29, 1e-5)
            if freq == 0.2:
                ax.text(0.025, 0.90, r'$k_h^4=[\ell(\ell+1)]^2$', rotation=0, transform=ax.transAxes, ha='left', va='center')
            ax.text(0.01, 0.02, r'$f = {{{}}}$ (d$^{{-1}}$)'.format(freq), ha='left', va='bottom', transform=ax.transAxes)
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(r'$|L_{w}|/L_{*}$')


for ax in [ax1_1, ax1_2, ax1_3, ax2_1, ax2_2, ax2_3]:
    ax.set_xticklabels(())
    ax.tick_params(axis="x",direction="in", which='both')

for i, Re in enumerate([263, 574, 1230, 3250, 6640, 9970]):
    if Re >= 1500:
        color='lightgrey'
    else:
        color='k'
    cax.text((1/12)+i/6, 0.4, Re, ha='center', va='center', transform=cax.transAxes, color=color)

cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cax.text(-0.02, 0.5, 'Re', ha='right', va='center', transform=cax.transAxes)
cb.set_ticks(())


plt.savefig('fig02_turbulence_waveflux_variation.png', dpi=300, bbox_inches='tight')
plt.savefig('fig02_turbulence_waveflux_variation.pdf', dpi=300, bbox_inches='tight')
