import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

from compstar.tools.mesa import DimensionalMesaReader, find_core_cz_radius
from astropy import units as u
from astropy import constants
import mesa_reader as mr

mesa_LOG = '../../MESA/15msol_ZLMC/LOGS/profile47.data'
p = mr.MesaData(mesa_LOG)
Lstar = p.header_data['photosphere_L']*p.header_data['lsun']
print('stellar lum: {:.2e}'.format(Lstar))




def read_mesa(mesa_file_path):
    dmr = DimensionalMesaReader(mesa_file_path)
    file_dict = dmr.structure
    file_dict['Lstar'] = file_dict['Luminosity'].max()
    file_dict['Rcore'] = find_core_cz_radius(mesa_file_path, dimensionless=False)
    r = file_dict['r']
    L = file_dict['Luminosity']
    Lconv = file_dict['Luminosity']*file_dict['conv_L_div_L']
    rho = file_dict['rho']
    cs = file_dict['csound']
    good = r.value < file_dict['Rcore'].value
    u = (L / (4 * np.pi * r**2 * rho))**(1/3)
    file_dict['t_conv'] = t_conv = np.sum((np.gradient(r)/u)[good])
    file_dict['Lconv_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*Lconv)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['rho_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*rho)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['u_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*(Lconv/(4*np.pi*r**2*rho))**(1/3))[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['cs_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*cs)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['Ma_cz'] = eval('u_cz/cs_cz', file_dict)
    file_dict['f_cz'] = eval('u_cz/Rcore', file_dict)
    print('Ma {:.3e}, f {:.3e}, t_conv {:.3e}'.format(file_dict['Ma_cz'].cgs, file_dict['f_cz'].cgs, t_conv.cgs))
    return file_dict

rv = '1.75'
hz_to_invday = 24*60*60



from palettable.colorbrewer.sequential import Blues_7_r 

#col, row
fig = plt.figure(figsize=(7.5, 4.5))
ax1_1 = fig.add_axes([0.025, 0.75, 0.4, 0.2])
ax1_2 = fig.add_axes([0.025, 0.55, 0.4, 0.2])
ax1_3 = fig.add_axes([0.025, 0.00, 0.4, 0.4])
ax2_1 = fig.add_axes([0.575, 0.75, 0.4, 0.2])
ax2_2 = fig.add_axes([0.575, 0.55, 0.4, 0.2])
ax2_3 = fig.add_axes([0.575, 0.00, 0.4, 0.4])
axleft = [ax1_1, ax1_2]
axright = [ax2_1, ax2_2]
axs = axleft + axright
axtop = [ax1_1, ax2_1]
axbot = [ax1_2, ax2_2]

data_dir='../../data/dedalus/wave_fluxes/'
subdirs = ['zams_15Msol_LMC/re03200', 'zams_15Msol_LMC/rotating']
colors = [Blues_7_r.mpl_colors[3], Blues_7_r.mpl_colors[0]]
freq_ranges = [(0.2, 1), (0.2, 1)]
mesa_LOGs = [   '../../MESA/15msol_ZLMC/LOGS/profile47.data',
                '../../MESA/15msol_ZLMC/LOGS/profile47.data']

for i, sdir in enumerate(subdirs):
    print(sdir)
    color = colors[i]
    mesa_LOG = mesa_LOGs[i]
    freq_range = freq_ranges[i]
    with h5py.File('{}/{}/wave_luminosities.h5'.format(data_dir, sdir), 'r') as f:
        freqs = f['cgs_freqs'][()]*hz_to_invday
        ells  = f['ells'][()].ravel()
        lum = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]
        file_dict = read_mesa(mesa_LOG)

        fitfunc = ((freqs/hz_to_invday)**(-13/2))[:,None] * np.sqrt(ells*(ells+1))[None,:]
        good1 = (freqs > freq_range[0])*(freqs < freq_range[1])
        good2 = ells == 1
        log_fitAmp = np.mean( np.log10(np.abs(lum)/fitfunc)[good1, good2] )
        print('{} / L_conv {:.3e}, fitAmp: {:.3e}'.format(sdir, file_dict['Lconv_cz'], 10**(log_fitAmp)))

    for ell, ax in zip([1,2], [ax1_1, ax1_2]):
        kh = np.sqrt(ell*(ell+1))
        print('ell = {}'.format(ell))
        ax.loglog(freqs,  np.abs(lum[:, ells == ell]), color=color)
        if i == len(subdirs)-1:
            ax.loglog(freqs, 10**log_fitAmp*(freqs/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$', zorder=7, lw=0.5)
            ax.text(0.99, 0.965, r'$\ell = {{{}}}$'.format(ell), ha='right', va='top', transform=ax.transAxes)
            ax.set_ylim(1e10, 1e33)
            ax.set_ylabel(r'$L_w$ (erg$\,\,$s$^{-1}$)')

    for freq, ax in zip([0.4, 0.8], [ax2_1, ax2_2]):
        kh = np.sqrt(ells*(ells+1))
        ax.loglog(ells, 10**log_fitAmp*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$', zorder=7, lw=0.5)
        print('f = {}'.format(freq))
        ax.loglog(ells,  lum[ freqs  > freq, :][0,:],    color=color)
        ax.text(0.01, 0.98, r'$f = {{{}}}$ d$^{{-1}}$'.format(freq), ha='left', va='top', transform=ax.transAxes)
        ax.set_ylim(1e10, 1e34)
        ax.set_ylabel(r'$L_w$ (erg$\,\,$s$^{-1}$)')

ax1_2.text(0.18, 0.50, r'Nonrotating', c=colors[0],  transform=ax1_2.transAxes, va='top', ha='left')
ax1_2.text(0.01, 0.75, r'Rotating', c=colors[1],  transform=ax1_2.transAxes, va='top', ha='left')

for ax in axleft:
    ax.set_xlabel('$f$ (d$^{-1}$)')
    ax.set_xlim(6e-2, 1e1)
for ax in axright:
    ax.set_xlabel('$\ell$')
    ax.set_xlim(1, 1e2)


for ax in [ax1_1, ax2_1, ax1_3, ax2_3]:
    ax.set_xticklabels(())
    ax.tick_params(axis="x",direction="in", which='both')


#Variability plots
def red_noise(nu, alpha0, nu_char, gamma=2):
    return alpha0/(1 + (nu/nu_char)**gamma)

t_conv = file_dict['t_conv'].value / (60 * 60 * 24) #d
Ro = 10 / t_conv
print('Ro', Ro)

output_file = '../../data/dedalus/predictions/magnitude_spectra.h5'
star_dirs = ['15msol_ZLMC', 'rot_15msol_ZLMC']
Lmax = [15, 15]
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()] * 24 * 60 * 60

alpha = 0.4e0 
alpha_latex = '0.4 $\mu$mag'
nu_char = 0.16
gamma = 3.9

for i, sdir in enumerate(star_dirs):
    magnitude_cube = out_f['{}_magnitude_cube'.format(sdir)][()]
    ell_list = np.arange(1, Lmax[i]+1)
    axl, axr = ax1_3, ax2_3

    axl.loglog(freqs, np.sqrt(np.sum(magnitude_cube[:Lmax[i],:]**2, axis=0)), c=colors[i])
    if i == 1:
        axr.loglog(freqs, np.sqrt(np.sum(magnitude_cube[:Lmax[i],:]**2, axis=0)), c='k')
        axr.loglog(freqs, red_noise(freqs, alpha, nu_char, gamma), c='orange')

    axl.set_xlim(6e-2, 1e1)
    axr.set_xlim(6e-2, 1e1)
    axl.set_ylim(1e-6, 3e2)

    axr.set_ylim(1e-6, 3e2)
    axl.set_xlabel('$f$ (d$^{-1}$)')
    axr.set_xlabel('$f$ (d$^{-1}$)')
    axl.set_ylabel('$\Delta m$ ($\mu$mag)')
    axr.set_ylabel('$\Delta m$ ($\mu$mag)')
#    axl.text(0.04, 0.93, r'$M = $ ' + '{}'.format(int(sdir.split('msol')[0])) + r'$M_{\odot}$', transform=axl.transAxes, ha='left', va='center')
#    axr.text(0.96, 0.93, r'$M = $ ' + '{}'.format(int(sdir.split('msol')[0])) + r'$M_{\odot}$', transform=axr.transAxes, ha='right', va='center')


alpha_lbl = r'$\alpha_0 = $' + alpha_latex
nu_lbl    = r'$\nu_{\rm char} = $' + '{:.2f}'.format(nu_char) + ' d$^{-1}$'
gamma_lbl = r'$\gamma = $' + '{:.1f}'.format(gamma)
label     = '{}\n{}\n{}'.format(alpha_lbl, nu_lbl, gamma_lbl)
ax2_3.text(0.01, 0.99, label, ha='left', va='top', transform=ax2_3.transAxes)





plt.savefig('fig13_rot_waveflux.png', bbox_inches='tight', dpi=300)
plt.savefig('fig13_rot_waveflux.pdf', bbox_inches='tight', dpi=300)
plt.clf()
