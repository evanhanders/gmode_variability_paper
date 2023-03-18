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
    file_dict['Lconv_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*Lconv)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['rho_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*rho)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['u_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*(Lconv/(4*np.pi*r**2*rho))**(1/3))[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['cs_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*cs)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['Ma_cz'] = eval('u_cz/cs_cz', file_dict)
    file_dict['f_cz'] = eval('u_cz/Rcore', file_dict)
    print('Ma {:.3e}, f {:.3e}'.format(file_dict['Ma_cz'].cgs, file_dict['f_cz'].cgs))
    return file_dict

rv = '1.25'
hz_to_invday = 24*60*60



from palettable.colorbrewer.sequential import Oranges_7_r 

#with h5py.File('twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
#    freqs_msol15 = f['cgs_freqs'][()]*hz_to_invday
#    ells_msol15  = f['ells'][()].ravel()
##    power = f['vel_power(r=1.1)'][0,:]
#    lum_msol15 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]
#    msol15_file_dict = read_mesa('gyre/15msol/LOGS/profile47.data')
#
#    fitfunc = ((freqs_msol15/hz_to_invday)**(-15/2))[:,None] * np.sqrt(ells_msol15*(ells_msol15+1))[None,:]
##    fitfunc = ((freqs_msol15/hz_to_invday)**(-13/2))[:,None] * np.sqrt(ells_msol15*(ells_msol15+1))[None,:]
#    good1 = (freqs_msol15 > 0.2)*(freqs_msol15 < 1)
#    good2 = ells_msol15 == 1
#    log_fitAmp = np.mean( np.log10(lum_msol15/fitfunc)[good1, good2] )
#    print('15 fitAmp: {:.3e}'.format(10**(log_fitAmp)))
#
#with h5py.File('other_stars/msol40_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
#    freqs_msol40 = f['cgs_freqs'][()]*hz_to_invday
#    ells_msol40  = f['ells'][()].ravel()
#    lum_msol40 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]
#    msol40_file_dict = read_mesa('gyre/40msol/LOGS/profile53.data')
#
#    fitfunc = ((freqs_msol40/hz_to_invday)**(-13/2))[:,None] * np.sqrt(ells_msol40*(ells_msol40+1))[None,:]
#    good1 = (freqs_msol40 > 0.1)*(freqs_msol40 < 0.7)
#    good2 = ells_msol40 == 1
#    log_fitAmp = np.mean( np.log10(lum_msol40/fitfunc)[good1, good2] )
#    print('40 fitAmp: {:.3e}'.format(10**(log_fitAmp)))
#
#
#
#with h5py.File('other_stars/msol3_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
#    freqs_msol3 = f['cgs_freqs'][()]*hz_to_invday
#    ells_msol3  = f['ells'][()].ravel()
#    lum_msol3 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]
#    msol3_file_dict = read_mesa('gyre/3msol/LOGS/profile43.data')
#
#    fitfunc = ((freqs_msol3/hz_to_invday)**(-13/2))[:,None] * np.sqrt(ells_msol3*(ells_msol3+1))[None,:]
#    good1 = (freqs_msol3 > 0.2)*(freqs_msol3 < 1.5)
#    good2 = ells_msol3 == 1
#    log_fitAmp = np.mean( np.log10(lum_msol3/fitfunc)[good1, good2] )
#    print('3 fitAmp: {:.3e}'.format(10**(log_fitAmp)))

#col, row
fig = plt.figure(figsize=(7.5, 4.5))
ax1_1 = fig.add_axes([0.025, 0.75, 0.4, 0.2])
ax1_2 = fig.add_axes([0.025, 0.55, 0.4, 0.2])
ax1_3 = fig.add_axes([0.025, 0.20, 0.4, 0.2])
ax1_4 = fig.add_axes([0.025, 0.00, 0.4, 0.2])
ax2_1 = fig.add_axes([0.575, 0.75, 0.4, 0.2])
ax2_2 = fig.add_axes([0.575, 0.55, 0.4, 0.2])
ax2_3 = fig.add_axes([0.575, 0.20, 0.4, 0.2])
ax2_4 = fig.add_axes([0.575, 0.00, 0.4, 0.2])
axleft = [ax1_1, ax1_2, ax1_3, ax1_4]
axright = [ax2_1, ax2_2, ax2_3, ax2_4]
axs = axleft + axright
axtop = [ax1_1, ax2_1]
axbot = [ax1_4, ax2_4]
#Fig ranges:
#40: 0.1 - 0.7 1/day
#15: 0.2 - 1 1/day
#3:  0.2 - 1.5 1/day
#all: ell = 1.

data_dir='../../data/dedalus/wave_fluxes/'
subdirs = ['zams_03Msol_solarZ/nr256', 'zams_15Msol_LMC/re03200', 'zams_40Msol_solarZ/nr256']
colors = [Oranges_7_r.mpl_colors[3], Oranges_7_r.mpl_colors[1], Oranges_7_r.mpl_colors[0]]
freq_ranges = [(0.2, 1.5), (0.2, 1), (0.1, 0.7)]
mesa_LOGs = [   '../../MESA/03msol_Zsolar/LOGS/profile43.data',
                '../../MESA/15msol_ZLMC/LOGS/profile47.data',
                '../../MESA/40msol_Zsolar/LOGS/profile53.data']

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
        log_fitAmp = np.mean( np.log10(lum/fitfunc)[good1, good2] )
        print('{} fitAmp: {:.3e}'.format(sdir, 10**(log_fitAmp)))



    for ell, axT, axB in zip([1,2], [ax1_1, ax1_2], [ax1_3, ax1_4]):
        kh = np.sqrt(ell*(ell+1))
        print('ell = {}'.format(ell))
        axT.loglog(freqs,  np.abs(lum[:, ells == ell]), color=color)
        axB.loglog(freqs,  np.abs(lum[:, ells == ell])/eval('Ma_cz*Lconv_cz', file_dict).value, color=color)
        if i == len(subdirs)-1:
            axB.fill_between(freqs, 4e-45*(freqs/hz_to_invday)**(-6.5)*kh**4, 4e-47*(freqs/hz_to_invday)**(-6.5)*kh**4, color='lightgrey')
            axB.loglog(freqs, 4e-45*(freqs/hz_to_invday)**(-6.5)*kh**4,lw=0.5, c='lightgrey')
            axB.loglog(freqs, 4e-46*(freqs/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(4 \times 10^{-46})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
            axB.loglog(freqs, 4e-47*(freqs/hz_to_invday)**(-6.5)*kh**4,lw=0.5, c='lightgrey')
            axT.text(0.99, 0.965, r'$\ell = {{{}}}$'.format(ell), ha='right', va='top', transform=axT.transAxes)
            axB.text(0.99, 0.965, r'$\ell = {{{}}}$'.format(ell), ha='right', va='top', transform=axB.transAxes)
            axT.set_ylim(1e10, 1e33)
            axB.set_ylim(1e-19, 1e-4)
            axT.set_ylabel(r'$L_w$ (erg$\,\,$s$^{-1}$)')
            axB.set_ylabel(r'$L_w / (\mathscr{M} L_{\rm conv})$')

    for freq, axT, axB in zip([0.4, 0.8], [ax2_1, ax2_2], [ax2_3, ax2_4]):
        kh = np.sqrt(ells*(ells+1))
        print('f = {}'.format(freq))
        axT.loglog(ells,  lum[ freqs  > freq, :][0,:],    color=color)
        axB.loglog(ells,  lum[ freqs  > freq, :][0,:]/eval('Ma_cz*Lconv_cz', file_dict).value,   color=color)
        axB.fill_between(ells, 4e-45*(freq/hz_to_invday)**(-6.5)*kh**4, 4e-47*(freq/hz_to_invday)**(-6.5)*kh**4, color='lightgrey')
        axB.loglog(ells, 4e-45*(freq/hz_to_invday)**(-6.5)*kh**4, lw=0.5, c='lightgrey', label=r'$(3\times10^{-15})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
        axB.loglog(ells, 4e-46*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
        axB.loglog(ells, 4e-47*(freq/hz_to_invday)**(-6.5)*kh**4, lw=0.5, c='lightgrey', label=r'$(3\times10^{-10})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
        axT.text(0.01, 0.98, r'$f = {{{}}}$ d$^{{-1}}$'.format(freq), ha='left', va='top', transform=axT.transAxes)
        axB.text(0.01, 0.98, r'$f = {{{}}}$ d$^{{-1}}$'.format(freq), ha='left', va='top', transform=axB.transAxes)
        axT.set_ylim(1e10, 1e34)
        axB.set_ylim(1e-19, 1e-4)
        axT.set_ylabel(r'$L_w$ (erg$\,\,$s$^{-1}$)')
        axB.set_ylabel(r'$L_w / (\mathscr{M} L_{\rm conv})$')


ax1_1.text(0.03, 0.98, r'$40 M_{\odot}$', c=Oranges_7_r.mpl_colors[0],  transform=ax1_1.transAxes, va='top', ha='left')
ax1_1.text(3e-2, 3e22, r'$15 M_{\odot}$', c=Oranges_7_r.mpl_colors[1], ha='left', va='center')
ax1_1.text(7e-2, 3e16, r'$3 M_{\odot}$', c=Oranges_7_r.mpl_colors[3], ha='left', va='center')

for ax in axleft:
    ax.set_xlabel('$f$ (d$^{-1}$)')
    ax.set_xlim(1e-2, 1e1)
for ax in axright:
    ax.set_xlabel('$\ell$')
    ax.set_xlim(1, 1e2)


for ax in [ax1_1, ax2_1, ax1_3, ax2_3]:
    ax.set_xticklabels(())
    ax.tick_params(axis="x",direction="in", which='both')
plt.savefig('fig03_mstar_waveflux.png', bbox_inches='tight', dpi=300)
plt.savefig('fig03_mstar_waveflux.pdf', bbox_inches='tight', dpi=300)
plt.clf()
