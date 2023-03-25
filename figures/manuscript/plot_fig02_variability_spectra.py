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
import pandas as pd

from read_mist_models import EEP
import mesa_reader as mr
from matplotlib.patches import ConnectionPatch
from palettable.colorbrewer.qualitative import Dark2_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


bowman_data_dir = '../../data/observation_tables/'
file1 = bowman_data_dir + 'bowman2020_tablea1.txt'
file2 = bowman_data_dir + 'bowman2020_tablea2.txt'
df1 = pd.read_csv(file1,
                 sep="|",
                 skiprows=6,
                 skipfooter = 1,
                 usecols=[2,4,5,6,7],
                 names=['TIC','LogT','LogL','vsini','vmacro'])


df2 = pd.read_csv(file2,
                 sep="|",
                 skiprows=6,
                 skipfooter = 1,
                 usecols=[2,3,5,7,9],
                 names=['TIC','alpha0','nuchar','gamma', 'Cw'])

nuchar = df2['nuchar']
alpha0 = df2['alpha0']
gamma  = df2['gamma']
Cw     = df2['Cw']
vsini  = df1['vsini']
vmacro = df1['vmacro']
LogT   = df1['LogT']
LogL   = df1['LogL']

nuchar = np.array([float(item) for item in nuchar])
alpha0 = np.array([float(item) for item in alpha0])
gamma  = np.array([float(item) for item in gamma])
Cw     = np.array([float(item) for item in Cw])
vsini  = np.array([float(item) for item in vsini])
LogT   = np.array([float(item) for item in LogT])
LogL   = np.array([float(item) for item in LogL])


#NOTE to compare to MESA model we need to calculate 'ell' not bolometric luminosity.
# from matteo:
#ell_sun=(5777)**4.0/(274*100)  
#ell = (10**logt)**4.0/(10**logg)
#ell=np.log10(ell/ell_sun)  
sLum_sun=(5777)**4.0/(274*100)  

output_file = '../../data/dedalus/predictions/magnitude_spectra.h5'
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()]


#Calculate transfer functions
eig_dir = 'gyre_output'


plt.figure()
star_dirs = ['03msol_Zsolar', '40msol_Zsolar', '15msol_ZLMC']
star_keys = star_dirs
Lmax = [15,15,15]

signals = []
specLums = []
logTeffs = []
for i, skey, sdir in zip(range(len(star_dirs)), star_keys, star_dirs):
    #MESA history for getting ell.
    mesa_history = '../../MESA/{}/LOGS/history.data'.format(sdir)
    history = mr.MesaData(mesa_history)
    mn = history.model_number
    log_g = history.log_g
    log_Teff = history.log_Teff

    sLum = (10**log_Teff)**4.0/(10**log_g)
    sLum=np.log10(sLum/sLum_sun)
    specLums.append(sLum[-1])
    logTeffs.append(log_Teff[-1])

    signals.append(np.sum(out_f['{}_magnitude_cube'.format(skey)][:Lmax[i],:], axis=0))

    #only get stars that are within 0.2 log10T and log10L on HRD.
    if i == 2:
        good =  (LogL > specLums[-1] - 0.2)*(LogL < specLums[-1] + 0.2)\
               *(LogT > logTeffs[-1] - 0.2)*(LogT < logTeffs[-1] + 0.2)

        nuchar = nuchar[good]
        alpha0 = alpha0[good]
        gamma  = gamma[good]
        Cw     = Cw[good]
        vsini  = vsini[good]
        LogT   = LogT[good]
        LogL   = LogL[good]

freqs *= 60*60*24 #1/s -> 1/day

#### MAKE PAPER FIGURE
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_axes((0.00,  0, 0.27, 1))
ax2 = fig.add_axes((0.33,  0, 0.27, 1))
ax3 = fig.add_axes((0.73,  0, 0.27, 1))


plt.subplots_adjust(hspace=0.5, wspace=0.7)


#Reads in non-rotating mist models.
#Data: https://waps.cfa.harvard.edu/MIST/model_grids.html (v/vcrit = 0; [Fe/H] = 0) EEP tracks.
#Read script: https://github.com/jieunchoi/MIST_codes/blob/master/scripts/read_mist_models.py
zamsT = []
zamsL = []
for mass_str in ['00300', '00500', '01000', '01500', '02000', '04000']:
    model = EEP('../../data/mist_models/{}M.track.eep'.format(mass_str), verbose=True)
    mass = model.minit
    center_h1 = model.eeps['center_h1']
    good = (center_h1 < center_h1[0]*0.999)*(center_h1 > 0.02)
    log_Teff = model.eeps['log_Teff']
    log_g = model.eeps['log_g']
    ell = (10**log_Teff)**4.0/(10**log_g)
    ell = np.log10(ell/sLum_sun)
    zamsT.append(log_Teff[good][0])
    zamsL.append(ell[good][0])
    ax3.plot(log_Teff[good], ell[good], label=mass, c='grey', lw=1, zorder=0)
#    ax3.text(0.01+log_Teff[good][0], -0.1+ell[good][0], '{:d}'.format(int(mass))+r'$M_{\odot}$', ha='right')
    if mass_str == '01500':
        ax3.text(0.03+log_Teff[good][0], -0.15+ell[good][0], '{:d}'.format(int(mass)), ha='right', size=10, color='grey')
    else:
        ax3.text(0.02+log_Teff[good][0], -0.1+ell[good][0], '{:d}'.format(int(mass)), ha='right', size=10, color='grey')

#make colormap based on main sequence distance for stars
#ax3.plot(zamsT, zamsL, c='grey', zorder=0, lw=1)
denseT = np.linspace(4.2, 4.7, 200)
denseL = interp1d(zamsT, zamsL, bounds_error=False, fill_value='extrapolate')(denseT)
distance = []
for T, L in zip(LogT, LogL):
    distance.append(np.min(np.sqrt((denseT - T)**2 + (denseL - L)**2)))
abs_star_distance = np.array(100*np.array(distance)/np.max(distance) - 1, dtype=int)
black = (0, 0, 0)
pink_rgbs = []
for i in range(3):
    pink_rgbs.append(np.linspace(cmap.mpl_colors[3][i], black[i], 150)[:100])
star_colors = []
for i in range(len(LogL)):
    ind = abs_star_distance[i]
    color = []
    for j in range(3):
        color.append(pink_rgbs[j][ind])
    star_colors.append(color)

#make legend & set clims, etc.
ax3.scatter(0.05, 0.04, c='white', marker='*', s=100, zorder=1, edgecolors='k', linewidths=0.5, transform=ax3.transAxes)
ax3.scatter(0.05, 0.09, c='white', marker='o', s=20, zorder=1, edgecolors='k',  linewidths=0.5, transform=ax3.transAxes)
ax3.text(0.10, 0.0375, 'Simulations', c='k', ha='left', va='center', transform=ax3.transAxes)
ax3.text(0.10, 0.09, 'Observed stars', ha='left', va='center', transform=ax3.transAxes, color='k')#cmap.mpl_colors[3])
ax3.set_xlim(4.75, 4.0)
ax3.set_ylim(1.3, 4.0)
ax3.set_ylabel(r'$\log_{10}\, \mathscr{L} / \mathscr{L}_\odot$')
ax3.set_xlabel(r'$\log_{10}\, $T$_{\rm eff}/$K')

plt.axes(ax1)
ax1.text(0.06, 1.5e-1, 'Predicted wave signal', ha='left')
ax1.text(0.25, 3.5e2, 'Observed red noise', ha='left', va='center')

for i in range(len(LogL)):
    color = star_colors[i]
    ax3.scatter(LogT[i], LogL[i], color=color, zorder=1, s=20, edgecolors='k', linewidths=0.5)
    alphanu = (alpha0[i] / (1 + (freqs/nuchar[i])**gamma[i]) + Cw[i]) #mags
    plt.loglog(freqs, alphanu, color=color)
    plt.ylim(5e-4, 1e3)
    plt.ylabel(r'$\Delta m$ ($\mu$mag)')
    plt.xlabel(r'frequency (d$^{-1}$)')

min_plot_freq = 1e-3
for i in range(3):
    star_log_Teff, star_log_Ell  = logTeffs[i], specLums[i]
    ax3.scatter(star_log_Teff, star_log_Ell, c=cmap.mpl_colors[i], marker='*', s=100, zorder=1, edgecolors='k', linewidths=0.5)
    good = freqs >= min_plot_freq
    ax2.loglog(freqs[good], signals[i][good], color=cmap.mpl_colors[i], lw=1)#, label=r'15 $M_{\odot}$ LMC sim')
    if i == 2:
        #plot on middle panel
        ax1.loglog(freqs[good], signals[i][good], color=cmap.mpl_colors[i], lw=1)#, label=r'15 $M_{\odot}$ LMC sim')

#con2 = ConnectionPatch(xyA=(1e1,1e-4), xyB=(4e-2,1e-4), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2, color='grey', lw=0.5)
#ax1.add_artist(con2)
con1 = ConnectionPatch(xyA=(1e1,1e0), xyB=(4e-2,1e0), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2, color='grey', lw=1)
ax1.add_artist(con1)
ax1.plot([7e0,1e1],[1e0, 1e0], c='grey', lw=1)

ax1.text(0.12, 0.04, r'15 $M_{\odot}$', color=cmap.mpl_colors[2], ha='center', va='center', size=8)
ax2.text(8e-2, 3.3e-1, r'40 $M_{\odot}$', color=cmap.mpl_colors[1], ha='center', va='center', size=8)
ax2.text(8e-2, 1.1e-1, r'15 $M_{\odot}$', color=cmap.mpl_colors[2], ha='center', va='center', size=8)
ax2.text(1.3e-1, 1.1e-2, r'3 $M_{\odot}$', color=cmap.mpl_colors[0], ha='center', va='center', size=8)
ax2.set_ylim(5e-4, 1e0)
ax2.set_xlabel(r'frequency (d$^{-1}$)')
for ax in [ax1, ax2]:
    ax.set_xlim(5e-2, 1e1)

plt.savefig('fig02_obs_prediction.png', bbox_inches='tight', dpi=300)
plt.savefig('fig02_obs_prediction.pdf', bbox_inches='tight', dpi=300)
