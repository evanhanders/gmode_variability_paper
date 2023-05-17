import numpy as np
import mesa_reader as mr
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from palettable.colorbrewer.qualitative import Dark2_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
sLum_sun=(5777)**4.0/(274*100)

#Get simulation details.
star_dirs = ['03msol_Zsolar', '40msol_Zsolar', '15msol_ZLMC']
sim_mass = [3, 40, 15]
sim_alpha = [5.5e-3, 0.16, 0.06]
sim_nuchar = [0.3, 0.13, 0.22]
sim_specLums = []
sim_logTeffs = []
for i, sdir in enumerate(star_dirs):
    #MESA history for getting ell.
    mesa_history = '../../MESA/{}/LOGS/history.data'.format(sdir)
    history = mr.MesaData(mesa_history)
    mn = history.model_number
    log_g = history.log_g
    log_Teff = history.log_Teff

    sLum = (10**log_Teff)**4.0/(10**log_g)
    sLum=np.log10(sLum/sLum_sun)
    sim_specLums.append(sLum[-1])
    sim_logTeffs.append(log_Teff[-1])

fig = plt.figure(figsize=(7.5, 2.5))
ax1 = fig.add_axes([0.02, 0.02, 0.43, 0.8])
ax2 = fig.add_axes([0.55, 0.02, 0.43, 0.8])
cax = fig.add_axes([0.25, 0.93, 0.50, 0.05])
axs = [ax1, ax2]

norm = mpl.colors.Normalize(vmin=4, vmax=4.7)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)

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
                 usecols=[2,3,4,5,6,],
                 names=['TIC','alpha0','e_alpha0', 'nuchar', 'e_nuchar'])

nuchar   = df2['nuchar']
e_nuchar = df2['e_nuchar']
alpha0   = df2['alpha0']
e_alpha0 = df2['e_alpha0']
vsini    = df1['vsini']
vmacro   = df1['vmacro']
LogT     = df1['LogT']
LogL     = df1['LogL']

nuchar   = np.array([float(item) for item in nuchar])
alpha0   = np.array([float(item) for item in alpha0])
e_nuchar = np.array([float(item) for item in e_nuchar])
e_alpha0 = np.array([float(item) for item in e_alpha0])
vsini    = np.array([float(item) for item in vsini])
LogT     = np.array([float(item) for item in LogT])
LogL     = np.array([float(item) for item in LogL])



for ax in axs:
    ax.set_xlabel(r'$\log_{10}\,\mathscr{L}/\mathscr{L}_{\odot}$')
ax1.set_ylabel(r'$\alpha_{0}$ ($\mu$mag)')
ax2.set_ylabel(r'$\nu_{\rm char}$ (d$^{-1}$)')

for j in range(LogT.size):
    ax1.errorbar(LogL[j], alpha0[j], yerr=e_alpha0[j], color=(1,1,1,0), marker='o', ecolor='k', elinewidth=0.5)
    ax2.errorbar(LogL[j], nuchar[j], yerr=e_nuchar[j], color=(1,1,1,0), marker='o', ecolor='k', elinewidth=0.5)
    ax1.scatter(LogL[j], alpha0[j], color=sm.to_rgba(LogT[j])[:3], alpha=0.7, marker='o', edgecolors='k', linewidths=0.5, s=20)
    ax2.scatter(LogL[j], nuchar[j], color=sm.to_rgba(LogT[j])[:3], alpha=0.7, marker='o', edgecolors='k', linewidths=0.5, s=20)
ax1.set_yscale('log')

for i in range(len(sim_mass)):
    ax1.scatter(sim_specLums[i], sim_alpha[i],  color=sm.to_rgba(sim_logTeffs[i]), marker='*', s=150, edgecolors='k', linewidths=1)
    ax2.scatter(sim_specLums[i], sim_nuchar[i], color=sm.to_rgba(sim_logTeffs[i]), marker='*', s=150, edgecolors='k', linewidths=1)
#    ax1.scatter(sim_specLums[i], sim_alpha[i],  color=cmap.mpl_colors[i], marker='*', s=200, edgecolors='k', linewidths=1)
#    ax2.scatter(sim_specLums[i], sim_nuchar[i], color=cmap.mpl_colors[i], marker='*', s=200, edgecolors='k', linewidths=1)

ax2.set_yscale('log')

cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cax.text(-0.02, 0.5, r'$\log_{10} T_{\rm eff}$', ha='right', va='center', transform=cax.transAxes)
cax.invert_xaxis()
cb.set_ticks((4, 4.2, 4.4, 4.6))

fig.savefig('fig03_scatter_plots.png', dpi=300, bbox_inches='tight')
fig.savefig('fig03_scatter_plots.pdf', dpi=300, bbox_inches='tight')

