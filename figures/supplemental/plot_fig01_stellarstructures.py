import h5py
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from palettable.colorbrewer.qualitative import Accent_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

from compstar.tools.mesa import DimensionalMesaReader

mesa_root = '../../MESA/'
mesa_files = [  '03msol_Zsolar/LOGS/profile43.data',\
                '15msol_ZLMC/LOGS/profile47.data',\
                '40msol_Zsolar/LOGS/profile53.data',\
                '15msol_ZLMC/LOGS/profile47.data'
              ]
mesa_files = ['{}/{}'.format(mesa_root, f) for f in mesa_files]
structures = [DimensionalMesaReader(f) for f in mesa_files]

dedalus_root = '../../dedalus'
star_files = [ 'zams_03Msol_solarZ/wave_generation/nr256/star/star_256+128_bounds0-2L_Re1.30e+04_de1.5_cutoff1.0e-12.h5',\
               'zams_15Msol_LMC/wave_generation/re10000/star/star_512+192_bounds0-2L_Re3.00e+04_de1.5_cutoff1.0e-10.h5',\
               'zams_40Msol_solarZ/wave_generation/nr256/star/star_256+128_bounds0-2L_Re7.00e+03_de1.5_cutoff1.0e-10.h5',\
               'zams_15Msol_LMC/wave_propagation/nr256/star/star_256+192+64_bounds0-0.93R_Re1.00e+04_de1.5_cutoff1.0e-10.h5'
             ]
star_files = ['{}/{}'.format(dedalus_root, f) for f in star_files]
star_handles = [h5py.File(f, 'r') for f in star_files]


def get_dedalus_stratification(i, basis_list):
    rs = []
    lnrhos = []
    N2s = []
    grad_s = []
    chi_rad = []
    for bases in basis_list:
        rs.append(star_handles[i]['r_{}'.format(bases)][()]*star_handles[i]['L_nd'][()])
        lnrhos.append(star_handles[i]['ln_rho0_{}'.format(bases)][()])
        N2s.append(-star_handles[i]['g_{}'.format(bases)][2,:]*star_handles[i]['grad_s0_{}'.format(bases)][2,:]/star_handles[i]['Cp'][()] *(star_handles[i]['tau_nd'][()])**(-2))
        grad_s.append(star_handles[i]['grad_s0_{}'.format(bases)][2,:]*star_handles[i]['s_nd'][()]/star_handles[i]['L_nd'][()])
        chi_rad.append(star_handles[i]['kappa_rad_{}'.format(bases)]/(np.exp(star_handles[i]['ln_rho0_{}'.format(bases)][()])*star_handles[i]['Cp'][()]) * star_handles[i]['L_nd'][()]**2 / star_handles[i]['tau_nd'][()] )
    r = np.concatenate([a.ravel() for a in rs])
    ln_rho = np.concatenate([a.ravel() for a in lnrhos])
    rho = np.exp(ln_rho)*star_handles[i]['rho_nd'][()]
    grad_s = np.concatenate([a.ravel() for a in grad_s])
    N2 = np.concatenate([a.ravel() for a in N2s])
    chi_rad = np.concatenate([a.ravel() for a in chi_rad])

    r_l = star_handles[i]['lum_r_vals'][()]*star_handles[i]['L_nd'][()]
    sim_lum = star_handles[i]['sim_lum'][()]*star_handles[i]['m_nd'][()]*star_handles[i]['L_nd'][()]**2/star_handles[i]['tau_nd'][()]**3

    return r, ln_rho, rho, grad_s, N2, chi_rad, r_l, sim_lum


fig = plt.figure(figsize=(7.5, 6))
ax1_1 = fig.add_axes([0.100, 0.820, 0.280, 0.160])
ax1_2 = fig.add_axes([0.100, 0.660, 0.280, 0.160])
ax1_3 = fig.add_axes([0.100, 0.500, 0.280, 0.160])
ax1_4 = fig.add_axes([0.100, 0.340, 0.280, 0.160])
ax1_5 = fig.add_axes([0.100, 0.080, 0.280, 0.160])

ax2_1 = fig.add_axes([0.400, 0.820, 0.280, 0.160])
ax2_2 = fig.add_axes([0.400, 0.660, 0.280, 0.160])
ax2_3 = fig.add_axes([0.400, 0.500, 0.280, 0.160])
ax2_4 = fig.add_axes([0.400, 0.340, 0.280, 0.160])
ax2_5 = fig.add_axes([0.400, 0.080, 0.280, 0.160])

ax3_1 = fig.add_axes([0.700, 0.820, 0.280, 0.160])
ax3_2 = fig.add_axes([0.700, 0.660, 0.280, 0.160])
ax3_3 = fig.add_axes([0.700, 0.500, 0.280, 0.160])
ax3_4 = fig.add_axes([0.700, 0.340, 0.280, 0.160])
ax3_5 = fig.add_axes([0.700, 0.080, 0.280, 0.160])

row1_axs = [ax1_1, ax2_1, ax3_1]
row2_axs = [ax1_2, ax2_2, ax3_2]
row3_axs = [ax1_3, ax2_3, ax3_3]
row4_axs = [ax1_4, ax2_4, ax3_4]
row5_axs = [ax1_5, ax2_5, ax3_5]

axs = row1_axs + row2_axs + row3_axs + row4_axs

ax1_1.set_ylabel(r'$\rho\,\left(\frac{\rm{g}}{\rm{cm}^{3}}\right)$')
ax1_2.set_ylabel(r'$N^2\, \left(\frac{\rm{rad}}{s}\right)^2$')
ax1_3.set_ylabel(r'$\nabla s\,\left(\frac{\rm{erg}}{\rm{g}\,\rm{K}\,\rm{cm}}\right)$')
ax1_4.set_ylabel(r'$\chi_{\rm rad}\, \left(\frac{\rm{cm}^{2}}{\rm{s}}\right)$')
ax1_5.set_ylabel(r'$L_{\rm conv}/L_*$')

for ax in axs:
    ax.set_yscale('log')

axs += row5_axs


for i in range(len(mesa_files)):
    if i < 3: 
        row1_axs[i].plot(structures[i].structure['r'], structures[i].structure['rho'],      c='k', lw=2, label='MESA')    
        row2_axs[i].plot(structures[i].structure['r'], structures[i].structure['N2'],       c='k', lw=2)    
        row3_axs[i].plot(structures[i].structure['r'], structures[i].structure['grad_s'],   c='k', lw=2)    
        row4_axs[i].plot(structures[i].structure['r'], structures[i].structure['rad_diff'], c='k', lw=2)
        row5_axs[i].plot(structures[i].structure['r'], structures[i].structure['L_conv']/structures[i].structure['Luminosity'].max(), c='k', lw=2)

        r, ln_rho, rho, grad_s, N2, chi_rad, r_l, sim_lum = get_dedalus_stratification(i, ['B', 'S1'])

        row1_axs[i].plot(r, rho,     c=cmap.mpl_colors[0],   lw=0.75, zorder=10, label='WG sim')    
        row2_axs[i].plot(r, N2,      c=cmap.mpl_colors[0],   lw=0.75, zorder=10)    
        row3_axs[i].plot(r, grad_s,  c=cmap.mpl_colors[0],   lw=0.75, zorder=10)    
        row4_axs[i].plot(r, chi_rad, c=cmap.mpl_colors[0],   lw=0.75, zorder=10)
        row5_axs[i].plot(r_l, sim_lum/structures[i].structure['Luminosity'].max(), c=cmap.mpl_colors[0], lw=0.75, zorder=10)
    else:
        r, ln_rho, rho, grad_s, N2, chi_rad, r_l, sim_lum = get_dedalus_stratification(i, ['B', 'S1', 'S2'])

        row1_axs[1].plot(r, rho,     c=cmap.mpl_colors[2], label='WP sim')    
        row2_axs[1].plot(r, N2,      c=cmap.mpl_colors[2])    
        row3_axs[1].plot(r, grad_s,  c=cmap.mpl_colors[2])    
        row4_axs[1].plot(r, chi_rad, c=cmap.mpl_colors[2])
        row5_axs[1].plot(r_l, sim_lum/structures[i].structure['Luminosity'].max(), c=cmap.mpl_colors[2],)


for ax in [ax1_1, ax2_1, ax3_1]:
    ax.set_ylim(1e-10, 1e2)
    ax.set_yticks((1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2))
for ax in [ax1_2, ax2_2, ax3_2]:
    ax.set_ylim(1e-10, 1e-4)
    ax.set_yticks((1e-9, 1e-8, 1e-7, 1e-6, 1e-5, ))
for ax in [ax1_3, ax2_3, ax3_3]:
    ax.set_ylim(1e-8, 1e1)
    ax.set_yticks(( 1e-7, 1e-5, 1e-3, 1e-1 ))
for ax in [ax1_4, ax2_4, ax3_4]:
    ax.set_ylim(1e5, 1e18)
    ax.set_yticks(( 1e6, 1e8, 1e10, 1e12, 1e14, 1e16 ))
ax1_4.set_xticks((0, 5e10, 1e11))
ax1_4.set_xticklabels(('0', r'$5 \times 10^{10}$', '$10^{11}$'))
ax2_4.set_xticks((0, 1e11, 2e11, 3e11))
ax2_4.set_xticklabels(('0', '$10^{11}$', r'$2 \times 10^{11}$', r'$3 \times 10^{11}$' ))
ax3_4.set_xticks((0, 2e11, 4e11))
ax3_4.set_xticklabels(('0', r'$2 \times 10^{11}$', r'$4 \times 10^{11}$' ))

for ax in row5_axs:
    ax.set_ylim(0, 0.65)
ax1_5.set_xlim(0, 2.5e10)
ax1_5.set_xticks((0, 1e10, 2e10))
ax1_5.set_xticklabels(('0', r'$10^{10}$', r'$2 \times 10^{10}$'))
ax2_5.set_xlim(0, 1.2e11)
ax2_5.set_xticks((0, 5e10, 1e11))
ax2_5.set_xticklabels(('0', r'$5 \times 10^{10}$', r'$10^{11}$'))
ax3_5.set_xlim(0, 2.5e11)
ax3_5.set_xticks((0, 1e11, 2e11))
ax3_5.set_xticklabels(('0', r'$10^{11}$', r'$2 \times 10^{11}$'))
ax2_5.set_yticklabels(())
ax3_5.set_yticklabels(())

for ax in axs[:-6]:
    ax.set_xticklabels(())
    ax.tick_params(axis="x",direction="in", which='both')
for ax in axs[-6:]:
    ax.set_xlabel('radius (cm)')

for i, ax in enumerate(axs):
    if i % 3 == 0: 
        if i < 3:
            ax.text(0.98, 0.88, r'$3\,M_{\odot}$',  ha='right', va='center', transform=ax.transAxes)
        else:
            ax.text(0.02, 0.88, r'$3\,M_{\odot}$',  ha='left', va='center', transform=ax.transAxes)
        continue
    elif (i + 1) % 3 == 0:
        if i < 3:
            ax.text(0.98, 0.88, r'$40\,M_{\odot}$',  ha='right', va='center', transform=ax.transAxes)
        else:
            ax.text(0.02, 0.88, r'$40\,M_{\odot}$',  ha='left', va='center', transform=ax.transAxes)
    elif (i + 2) % 3 == 0:
        if i < 3:
            ax.text(0.98, 0.88, r'$15\,M_{\odot}$',  ha='right', va='center', transform=ax.transAxes)
        else:
            ax.text(0.02, 0.88, r'$15\,M_{\odot}$',  ha='left', va='center', transform=ax.transAxes)
    ax.set_yticklabels(())

row1_axs[1].legend(loc='lower left')




[h.close() for h in star_handles]
fig.savefig('fig01_mesa_dedalus_stratification.png', dpi=300)
fig.savefig('fig01_mesa_dedalus_stratification.pdf', dpi=300)
