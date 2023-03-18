import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py

from dedalus.tools.general import natural_sort

files = natural_sort(glob.glob('eigenvalues/ell*.h5'))

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)




fig = plt.figure(figsize=(8.5,5))
ncol = 3
hwidth = 0.25
pad = (1 - 3*hwidth)/(ncol-1)
ax1 = fig.add_axes([0.00,       0.55, hwidth, 0.4])
ax2 = fig.add_axes([hwidth+pad, 0.55, hwidth, 0.4])
ax3 = fig.add_axes([1-hwidth,   0.55, hwidth, 0.4])
ax4 = fig.add_axes([0.00,       0.0, hwidth, 0.4])
ax5 = fig.add_axes([hwidth+pad, 0.0, hwidth, 0.4])
ax6 = fig.add_axes([1-hwidth,   0.0, hwidth, 0.4])

axs = [ax1,ax2,ax3, ax4, ax5, ax6]
axs_left = [ax1, ax4]
axs_bot = [ax4, ax5, ax6]

brunt_pow_adj = 0

for file in files:
    print('plotting from {}'.format(file))
    out_dir = file.split('.h5')[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with h5py.File(file, 'r') as f:
        r = f['r'][()].ravel().real
        evalues = f['good_evalues'][()]
        efs_s = f['entropy_eigenfunctions'][()]
        efs_lnrho = f['ln_rho_eigenfunctions'][()]
        efs_u = f['velocity_eigenfunctions'][()]
        efs_enth = f['enthalpy_fluc_eigenfunctions'][()]
        rho = f['rho_full'][()].ravel().real
        bruntN2 = f['bruntN2'][()].ravel().real
        N2_scale = np.copy(bruntN2)
        N2_scale[N2_scale < 1] = 1
        print(N2_scale)


    for i, ev in enumerate(evalues):
        print('plotting {:.7e}'.format(ev))
        ax1.plot(r, rho**(-1/2)*r**(1/2)*N2_scale**(-1/4)*efs_enth[i,:].real, label='real')
        ax1.plot(r, rho**(-1/2)*r**(1/2)*N2_scale**(-1/4)*efs_enth[i,:].imag, label='imag')
        ax2.plot(r, rho**(1/2)*r**(3/2)*N2_scale**(1/4)*efs_u[i,0,:].real, label='real')
        ax2.plot(r, rho**(1/2)*r**(3/2)*N2_scale**(1/4)*efs_u[i,0,:].imag, label='imag')
        ax3.plot(r, rho**(1/2)*r**(3/2)*N2_scale**(1/4)*efs_u[i,2,:].real, label='real')
        ax3.plot(r, rho**(1/2)*r**(3/2)*N2_scale**(1/4)*efs_u[i,2,:].imag, label='imag')

        wave_lum = 4*np.pi*r**2 * np.conj(efs_u[i,2,:]) * efs_enth[i,:]
        ax4.plot(r, wave_lum.real, label='real')
        ax4.plot(r, wave_lum.imag, label='imag')

        ax5.plot(r, rho**(1/2)*r**(1/2)*N2_scale**(-(3/4))*efs_s[i,:].real, label='real')
        ax5.plot(r, rho**(1/2)*r**(1/2)*N2_scale**(-(3/4))*efs_s[i,:].imag, label='imag')

#        ax6.axhline(ev.real**2, c='k', label='$\omega^2$')
#        ax6.plot(r, bruntN2, lw=2)
#        ax6.plot(r, S*r**(2+brunt_pow_adj), c='orange', ls='--', label=r'${{{}}} r^{{{}}}$'.format(S, 2+brunt_pow_adj))
#        ax6.legend()
        ax6.plot(r, rho**(1/2)*r**(3/2)*N2_scale**(1/4)*efs_u[i,1,:].real, label='real')
        ax6.plot(r, rho**(1/2)*r**(3/2)*N2_scale**(1/4)*efs_u[i,1,:].imag, label='imag')

        ax1.legend()
        ax1.set_ylabel(r'$r^{1/2}(N^2)^{-1/4} \,\rho^{{-1/2}}\, h_{\rm fluc}$')
        ax2.set_ylabel(r'$r^{3/2}(N^2)^{1/4}\,\rho^{{1/2}}\,u_\phi$')
        ax3.set_ylabel(r'$r^{3/2}(N^2)^{1/4}\,\rho^{{1/2}}\,u_r$')
        ax4.set_ylabel(r'Lum = $4\pi r^2 u_r^* h_{\rm fluc}$')
        ax5.set_ylabel(r'$r^{1/2}(N^2)^{-3/4}\rho^{{1/2}}\,s$')
        ax6.set_ylabel(r'$r^{3/2}(N^2)^{1/4}\,\rho^{{1/2}}\,u_\theta$')
#        ax6.set_ylabel(r'$N^2$')
        plt.suptitle('ev = {:.3e}'.format(ev))


        for ax in axs:
            ax.set_xlim(0, r.max())

        for ax in axs_bot:
            ax.set_xlabel('r')
        
        plt.savefig('{}/ef_{:03d}.png'.format(out_dir, i), bbox_inches='tight')
        for ax in axs:
            ax.clear()



