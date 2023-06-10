"""
Script for analytically computing the Transfer function of damped sound waves in a 1D room of length L.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

D = 10
nu    = 1.349e-5 #m^2/s
Pr_air = 0.7
kappa = nu/Pr_air
c2 = (331)**2 #m/s
Lx = 10

Nz = 5000
n_modes = 600
omega_f = 2*np.pi*np.logspace(1, np.log10(30000), 5*n_modes) #Hz

n = 1 + np.arange(n_modes)
k = np.pi*n / Lx
omega = np.sqrt(c2)*k*np.sqrt(1 - (D**2 + 2*D*k**2*(kappa-nu) + k**4*(kappa-nu)**2*k**2)/(4*c2*k**2)) - 1j*(D + k**2*(kappa+nu))/2
z = np.linspace(0, Lx, Nz)
u = np.zeros((n.size, Nz), dtype=np.complex128)
p = np.zeros((n.size, Nz), dtype=np.complex128)
u_dual = np.zeros((n.size, Nz), dtype=np.complex128)
forcing = np.zeros_like(z)
forcing[(z > 0.5)*(z < 1.5)] = 1
for i in range(n.size):
    u[i,:] = np.sin(k[i]*z)
    u_dual[i,:] = (2/Lx)*u[i,:]
    p[i,:] = (-1j*k[i]*c2 / (-1j * omega[i] + kappa*k[i]**2)) * u[i,:]

dual_IP_force = np.zeros(n.size)
IP_check = np.zeros((n.size, n.size))
IP = lambda u1, u2: np.sum(np.gradient(z)*np.conj(u1)*u2)
for i in range(n.size):
    dual_IP_force[i] = IP(u_dual[i], forcing)
    for j in range(n.size):
        IP_check[i,j] = IP(u_dual[i], u[j])

if not np.allclose(IP_check.real, np.eye(IP_check.shape[0])):
    raise ValueError("Dual basis not properly specified; cannot calculate transfer")

z_listen = np.argmin(np.abs(z - 0.83*Lx))

#create transfer function.
omegas  = omega[:,None]
evp_spatial = (u[:,z_listen] * dual_IP_force)[:,None]

def transfer_function(om):
    om = om[None,:]
    cos_term = np.sum(      evp_spatial * om / ((omegas - om)*(omegas + om)), axis=0).real
    sin_term = np.sum(-1j * evp_spatial * omegas / ((omegas - om)*(omegas + om)), axis=0).real
    Transfer = np.sqrt(cos_term**2 + sin_term**2)
    return Transfer

def refined_transfer(om):
    T = transfer_function(om)

    peaks = 1
    while peaks > 0:
        i_peaks = []
        for i in range(1,len(om)-1):
            if (T[i]>T[i-1]) and (T[i]>T[i+1]):
                delta_m = np.abs(T[i]-T[i-1])/T[i]
                delta_p = np.abs(T[i]-T[i+1])/T[i]
                if delta_m > 0.01 or delta_p > 0.01:
                    i_peaks.append(i)

        peaks = len(i_peaks)
        print("number of peaks: %i" % (peaks))

        om_new = np.array([])
        for i in i_peaks:
            om_low = om[i-1]
            om_high = om[i+1]
            om_new = np.concatenate([om_new,np.linspace(om_low,om_high,10)])

        T_new = transfer_function(om_new)

        om = np.concatenate([om,om_new])
        T = np.concatenate([T,T_new])

        om, sort = np.unique(om, return_index=True)
        T = T[sort]

    return om, T

omega_f, Transfer = refined_transfer(omega_f.ravel())

with h5py.File('transfer_nmode{}_D{}.h5'.format(n_modes, D), 'w') as f:
    f['u_evecs'] = u
    f['p_evecs'] = p
    f['omegas'] = omega
    f['ks'] = k
    f['u_dual'] = u_dual

    f['transfer_freqs'] = omega_f/(2*np.pi)
    f['transfer'] = Transfer

plt.figure()
plt.loglog(omega_f.ravel()/(2*np.pi), Transfer)
plt.xlabel('freq (Hz)')
plt.ylabel('transfer')
plt.savefig('transfer.png')

