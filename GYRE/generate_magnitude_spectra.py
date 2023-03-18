import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

#Calculate transfer functions
eig_dir = 'gyre_output'
output_file = 'magnitude_spectra.h5'

star_dirs = ['03msol_Zsolar', '40msol_Zsolar', '15msol_ZLMC']
luminosity_amplitudes = [7.34e-15, 5.3e-10, 2.33e-11]
Lmax = 15
obs_length_days = 365
obs_length_sec  = obs_length_days*24*60*60
obs_cadence = 30*60 #30 min
df = 1/obs_length_sec
N_data = int(obs_length_sec/obs_cadence)
freqs = np.arange(N_data)*df

out_f = h5py.File(output_file, 'w')
out_f['frequencies'] = freqs

signals = []
specLums = []
logTeffs = []
for i, sdir in enumerate(star_dirs):
    wave_luminosity = lambda f, l: luminosity_amplitudes[i]*f**(-6.5)*np.sqrt(l*(l+1))**4
    transfer_oms = []
    transfer_signal = []
    pure_transfers  = []

    ell_list = np.arange(1, Lmax+1)
    for ell in ell_list:
        print(sdir, " ell = %i" % ell)


        with h5py.File('{:s}/{:s}/ell{:03d}_eigenvalues.h5'.format(sdir, eig_dir, ell), 'r') as f:
            om = f['transfer_om'][()]
            transfer_root_lum = f['transfer_root_lum'][()].real
        micromag = transfer_root_lum*np.sqrt(np.abs(wave_luminosity(om/(2*np.pi), ell)))
        transfer_oms.append(om[np.isfinite(micromag)])
        transfer_signal.append(micromag[np.isfinite(micromag)])
        pure_transfers.append(transfer_root_lum[np.isfinite(micromag)])



    transfers  = np.zeros((Lmax, N_data))
    magnitudes = np.zeros((Lmax, N_data))
    for j in range(freqs.size-1):
        for k, oms, signal, transfer in zip(range(Lmax), transfer_oms, transfer_signal, pure_transfers):
            good = (2*np.pi*freqs[j+1] >= oms)*(2*np.pi*freqs[j] < oms)
            if np.sum(good) > 0:
                magnitudes[k,j] = np.max(signal[good])
                transfers[k,j] = np.max(transfer[good])
            elif 2*np.pi*freqs[j] > oms.min() and 2*np.pi*freqs[j] <= oms.max():
                magnitudes[k,j] = signal[np.argmin(np.abs(2*np.pi*freqs[j] - oms))]
                transfers[k,j]  = transfer[np.argmin(np.abs(2*np.pi*freqs[j] - oms))]

    total_signal = np.sqrt(np.sum(magnitudes**2, axis=0))
    out_f['{}_magnitude_cube'.format(sdir)] = magnitudes
    out_f['{}_transfer_cube'.format(sdir)] = transfers
    out_f['{}_magnitude_sum'.format(sdir)] = total_signal
    plt.loglog(freqs, total_signal, c='k')
    plt.legend()
    plt.xlim(1e-7, 1e-3)
    plt.ylim(1e-6, 1)
    plt.xlabel('freq (Hz)')
    plt.ylabel(r'$\Delta m\,(\mu\rm{mag})$')
    plt.savefig('obs_ell_contributions_{}.png'.format(sdir), bbox_inches='tight')
    plt.clf()
out_f.close()
