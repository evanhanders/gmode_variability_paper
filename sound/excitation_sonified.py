
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import h5py
from scipy.io.wavfile import read, write
from IPython.display import Audio

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

def sonify(mass):
    if mass == 3:
        filename = '../data/dedalus/wave_fluxes/zams_03Msol_solarZ/nr256/wave_luminosities.h5'
    elif mass == 15:
        filename = '../data/dedalus/wave_fluxes/zams_15Msol_LMC/re03200/wave_luminosities.h5'
    elif mass == 40:
        filename = '../data/dedalus/wave_fluxes/zams_40Msol_solarZ/nr256/wave_luminosities.h5'
    else:
        raise ValueError("no wave luminosity for mass = {}".format(mass))
    h5 = h5py.File(filename,'r')
    luminosity = h5['cgs_wave_luminosity(r=1.25)']
    luminosity= np.sum(luminosity, axis=2)
    luminosity = luminosity[0,:]
    frequ = h5['cgs_freqs']
    FsTrue = 1/1800
    FsSound = 45000
    shift_factor = FsSound/FsTrue
    shifted = np.full_like(frequ, shift_factor)
    shifted_frequencies = shifted*frequ
    luminosity[shifted_frequencies<20] = 0
    phis = np.random.default_rng().uniform(0,2*np.pi,luminosity.shape)
    phase_msol = luminosity * (np.cos(phis) + 1j*np.sin(phis))
    base = ifft(phase_msol)
    print(len(base))
    music = np.tile(base,20)
    norm = np.abs(music).max()
    music /= norm
    luminosity /= norm
    write('E_in_{}_msol.wav'.format(mass), FsSound, music.real/15)
    return shifted_frequencies, FsSound, music, luminosity

f3, Fs3, music3, l3 = sonify(3)
f15, Fs15, music15, l15 = sonify(15)
f40, Fs40, music40, l40 = sonify(40)
#definining colors for plotting
m3c = '#1b9e77'
m3cl = '#66c2a5'
m15c = '#7570b3'
m15cl = '#8da0cb'
m40c = '#d95f02'
m40cl = '#fc8d62'
#funtion to make time series
def timeseries(data,samplerate):
    duration = len(data)/samplerate
    time = np.arange(0,duration,1/samplerate) #time vector
    return time
fig = plt.figure(figsize=(7.5,4.0))
ax2 = fig.add_axes([0.55,0.67,0.45,.33])
ax1 = fig.add_axes([0,0.67,0.45,.33])
ax3 = fig.add_axes([0,0.34,0.45,.33])
ax4 = fig.add_axes([0.55,0.34,0.45,.33])
ax5 = fig.add_axes([0,0.01,0.45,.33])
ax6 = fig.add_axes([0.55,0.01,0.45,.33])
ax1.plot(timeseries(music3,Fs3), music3, color=m3cl)
ax3.plot(timeseries(music15,Fs15), music15, color=m15cl)
ax5.plot(timeseries(music40,Fs40), music40, color=m40cl)
ax5.set_xlabel('Time (s)')
for ax in [ax1,ax3,ax5]:
    ax.set_xlim(0,0.09)
    ax.set_ylim(-1,1)
    ax.set_yticks([-0.5,0,0.5])
    ax.set_ylabel('Amplitude')
ax1.set_xticks([])
ax3.set_xticks([])
ax2.loglog(f3, l3, label='3 $M_\odot$', color=m3cl)
ax4.loglog(f15, l15, label='15 $M_\odot$', color=m15cl)
ax6.loglog(f40, l40, label='40 $M_\odot$', color=m40cl)
ax6.set_xlabel('Frequency (Hz)')
for ax, label in zip([ax2, ax4, ax6],['3 $M_\odot$','15 $M_\odot$','40 $M_\odot$']):
    ax.set_ylim(1e-12,6e3)
    ax.text(0.98,0.9,label, ha='right',va='center',transform=ax.transAxes)
    ax.set_xlim(21,20000)
    ax.set_ylabel('Power')
    ax.set_yticks([1e-10,1e-6,1e-2,1e2])

plt.savefig("excitation_ts.pdf",dpi = 400, bbox_inches="tight")
