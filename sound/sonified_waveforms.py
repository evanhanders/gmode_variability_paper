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

#read in h5 file
filename = '../data/dedalus/predictions/magnitude_spectra.h5'
h5 = h5py.File(filename,'r')
frequ = h5['frequencies'][()]
msol_15_amp = h5['15msol_ZLMC_magnitude_sum'][()]
msol_3_amp = h5['03msol_Zsolar_magnitude_sum'][()]
msol_40_amp = h5['40msol_Zsolar_magnitude_sum'][()]

#cutoff low frequency noise which can dominate
cutoff = 1e-7
msol_15_amp[frequ<cutoff] = 0
msol_40_amp[frequ<cutoff] = 0
msol_3_amp[frequ<cutoff] = 0

#shift to audible frequency ranges
FsTrue = 1/1800
FsSound = 45000
shift_factor = FsSound/FsTrue
shifted = np.full_like(frequ, shift_factor)
shifted_frequencies = shifted*frequ

msol_15_amp[shifted_frequencies<20] = 0
msol_40_amp[shifted_frequencies<20] = 0
msol_3_amp[shifted_frequencies<20] = 0

#generate random power phases for each frequency
phis = np.random.default_rng().uniform(0,2*np.pi,msol_15_amp.shape)
phase_15msol = msol_15_amp * (np.cos(phis) + 1j*np.sin(phis))
phase_40msol = msol_40_amp * (np.cos(phis) + 1j*np.sin(phis))
phase_3msol = msol_3_amp * (np.cos(phis) + 1j*np.sin(phis))

#Take back to time domain and make longer
base15 = ifft(phase_15msol)
base40 = ifft(phase_40msol)
base3 = ifft(phase_3msol)
music15 = np.tile(base15,40)
music40 = np.tile(base40,40)
music3 = np.tile(base3,40)

#normalization
norm15 = np.abs(music15).max()
norm40 = np.abs(music40).max()
norm3 = np.abs(music3).max()
music15 /= norm15
music40 /= norm40
music3 /= norm3
msol_40_amp /= norm40
msol_15_amp /= norm15
msol_3_amp /= norm3

#write files
write("sonified_3msol.wav", FsSound, music3.real/35)
write("sonified_15msol.wav", FsSound, music15.real/16)
write("sonified_40msol.wav", FsSound, music40.real/8)

#Plot timeseries data and power spectrum
m3c = '#1b9e77'
m15c = '#7570b3'
m40c = '#d95f02'
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
ax1.plot(timeseries(music3,FsSound), music3, color=m3c)
ax3.plot(timeseries(music15,FsSound), music15, color=m15c)
ax5.plot(timeseries(music40,FsSound), music40, color=m40c)
ax1.set_ylabel('Amplitude')
ax3.set_ylabel('Amplitude')
ax5.set_ylabel('Amplitude')
ax5.set_xlabel('Time (s)')
for ax in [ax1,ax3,ax5]:
    ax.set_xlim(0,0.5)
    ax.set_ylim(-1,1)
    ax.set_yticks([-0.5,0,0.5])
ax1.set_xticks([])
ax3.set_xticks([])
ax2.loglog(shifted_frequencies, msol_3_amp, label='3 $M_\odot$', color=m3c)
ax4.loglog(shifted_frequencies, msol_15_amp, label='15 $M_\odot$', color=m15c)
ax6.loglog(shifted_frequencies, msol_40_amp, label='40 $M_\odot$', color=m40c)
ax6.set_xlabel('Frequency (Hz)')
for ax, label in zip([ax2, ax4, ax6],['3 $M_\odot$','15 $M_\odot$','40 $M_\odot$']):
    ax.set_ylim(1e-5,1e4)
    ax.text(0.98,0.9,label, ha='right',va='center',transform=ax.transAxes)
    ax.set_xlim(21,20000)
    ax.set_ylabel('Power')
    ax.set_yticks([1e-4,1e-2,1,1e2])

plt.savefig("sonified_timeseries_PS.pdf",dpi = 400, bbox_inches="tight")
