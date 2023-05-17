# IMPORTING PACKAGES
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
import time
from multiprocessing import Process
from IPython.display import Audio
from numpy.fft import rfft, irfft, rfftfreq
from moviepy.editor import *
from moviepy.video.io.bindings import mplfig_to_npimage
import h5py

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

#funtion to make time series
def timeseries(data,samplerate):
    duration = len(data)/samplerate
    time = np.arange(0,duration,1/samplerate) #time vector
    return time

#defining fourier stuff
#fft
def four(data):
    return rfft(data)
#ifft
def ifour(data):
    return irfft(data)
#Frequencies
def freq(data,Fs):
    return rfftfreq(len(data),d=1/Fs)

songname = 'Jupiter_Holst_Clip.wav'
#reading in files
data, Fs = sf.read(songname)
#saving norm constant
normjupiter = np.abs(data[:,0]).max()

#Fourier transforms
filtereddata = four(data[:,0])
frequ = freq(data, Fs)

#definining colors for plotting
m3c = '#1b9e77'
m3cl = '#66c2a5'
m15c = '#7570b3'
m15cl = '#8da0cb'
m40c = '#d95f02'
m40cl = '#fc8d62'

fig = plt.figure(figsize=(7.5,4.0))
ax2 = fig.add_axes([0.55,0.67,0.45,.33])
ax1 = fig.add_axes([0,0.67,0.45,.33])
ax3 = fig.add_axes([0,0.34,0.45,.33])
ax4 = fig.add_axes([0.55,0.34,0.45,.33])
ax5 = fig.add_axes([0,0.01,0.45,.33])
ax6 = fig.add_axes([0.55,0.01,0.45,.33])
ax1.plot(timeseries(data[:,0]/normjupiter,Fs), data[:,0]/normjupiter, color=m3cl)
ax3.plot(timeseries(data[:,0]/normjupiter,Fs), data[:,0]/normjupiter, color=m15cl)
ax5.plot(timeseries(data[:,0]/normjupiter,Fs), data[:,0]/normjupiter, color=m40cl)
ax5.set_xlabel('Time (s)')
for ax in [ax1,ax3,ax5]:
    ax.set_xlim(0,20)
    ax.set_ylim(-1,1)
    ax.set_yticks([-0.5,0,0.5])
    ax.set_ylabel('Amplitude')
ax1.set_xticks([])
ax3.set_xticks([])
ax2.loglog(frequ, np.conj(filtereddata)*filtereddata, label='3 $M_\odot$', color=m3cl)
ax4.loglog(frequ, np.conj(filtereddata)*filtereddata, label='15 $M_\odot$', color=m15cl)
ax6.loglog(frequ, np.conj(filtereddata)*filtereddata, label='40 $M_\odot$', color=m40cl)
ax6.set_xlabel('Frequency (Hz)')
for ax, label in zip([ax2, ax4, ax6],['3 $M_\odot$','15 $M_\odot$','40 $M_\odot$']):
    ax.set_ylim(1e-3,1e9)
    ax.text(0.98,0.9,label, ha='right',va='center',transform=ax.transAxes)
    ax.set_xlim(21,20000)
    ax.set_ylabel('Amplitude')
    ax.set_yticks([1e-1,1e2,1e5,1e8])

plt.savefig("original_waveform.pdf",dpi = 400, bbox_inches="tight")

