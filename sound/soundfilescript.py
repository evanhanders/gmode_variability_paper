# IMPORTING PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from numpy.fft import rfft, irfft, rfftfreq
import h5py
from scipy.io.wavfile import read, write

#INPUTS GO HERE
filename = 'sound_clips/Jupiter_Holst_Clip.wav' #this is “pure sound” we want to "record"
transfername = "transfer_nmode600_D10.h5" #this is the transfer function that we've already calculated
dampinglevel = 10 #This changes the strength of damping in our room

#VARIOUS FUNCTIONS
#fft
def four(data):
    return rfft(data)
#ifft
def ifour(data):
    return irfft(data)
#Frequencies
def freq(data,Fs):
    return rfftfreq(len(data),d=1/Fs)
#funtion to make time series
def timeseries(data,samplerate):
    duration = len(data)/samplerate
    time = np.arange(0,duration,1/samplerate) #time vector
    return time

#definining colors for plotting
purple1 = "#762a83"
purple2 = '#af8dc3'
purple3 = '#e7d4e8'
green1 = "#1b7837"
green2 = '#7fbf7b'
green3 = '#d9f0d3'

#interpolation function
def interp(transfer_freq, frequ, transfer):
    minfreq = transfer_freq[0]
    maxfreq = transfer_freq[-1]
    mask = np.zeros_like(frequ)
    mask[frequ<minfreq] = transfer[0]
    mask[frequ>maxfreq] = transfer[-1]
    good = (frequ>=minfreq)*(frequ<=maxfreq)
    idx = np.where(good)
    for i in range(np.sum(good)):
        mask[idx[0][i]] = transfer[np.argmin(np.abs(transfer_freq - frequ[idx[0][i]]))]
        if i%90000 ==0:
            print(i, mask[idx[0][i]], frequ[idx[0][i]])
    return mask

#function for normalizing volume
def norm(bad_sound, original, fn):
    dt = 1/fn
    original_power = np.sum(original**2*dt)
    new_power = np.sum(bad_sound**2*dt)
    return np.sqrt(original_power/new_power)*bad_sound

#READING IN INITIAL AUDIO FILE
Fs, data = read(filename)
filtereddata = four(data[:,0])
frequ = freq(data, Fs)
audible = frequ[np.argmin(np.abs(frequ - 20)):np.argmin(np.abs(frequ - 20000))]
# read in transfer function from h5py file 
h5 = h5py.File(transfername,'r')
outname = transfername.replace(".h5",".wav")
transfer = h5['transfer']  
transfer_freq = h5['transfer_freqs'] 

#interpolate over frequencies that our song is in
print("Interpolating...")
print("Iteration, Mask Value, Frequency")
mask = interp(transfer_freq, frequ, transfer)
h5.close()
print("...Interpolation Finished")
#apply mask to data and normalize
filtered = filtereddata*mask
norm_filtered = norm(filtered, filtereddata, Fs)

#Plot normalized power spectrum
plt.clf()
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

fig = plt.figure(figsize=(7.5,4.0))
ax2 = fig.add_axes([0.55,0,0.45,1])
ax1 = fig.add_axes([0,0,0.45,1])
ax1.loglog(frequ, mask, color=green1)
ax1.set_xlabel("frequency (Hz)")
ax1.set_ylabel("Transfer")
ax2.loglog(frequ, np.conj(filtereddata)*filtereddata,label='original', color=purple2)
ax2.loglog(frequ, np.conj(norm_filtered)*norm_filtered, label='filtered', color=green2)

ax2.legend()
ax1.set_xlim(audible[0],audible[-1])

ax2.set_xlim(audible[0],audible[-1])
ax2.set_ylim(1e-1,1e18)
ax2.set_xlabel("frequency (Hz)")
ax2.set_ylabel("Power")
plt.tight_layout()
plt.savefig("Transfer_and_PowerSpectrum.pdf", dpi = 400, bbox_inches="tight")

#take back to time space
filteredwrite = ifour(norm_filtered)
#Write sound file! 
write(outname, Fs, filteredwrite)
