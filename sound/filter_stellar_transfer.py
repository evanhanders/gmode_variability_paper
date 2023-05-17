import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import h5py
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import find_peaks

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

m3c = '#1b9e77'
m15c = '#7570b3'
m40c = '#d95f02'

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

#funtion to make time series
def timeseries(data,samplerate):
    duration = len(data)/samplerate
    time = np.arange(0,duration,1/samplerate) #time vector
    return time

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


#read in h5 file to get stellar transfer functions TODO evan add filename
filename = 'magnitude_spectra.h5'
h5 = h5py.File(filename,'r')
msol_15_transfer = h5['15msol_ZLMC_transfer_cube'][()][0]
msol_40_transfer = h5['40msol_Zsolar_transfer_cube'][()][0]
msol_3_transfer = h5['03msol_Zsolar_transfer_cube'][()][0]
transfer_frequencies = h5['frequencies'][()]
h5.close()

FsTrue = 1/1800
FsSound = 19000

shift_factor = FsSound/FsTrue
shifted = np.full_like(transfer_frequencies, shift_factor)
shifted_frequencies = shifted*transfer_frequencies

#Read in jupiter clip
filename = 'Jupiter_Holst_Clip.wav'
Fs, data = read(filename)

#take fourier transform
filtereddata = four(data[:,0])
frequ = freq(data, Fs)
audible = frequ[np.argmin(np.abs(frequ - 20)):np.argmin(np.abs(frequ - 20000))]

#interpolate over frequencies that our song is in
print("Interpolating...")
print("Iteration, Mask Value, Frequency")
mask15 = interp(shifted_frequencies, frequ, msol_15_transfer)
mask40 = interp(shifted_frequencies, frequ, msol_40_transfer)
mask3 = interp(shifted_frequencies, frequ, msol_3_transfer)
print("...Interpolation Finished")

fig = plt.figure(figsize=(7.5,4.0))
plt.loglog(frequ, np.cbrt(mask3),  label='Mass={}'.format(3), color=m3c)
plt.loglog(frequ, np.cbrt(mask15),  label='Mass={}'.format(15), color=m15c)
plt.loglog(frequ, np.cbrt(mask40),  label='Mass={}'.format(40), color=m40c)
plt.legend(loc='upper left')
plt.xlim(20,20000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.savefig("transfer_funcs.pdf")

#apply mask to data and normalize
filtered3 = filtereddata*np.cbrt(mask3)
filtered15 = filtereddata*np.cbrt(mask15)
filtered40 = filtereddata*np.cbrt(mask40)

#take back to time space
filteredwrite3 = ifour(filtered3)
filteredwrite15 = ifour(filtered15)
filteredwrite40 = ifour(filtered40)

# normalization 
norm15 = np.abs(filteredwrite15).max()
norm40 = np.abs(filteredwrite40).max()
norm3 = np.abs(filteredwrite3).max()
filteredwrite15 /= norm15
filteredwrite40 /= norm40
filteredwrite3 /= norm3

#write files
write('J_in_40_msol.wav', Fs, filteredwrite40/4)
write('J_in_15_msol.wav', Fs, filteredwrite15/4)
write('J_in_3_msol.wav', Fs, filteredwrite3/18)

#Plotting
fig = plt.figure(figsize=(7.5,4.0))
ax2 = fig.add_axes([0.55,0.67,0.45,.33])
ax1 = fig.add_axes([0,0.67,0.45,.33])
ax3 = fig.add_axes([0,0.34,0.45,.33])
ax4 = fig.add_axes([0.55,0.34,0.45,.33])
ax5 = fig.add_axes([0,0.01,0.45,.33])
ax6 = fig.add_axes([0.55,0.01,0.45,.33])
ax1.plot(timeseries(filteredwrite3,Fs), filteredwrite3, color=m3c)
ax3.plot(timeseries(filteredwrite15,Fs), filteredwrite15, color=m15c)
ax5.plot(timeseries(filteredwrite40,Fs), filteredwrite40, color=m40c)
ax1.set_ylabel('Amplitude')
ax3.set_ylabel('Amplitude')
ax5.set_ylabel('Amplitude')
ax5.set_xlabel('Time (s)')
for ax in [ax1,ax3,ax5]:
    ax.set_xlim(0,20)
    ax.set_ylim(-1,1)
    ax.set_yticks([-0.5,0,0.5])
ax1.set_xticks([])
ax3.set_xticks([])
ax2.loglog(frequ, np.conj(filtered3)*filtered3/norm3, label='3 $M_\odot$', color=m3c)
ax4.loglog(frequ, np.conj(filtered15)*filtered15/norm15, label='15 $M_\odot$', color=m15c)
ax6.loglog(frequ, np.conj(filtered40)*filtered40/norm40, label='40 $M_\odot$', color=m40c)
ax6.set_xlabel('Frequency (Hz)')
for ax, label in zip([ax2, ax4, ax6],['3 $M_\odot$','15 $M_\odot$','40 $M_\odot$']):
    #ax.set_ylim(1e-15,1)
    ax.text(0.98,0.9,label, ha='right',va='center',transform=ax.transAxes)
    ax.set_xlim(21,20000)
    ax.set_ylabel('Amplitude')
    #ax.set_yticks([1e-14,1e-10,1e-6,1e-2])

plt.savefig("gmodes_timeseries_PS.pdf",dpi = 400, bbox_inches="tight")
