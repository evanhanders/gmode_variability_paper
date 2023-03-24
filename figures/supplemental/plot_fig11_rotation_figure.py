import matplotlib.pyplot as plt 
import mesa_reader as mr
import numpy as np
from pylab import *
from math import log10, pi
from matplotlib import rc
#rc('mathtext', default='regular')
import matplotlib as mpl 
import matplotlib.tri as tri 
import pandas as pd  
import pickle

import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from matplotlib.patches import ConnectionPatch



from numpy import loadtxt
import pandas as pd  
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
 

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


############### SOME DEFINITIONS
fiducial_ell = 3.111
fiducial_T = 4.517
fiducial_Ro = 0.6785 #calculated from MESA model -- see fig13 file
fiducial_Rop = fiducial_Ro / 0.7 #taking <sin i> = 0.7.
rsun=6.9598e10
msun=1.9892e33
lsun=3.8418e33
tsun = 5777
G=6.67428e-8
ell_sun=(5777)**4.0/(274*100)  # Used for Spectroscopic HRD 
alpha_mlt=1.6
lgteff = r'$\log\, $T$_{\rm eff}/$K'
lgl= r'$\log\, L / L_\odot$'
lgell= r'$\log\, \mathscr{L} / \mathscr{L}_\odot$'

def find_h(dh,center_h1,model):
    zams=1
    while (center_h1[zams] > (center_h1[1] - dh)):
     zams=zams+1
    return zams;

def find_tams(center_h1,model):
    tams=1
    n=len(model)
    while (center_h1[tams] > 0.05) and (tams < n-1):
     tams=tams+1
    return tams;

prefix,DIR,mods,hs,hrdlines = pickle.load(open('../../data/Atlas/parsedAtlas.data','rb'))

DIR = '../../data/observation_tables/'
file1='{}/bowman2020_tablea1.txt'.format(DIR)
file2='{}/bowman2020_tablea2.txt'.format(DIR)
file3='{}/bowman2022_tableB1.csv'.format(DIR)

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
                 usecols=[2,3,4],
                 names=['TIC','alpha0','nuchar'])

df3 = pd.read_csv(file3,
                 sep=",",
                 skiprows=1,
                 skipfooter = 1,
                 usecols=[1,2,4,5,6,7,8,9],
                 names=['TIC','nuchar_20','nuchar_22','err_p','err_m','sigma','sigma_p','sigma_m'])

nuchar=df2['nuchar']
alpha0=df2['alpha0']
vsini=df1['vsini']
vmacro=df1['vmacro']
LogT=df1['LogT']
LogL=df1['LogL']

nuchar = np.array([float(item) for item in nuchar])
alpha0 = np.array([float(item) for item in alpha0])
vsini  = np.array([float(item) for item in vsini])
LogT   = np.array([float(item) for item in LogT])
LogL   = np.array([float(item) for item in LogL])
#vmacro = [float(item) for item in vmacro]

print (df1)
print (df2)
print (df3)

x = []
y = []
z = []
zz = []

k = 0
lzams = []
tzams = []
gzams = []
mask_beyond_tams=[]

for i in range(0,len(mods)):
  h=hs[i]
  model = h.model_number 
  logl = h.log_L   
  center_h1 = h.center_h1 
  logg = h.log_g   
  logt= h.log_Teff 
  ell = (10**logt)**4.0/(10**logg)
  ell=np.log10(ell/ell_sun)  

  zams=find_h(0.001,center_h1,model)
  tams=find_tams(center_h1,model)  

  # Create Lists  
  lzams.append(ell[zams])
  gzams.append(logg[zams])
  tzams.append(ell[zams])
  x.append(logt[zams:tams])
  y.append(ell[zams:tams])
  # z.append(h.Core_turnover_time[zams:]/24/3600)
  z.append(h.turnover_core[zams:tams]/24/3600)
  zz.append(10**h.log_R[zams:tams])
  mask_beyond_tams.append([logt[tams],ell[tams]]) # To patch region of interpolation outside tams  

x=array(list(flatten(x)))
y=array(list(flatten(y)))
z=array(list(flatten(z)))
zz = array(list(flatten(zz)))

numcols, numrows = 200, 200
xi = np.linspace(x.min(), x.max(), numcols)
yi = np.linspace(y.min(), y.max(), numrows)



#Interpolate convective turnover time
triang = tri.Triangulation(x,y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

# Interpolate radii
interpolator_r = tri.LinearTriInterpolator(triang, zz)


fig = plt.figure()
ax = fig.add_axes([0.13, 0.12, 0.84, 0.81])
ax.contour(xi, yi, zi, 14, colors='k')

# Patch contour region outside TAMS
patches = []   
polygon = matplotlib.patches.Polygon(mask_beyond_tams, closed=True,fill=True, color='white')
patches.append(polygon)
coll = PatchCollection(patches, zorder=2,color='white')
ax.add_collection(coll) 
    
ax.scatter(LogT,LogL,color='k',zorder=3)

ax.set_xlabel(lgteff)
ax.set_ylabel(lgell)
ax.set_xlim([4.83,4.1])

#ax.set_ylim([2.4,6.3])


#result=interpolator(xi,yi)
i_turnover=interpolator(LogT,LogL)
i_radii = interpolator_r(LogT,LogL)



print(len(i_turnover))

print(i_radii)
#plt.show()

# Copy Mask
mask=i_radii.mask

# Remove Masked elements (these are elements out of the interpolation range)
i_turnover = np.array(i_turnover[~i_turnover.mask])
i_radii = np.array(i_radii[~i_radii.mask])

# Remove elements for which we don't have interpolated values for radii and turnover
# First apply mask, then remove masked elemeents
i_vsini = np.ma.masked_array(vsini, mask=mask)
i_LogL = np.ma.masked_array(LogL, mask=mask)
i_LogT = np.ma.masked_array(LogT, mask=mask)
i_alpha0 = np.ma.masked_array(alpha0, mask=mask)

i_vsini = np.array(i_vsini[~i_vsini.mask])
i_LogL = np.array(i_LogL[~i_LogL.mask])
i_LogT = np.array(i_LogT[~i_LogT.mask])
i_alpha0 = np.array(i_alpha0[~i_alpha0.mask])

i = 0
print(len(i_turnover))
for i in range(0,len(i_turnover)-1):
    print(i_LogL[i],i_LogT[i],i_turnover[i],i_radii[i],i_alpha0[i])
    

rot_period = 2.0*3.1415*i_radii*rsun/(i_vsini*1e5)
rossby = rot_period/(i_turnover*24*3600)

#Now make the figure.
fig = plt.figure(figsize=(7.5, 5))
ax1 = fig.add_axes([0.03, 0.55, 0.42, 0.40])
ax2 = fig.add_axes([0.55, 0.55, 0.42, 0.40])
ax3 = fig.add_axes([0.03, 0.05, 0.42, 0.40])
ax4 = fig.add_axes([0.55, 0.05, 0.42, 0.40])
cax1 = fig.add_axes([0.98, 0.55, 0.02, 0.40])
cax2 = fig.add_axes([0.98, 0.05, 0.02, 0.40])


sizes = [2.6**i for i in LogL]  # Get unique values of i_LogL and compute corresponding marker sizes


fiducial  = (LogT > fiducial_T   - 0.2)*(LogT < fiducial_T + 0.2)
fiducial *= (LogL > fiducial_ell - 0.2)*(LogL < fiducial_ell + 0.2)

non_fiducial = np.logical_or(LogT <= fiducial_T   - 0.2, LogT >= fiducial_T + 0.2)
non_fiducial = np.logical_or(non_fiducial,LogL <= fiducial_ell - 0.2)
non_fiducial = np.logical_or(non_fiducial,LogL >= fiducial_ell + 0.2)



norm = mpl.colors.Normalize(vmin=4, vmax=4.7)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
norm2 = mpl.colors.Normalize(vmin=4.35, vmax=4.53)
sm2 = mpl.cm.ScalarMappable(norm=norm2, cmap=mpl.cm.viridis)

for i in range(LogL.size):
    if fiducial[i]:
        ax1.scatter(vsini[i],alpha0[i],s=sizes[i], c=sm.to_rgba(LogT[i]), alpha=0.7, edgecolors='k', linewidths=0.5)
        ax3.scatter(vsini[i],alpha0[i],s=sizes[i], c=sm2.to_rgba(LogT[i]), alpha=0.7, edgecolors='k', linewidths=0.5)
    else:
        ax1.scatter(vsini[i],alpha0[i],s=sizes[i], c=sm.to_rgba(LogT[i]), alpha=0.7, linewidth=0)

#for i in range(i_LogL.size):
#    if fiducial[i]:
#        ax1.scatter(i_vsini[i],i_alpha0[i],s=sizes[i], c=sm.to_rgba(i_LogT[i]), alpha=0.7, edgecolors='k', linewidths=0.5)
#        ax3.scatter(i_vsini[i],i_alpha0[i],s=sizes[i], c=sm2.to_rgba(i_LogT[i]), alpha=0.7, edgecolors='k', linewidths=0.5)
#    else:
#        ax1.scatter(i_vsini[i],i_alpha0[i],s=sizes[i], c=sm.to_rgba(i_LogT[i]), alpha=0.7, linewidth=0)
#
for ax in [ax1, ax3]:
    ax.set_yscale('log')
    ax.set_ylabel(r'$\alpha_0$ ($\mu$mag)')
    ax.set_xlabel(r'$v\,\sin\,i$ (km s$^{-1}$)')



sizes = [2.6**i for i in i_LogL]  # Get unique values of i_LogL and compute corresponding marker sizes
sc = ax2.scatter(rossby, i_alpha0, s=sizes, c=i_LogT, cmap='viridis')
handles, labels = sc.legend_elements("sizes")
ax2.cla()
for ax in [ax2, ax4]:
    ax.set_xlabel(r'Ro$_{\rm p}$ ')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(fiducial_Rop, c='k', lw=0.5)
    ax.fill_between([fiducial_Rop, 1e4], 1e-4, 1e5, facecolor=(0.5,0.5,0.5,0.1))
    ax.text(0.98, 0.08, 'Slow Rotators', ha='right', va='center', transform=ax.transAxes)

fiducial  = (i_LogT > fiducial_T   - 0.2)*(i_LogT < fiducial_T + 0.2)
fiducial *= (i_LogL > fiducial_ell - 0.2)*(i_LogL < fiducial_ell + 0.2)

non_fiducial = np.logical_or(i_LogT <= fiducial_T   - 0.2, i_LogT >= fiducial_T + 0.2)
non_fiducial = np.logical_or(non_fiducial,i_LogL <= fiducial_ell - 0.2)
non_fiducial = np.logical_or(non_fiducial,i_LogL >= fiducial_ell + 0.2)


for i in range(i_LogL.size):
    if fiducial[i]:
        ax2.scatter(rossby[i],i_alpha0[i],s=sizes[i], c=sm.to_rgba(i_LogT[i]), alpha=0.7, edgecolors='k', linewidths=0.5)
        ax4.scatter(rossby[i],i_alpha0[i],s=sizes[i], c=sm2.to_rgba(i_LogT[i]), alpha=0.7, edgecolors='k', linewidths=0.5)
        print(i_alpha0[i], i_LogT[i], i_LogL[i])
    else:
        ax2.scatter(rossby[i],i_alpha0[i],s=sizes[i], c=sm.to_rgba(i_LogT[i]), alpha=0.7, linewidth=0)


# Get six representative i_LogL values to display in the legend
unique_i_LogL = np.unique(i_LogL)
step = max(int(len(unique_i_LogL) / 6), 1)  # Compute step size for selecting representative values
selected_i_LogL = unique_i_LogL[::step][:6]

labels = [f'{i}' for i in selected_i_LogL]  # Create custom legend labels with selected i_LogL values
legend1 = ax1.legend(handles, labels, loc='lower right', title=lgell, ncol=2, frameon=False)

# Add legend for color map
cbar = plt.colorbar(sm, cax=cax1)
cbar.set_label(lgteff)

cbar = plt.colorbar(sm2, cax=cax2)
cbar.set_label(lgteff)


ax1.set_xlim(0, None)
ax1.set_ylim(4e0, 1e4)
ax2.set_ylim(4e0, 1e4)
ax3.set_ylim(1e-1, 3e2)
ax4.set_ylim(1e-1, 3e2)

#ax2.set_ylabel(r'$\log_{10} \alpha_0$')
ax2.set_xlim(2e-2, 3e1)
ax4.set_xlim(1e-1, 3e0)
ax3.set_xlim(0, 35)
#ax.text(1,3.5,'Bowman+2020',fontsize=22)



con1 = ConnectionPatch(xyA=(35,4e0), xyB=(35,3e2), coordsA='data', coordsB='data', axesA=ax1, axesB=ax3, color='grey', lw=1, alpha=0.5)
ax1.add_artist(con1)
ax1.plot([35, 35], [4e0, 8e0], color='grey', alpha=0.5)


con2 = ConnectionPatch(xyA=(1e-1,4e0), xyB=(1e-1,3e2), coordsA='data', coordsB='data', axesA=ax2, axesB=ax4, color='grey', lw=1, alpha=0.5)
ax2.add_artist(con2)
ax2.plot([1e-1, 1e-1], [4e0, 8e0], color='grey', alpha=0.5)
con2 = ConnectionPatch(xyA=(3e0,4e0), xyB=(3e0,3e2), coordsA='data', coordsB='data', axesA=ax2, axesB=ax4, color='grey', lw=1, alpha=0.5)
ax2.add_artist(con2)
ax2.plot([3e0, 3e0], [4e0, 8e0], color='grey', alpha=0.5)

#Fiducial rotating star
from palettable.colorbrewer.qualitative import Dark2_5 as cmap
ax3.scatter(21.7, 0.4, c=sm2.to_rgba(fiducial_T), marker='*', edgecolors='k', linewidth=1, s=200, zorder=100)
ax4.scatter(fiducial_Rop, 0.4, c=sm2.to_rgba(fiducial_T), marker='*', edgecolors='k', linewidth=1, s=200, zorder=100)

plt.savefig("fig11_rednoise_Ro.pdf",bbox_inches='tight', dpi=300)
plt.savefig("fig11_rednoise_Ro.png",bbox_inches='tight', dpi=300)
