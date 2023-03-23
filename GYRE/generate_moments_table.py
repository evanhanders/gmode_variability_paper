"""
Calculate transfer function to get surface response of convective forcing.
Outputs a function which, when multiplied by sqrt(wave flux), gives you the surface response.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser

import mesa_reader as mr
from matplotlib.patches import ConnectionPatch
from palettable.colorbrewer.qualitative import Dark2_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

def tex_exponential(number):
    """ Returns a tex-math version of an exponential string """
    neg = number < 0
    log10 = np.log10(np.abs(number))
    power = int(np.floor(log10))
    stem  = 10**(log10 - power)
    if neg:
        string = r'$- '
    else:
        string = '$ '
    if power != 0:
        string += '{:>4.1f}'.format(stem) + r' \times 10^{' + '{:>2d}'.format(power) + r'}$'
    else:
        string += '{:>4.1f}'.format(stem) + '{:15s}'.format('') + r'$'
    return string



#Calculate transfer functions
eig_dir = 'gyre_output'
output_file = 'moments_table.csv'

#star_dirs = ['3msol', '15msol', '40msol']
star_dirs = ['03msol_Zsolar', '15msol_ZLMC', '40msol_Zsolar']
Lmax = 15
ell_list = np.arange(1, Lmax+1)
I0s = []
Ys  = []
Is  = []
for i, sdir in enumerate(star_dirs):
    I_vals = []
    Y_vals = []
    for ell in ell_list:
        print(sdir, " ell = %i" % ell)
        with h5py.File('{:s}/{:s}/ell{:03d}_eigenvalues.h5'.format(sdir, eig_dir, ell), 'r') as f:
            I_0 = f['I_0_Red'][()]
            I_l = f['I_l_Red'][()]
            dI_l_dlnTeff = f['dI_l_dlnTeff_Red'][()]
            dI_l_dlng    = f['dI_l_dlng_Red'][()]
            Y_l = f['Y_l'][()]
        if ell == 1:
            I0s.append(I_0)
        Y_vals.append(Y_l)
        I_vals.append((I_l/I_0, dI_l_dlnTeff/I_0, dI_l_dlng/I_0))
    print(sdir, ', I_0: {:.3e}'.format(I_0))
    Ys.append(Y_vals)
    Is.append(I_vals)

csv = open(output_file, 'w')
string = ""
tex_string = ""
for i, ell in enumerate(ell_list):
    header = '{:>03s},  '.format('ell')
    string = '{:>03d},  '.format(ell)
    tex_string = '{:>2d}  & '.format(ell)
    header += '{:>6s},    '.format('Y_ell')
    string += '{:>6.04f},    '.format(Ys[0][i])
    tex_string += '{:>6.03f} & '.format(Ys[0][i])
    for j in range(len(star_dirs)):
        string += '{:>14.02e}, {:>23.02e}, {:>20.02e},    '.format(*Is[j][i])
        header += '{:>14s}, '.format(star_dirs[j]+' I_l/I_0')
        header += '{:>23s}, '.format(star_dirs[j]+' dI_l_dlnTeff/I_0')
        header += '{:>20s},    '.format(star_dirs[j]+' dI_l_dlng/I_0')
        tex_string += '{:>10s} & {:>10s} & {:>10s} &    '.format(*(tex_exponential(n) for n in Is[j][i]))

    header = header[:-5] + '\n'
    string = string[:-5] + '\n'
    tex_string = tex_string[:-5] + r' \\'

    if i == 0:
        print(header)
        csv.write(header)
    csv.write(string)
    print(tex_string)
csv.close()

