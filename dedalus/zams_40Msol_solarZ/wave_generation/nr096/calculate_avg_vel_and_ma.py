"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    calculate_avg_vel_and_ma.py [options]

Options:
    --root_dir=<str>         Path to root run directory [default: ./]
    --data_dir=<str>         Name of data handler directory [default: scalars]
    --fig_name=<str>         Name of figure output directory & figures [default: fundamentals]
    --start_file=<int>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>          Total number of files to plot
    --roll_writes=<int>      Number of writes over which to take average
    --dpi=<int>              Image pixel density [default: 200]

    --col_inch=<float>        Figure width (inches) [default: 6]
    --row_inch=<float>       Figure height (inches) [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
import h5py
import numpy as np

from plotpal.profiles import RolledProfilePlotter
from plotpal.scalars import ScalarPlotter
from plotpal.file_reader import match_basis

from compstar.defaults import config
from compstar.dedalus.parser import name_star

# Read in master output directory
root_dir    = args['--root_dir']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

roll_writes = args['--roll_writes']
if roll_writes is not None:
    roll_writes = int(roll_writes)

bases_keys = []
for i, resolution in enumerate(config.star['nr']):
    if i == 0:
        bases_keys.append('B')
    else:
        bases_keys.append('S{}'.format(i))

star_dir, star_file = name_star()
with h5py.File(star_file, 'r') as f:
    r_stitch = f['r_stitch'][()]
    rho0 = np.exp(f['ln_rho0_B'][()])
    pom0 = f['pom0_B'][()]
    r    = f['r_B'][()]
    L_nd     = f['L_nd'][()]
    tau_nd   = f['tau_nd'][()]
    nu       = f['nu_diff_B'][()].ravel().min() * L_nd**2 / tau_nd
    dr = np.gradient(r, axis=-1)

c2_ball = (L_nd/tau_nd)**2*np.sum((4*np.pi*r**2*dr*pom0)[r < 1]) / (4*np.pi*1**3/3)

# Create Plotter object, tell it which fields to plot
#plotter = RolledProfilePlotter(root_dir, file_dir=data_dir, out_name=fig_name, roll_writes=roll_writes, start_file=start_file, n_files=n_files)
plotter = ScalarPlotter(root_dir, data_dir, fig_name, start_file=start_file, n_files=n_files, roll_writes=roll_writes)

u2 = []
ma2 = []
while plotter.writes_remain():

    dsets, ni, a, b = plotter.get_dsets(['vol_avg(u_squared_B)', 'vol_avg(Re_B)'])
#    dsets, ni = plotter.get_dsets(['s2_avg(KE_lum_r_B)'])
#    KE = dsets['s2_avg(KE_lum_r_B)'][ni]
#    u2_ball = 2*np.sum((4*np.pi*r**2*dr*KE/rho0)[r < 1]) / (4*np.pi*1**3/3)
    u2_cgs = (L_nd/tau_nd)**2*dsets['vol_avg(u_squared_B)'][ni]*(1.1/1.0)**3 #adjust volume average to just account for CZ
    u2.append(u2_cgs.ravel())
    ma2.append(u2_cgs.ravel()/c2_ball)

u = (np.abs(np.array(u2)))**(1/2)
ma = (np.abs(np.array(ma2)))**(1/2)
#print(u2, u)

print('u: {:.3e} // ma: {:.3e} // re: {:.3e}'.format(np.mean(u[u.size // 2:]), np.mean(ma[ma.size //2:]), np.mean(u[u.size // 2:])*L_nd/nu))

