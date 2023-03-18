"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_fluxes.py [options]

Options:
    --root_dir=<str>         Path to root run directory [default: ./]
    --data_dir=<str>         Name of data handler directory [default: profiles]
    --fig_name=<str>         Name of figure output directory & figures [default: flux_plots]
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
    sim_lum_r = f['lum_r_vals'][()]
    sim_lum   = f['sim_lum'][()]

op = config.handlers[data_dir]['tasks'][0]['type']
base_tasks = config.handlers[data_dir]['tasks'][0]['fields']
handler_tasks = []
for bn in bases_keys:
    handler_tasks += ['{}({}_{})'.format(op, t, bn) for t in base_tasks]

from palettable.colorbrewer.qualitative import Dark2_7
Dark2_7 = Dark2_7.mpl_colors
def luminosities(ax, dictionary, index):
    rs = []
    KEs = []
    PEs = []
    enths = []
    viscs = []
    conds = []
    for bn in bases_keys:
        rs.append(match_basis(dictionary['s2_avg(KE_lum_r_{})'.format(bn)], 'r'))
        KEs.append(dictionary['s2_avg(KE_lum_r_{})'.format(bn)][index].ravel())
        enths.append(dictionary['s2_avg(enth_lum_r_{})'.format(bn)][index].ravel())
        viscs.append(dictionary['s2_avg(visc_lum_r_{})'.format(bn)][index].ravel())
        conds.append(dictionary['s2_avg(cond_lum_r_{})'.format(bn)][index].ravel())
        PEs.append(dictionary['s2_avg(PE_lum_r_{})'.format(bn)][index].ravel())
    legend = False
    ax.plot(sim_lum_r.ravel(), sim_lum.ravel(), c='k', lw=3)
    for r, KE, enth, visc, cond, PE in zip(rs, KEs, enths, viscs, conds, PEs):
        ax.plot(r, KE, label='KE', c=Dark2_7[0])
        ax.plot(r, enth, label='enth', c=Dark2_7[1])
        ax.plot(r, visc, label='visc', c=Dark2_7[3])
        ax.plot(r, cond, label='cond', c=Dark2_7[4])
        ax.plot(r, PE, label='PE', c=Dark2_7[6])
        sum = KE +enth + visc + cond + PE
        ax.plot(r, sum, label='sum', c=Dark2_7[5], ls='--')
        if not legend:
            ax.legend()
            legend = True
    

# Create Plotter object, tell it which fields to plot
plotter = RolledProfilePlotter(root_dir, file_dir=data_dir, out_name=fig_name, roll_writes=roll_writes, start_file=start_file, n_files=n_files)
plotter.setup_grid(num_rows=1, num_cols=1, col_inch=float(args['--col_inch']), row_inch=float(args['--row_inch']))
plotter.add_line('r', luminosities, grid_num=0, needed_tasks=handler_tasks)
plotter.plot_lines()
