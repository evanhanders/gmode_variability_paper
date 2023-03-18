"""
Script for plotting traces of evaluated scalar quantities vs. time.

Usage:
    plot_scalar_traces.py [options]

Options:
    --root_dir=<str>                    Root data directory [default: .]
    --data_dir=<dir>                    Name of data handler directory [default: scalars]
    --out_name=<out_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]

    --roll_writes=<r>                   Number of writes to roll a comparison average over
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.scalars import ScalarFigure, ScalarPlotter

# Read in master output directory
root_dir    = args['--root_dir']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

out_name    = args['--out_name']
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
dpi         = int(args['--dpi'])
if n_files is not None: 
    n_files = int(n_files)

roll_writes = args['--roll_writes']
if roll_writes is not None:
    roll_writes = int(roll_writes)

figs = []

# Nu vs time
fig2 = ScalarFigure(num_rows=2, num_cols=1, col_inch=6, fig_name='energy_fluc')
fig2.add_field(0, 'integ(FlucE)')
fig2.add_field(0, 'integ(KE)')
fig2.add_field(0, 'integ(IE1)')
fig2.add_field(0, 'integ(PE1)')
fig2.add_field(1, 'integ(FlucE)')
figs.append(fig2)

fig3 = ScalarFigure(num_rows=1, num_cols=1, col_inch=6, fig_name='mass')
fig3.add_field(0, 'integ(rho_fluc)')
figs.append(fig3)



# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir=data_dir, out_name=out_name, start_file=start_file, n_files=n_files, roll_writes=roll_writes)
plotter.load_figures(figs)
plotter.plot_figures(dpi=dpi)
plotter.plot_convergence_figures(dpi=dpi)
