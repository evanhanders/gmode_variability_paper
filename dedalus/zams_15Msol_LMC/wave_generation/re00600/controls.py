from collections import OrderedDict
from compstar.defaults.controls import *

#### Default star info / stratification choices
#path: path to MESA profile to load (searchings in $PATH and  d3_stars/stock_models
#r_bounds: basis boundaries -- Only works with L if you have core convection zone; how to generalize that?
#nr: radial resolution(s) -> length = len(r_bounds) - 1
star = OrderedDict()
star['path'] = '../../../../MESA/15msol_ZLMC/LOGS/profile47.data'
star['r_bounds'] = (0, '1.1L', '2L')
star['nr'] = (128, 96)
star['smooth_h'] = True


### Numerical choices
#Need to add a tag saying what the default diffusivity formalism is
#reynolds_target: Target reynolds number of simulation; higher needs more resolution
#ncc_cutoff: non-constant-coefficient cutoff needs to be small enough to resolve star non-constant coefficients but large enough that machine precision isn't a problem    
numerics = OrderedDict()
numerics['equations'] = 'FC_HD' #fully compressible hydro
numerics['reynolds_target'] = 2e3
numerics['prandtl'] = 1
numerics['ncc_cutoff'] = 1e-10
numerics['N_dealias'] = 1.5
numerics['L_dealias'] = 1.5

### Choices for dynamical simulation / initial value problem
# ntheta: Lmax - 1
# timestepper: SBDF2, RK222, or RK443
# safety: CFL safety factor (recommend 0.2 for SBDF2 or RK222; 0.4 for RK443)
# CFL_max_r: maximum radial value to measure CFL criterion In nondimensional units
# Stop conditions include 'wall_hours' and 'buoy_end_time' (sim time in heating timescales)
# sponge: if true, include a damping layer near the outside of the simulation.
# tau_factor: multiply damping term by this factor.
# A0: Initial noise amplitude.
dynamics = OrderedDict()
dynamics['ntheta'] =128 
dynamics['timestepper'] = 'SBDF2'
dynamics['safety'] = 0.2
dynamics['CFL_max_r'] = 1.05
dynamics['A0'] = 1e-8
dynamics['wall_hours'] = 23.5
dynamics['buoy_end_time'] = 200
dynamics['sponge'] = True 
dynamics['tau_factor'] = 1
dynamics['restart'] = '../re00300/final_checkpoint/final_checkpoint_s1.h5'

