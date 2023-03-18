"""
d3 script for a dynamical simulation of compressible convection in a star.
"""
import h5py
import numpy as np
from mpi4py import MPI

import dedalus.public as d3
from compstar.defaults import config
from compstar.dedalus.compressible_functions import SphericalCompressibleProblem
from compstar.dedalus.outputs import initialize_outputs, output_tasks
from compstar.dedalus.parser import name_star
from compstar.tools.general import one_to_zero, zero_to_one

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Read options
    out_dir, ncc_file = name_star()
    out_dir = './'
    
    ntheta = config.dynamics['ntheta']
    nphi = 2*ntheta
    L_dealias  = config.numerics['L_dealias']
    N_dealias  = config.numerics['N_dealias']
    ncc_cutoff = config.numerics['ncc_cutoff']
    dtype      = np.float64
    sponge     = config.dynamics['sponge'] #If True, use damping layer.

    # Parameters
    resolutions = []
    for nr in config.star['nr']:
        resolutions.append((nphi, ntheta, nr))

    # Read in domain bounds 
    if ncc_file is not None:
        with h5py.File(ncc_file, 'r') as f:
            r_stitch = f['r_stitch'][()]
            r_outer = f['r_outer'][()]
    else:
        raise ValueError("NCC file cannot be 'None'")
    logger.info('r_stitch: {} / r_outer: {:.2f}'.format(r_stitch, r_outer))

    # Simulation end conditions
    wall_hours    = config.dynamics['wall_hours']
    buoy_end_time = config.dynamics['buoy_end_time']

    # rotation -- explicitly turned off.
    do_rotation = False
    if 'rotation_time' in config.dynamics.keys():
        do_rotation = True
        rotation_time = config.dynamics['rotation_time']
        dimensional_Omega = 2*np.pi / rotation_time  #radians / day [in MESA units]

    # Can specify to read initial conditions from a file
    if 'restart' in config.dynamics.keys():
        restart = config.dynamics['restart']
        A0 = None
    else:
        restart = None
        A0 = config.dynamics['A0']

    # Timestepper
    ts = None
    if 'timestepper' in config.dynamics.keys():
        if config.dynamics['timestepper'] == 'SBDF4':
            logger.info('using timestepper SBDF4')
            ts = d3.SBDF4
        elif config.dynamics['timestepper'] == 'RK222':
            logger.info('using timestepper RK222')
            ts = d3.RK222
        elif config.dynamics['timestepper'] == 'RK443':
            logger.info('using timestepper RK443')
            ts = d3.RK443
    if ts is None:
        logger.info('using default timestepper SBDF2')
        ts = d3.SBDF2

    # Processor mesh -- works if ncpu is a power of 2.
    ncpu = MPI.COMM_WORLD.size
    mesh = None
    if 'mesh' in config.dynamics.keys():
        mesh = config.dynamics['mesh']
    else:
        log2 = np.log2(ncpu)
        if log2 == int(log2):
            mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

    # Create sponge function template 
    if len(r_stitch) > 0:
        sponge_function = lambda r: zero_to_one(r, r_outer - 0.15, 0.07)
    else:
        sponge_function = lambda r: 0*r

    # Construct bases, domains, fill NCC structure, etc.
    stitch_radii = r_stitch
    radius = r_outer
    compressible = SphericalCompressibleProblem(resolutions, stitch_radii, radius, ncc_file, dealias=(L_dealias, L_dealias, N_dealias), dtype=dtype, mesh=mesh, sponge=sponge, do_rotation=do_rotation, sponge_function=sponge_function)
    compressible.make_fields()
    variables, timescales = compressible.fill_structure()
    compressible.set_substitutions()
    variables = compressible.namespace

    # Report timescales.
    t_kep, t_heat, t_rot = timescales
    logger.info('timescales -- t_kep {}, t_heat {}, t_rot {}'.format(t_kep, t_heat, t_rot))

    # Update sponge field with user-specified boosting factor; default = 1.
    if sponge:
        tau_factor    = config.dynamics['tau_factor']
        for i, bn in enumerate(compressible.bases.keys()):
            variables['sponge_{}'.format(bn)]['g'] *= tau_factor

    # Put nccs and fields into locals() for easier access throughout rest of file.
    locals().update(variables)

    # Build Problem
    prob_variables = compressible.get_compressible_variables()
    problem = d3.IVP(prob_variables, namespace=locals())
    problem = compressible.set_compressible_problem(problem)
    logger.info("Problem built")

    # Build Solver
    solver = problem.build_solver(ts, ncc_cutoff=config.numerics['ncc_cutoff'])
    solver.stop_sim_time = buoy_end_time*t_heat
    solver.stop_wall_time = wall_hours * 60 * 60
    logger.info("Solver built")

    # Initial conditions / Checkpoint
    write_mode = 'overwrite'
    timestep = None
    if restart is not None:
        write, timestep = solver.load_state(restart)
        timestep *= 0.5 #bootstrap safely
        write_mode = 'append'
    else:
        # Noise Initial conditions in the inner 80% of the simulation domain.
        for bk in compressible.bases_keys:
            variables['s1_{}'.format(bk)].fill_random(layout='g', seed=42, distribution='normal', scale=A0)
            variables['s1_{}'.format(bk)].low_pass_filter(scales=0.5)
            variables['s1_{}'.format(bk)]['g'] *= np.sin(variables['theta1_{}'.format(bk)])
            variables['s1_{}'.format(bk)]['g'] *= one_to_zero(variables['r1_{}'.format(bk)], r_outer*0.8, width=r_outer*0.1)
            #make perturbations pressure-neutral.
            variables['ln_rho1_{}'.format(bk)].change_scales(compressible.bases[bk].dealias)
            variables['ln_rho1_{}'.format(bk)]['g'] = (variables['s1_{}'.format(bk)]/variables['Cp']).evaluate()['g']

    # Setup output tasks based on outputs.py
    analysis_tasks, even_analysis_tasks = initialize_outputs(solver, compressible.coords, variables, compressible.bases, timescales, out_dir=out_dir)
    logger.info('outputs initialized')

    # Setup flow tools for simulation text output during timesteps..
    from dedalus.extras.flow_tools import GlobalFlowProperty
    flow = GlobalFlowProperty(solver, cadence=1)
    for bn, basis in compressible.bases.items():
        re = eval(output_tasks['Re'].format(bn), dict(solver.problem.namespace))
        flow.add_property(d3.Grid(re), name='Re_{}'.format(bn))

    #CFL setup
    #can limit maximum radius for CFL evaluation
    safety = config.dynamics['safety']
    if 'CFL_max_r' in config.dynamics.keys():
        CFL_max_r = config.dynamics['CFL_max_r'] 
    else:
        CFL_max_r = np.inf

    heaviside_cfl = compressible.dist.Field(name='heaviside_cfl', bases=compressible.bases['B'])
    heaviside_cfl['g'] = 1
    if np.sum(r1_B > CFL_max_r) > 0:
        heaviside_cfl['g'][:,:, r1_B.flatten() > CFL_max_r] = 0
    heaviside_cfl = d3.Grid(heaviside_cfl).evaluate()

    max_dt = t_kep
    initial_max_dt = 0.05*t_heat 
    while initial_max_dt < max_dt:
        max_dt /= 2
    if timestep is None:
        timestep = initial_max_dt
    my_cfl = d3.CFL(solver, timestep, safety=safety, cadence=1, max_dt=initial_max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
    my_cfl.add_velocity(heaviside_cfl*u_B)
    logger.info('cfl constructed') 

    # Main loop
    start_iter = solver.iteration
    Re0 = 0
    try:
        while solver.proceed:
            effective_iter = solver.iteration - start_iter
            timestep = even_analysis_tasks.compute_timestep(my_cfl) #compute nice timestep size for even time sampling, if required.
            solver.step(timestep)

            if solver.iteration % 10 == 0:
                Re0 = flow.max('Re_B')
                this_str = "iteration = {:08d}, t/th = {:f}, t = {:f}, timestep = {:f}, maxRe = {:.4e}".format(solver.iteration, solver.sim_time/t_heat, solver.sim_time, timestep, Re0)
                logger.info(this_str)

            if np.isnan(Re0):
                logger.info('exiting with NaN')
                break
    except:
        import traceback
        logger.info('something went wrong in main loop.')
        logger.info(traceback.format_exc())
        raise
    finally:
        solver.log_stats()

        logger.info('making final checkpoint')
        fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_heat)
        fcheckpoint.add_tasks(solver.state, layout='g')
        solver.step(timestep)


