# gmode_variability_paper

This repository contains the scripts used to generate the data and figures presented in "The photometric variability of massive stars due to gravity waves excited by core convection" (Anders et al. 2023). See below for more details of the repository breakdown:

## MESA Stellar Models

This work relies on near-ZAMS MESA stellar models.
Working directories for each of the three stars studied in this project can be found in the MESA/ folder.
The MESA profile file for each star that we used in the work is found in the LOGS/ directory of the work directories, but the user can also recompute these models using MESA r21.12.1.
See [the MESA documentation of this release](https://docs.mesastar.org/en/release-r21.12.1/) for more details.

## GYRE g-mode pulsations

We use GYRE version 7.0 to identify the nonadiabatic gravity wave eigenvalues and eigenvectors associated with each of our MESA stellar models.
Working directories containing GYRE inlists for each of the stars can be found in the GYRE/ folder.
With GYRE installed, to solve for the eigenvalues, simply navigate into a working directory, open the 'run_gyre.sh' file, change the relevant paths to point to the MESA_DIR and GYRE_DIR on your local computer, then execute run_gyre.sh.
See [the GYRE documentation](https://gyre.readthedocs.io/en/v7.0/index.html) for more details.

### Computing GYRE magnitude perturbations and computing the transfer function

To compute the magnitude perturbation associated with each GYRE eigenmode, we use MSG version 1.1.2, [documented here](https://msg.readthedocs.io/en/v1.1.2/).
MSG relies on a spectroscopic grid file and passband file; to recreate our work, download the 'msg_grid_passbands.tar' tarball from (TODO) and unpack it into the data/ directory of this repository (after unpacking the tarball you should have a folder called data/MSG/).

The transfer function logic is implemented under-the-hood in the [evanhanders/compressible_stars](https://github.com/evanhanders/compressible_stars) directory, see 'Dedalus Simulations' section below.
To generate the transfer function with MSG and compressible_stars installed, execute `python3 transfer_gyre.py` inside of each of the working directories.
Output files are stored in the working directory's gyre_output/ folder.

After computing the transfer functions for the various stars, to compute the magnitude variability associated with waves in each star, navigate to the `GYRE/` directory and run `python3 generate_magnitude_spectra.py`, which will output data to the `data/dedalus/predictions/magnitude_spectra.h5` file.

## Dedalus Simulations

We use Dedalus version 3 to simulate fully compressible dynamics in a 3D spherical domain based upon MESA models.
The implementation of the fully compressible equations in Dedalus as well as the code used to build a Dedalus background state from a 1D stellar model is publicly available in the pip-installable repository [evanhanders/compressible_stars](https://github.com/evanhanders/compressible_stars), and we used version 0.1.0 of that repository in this work.

To run a Wave Generation simulation, navigate into a working directory (e.g., `dedalus/zams_15Msol_LMC/wave_generation/re00300`), then build the Dedalus fields associated with the stellar model (`python3 build_star.py`).
This may take a few minutes to complete.
I recommend looking at the figures in the star/ folder to ensure that the star seems to have been built properly.
To run the wave generation simulation on e.g., 64 cores, just run `mpirun -n 64 python3 compressible_dynamics.py`.
The resolution, timestepper choices, etc can be modified in the `controls.py` file.
The data which is output to file can be modified in the `outputs.py` file.
After the simulation has completed, various output operations can be performed, most of which rely on the pip-installable [evanhanders/plotpal](https://github.com/evanhanders/plotpal) repository, version 1.0.
To look at slices of dynamics in the equatorial plane, you could run `mpirun -n 64 python3 ball_shell_equatorial_slices.py`.
To measure the wave luminosity, first run `mpirun -n 64 post_ivp_SH_transform.py` then `python3 post_ivp_SH_wave_flux.py`.

Running a wave propagation simulation is similar.
First navigate into a working directory (e.g., `dedalus/zams_15Msol_LMC/wave_propagation/nr128`), then build the star and run the simulation just as you would for the Wave Generation simulation.
To measure the power spectrum at the outer boundary of the simulation, first run `mpirun -n 64 post_ivp_SH_transform.py` then `python3 post_ivp_SH_surface_power.py`.
To compute the eigenvalues and eigenvectors associated with a Dedalus simulation, run `python3 eigenvalues.py`.
To compute the Dedalus transfer function, run `python3 transfer.py` after completing the eigenvalue solve.

Data products associated with running all of these types of simulations are available online.
To make them locally accessible, download the 'dedalus_and_obs.tar' tarball from (TODO) and unpack it into the data/ directory of this repository (after unpacking the tarball you should have a folder called data/dedalus/).

## Producing Figures

The scripts used to generate all of the figures in the main manuscript and supplemental materials can be found in the `figures/` folder.

## Supplemental Information

An early draft of the supplemental information containing the precise implementation of equations can be found in the `supplemental/dense_supplemental_materials_v01.pdf` file.
A shorter version similar to the one published with the manuscript is available in the same directory.

## Sonification

To make our results more accessible, we have created a sonification of the data presented in the manuscript, available in the `sound/` directory.
Please refer to the `sound/gmode_sonification.pdf` file for a narrative description of these sonified results.
