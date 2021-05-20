"""
Example of aimless shooting with cp2k from initial guesses all the way to CV
generation.
"""

# Disable some logging stuff for the time being
import logging
logger = logging.getLogger()
logger.disabled = True

from transition_sampling.engines import CP2KEngine
from transition_sampling.colvar import PlumedDriver
from transition_sampling.algo import AimlessShootingDriver

# cp2k input to use
CP2K_INPUT = "./input_files/md.inp"

# Delta t to use for aimless shooting in fs
DELTA_T = 5

# plumed file containing committor basin inputs
PLUMED_COMMITTOR = "./input_files/plumed.dat"

# plumed file with CVs to be calculated after simulations are done
PLUMED_COLVARS = "./input_files/plumed_colvar.dat"

# Command to launch a single simulation of CP2K with. For this system,
# using 1 mpi process therefore, should run on a job with at least 8 cores.
# (4 parallel shootings * 2 simulations each * 1 core/simulation = 8 cores)
CP2K_CMD = "mpirun -n 1 -genv OMP_NUM_THREADS 1 /gscratch/pfaendtner/codes/cp2k/cp2k/cp2k/exe/Linux-x86-64-intel/cp2k.psmp"

# Plumed binary
PLUMED_CMD = "mpirun plumed"

# Inputs for the cp2k engine
INPUTS = {"engine": "cp2k",
          "cp2k_inputs": CP2K_INPUT,
          "md_cmd": CP2K_CMD,
          "plumed_file": PLUMED_COMMITTOR,
          "delta_t": DELTA_T}

# Directory containing initial starts
STARTS_DIR = "./input_files/initial_starts"

# Base name of all results files. `result_files` directory must already exist,
# all results will be placed in it
AIMLESS_OUTPUT_NAME = "./result_files/results"
COLVAR_OUTPUT_NAME = "./result_files/result_COLVAR"

# Directory where CP2K will put all the files it generates
# WARNING: I think you need the full path here, will look more later
# CHANGE THIS
ENGINE_DIR = "/FULL/FILE/PATH/transition-sampling/example/engine_working_dir/"


def main():
    # Create the engine
    engine = CP2KEngine(INPUTS, ENGINE_DIR)

    # Create an AimlessShooting class. Will write to results.xyz and
    # results.csv No acceptor defaults to accepting so long as both committed
    # basins are different. Can create your own acceptor here if desired (see
    # MultiBasinAcceptor)
    algo = AimlessShootingDriver(engine, STARTS_DIR, AIMLESS_OUTPUT_NAME,
                                 acceptor=None)

    # Run 4 aimless shooting until generating 5 accepteds. 3 state retries
    # and 5 velocity retries. Keywords are required for these arguments.
    algo.run(4, n_points=5, n_state_tries=3, n_vel_tries=5)

    # After running simulations, run the CV calculation on the results file.
    # Output CVs to the COLVAR_OUTPUT_NAME
    plumed_driver = PlumedDriver(PLUMED_CMD)
    plumed_driver.run(PLUMED_COLVARS, f"{AIMLESS_OUTPUT_NAME}.xyz",
                      f"{AIMLESS_OUTPUT_NAME}.csv", COLVAR_OUTPUT_NAME)


if __name__ == "__main__":
    main()
