"""
Scratch file for testing running cp2k in parallel from python
"""
import os
import subprocess
import tempfile
import time
import shutil
from cp2k_input_tools.parser import CP2KInputParser
from cp2k_input_tools.generator import CP2KInputGenerator

# executable sourced from /gscratch/pfaendtner/lflo/scripts/activate_cp2k.sh
# mpirun is also sourced from here
CP2K_EXEC = "/gscratch/pfaendtner/codes/cp2k/cp2k/cp2k/exe/Linux-x86-64-intel/cp2k.psmp"
INP_FILE = "argon.inp"

# tasks_per_node * num_nodes
N_TASKS = int(os.environ["SLURM_NTASKS"])
N_TASKS_PER_INST = str(N_TASKS // 2)


# change the set project name to this by parsing the input file
PROJ_NAME = "argon_proj"
cp2k_parser = CP2KInputParser()
cp2k_gen = CP2KInputGenerator()

with open(INP_FILE) as f:
    cp2k_inputs = cp2k_parser.parse(f)

cp2k_inputs["+global"]["project_name"] = f"{PROJ_NAME}_fwd"

# Uncomment to throw an error in input file
# calc.CP2K_INPUT.MOTION.MD.Ensemble = ""

# write the modified input file to a temporary directory
temp_dir = tempfile.TemporaryDirectory(prefix="aimless_cp2k_")

with open(f"{temp_dir.name}/{INP_FILE}_fwd", "w") as out:
    for line in cp2k_gen.line_iter(cp2k_inputs):
        out.write(f"{line}\n")

proc_fwd = subprocess.Popen(["mpirun", "-n", N_TASKS_PER_INST, "-genv",
                            "OMP_NUM_THREADS", "1", CP2K_EXEC, "-i",
                             f"{INP_FILE}_fwd", "-o", "fwd.out"],
                            cwd=temp_dir.name, stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE)

# Create a new project with the reverse tag and fork a new proc
cp2k_inputs["+global"]["project_name"] = f"{PROJ_NAME}_rev"
with open(f"{temp_dir.name}/{INP_FILE}_rev", "w") as out:
    for line in cp2k_gen.line_iter(cp2k_inputs):
        out.write(f"{line}\n")

proc_rev = subprocess.Popen(["mpirun", "-n", N_TASKS_PER_INST, "-genv",
                            "OMP_NUM_THREADS", "1", CP2K_EXEC, "-i",
                             f"{INP_FILE}_rev", "-o", "rev.out"],
                            cwd=temp_dir.name, stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE)


# Wait until both procs are done
while proc_fwd.poll() is None or proc_rev.poll() is None:
    if proc_fwd.poll() is None:
        print("Foward Running")
    if proc_rev.poll() is None:
        print("Reverse Running")
    time.sleep(.3)

# TODO: Under what conditons is the error code set? Warnings? Fatals?
if proc_fwd.returncode != 0:
    stdout, stderr = proc_fwd.communicate()

    # Think we need to just look through the output file :(

    # Tested so far: fatal in input prints to stdout, nothing to stderr, and
    # set the error code.
    print(stdout.decode('ascii'))
    shutil.copyfile(temp_dir.name + "/fwd.out",
                    "fwd_failed.out")
    raise Exception("Forward process failed")

if proc_rev.returncode != 0:
    stdout, stderr = proc_rev.communicate()

    print(stdout.decode('ascii'))
    shutil.copyfile(temp_dir.name + "/rev.out",
                    "rev_failed.out")
    raise Exception("Reverse process failed")

# Copy the coordinates (results for now) to the starting directory
# and clean up the temp
shutil.copyfile(temp_dir.name + "/argon_proj_fwd-pos-1.xyz",
                "argon_proj_fwd-pos-1.xyz")
shutil.copyfile(temp_dir.name + "/argon_proj_rev-pos-1.xyz",
                "argon_proj_rev-pos-1.xyz")

temp_dir.cleanup()
print("success")
