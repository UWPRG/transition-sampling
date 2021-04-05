# Example of Aimless Shooting with CP2K

This provides a basic idea of how the components of the `transition_sampling` package fit together to
run aimless shooting and calculate CVs for resultant shooting points. 
The specific system is a chloromethane Sn2 reaction with PM6 theory. 

## Directory Structure
 - This directory (`example`) - Working directory that aimless shooting is launched from using 
   `main.py`
 - `input_files` - Files that are required to start aimless shooting
 - `result_files` - Files generated as output from aimless shooting
 - `engine_working_dir` - Empty directory for CP2K to put its temporary files. We avoid using a 
   directory in `/tmp/` because it often breaks when distributed over multiple nodes.
   
## Files in This Directory
 - `sourceme.sh` - Sets up python environment, CP2K, and plumed binaries. Adjust as needed.
 - `job.sh` - Script to launch `main.py` with `slurm`. **NOTE: CURRENTLY MUST USE AN INTERACTIVE NODE
   to have parallel cp2k simulations i.e. don't use this**
 - #### `main.py` - Python script to be executed. Puts everything together.

## `main.py` Walkthrough

The options within `main.py` are well documented with comments, but the general procedure is to 
 1. Gather all the inputs that will be needed
 2. Create a `CP2KEngine` with the engine inputs
 3. Create an `AimlessShootingDriver` with that engine and aimless shooting inputs
 4. Run that driver with `.run`

Here it is set up to run 4 parallel aimless shootings, each until at least `n_points` (5)
accepted shooting points are generated. If a shooting point is rejected, it's velocity is 
regenerated `n_vel_tries` (5) times before giving up on that shooting point. Each of these rejects
is still recorded. A random state from known accepted shooting points is then chosen, and the
process repeated. This process (retrying with a random known accepted) can occur `n_state_retries`
(3) before completely failing.
    
Each individual aimless shooting generates a pair of files:
  - A `.xyz` that contains that `xyz` structure of each shooting point concatanted into one
    file
  - A `.csv` that contains metadata, such as accepeted or rejected, for each structure. The `index`
    column gives the corresponding structure in the `.xyz`
    
Since there are 4 parallel aimless shootings, 4 pairs of files will be generated, each with a digit
appended to the result name. In addition, a 'master' file pair without a digit will be generated 
that holds the results of all parallel shootings together, continually updated. This is typically
what's used when calculating CVs.

 5. Create a `PlumedDriver` with the plumed inputs
 6. Use the driver and the results from the aimless shooting to calculate CVs for each state
    
After calculating the CVs, likelihood maximization can be performed using the metadata
`csv`. (Soon to come)

