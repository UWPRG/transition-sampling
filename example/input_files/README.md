# Input Files for Aimless Shooting with CP2K

This directory contains all the files that are required for running aimless shooting with
CP2K.

## Directory Structure
- This directory (`input_files`) - Required input files. 
- `initial_starts` - Directory containing only `xyz` files of starting structures to try.

## Files in This Directory
- `md.inp` - Full CP2K input for this system. Highly customizable.
- `plumed.dat` - Plumed file describing committor basins. Must have at least two basins,
  can be either single line or multiline format.
- `plumed_colvar.dat` - Plumed file to calculate CVs with. All named CVs will 
   be calculated and added to the output colvar file. Any other print statement already
   present is respected and may result in duplicate files.

## Notes About `md.inp`
Some specific things for the CP2K input.
 ### Required Fields:
   - `&TEMPERATURE` Any CP2K units
   - `&TIMESTEP` - Any CP2K units
   - `&CELl` - Only supports `ABC` for the time being
   - `&COORD` - Required for the time being. Must have an entry for each atom in the
      system and each atom type must correspond exactly to those provided in the 
      `initial_starts` `xyz` files.
     
  ### Not Allowed
   - Comments with a `'` or `"` character in them. This causes an error in the upstream
     `CP2KOutputTools` package. 
     [See current issue.](https://github.com/UWPRG/transition-sampling/issues/21)