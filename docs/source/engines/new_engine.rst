Implementing a New Engine
==================

..  toctree::
    :maxdepth: 1

CP2K Specific Inputs
--------------------
In addition to the inputs required by all :doc:`engines <engines>`, the CP2K specific inputs are listed below.

``cp2k_inputs`` - path of the CP2K input file
    The path to a well formed CP2K input file that describes the system you want to run. This original file will not be
    modified. An exception will be thrown if this does not pass the linter provided by
    `cp2k-input-tools <https://github.com/cp2k/cp2k-input-tools>`_. All existing parameters will be used, except:

    #. Any velocity generation - Aimless shooting will generate its own starting velocities
    #. The trajectory print frequency - This is modified for individual shootings to extract to relevant :math:`\Delta t` positions

    ..  warning::
        At the time being, the coordinates of all atoms need to be specified in the ``&COORD`` section of the input file
        in order to define the present atoms. The coordinates do not need to reflect a transition state, but the atoms
        and their indices are fixed by this.

CP2K Outputs
------------
If CP2K exits with a non-zero exit code, the output file is copied from the engine's working directory with the name
``{projname}_FATAL.out``, where ``projname`` is the root name of that shooting point. Additionally, all std out and std
err is logged as a warning, with a note that this simulation failed. Note that this is not necessarily terminal, as
multiple velocities are resampled. If many of these are occurring, something is likely wrong.

CP2K .out files are parsed for warnings, barring those about filename length. If any are found, they are logged.

Additional Logging
------------------
Setting ``log_level = DEBUG`` will log the individual commands used for each simulation so you can ensure they match
your expectations.
