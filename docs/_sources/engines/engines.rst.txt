MD Engines
==========
Below are all currently supported engines to generate shooting points with.

..  toctree::
    :maxdepth: 2
    :caption: Available Engines:

    CP2K <cp2k>

Intro to Engines
----------------
All engines share some common features that need to be specified in
the inputs. In particular, these are

``engine`` -  the engine name
    String defined by the engine implementation of the method
    :func:`~transition_sampling.engines.abstract_engine.AbstractEngine.get_engine_str`

    Example:

    * ``"engine": "cp2k"`` - use the cp2k engine implementation

``cmd`` - the command line command to invoke the engine
    The string you would use to run the MD engine from the command line normally, leading
    up to the path of the engine executable. Any additional commands (such as ``mpirun``)
    can be prepended, but the final argument should be the engine executable.

    ..  warning::
        You should not rely on relative paths of executables. They should either be
        in the PATH environment variable when invoked, or the full path should be given.

    Examples of valid commands:

    * ``"cmd": "/path/to/cp2k/exec"`` - just use the standard cp2k executable.
    * ``"cmd": "mpirun -n 2 -genv 1 /path/to/cp2k/exec"`` -  Run cp2k with MPI with two processes

        ..  note::
            Since the aimless shooting algorithm runs the forwards and backwards trajectories
            in parallel, the ideal number of MPI processes is equal to ``num_cores/2``, so
            cores are distributed evenly across the two trajectories.

    Examples of **invalid** commands:

    * ``"cmd": "/path/to/cp2k/exec -i my_input_file -o my_output_file"`` - arguments after the engine executable should be left to the module

