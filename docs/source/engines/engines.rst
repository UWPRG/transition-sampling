MD Engines
==========
Below are all currently supported engines to generate shooting points with.

..  toctree::
    :maxdepth: 1
    :caption: Available Engines:

    CP2K <cp2k>
    GROMACS <gromacs>
    Implementing a New Engine <new_engine>

.. warning::
    Constant pressure (changing box sizes) is not currently supported

Intro to Engines
----------------
All engines share some common features that need to be specified in
the inputs. In particular, these are

``engine`` -  the engine name
    String defined by the engine implementation of the method
    :func:`~transition_sampling.engines.abstract_engine.AbstractEngine.get_engine_str`

    Example:

    * ``"engine": "cp2k"`` - use the cp2k engine implementation

``md_cmd`` - the command line command to invoke the engine
    The string you would use to run the MD engine from the command line normally, leading
    up to the path of the engine executable. Any additional commands (such as ``mpirun``)
    can be prepended, but the final argument should be the engine executable.

    If your engine command is more complicated, such being wrapped in quotes or
    needing multiple commands chained together, you can use argument substitution.
    This allows exact specification of where arguments should be placed, **and
    executes in shell mode**, allowing operators like ``&&`` or ``|`` that would
    otherwise produce bad input. This is done placing the ``%CMD_ARGS%`` where
    the engine arguments should be substituted

    ..  warning::
        You should not rely on relative paths of executables. They should either be
        in the PATH environment variable when invoked, or the full path should be given.

    Examples of valid commands:

    * ``"md_cmd": "/path/to/cp2k/exec"`` - just use the standard cp2k executable.
    * ``"md_cmd": "mpirun -n 2 -genv 1 /path/to/cp2k/exec"`` -  Run cp2k with MPI with two processes
    * ``"md_cmd": "echo 'hello there' && /path/to/cp2k/exec %CMD_ARGS%"`` - Running chained commands requires using argument substitution
    * ``"md_cmd": "sbatch -n 2 --wrap '/path/to/cp2k/exec %CMD_ARGS%'`` - Exact placement of arguments requires using argument substitution

        ..  note::
            Since the aimless shooting algorithm runs the forwards and backwards trajectories
            in parallel, the ideal number of MPI processes is equal to ``num_cores/2``, so
            cores are distributed evenly across the two trajectories. (Assuming resources are
            shared).

    Examples of **invalid** commands:

    * ``"md_cmd": "/path/to/cp2k/exec -i my_input_file -o my_output_file"`` - arguments after the engine executable should be left to the module
    * ``"md_cmd": "echo 'hello there' && /path/to/cp2k/exec"`` - Shell specific operators such as ``&&`` will not work without argument substitution

    .. hint::
        Using ``aimless_driver --log-level DEBUG`` will log the commands used
        to launch each trajectory and can be useful in debugging.

``plumed_file`` - the path to the `PLUMED <https://www.plumed.org/>`_ file with the committor basins for the system
    The plumed file that defines the basins associated with a complete reaction. This should
    stop the system when the simulation commits to a basin (``NOSTOP=off``, the default) and the output
    file name should not be defined (``FILE`` not specified). See
    `COMMITOR <https://www.plumed.org/doc-v2.5/user-doc/html/_c_o_m_m_i_t_t_o_r.html>`_ for more details.

    Suppose we had a system with two atoms. The following PLUMED would consider the ending reaction states to be when
    the atoms were closer than 0.15nm or further than 1nm apart::

        d1:  DISTANCE ATOMS=1,2
        COMMITTOR ...
          ARG=d1
          BASIN_LL1=0
          BASIN_UL1=.15
          BASIN_LL2=1
          BASIN_UL2=100
        ... COMMITTOR

    .. note::
        Take care that PLUMED has its atoms indexed starting at 1.

    An equivalent plumed file with the ``COMMITTOR`` on one line could also be used::

        d1:  DISTANCE ATOMS=1,2
        COMMITTOR ARG=d1 BASIN_LL1=0 BASIN_UL1=.15 BASIN_LL2=1 BASIN_UL2=100

