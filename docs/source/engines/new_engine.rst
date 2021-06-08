Implementing a New Engine
=========================

..  toctree::
    :maxdepth: 1

Implementing a new engine only requires figuring out how to fulfill the methods of
:class:`~transition_sampling.engines.abstract_engine.AbstractEngine` and adding it as case in
:func:`~transition_sampling.driver.parse_engine`. It should then seamlessly work with existing aimless shooting
infrastructure.

.. note::
    A note on **units**: You are free to internally use whatever units are easiest to interact with your engine, however
    anything received or returned to aimless shooting via the public methods defined in
    :class:`~transition_sampling.engines.abstract_engine.AbstractEngine` are expected to be in Å (coordinates, box lengths)
    and m/s (velocities). You must convert between them yourself if you use any other units.


Suggested Workflow
------------------

Define the additional inputs your engine requires
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Think about running an MD simulation by hand. Aimless shooting essentially requires that you can change the starting
coordinates and velocities of a simulation. What's the closest level of input you can get to running a simulation while
still being able to change those parameters? Examples of existing engines:

* CP2K:
    In CP2K, everything is specified in a ``.inp`` file, including coordinates and velocities. Therefore, we can take
    a template ``.inp`` file that contains all the other simulation parameters and just modify the coords and vels as necessary.
    When we're ready to run a new simulation, we can write out an updated version of the original ``.inp`` file and pass that
    to the CP2K executable.

* GROMACS:
    GROMACS has multiple stages to running a simulation. For example, a ``.top`` file needs to be generated from a ``.pdb``.
    Then a ``.gro`` (coordinates and velocities), a ``.mdp`` file (parameters) and ``.top`` (topology) need to be
    compiled into a ``.tpr`` with ``gmx grompp``. Finally, this ``.tpr`` can be run with ``gmx mdrun``.

    Bonds are not changing during aimless shooting so we can use a static ``.top`` file. However, to change coordinates
    and velocities, we need to modify the ``.gro`` file, which requires recompiling with ``grompp``. Therefore, the
    additional inputs we will need are:

    * ``top_file``: constant file used to compile a new simulation
    * ``mdp_file``: parameter file used to compile a new simulation
    * ``gro_file``: coordinate file that gets modified for each shooting point before recompiling
    * ``grompp_cmd``: command to compile the above into a new ``.tpr`` to run

The inputs you define can then be passed in in the ``inputs`` dictionary. You can then require and validate them by
adding them as cases in :func:`~transition_sampling.engines.abstract_engine.AbstractEngine.validate_inputs`

Return atoms
^^^^^^^^^^^^
To override :func:`~transition_sampling.engines.abstract_engine.AbstractEngine.atoms`, you need to return a sequence of
the atoms in your system as strings of their periodic table representation (e.g. Argon -> ``"Ar"``).

* CP2K
    The template ``.inp`` file has its ``&COORD`` section parsed to read the element names

* GROMACS
    The template ``.gro`` file has the atom names extracted from it.

Store and write positions
^^^^^^^^^^^^^^^^^^^^^^^^^
Your engine needs to be able to hold coordinates as internal state and then pass them to the simulation when requested
to run. These are passed to you in Å in :func:`~transition_sampling.engines.abstract_engine.AbstractEngine.set_positions`.

* CP2K
    ``CP2KEngine`` stores the template ``.inp`` file in memory as a dictionary and stores positions by modifying the
    ``&COORDS`` section. This dictionary is written out with the updated coordinates when running a simulation.

* GROMACS
    ``GROMACSEngine`` stores the template ``.gro`` file in memory and stores positions by modifying the positions array
    A new ``.gro`` file is written out with the updated coordinates when running a simulation.

Store and write velocities
^^^^^^^^^^^^^^^^^^^^^^^^^^
Your engine needs to be able to hold velocities as internal state and then pass them to the simulation when requested
to run. These are passed to you in m/s in :func:`~transition_sampling.engines.abstract_engine.AbstractEngine.set_velocities`,
and will likely have a similar solution to setting positions.

* CP2K
    ``CP2KEngine`` stores the template ``.inp`` file in memory as a dictionary and stores velocities by modifying the
    ``&VELOCITY`` section. This dictionary is written out with the updated coordinates when running a simulation.

* GROMACS
    ``GROMACSEngine`` stores the template ``.gro`` file in memory and stores velocities by modifying the velocities array
    A new ``.gro`` file is written out with the updated velocities when running a simulation.

Return :math:`\Delta t, \ 2\Delta t` output frames
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After running a single simulation with :func:`~transition_sampling.engines.abstract_engine.AbstractEngine._launch_traj`,
you need to return the coordinates of the :math:`\Delta t, \ 2\Delta t` frames, as defined by when
:func:`~transition_sampling.engines.abstract_engine.AbstractEngine.set_delta_t` is called. The easiest way to do this has
been to set the print frequency of the trajectory to be the number of frames equivalent to :math:`\Delta t`. Then, the
second printed frame will be at :math:`\Delta t` and the third printed frame will be at :math:`2\Delta t` (the first
printed frame is generally the starting position). The problem then boils down to two parts:

    #. Setting the print frequency correctly
    #. Reading the printed trajectory

In both CP2K and GROMACS, the simulation timestep is parsed (from the ``.inp`` and ``.mdp`` respectively). The number of
frames corresponding to one :math:`\Delta t` is then calculated, and this set as trajectory print frequency in the
parameter file.

After the simulation completes, ``mdtraj`` is already included and can be used to parse the trajectory. Prefer to use the
low level readers so that only the first three frames need to be read before closing the file. Ensure the coordinates
are converted to Å before returning them.

Figure out how to integrate PLUMED
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The base engine is given a :class:`~transition_sampling.engines.plumed.PlumedInputHandler` that handles moving the plumed
file around. You can write a unique plumed file for the trajectory by calling
:func:`~transition_sampling.engines.plumed.PlumedInputHandler.write_plumed` with the desired location and output file name.

The simulation then needs to be told to use this file. In GROMACS, this is just the ``-plumed`` flag added to the command
list, while in CP2K, the ``.inp`` file has to be modified before being written.

After the simulation is complete, :class:`~transition_sampling.engines.plumed.PlumedOutputHandler` can be used to determine
which basin was committed to.

Run a simulation with :func:`~transition_sampling.engines.abstract_engine.AbstractEngine._launch_traj`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With all the above figured out, this should be just slight modifications of existing implementations. Essentially

#. Write all needed input files to the unique name for this trajectory (given by ``projname``)

    #. If necessary as with GROMACS, preprocess these

#. Run asynchronously with the given ``md_cmd`` by opening a subprocess and periodically checking for completion,
   ``await`` ing in the interim to allow other shootings to process.
#. Check for any errors or warnings, log or copy files as appropriate
#. Read the committed basins and :math:`\Delta t, \ 2\Delta t` output frames and return in the specified dictionary

