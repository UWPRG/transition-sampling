Aimless Shooting
----------------

For exact specifications, refer to the original paper by Peters and Trout,
`10.1063/1.2234477 <https://aip.scitation.org/doi/10.1063/1.2234477>`_

Generation of shooting points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Aimless shooting takes an initial (or multiple) configuration that are guessed to be near the transition state, or on
the dividing surface between two or more stable basins. Velocities are randomly generated from the Maxwell-Boltzmann
distribution and two trajectories are launched in parallel

#. One with the initial velocities set as generated. This is the *forward* trajectory
#. One with the generated velocities flipped (multiplied by -1). This is the *reverse* trajectory

These simulations are run until committing to one of the defined basins, or until completing the number of steps
specified in the MD parameters. If the two trajectories commit to different basins (or different categories of basins in
the case of multiple reactant or product states), this shooting point is considered as *accepted*. Otherwise, its said
to be *rejected*. The core ideas of aimless shooting are that

#. An accepted shooting point is more likely to be on the dividing surface than a rejected shooting point

#. If we take an accepted shooting point and perturb its configuration slightly, it's likely to result in another accepted
   shooting point. I.e., it should remain close to the dividing surface because the perturbation was small.

Using these principles, we generate more accepted shooting points by randomly perturbing an already accepted shooting point
along it's forward or reverse trajectory by :math:`\Delta t`, where :math:`\Delta t` is a time values that's a small
proportion of the overall reaction time. The specifics of this are defined in the above paper, but it essentially results
in trying the new shooting point randomly from one of the configurations in :math:`-2\Delta t \ -\Delta t, \ 0, \ +\Delta t, \ +2\Delta t`
from the previous accepted shooting point's forward and reverse trajectories. It's therefore possible that the old shooting
point could be chosen again as the new one.

If a shooting point is not accepted, it's not clear what step should be taken next. If a new shooting point is rejected,
we try resampling the velocities up to ``n_vel_tries``. If one of these is accepted, the shooting point is immediatley
considered accepted. If all of these are rejected, the shooting point is entirely rejected. The results of each shooting
are considered because they are still useful in the likelihood maximization. For example, a shooting point that is rejected
3 times and accepted on the 4th time indicates that it is further from the dividing surface than one that immediately accepts.

When a shooting point is entirely rejected, it is again unclear what to do. In our case, we keep a list of all shooting
points that have ever been accepted and randomly select one of them to restart with after a complete rejection of a point.
There's then an additional parameter ``n_state_tries`` that controls how many total rejections can happen in a row before
totally failing and exiting.

Initial starting
^^^^^^^^^^^^^^^^
A directory of initial starting guesses (containing one or more ``.xyz`` files that store coordinates in Å) is used to
'seed' the aimless shooting processes. These should be points you believe to be on the dividing surface. Each point will
try to be accepted up to ``n_vel_tries`` by resampling velocities. If all of these attempts are rejected, the point
will be considered rejected. At least one of your starting guesses needs to be accepted in order to proceed, otherwise the
program will exit.

Parallel shootings
^^^^^^^^^^^^^^^^^^
One instance of aimless shooting can only run two MD simulations at a time (forwards and reverse) because new shooting
points are generated sequentially. If you have more computational resources, you can run *multiple* aimless shootings
independently in parallel, and their configurations are recombined at the end. For example, you could run 4 parallel
aimless shootings to have 8 MD simulations running simultaneously. All parallel shootings are started from the same
initial configurations, but will diverge since velocities are randomly generated.

Each parallel instance will generate its own pair of `result files <Aimless shooting outputs_>`_, as well as one
cumulative combined pair of result files.

Basins
^^^^^^
One or more stable basins are defined using PLUMED's
`COMMITTOR directive <https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_m_m_i_t_t_o_r.html>`_ in a ``plumed.dat`` file.
This allows the flexibility to define any basin in terms of existing PLUMED CVs, while also ending a simulation upon
reaching one of these basins.

Aimless shooting outputs
^^^^^^^^^^^^^^^^^^^^^^^^
Our aimless shooting results in two files: a ``.csv`` and a ``.xyz``.

``.xyz``
    stores coordinates of attempted shooting points in Å appended to one another. Each velocity resampling records to
    this file, so it's not uncommon to see multiple of the same configuration in a row.

``.csv``
    Each row corresponds to the configuration on the ``.xyz`` file and stores metadata about it. It stores if that
    configuration was considered to be accepted, the integer of the basin the forward trajectory committed to, the basin
    the reverse trajectory committed to, and the PBC box size (in Å) of the configuration. If a trajectory did not
    commit, ``None`` is stored.



