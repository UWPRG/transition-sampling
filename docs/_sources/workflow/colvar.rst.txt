Collective Variable Calculation
-------------------------------

Given a pair ``.xyz`` and ``.csv`` result files from aimless shooting, PLUMED can be used to calculate any number of
configurational collective variables for each entry in the ``.xyz`` file. Simply define all the collective variables to
be calculated in a ``plumed.dat`` and provide the path to the ``plumed`` binary, and the standard PLUMED output ``COLVAR``
file will be generated.

Again, this can be repeated with any combination of different CVs over one aimless shooting result pair.