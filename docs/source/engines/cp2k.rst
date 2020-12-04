CP2K Documentation
==================

..  toctree::
    :maxdepth: 1

CP2K Specific Inputs
--------------------
In addition to the inputs required by all :doc:`engines <engines>`, the CP2K specific inputs are listed below.

``cp2k_inputs`` - path of the CP2K input file
    The path to a well formed CP2K input file that describes the system you want to run. This original file will not be
    modified. An exception will be thrown if this does not pass the linter provided by
    `cp2k-input-tools <https://github.com/cp2k/cp2k-input-tools>`_
