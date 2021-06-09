``colvar_inputs`` (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the required inputs to calculate new CVs from the results of aimless shooting.
If this section is excluded, new CVs will not be calculated and we will continue to the next present part.

``plumed_cmd`` : `str` (Required)
    Command to launch PLUMED with, the PLUMED binary

``plumed_file`` : `str` (Required)
    Path to the ``plumed.dat`` that contains CV definitions

``output_name`` : `str` (Required)
    Path to file that PLUMED should dump CVs to in its COLVAR format

``csv_input`` : `str` (Optional)
    If you've excluded ``md_inputs`` and don't want to run it, you need to explicitly
    set the file name of the ``.csv`` file produced by a previous run. If ``md_inputs``
    is defined and this field is missing or ``null``, the file name will be inferred from
    the ``md_inputs`` section.

``xyz_input`` : `str` (Optional)
    If you've excluded ``md_inputs`` and don't want to run it, you need to explicitly
    set the file name of the ``.xyz`` file produced by a previous run. If ``md_inputs``
    is defined and this field is missing or ``null``, the file name will be inferred from
    the ``md_inputs`` section.

