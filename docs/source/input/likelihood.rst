``likelihood_inputs`` (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the required inputs to perform likelihood maximization on the results of aimless shooting and calculated CVs
If this section is excluded, likelihood maximization will not be performed and the program will exit after completing
any preceding sections.

``max_cvs``: `int` (Optional)
  Set the maximum length of the set of CVs to be considered. If null or missing, all may be
  considered if the improvement for length i to i+1 is enough as given by Peters.

``output_name`` : `str` (Required)
    Path to the file to put results of maximization in.

``csv_input`` : `str` (Optional)
    If you've excluded ``md_inputs`` and don't want to run it, you need to explicitly
    set the file name of the ``.csv`` file produced by a previous run. If ``md_inputs``
    is defined and this field is missing or ``null``, the file name will be inferred from
    the ``md_inputs`` section.

``colvar_input`` : `str` (Optional)
    If you've excluded ``colvar_inputs`` and don't want to run it, you need to explicitly
    set the file name of the ``COLVAR`` file produced by a previous run. If ``colvar_inputs``
    is defined and this field is missing or ``null``, the file name will be inferred from
    the ``colvar_inputs`` section.

``n_iter`` : `int` (Optional)
    Global optimization is performed by basinhopping. This is the number of local optimization to perform per global
    optimizations. If missing, defaults to ``100``.

``use_jac`` : `bool` (Optional)
  Set to ``true`` if you want to use use the analytical calculation for the jacobian
  during optimization, which generally is faster. Set to ``false`` to use a finite
  difference approximation. Defaults to ``true`` if excluded.
