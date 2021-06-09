Using transition_sampling
=========================

Upon installing ``transition_sampling``, a command line entry point ``aimless_driver`` is created.

.. code-block::

    aimless_driver inputs.yml [--log-level=INFO] [--log-file=aimless.log]

``inputs.yml``
    describes all necessary inputs for the full pipeline, shown below.

``log-level``
    defaulted to ``INFO``, but can take values ``DEBUG, INFO, WARNING, ERROR, CRITICAL`` in order of
    decreasing granularity.

``log-file``
    defaulted to ``aimless.log``, but can be set to the path log messages should be directed to.

``inputs.yml`` Format
=====================
We've chosen the YAML format to describe the inputs to ``aimless_driver``. It's split into three main sections, one for
each present in :ref:`workflow`

.. include:: aimless.rst

.. include:: colvar.rst

.. include:: likelihood.rst
