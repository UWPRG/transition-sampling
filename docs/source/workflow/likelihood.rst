Likelihood Maximization
-----------------------

With the desired set of calculated collective variables and the corresponding result ``.csv`` from aimless shooting,
likelihood maximization can be performed. Here we find a linear combination of a subset of the given collective variables
that best represents the reaction coordinate for this system.

Individual optimization of a set of CVs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a single optimization of all CVs in a set, we perform the following:

The reaction coordinate :math:`r` is linear combination of :math:`m`
collective variables :math:`\mathbf{x}` defined for a shooting point with weights :math:`\boldsymbol{\alpha}`
plus a constant :math:`\alpha_0`:

.. math ::
    r(\mathbf{x}) = \alpha_0 + \sum_{i=1}^m \alpha_ix_i

The probability that a shooting point is on a transition path (TP) is chosen
to be modeled as a non-gaussian bell curve and is given by:

.. math ::
    \Pr(TP|\mathbf{x}) = p_0 (1 - \tanh^2(r(\mathbf{x}))

This probability is maximized for CVs of accepted shooting points and
minimized for rejected shooting points by optimizing
:math:`\boldsymbol{\alpha}`, :math:`\alpha_0`, and :math:`p_0` and considering each to be drawn i.i.d.

.. math ::
    \arg\max_{\boldsymbol{\alpha}, \alpha_0, p_0} \prod_{\mathbf{x} \in \text{Accepted}}\Pr(TP|\mathbf{x}) \prod_{\mathbf{x} \not\in \text{Accepted}} 1 - \Pr(TP|\mathbf{x})

The log-likelihood :math:`l` is evaluated, as it has an equivalent solution.

This is a constrained (:math:`p_0 \in (0, 1]`) non-convex optimization, so we use `basin-hopping
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_ to perform a series of local
optimizations in attempt to find a global maximum, the number of iterations for which can be set with ``n_iter``.

Optimization of all candidate CVs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It's likely that some candidate CVs will contribute more to the likelihood than others. We don't want to include CVs that
don't significantly increase it - we want to find the minimal set of CVs that make a good reaction coordinate. To facilitate
this, we start by evaluating the above objective function with only one CV in the available set. We take the maximum value
as the best combination for length :math:`i`, (where :math:`i=1` initially). Then:

#. Evaluate the objective function of each combination of length :math:`i+1`. Take the best value.
#. If the best value for length :math:`i+1` increased by more than :math:`0.5 \ln (N)`, where :math:`N` is the number of
   sample shooting points:

    * Then repeat with :math:`i=i+1` for all combinations of the next highest length
    * Otherwise, this is our optimal solution

Likelihood maximization results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The result file for likelihood maximization is ``.csv`` that stores the results of all maximized combinations, not just the
final maximum. It has columns ``# of CVs, obj. func. value, p0, alpha0, <one column for the weight of each CV>``.
For CVs that are not included in a row's optimization, their value is empty or ``nan``