---
title: 'An MD Engine Agnostic Implementation of Aimless-Shooting Likelihood Maximization'
tags:
  - Python
  - Molecular Dynamics
  - Reaction Coordinates
  - Aimless Shooting
authors:
  - name: Isaiah Lemmon
  - name: Luke Gibson
  - name: Jim Pfaendtner
affiliations:
  - as
date: 01 December 2021
bibliography: paper.bib
---
  
# Summary

Chemical reactions are extremely complex and difficult to analyze in the context of molecular dynamics simulations.
Reaction coordinates are useful because they provide a quantitative measure of a reaction’s progress, but obtaining a
reasonable coordinate system is computationally expensive. Aimless-shooting likelihood maximization is a well known
algorithm for finding reaction coordinates by running specific simulations [@peters], and the `transition-sampling` package
implements its workflow as a simple and extensible command line program written in Python. 

# Statement of need 

Many different simulation engines exist for performing molecular dynamics, each with a different workflow or features that
researchers may want to use. Many times, researchers cannot simply switch to a different engine because of the specific
features they require, or because the barrier to entry of compiling and changing workflows is too high. The
aimless-shooting algorithm is independent of any specific engine’s implementation, so to support the maximum number of
users, `transition-sampling` was designed to expose a simple and flexible Python interface that can be implemented for any
engine. This interface allows the algorithm to be applied reusably to any simulation engine, with specifics like
launching a simulation and reading its output abstracted away. There is existing support for `CP2K` [@cp2k] and `GROMACS` [@gromacs], along
with a detailed guide for implementing any additional engines. Two other implementations of aimless-shooting, `ATESA` [@atesa] and
`openpath-sampling` [@ops1, @ops2], have support for `AMBER` [@amber] and `GROMACS` respectively.

On top of the choice of molecular dynamics engine,
there is an enormous number of potential collective variables (CVs) that can be applied to a chemical system. CVs are
any differentiable function of the system’s atomic coordinates, and finding a reaction coordinate requires calculating
many of them for screening. Rather than attempt to implement these, `transition-sampling` relies on `PLUMED`, an actively
developed and widely used plugin that specifically supports robust and efficient CV calculation [@plumed]. CVs are defined in a
familiar format, exactly as they would be for `PLUMED`. If a user wants to use a CV that PLUMED doesn’t support, they can
either implement it themselves via the extensive developer documentation, or submit a request to the `PLUMED` developers.
Additionally, for engines that support `PLUMED` integration, simulations can be stopped once they reach their target,
reducing unnecessary computation. To our knowledge, no other packages support integration with PLUMED.

Finally, there are many different job schedulers for the computing clusters that molecular dynamics simulations are typically performed
on. `transition-sampling` was designed to be flexible and work with any of them, as the user defines the command to launch
a simulation.

`transition-sampling` is currently being utilized to explore the impact of interfacial effects on key
degradative chemical reactions that occur in the liquid electrolyte of a lithium-ion battery.





