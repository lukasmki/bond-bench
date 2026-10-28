bond-bench
==========

A benchmark dataset and evaluation harness for bond order assignment methods.

Bond order assignment takes a 3D structure (elements, coordinates, total
charge and multiplicity) and returns a chemical graph: bond orders, formal
charges and hydrogen counts. bond-bench provides

- a **dataset** of labeled structures, organized into strata that cover
  experimental crystal structures, computed geometries and deliberately
  perturbed inputs, and
- a **harness** that runs any method over those strata, checks every output
  for chemical validity and scores it against the reference label.

.. code-block:: bash

   uv sync
   uv run bond-bench pull                     # prebuilt dataset into dataset/
   uv run bond-bench run canonical-O --strata B1 --n 200
   uv run bond-bench score

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   dataset
   evaluation
   methods
   full-benchmark

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/formulation
   theory/canonical-algorithms

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli
   extending
   api

The dataset is derived from third-party sources with their own licenses; see
``LICENSE.md`` in the repository.
