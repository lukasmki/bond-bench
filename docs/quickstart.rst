Quickstart
==========

This page goes from a fresh clone to a scored, plotted result in a few
minutes.

1. Get the dataset
------------------

The prebuilt dataset is published as a GitHub release. Pulling it is much
faster than building it:

.. code-block:: bash

   uv run bond-bench pull            # every stratum into dataset/
   uv run bond-bench pull A1 B1      # or only selected strata
   uv run bond-bench validate

While the repository is private, ``pull`` needs a GitHub token. It uses
``GITHUB_TOKEN``, ``GH_TOKEN`` or your ``gh auth login``.

2. Run a method
---------------

.. code-block:: bash

   uv run bond-bench methods                                 # what is available
   uv run bond-bench run canonical-O --strata B1 A1 --n 200 --workers 8

``--n 200`` samples 200 structures per stratum. The same ``--n`` and
``--seed`` select the same structures for every method, so results stay
comparable. Without ``--n`` every structure is used.

Predictions and per-structure scores go to ``results/canonical-O/``.

3. Score and plot
-----------------

.. code-block:: bash

   uv run bond-bench score                                   # table of every result
   uv sync --extra plot && uv run bond-bench plot            # figures in results/plots/

.. tip::

   To compare against the published results without rerunning every method,
   pull them into ``results/`` first:

   .. code-block:: bash

      uv run bond-bench pull --results
      uv run bond-bench score

4. Use the data from Python
---------------------------

.. code-block:: python

   from bond_bench.data.store import StratumReader

   reader = StratumReader("dataset/B1.h5")
   record = reader[0]                          # MoleculeRecord with explicit H
   print(record.smiles, record.atomic_numbers, record.bond_orders)

   for rec in reader.iter(strip_h=True):       # H-stripped variant, derived on the fly
       ...

Next steps: :doc:`dataset` describes the strata and the file format,
:doc:`evaluation` explains how outputs are scored, and :doc:`extending` shows
how to benchmark your own method.
