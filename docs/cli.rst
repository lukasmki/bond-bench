Command-line reference
======================

.. code-block:: text

   bond-bench [--manifest PATH] [--out DIR] [-v] <command> ...

Global options come before the command:

``--manifest PATH``
   Strata manifest (default: ``assets/benchmark_strata_manifest.csv``).
``--out DIR``
   Dataset directory (default: ``dataset/``).
``-v``, ``--verbose``
   Debug logging.

``build``
---------

Build stratum files from the raw sources.

.. code-block:: text

   bond-bench build STRATUM [STRATUM ...] [--n N] [--seed S] [--cache-dir DIR]

``STRATUM``
   Stratum ids, built in the order given (e.g. ``A1 B1 C1``).
``--n N``
   Override ``n_structures`` from the manifest, for development builds.
``--seed S``
   Random seed (default: 0).
``--cache-dir DIR``
   Raw download cache (default: ``$BOND_BENCH_CACHE`` or ``data/raw/``).

``pull``
--------

Download the prebuilt dataset, or the published results, from a GitHub
release. Files are checked against the release's ``SHA256SUMS``.

.. code-block:: text

   bond-bench pull [STRATUM ...] [--results [DIR]] [--tag TAG] [--repo OWNER/NAME] [--force]

``STRATUM``
   Stratum ids (default: every released stratum).
``--results [DIR]``
   Pull benchmark results instead, into ``DIR`` (default: ``results/``).
``--tag TAG``
   Release tag (default: ``dataset-v<schema version>``, or
   ``results-v<schema version>`` with ``--results``).
``--repo OWNER/NAME``
   GitHub repository to pull from.
``--force``
   Overwrite local files that differ from the release. Without it, such files
   (for example your own rebuilds) are kept.

``validate``
------------

Check built stratum files: schema, offsets, label chemistry and the index.

.. code-block:: text

   bond-bench validate [--sample N] [--full]

``--sample N``
   Records per stratum to check chemically (default: 200).
``--full``
   Check every record.

``info``
--------

List the strata in the manifest and whether each is implemented and built.

``methods``
-----------

List registered methods and whether each is available on this machine.

``run``
-------

Run a method on built strata and score it.

.. code-block:: text

   bond-bench run METHOD [--strata ID ...] [--obs {auto,explicit_h,stripped}] [--n N]
                  [--workers W] [--seed S] [--param KEY=VALUE ...] [--noise SIGMAS]
                  [--reps R] [--equivariance N] [--timeout SECONDS] [--results DIR]

``METHOD``
   Method name, optionally with wrappers, e.g. ``openbabel+project``.
``--strata ID ...``
   Strata to run on (default: every built stratum).
``--obs``
   Observation: ``auto`` (default) follows the manifest's
   ``hydrogens_default``.
``--n N``
   Structures per stratum; groups (C1 ladders, B5 tautomer groups) are kept
   whole.
``--workers W``
   Parallel worker processes for in-process methods (default: 1).
``--seed S``
   Seed for sampling, noise and equivariance transforms (default: 0).
``--param KEY=VALUE``
   Method parameter; repeatable. Values are parsed as JSON when possible.
``--noise SIGMAS``
   Comma-separated on-the-fly noise levels in Å, e.g. ``0.05,0.1``.
``--reps R``
   Noisy replicates per structure and σ (default: 1).
``--equivariance N``
   Check permutation/Euclidean equivariance on N structures per stratum.
``--timeout SECONDS``
   Per-structure limit for in-process methods (default: 120; 0 = none).
``--results DIR``
   Results directory (default: ``results/``).

``score``
---------

Aggregate per-structure scores into ``results/summary.parquet`` and print a
summary table.

.. code-block:: text

   bond-bench score [--results DIR] [--method TAG ...] [--all-subgroups] [--rescore] [--no-ci]

``--method TAG ...``
   Restrict to these method tags.
``--all-subgroups``
   Also report subgroups (A1 resolution bin, C1 σ, on-the-fly noise σ).
``--rescore``
   Recompute score files from the stored predictions first.
``--no-ci``
   Skip confidence intervals.

``plot``
--------

Plot results with matplotlib (needs ``uv sync --extra plot``).

.. code-block:: text

   bond-bench plot [--results DIR] [--plot-dir DIR] [--metric NAME ...] [--format {png,pdf,svg}]

``--plot-dir DIR``
   Output directory (default: ``<results>/plots``).
``--metric NAME ...``
   Metrics to plot (default: ``smiles_match valid f1_macro``).
``--format``
   Image format (default: ``png``).
