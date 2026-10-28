Running the complete benchmark
==============================

These steps run every registered method on every built stratum, then score,
plot and publish the results. They assume a Linux machine with an NVIDIA GPU.
The ML adapters fall back to CPU without one, which is much slower.

Prerequisites
-------------

- ``uv``, ``git`` and ``curl`` on ``PATH``.
- `pixi <https://pixi.sh>`_ for the ``antechamber``, ``indigo`` and ``ball``
  adapters. It installs without root:

  .. code-block:: bash

     curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash   # installs to ~/.pixi/bin
     export PATH="$HOME/.pixi/bin:$PATH"

  Put the ``PATH`` line in job scripts too, since compute nodes may not read
  ``~/.bashrc``. Without pixi, skip those three adapters.
- Several GB of disk for the adapter environments and weights. To keep
  caches off a small home quota:

  .. code-block:: bash

     export PIXI_CACHE_DIR=/scratch/$USER/pixi-cache
     export UV_CACHE_DIR=/scratch/$USER/uv-cache

  The environments themselves live inside the clone, so clone where there is
  room.

1. Code and dataset
-------------------

.. code-block:: bash

   git clone <repository url> bond-bench && cd bond-bench
   uv sync
   uv run bond-bench pull
   uv run bond-bench validate

2. Adapters (once)
------------------

.. code-block:: bash

   adapters/mambax/setup.sh
   adapters/yuelbond/setup.sh
   adapters/bondnet/setup.sh
   adapters/mperformer/setup.sh       # Google Drive download; rerun later if the quota is exceeded
   adapters/indigo/setup.sh           # pixi; builds C++
   adapters/ball/setup.sh             # pixi; builds BALL from source (long)
   (cd adapters/openbabel && uv sync)
   pixi install --manifest-path adapters/antechamber/pixi.toml

   uv run bond-bench methods          # every row should say "ok"

Limit C++ build parallelism on small machines with ``BUILD_JOBS=4``. Each
``setup.sh`` skips what is already in place.

3. Train canonical M (once)
---------------------------

.. code-block:: bash

   uv sync --extra ml
   uv run python scripts/train_canonical_m.py --device cuda      # ~15 min on one A100

The checkpoint goes to ``models/canonical-M.pt`` (not committed). It records
the training strata and their checksums, the configuration and the training
ids. An optional noise-augmented variant:

.. code-block:: bash

   uv run python scripts/train_canonical_m.py --device cuda --sigma-aug 0.05 --out canonical-M-sigma0.05
   uv run bond-bench run canonical-M --param checkpoint=canonical-M-sigma0.05

4. Run every method
-------------------

.. code-block:: bash

   # built-in methods (in-process; --workers parallelizes over structures)
   for m in canonical-O canonical-R canonical-M rdkit-determinebonds; do
     uv run bond-bench run $m --workers 16 --equivariance 30
   done
   uv run bond-bench run canonical-O --param preset=atom_penalty --workers 16 --equivariance 30

   # wrapped published methods (one adapter process per batch of 256 structures)
   for m in openbabel antechamber indigo ball mambax yuelbond bondnet mperformer; do
     uv run bond-bench run $m --equivariance 30
   done

   # valid-set wrappers
   for m in openbabel rdkit-determinebonds antechamber indigo ball mambax yuelbond bondnet mperformer; do
     uv run bond-bench run $m+project
   done
   for m in mambax yuelbond bondnet; do
     uv run bond-bench run $m+hybrid
   done
   uv run bond-bench run canonical-M+hybrid --workers 16 --equivariance 30   # canonical H

``--workers`` and ``--timeout`` apply only to the built-in methods. Adapters
run one process per batch of 256 structures, with a one-hour limit per batch.

5. Score and plot
-----------------

.. code-block:: bash

   uv run bond-bench score --all-subgroups
   uv sync --extra plot && uv run bond-bench plot

6. Publish
----------

.. code-block:: bash

   uv run python scripts/publish_results.py

This regenerates ``results/summary.parquet``, packs ``results/`` into
``build/results-release/`` with a ``SHA256SUMS`` file, and prints the
``gh release create`` command. Runs made with ``--n`` are refused unless you
pass ``--allow-partial``.
