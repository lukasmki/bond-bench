Installation
============

bond-bench requires Python 3.13 or later and is managed with
`uv <https://docs.astral.sh/uv/>`_.

.. code-block:: bash

   git clone <repository url> bond-bench && cd bond-bench
   uv sync

This installs the ``bond-bench`` command into the project environment; run it
with ``uv run bond-bench``.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - extra
     - needed for
   * - ``plot``
     - ``bond-bench plot`` (matplotlib): ``uv sync --extra plot``
   * - ``ml``
     - training and running ``canonical-M`` (PyTorch): ``uv sync --extra ml``

The published methods that ``bond-bench`` wraps (Open Babel, antechamber,
MPerformer and others) run in their own environments under ``adapters/``.
Setting those up is covered in :doc:`full-benchmark`.

Building the documentation
--------------------------

Sphinx and the furo theme are in the ``dev`` dependency group, which
``uv sync`` installs by default:

.. code-block:: bash

   uv run make -C docs html          # output in docs/_build/html/

Running the tests
-----------------

.. code-block:: bash

   uv run pytest
