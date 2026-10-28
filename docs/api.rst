Python API
==========

The modules most useful from Python. Everything else is internal and may
change.

Data
----

.. automodule:: bond_bench.data.schema
   :members: MoleculeRecord

.. automodule:: bond_bench.data.store
   :members: StratumReader, write_stratum, write_index

.. automodule:: bond_bench.data.builders.base
   :members: StratumBuilder

Queries and predictions
-----------------------

.. automodule:: bond_bench.eval.query
   :members: Observation, Query

.. automodule:: bond_bench.eval.prediction
   :members: Status, Prediction

.. automodule:: bond_bench.eval.core.graph
   :members: Graph

Methods
-------

.. automodule:: bond_bench.eval.methods.base
   :members: Capabilities, Method, Wrapper, register, get_method, available_methods

.. The module docstring of ``external`` holds the JSONL protocol, which :doc:`extending` covers.

.. autoclass:: bond_bench.eval.methods.external.ExternalMethod

.. automodule:: bond_bench.eval.methods.adapters
   :members: AdapterMethod

Scoring
-------

.. automodule:: bond_bench.eval.metrics
   :members: grade, f1_from_confusion

.. automodule:: bond_bench.eval.runner
   :members: run, run_stratum, check_equivariance, rescore
