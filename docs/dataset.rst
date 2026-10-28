Dataset
=======

The dataset is defined by ``assets/benchmark_strata_manifest.csv``. Each row
is a *stratum*: a set of structures from one source, with one kind of
geometry and one reference label. Strata are grouped into three tracks:

- **A, experimental**: crystal structures (PDB ligands, COD small molecules).
- **B, computed**: optimized geometries from quantum-chemistry datasets.
- **C, perturbed**: noisy, off-equilibrium or otherwise non-ideal geometries.

Each stratum is stored as ``dataset/<stratum_id>.h5``, and
``dataset/index.parquet`` lists every structure across all strata. Run
``bond-bench info`` to see which strata are implemented and built.

Implemented strata
------------------

.. list-table::
   :header-rows: 1
   :widths: 6 22 72

   * - id
     - source
     - notes
   * - A1
     - PDB instances + wwPDB CCD
     - X-ray, three resolution bins. Platinum-style density filter (RSCC ≥ 0.9,
       RSR ≤ 0.3, complete, occupancy 1, no altlocs, not covalent). At most 3
       instances per CCD id. Hydrogens are stripped; per-atom H counts are part
       of the label.
   * - A2
     - COD + curated SMILES
     - Entries whose curated SMILES Merkys et al. (2023) found isomorphic with
       the coordinate-derived structure. Single-crystal X-ray, R ≤ 0.05, no
       disorder, experimental H. Each record is one formula unit.
   * - B1
     - QMugs (GFN2-xTB)
     - Drug-like molecules whose graph matches ChEMBL 27, excluding anything
       in GEOM-DRUGS. The primary computed test set.
   * - B2
     - GEOM-DRUGS
     - Lowest-energy conformer, kept only if its graph and charge match the
       GEOM SMILES. In-distribution reference; ``canonical-M`` trains on it.
   * - B3
     - QCML + GEOM-QM9
     - Small, elementally diverse molecules (≤ 8 heavy atoms for the QCML
       half).
   * - B5
     - Tautobase + Dhaked–Nicklaus
     - Tautomer groups, each structure ETKDG-embedded and GFN2-xTB relaxed.
       Each structure records whether it is the preferred tautomer.
   * - C1
     - derived from B1 (+ B4)
     - Gaussian noise ladder: σ ∈ {0, 0.05, 0.10, 0.20} Å on every parent.
   * - C5
     - QCML + Transition1x
     - Off-equilibrium structures: normal-mode samples and NEB images near a
       reaction endpoint.

Every label goes through RDKit sanitization, a charge-sum check and a
valence-plausibility check that rejects undeclared radicals and hypervalent
S/P without multiple bonds.

Hydrogens
---------

The manifest's ``hydrogens_default`` column says how each stratum is
presented to methods:

- ``Present``: hydrogens are explicit atoms with coordinates.
- ``Stripped``: hydrogens are removed, and the per-atom H count becomes part
  of what the method must predict.
- ``Both variants`` / ``Present + stripped``: the stratum is evaluated both
  ways.

Strata stored with explicit H can always be read H-stripped
(``StratumReader.iter(strip_h=True)``); the reverse is not possible.

File format
-----------

Each ``.h5`` file stores ragged arrays, indexed by per-molecule offsets.

``/molecules/*``
   One row per structure: ``structure_id``, ``source_id``, ``smiles``,
   ``inchikey``, ``total_charge``, ``multiplicity``, ``n_atoms``,
   ``n_bonds``, ``atom_offset``, ``bond_offset``, and stratum-specific
   ``meta_<key>`` columns (for example ``meta_res_bin`` in A1 or
   ``meta_sigma`` in C1).

``/atoms/*``
   Per-atom arrays:

   - ``atomic_numbers``
   - ``positions`` (Å)
   - ``formal_charges``
   - ``implicit_hydrogens``: -1 where H are explicit atoms; otherwise the
     labeled H count
   - ``radical_electrons``

``/bonds/*``
   ``bonds`` (atom indices within the molecule), ``bond_orders`` (Kekulé
   1/2/3; 9 is reserved for coordination bonds) and ``aromatic``.

Root attributes
   The manifest row, the schema version, the seed, the build parameters and
   provenance.

You rarely need to read these arrays directly:
:class:`~bond_bench.data.store.StratumReader` returns
:class:`~bond_bench.data.schema.MoleculeRecord` objects.

.. code-block:: python

   from bond_bench.data.store import StratumReader

   reader = StratumReader("dataset/A1.h5")
   reader.molecule_table()              # polars DataFrame of /molecules
   rec = reader.get(reader.ids()[0])    # one record by structure id

Building the dataset
--------------------

Each stratum has its own builder. Building from scratch downloads the raw
sources, which can be large (B1 streams the ~50 GB GEOM ``rdkit_folder`` once).

.. code-block:: bash

   uv run bond-bench build A1 B1 C1       # full size (n_structures from the manifest)
   uv run bond-bench build A1 --n 30      # quick development build
   uv run bond-bench validate             # schema, offsets, label chemistry, index

- Raw downloads are cached in ``data/raw/``. Change the location with
  ``--cache-dir`` or the ``BOND_BENCH_CACHE`` environment variable.
- Builds are deterministic for a given ``--seed`` and cache.
- Each build writes a funnel report of counts and rejection reasons to
  ``dataset/reports/<stratum_id>.json``.
- Derived strata are built after their parents: build B1 before C1.

Publishing a dataset release
----------------------------

After a full build, ``uv run python scripts/publish_dataset.py`` stages the
assets in ``build/release/`` and prints the ``gh release create`` command.
Dev builds made with ``--n`` are refused. Releases are tagged
``dataset-v<schema version>``, and every file is listed in ``SHA256SUMS``,
which ``bond-bench pull`` checks.
