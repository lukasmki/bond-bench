# antechamber adapter

Wraps AmberTools antechamber bond-type perception (bondtype, Wang et al. 2006), from conda-forge `ambertools` ≥ 24, in a pixi environment. Method name: `antechamber`. The environment is created on first run, so there is no `setup.sh`.

## Known issues

- **No formal charges.** antechamber assigns bond types but no formal charges, so raw output is invalid for every charged structure. Use `+project`.
- **Explicit H only.** bondtype needs every H. Stripped queries are reported as unsupported.
- **antechamber finds its own connectivity.** Each query is written as a Gaussian Cartesian input (`-fi gcrt`) with elements, coordinates, charge and multiplicity, and antechamber bonds atoms by distance. Connectivity errors are therefore antechamber's own, not the harness contact map's.
- **The flags are a choice of the adapter:**
  - `-j 4`: assign full bond types;
  - `-at sybyl`: the most permissive atom typing, which matters only so typing does not abort;
  - `-dr no`: skip acdoctor's sanity checks, which would otherwise reject unusual valences.
- **Bond types are read from the mol2 output.**
  - `ar` bonds are Kekulized with RDKit.
  - `am` (amide) bonds are read as single.
  - Any other type is read as single and flagged.
- **Elements are those of the atomic penalty table:** H, C, N, O, F, Si, P, S, Cl, Br, I.

## Verified

- Tested on Linux aarch64 (Docker, pixi).
- Bond orders were in the label's resonance class for all six test molecules.
- The two charged molecules were invalid, as expected without formal charges.
- Not yet run on linux-64.
