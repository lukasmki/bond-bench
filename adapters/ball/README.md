# ball adapter

Wraps BALL's AssignBondOrderProcessor (Dehof et al. 2011) through its `assign_bond_orders` utility. BALL comes from github.com/BALL-Project/ball (LGPL-2.1), pinned at commit `54e7895` (v1.7.4 era), and is built from source in a pixi environment. Method name: `ball`. Choose the strategy with `--param strategy=fpt|a_star|ilp`; the default is `fpt`, the utility's own default.

## Known issues

- **Connectivity comes from the harness.** BALL assigns orders to the bonds in its input. Each query is written as a mol2 file whose bonds are the contact map A(X) that `canonical-O` uses, all typed single, so its connectivity errors are those of `canonical-O`.
- **No formal charges, and the total charge is ignored.** The processor never reads either, so raw output is invalid for every charged structure. Use `+project`.
- **Connected molecules only.** The processor aborts on more than one component, so disconnected contact graphs are reported as unsupported.
- **Explicit H only.** Penalty classes are matched by SMARTS with atom degree, and the utility cannot add hydrogens.
- **The penalty table is fixed.** `assign_bond_orders` always uses the BALL table (`bond_lengths/BondOrder.xml`). The Antechamber-derived table that BALL's `BondOrderAssigner` tool offers is not used.
- **Only the first optimal solution is reported.**
- **Linux only, with a long build.**
  - Qt6 Core, Network and Xml are required even with the GUI off.
  - `--strategy=ilp` needs conda-forge `lp_solve`. The similarly named `lpsolve55` is a Python binding for Python ≤ 3.7 and does not work here.
  - As with `indigo`, the build needs `binutils` and an explicit `CMAKE_AR`.
  - On a memory-limited machine, set `BUILD_JOBS`, e.g. `BUILD_JOBS=3 ./setup.sh`.
- **`BALL_DATA_PATH` is set at run time.** It points at `vendor/ball/data`, where the penalty table and the mol2 writer's type tables live.

## Verified

- Built and tested on Linux aarch64 (Docker, pixi, with lp_solve).
- With `fpt`, `a_star` and `ilp`, bond orders were in the label's resonance class for all six test molecules.
- The two charged molecules were invalid, as expected without formal charges.
- Not yet run on linux-64.
