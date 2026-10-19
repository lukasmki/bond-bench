# indigo adapter

Wraps indigo-bondorder (Welsh & Allison 2019), from github.com/allison-group/indigo-bondorder (MIT), pinned at commit `77b1574`. Its `pyindigox` Python module is built in a pixi environment. Method name: `indigo`. Choose the algorithm with `--param algorithm=fpt|astar|local_optimisation`; the default is `fpt`.

## Known issues

- **Connectivity comes from the harness.** indigoX assigns electrons to a given graph. It receives the contact map A(X) that `canonical-O` uses, so its connectivity errors are those of `canonical-O`.
- **Connected molecules only.** `AssignElectrons` returns nothing for a graph with more than one component. Queries whose contact graph is disconnected, such as salts, are reported as unsupported. This lowers coverage on A2.
- **Explicit H only.**
- **Elements are limited to H, C, N, O, F, P, S, Cl and Br.** Those are the elements of the default def2-SVPD energy tables, so I, B and Si are unsupported.
- **Only the first co-optimal assignment is reported.** It is the one upstream applies. The other degenerate minimum-score assignments are discarded.
- **Non-integer bond orders are reported as errors.** Odd electron counts make indigoX produce 1.5 or 2.5 orders.
- **Old toolchain requirements.** Upstream has been dormant since 2022, and its bundled pybind11 2.2.2 does not compile against Python ≥ 3.11, so the environment uses Python 3.10. The build also needs the conda-forge `binutils` package and an explicit `CMAKE_AR`; without them CMake finds no archiver for the static library.

## Verified

- Built and tested on Linux aarch64 (Docker, pixi).
- With both `fpt` and `astar`, all six test molecules were valid and in the label's resonance class, formal charges included.
- Not yet run on linux-64.
