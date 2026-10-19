"""Command-line interface: ``bond-bench build|validate|info|methods|run|score``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from .data.builders.registry import get_builder, is_implemented
from .data.download import default_cache_dir
from .data.manifest import DEFAULT_MANIFEST, REPO_ROOT, load_manifest
from .data.store import write_index
from .data.validate import validate_index, validate_stratum

DEFAULT_OUT = REPO_ROOT / "dataset"
DEFAULT_RESULTS = REPO_ROOT / "results"


def cmd_build(args: argparse.Namespace) -> int:
    specs = load_manifest(args.manifest)
    unknown = [s for s in args.strata if s not in specs]
    if unknown:
        print(f"unknown strata: {unknown}", file=sys.stderr)
        return 2
    for stratum_id in args.strata:
        builder_cls = get_builder(stratum_id)
        builder = builder_cls(
            specs[stratum_id],
            cache_dir=args.cache_dir,
            out_dir=args.out,
            seed=args.seed,
            n=args.n,
        )
        builder.build()
    write_index(args.out)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    specs = load_manifest(args.manifest)
    paths = sorted(Path(args.out).glob("*.h5"))
    if not paths:
        print(f"no stratum files in {args.out}", file=sys.stderr)
        return 1
    results = []
    for path in paths:
        res = validate_stratum(path, specs.get(path.stem), sample=None if args.full else args.sample)
        results.append(res)
        status = "OK" if res.ok else "FAIL"
        print(f"{path.stem:4s} {status:4s} n={res.n_structures}")
        for msg in res.errors[:20]:
            print(f"     error: {msg}")
        if len(res.errors) > 20:
            print(f"     ... {len(res.errors) - 20} more errors")
        for msg in res.warnings:
            print(f"     warning: {msg}")
    index_errors = validate_index(Path(args.out), results)
    for msg in index_errors:
        print(f"index error: {msg}")
    return 0 if all(r.ok for r in results) and not index_errors else 1


def cmd_info(args: argparse.Namespace) -> int:
    specs = load_manifest(args.manifest)
    print(f"{'id':4s} {'track':12s} {'n':>6s}  {'impl':5s} {'built':5s} stratum")
    for sid, spec in specs.items():
        built = (Path(args.out) / f"{sid}.h5").exists()
        print(
            f"{sid:4s} {spec.track:12s} {spec.n_structures:6d}  "
            f"{'yes' if is_implemented(sid) else '-':5s} {'yes' if built else '-':5s} {spec.stratum}"
        )
    return 0


def _parse_params(items: list[str]) -> dict:
    from .eval.methods.base import parse_value

    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        params[key] = parse_value(value)
    return params


def cmd_methods(args: argparse.Namespace) -> int:
    from .eval.methods.base import WRAPPERS, available_methods

    print(f"{'method':22s} {'family':13s} {'stripped_h':10s} status")
    for name, cls in sorted(available_methods().items()):
        method = cls()
        status = "ok"
        if hasattr(method, "check_available"):
            status = method.check_available() or "ok"
        print(f"{name:22s} {cls.family:13s} {'yes' if cls.capabilities.stripped_h else '-':10s} {status}")
    print("wrappers (append with +):", ", ".join(sorted(WRAPPERS)))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .eval.report import equivariance_report, format_table, report
    from .eval.runner import run

    specs = load_manifest(args.manifest)
    strata = args.strata or sorted(p.stem for p in Path(args.out).glob("*.h5"))
    out_dir = run(
        args.method, strata, args.out, specs, results_dir=args.results, params=_parse_params(args.param),
        observation=args.obs, n=args.n, workers=args.workers, seed=args.seed,
        noise=[float(s) for s in args.noise.split(",")] if args.noise else (), reps=args.reps,
        equivariance=args.equivariance, timeout=args.timeout or None,
    )
    summary = report(args.results)
    print(format_table(summary.filter(pl.col("method") == out_dir.name)))
    if args.equivariance:
        print(equivariance_report(args.results).filter(pl.col("method") == out_dir.name))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .eval.report import equivariance_report, format_table, report
    from .eval.runner import rescore

    if args.rescore:
        print(f"rescored {rescore(args.results, args.out)} prediction files")
    summary = report(args.results, ci=not args.no_ci)
    if args.method:
        summary = summary.filter(pl.col("method").is_in(args.method))
    if args.all_subgroups is False:
        summary = summary.filter(pl.col("subgroup") == "all")
    print(format_table(summary))
    eq = equivariance_report(args.results)
    if not eq.is_empty():
        print(eq)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bond-bench")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dataset directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build stratum files")
    build.add_argument("strata", nargs="+", help="stratum ids, e.g. A1 B1 C1 (built in order)")
    build.add_argument("--n", type=int, default=None, help="override n_structures (dev builds)")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    build.set_defaults(func=cmd_build)

    validate = sub.add_parser("validate", help="check built stratum files")
    validate.add_argument("--sample", type=int, default=200, help="records per stratum to check chemically")
    validate.add_argument("--full", action="store_true", help="check every record")
    validate.set_defaults(func=cmd_validate)

    info = sub.add_parser("info", help="list strata and build status")
    info.set_defaults(func=cmd_info)

    methods = sub.add_parser("methods", help="list bond order assignment methods")
    methods.set_defaults(func=cmd_methods)

    run = sub.add_parser("run", help="run a method on built strata and score it")
    run.add_argument("method", help="method name, optionally with wrappers, e.g. openbabel+project")
    run.add_argument("--strata", nargs="*", default=None, help="default: every built stratum")
    run.add_argument("--obs", default="auto", choices=["auto", "explicit_h", "stripped"])
    run.add_argument("--n", type=int, default=None, help="structures per stratum (groups kept whole)")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--noise", default="", help="on-the-fly σ grid in Å, e.g. 0.05,0.1")
    run.add_argument("--reps", type=int, default=1)
    run.add_argument("--equivariance", type=int, default=0, metavar="N", help="check N structures per stratum")
    run.add_argument("--timeout", type=float, default=120.0, help="seconds per structure; 0 = no limit")
    run.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="summarize results")
    score.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    score.add_argument("--method", nargs="*", default=None)
    score.add_argument("--all-subgroups", action="store_true")
    score.add_argument("--rescore", action="store_true", help="recompute scores from stored predictions")
    score.add_argument("--no-ci", action="store_true")
    score.set_defaults(func=cmd_score)
    return parser


def app(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.func(args)
