"""bond-bench: bond order assignment benchmark."""


def main() -> None:
    from .cli import app

    raise SystemExit(app())
