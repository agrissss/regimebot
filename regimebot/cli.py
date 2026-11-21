"""Command-line entry point."""

from importlib.metadata import version as package_version

import typer

app = typer.Typer(help="Regime-aware crypto trading research system.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """regimebot commands."""


@app.command()
def version() -> None:
    """Print the installed regimebot version."""
    typer.echo(package_version("regimebot"))
