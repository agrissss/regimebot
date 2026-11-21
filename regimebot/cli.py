"""Command-line entry point."""

from importlib.metadata import version as package_version

import typer

from regimebot.config import load_config
from regimebot.db.session import upgrade_to_head

app = typer.Typer(help="Regime-aware crypto trading research system.", no_args_is_help=True)

db_app = typer.Typer(help="Database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")


@app.callback()
def main() -> None:
    """regimebot commands."""


@app.command()
def version() -> None:
    """Print the installed regimebot version."""
    typer.echo(package_version("regimebot"))


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply all database migrations to the configured database."""
    cfg = load_config()
    upgrade_to_head(cfg.database_url)
    typer.echo("database is up to date")
