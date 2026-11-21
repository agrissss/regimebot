"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-11-21 08:05:36.753064
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candles",
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=8), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Double(), nullable=False),
        sa.Column("high", sa.Double(), nullable=False),
        sa.Column("low", sa.Double(), nullable=False),
        sa.Column("close", sa.Double(), nullable=False),
        sa.Column("volume", sa.Double(), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "timeframe", "ts"),
    )
    op.create_table(
        "funding_rates",
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rate", sa.Double(), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "ts"),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("git_sha", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "equity_snapshots",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", sa.Double(), nullable=False),
        sa.Column("gross_exposure", sa.Double(), nullable=False),
        sa.Column("drawdown", sa.Double(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("run_id", "ts"),
    )
    op.create_table(
        "fills",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Double(), nullable=False),
        sa.Column("price", sa.Double(), nullable=False),
        sa.Column("ref_price", sa.Double(), nullable=False),
        sa.Column("fee", sa.Double(), nullable=False),
        sa.Column("slippage_bps", sa.Double(), nullable=False),
        sa.Column("liquidity", sa.String(length=8), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fills_run_id", "fills", ["run_id"], unique=False)
    op.create_index("ix_fills_symbol_ts", "fills", ["symbol", "ts"], unique=False)
    op.create_table(
        "open_positions",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("state_json", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("run_id", "symbol"),
    )
    op.create_table(
        "regimes",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=False),
        sa.Column("inputs_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("run_id", "symbol", "ts"),
    )
    op.create_table(
        "risk_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_events_run_id", "risk_events", ["run_id"], unique=False)
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("strategy", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("entry", sa.Double(), nullable=False),
        sa.Column("stop", sa.Double(), nullable=False),
        sa.Column("target", sa.Double(), nullable=False),
        sa.Column("atr", sa.Double(), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_signals_run_id", "signals", ["run_id"], unique=False)
    op.create_index("ix_signals_symbol_ts", "signals", ["symbol", "ts"], unique=False)
    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("strategy", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=False),
        sa.Column("qty", sa.Double(), nullable=False),
        sa.Column("entry_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.Double(), nullable=False),
        sa.Column("exit_price", sa.Double(), nullable=False),
        sa.Column("initial_stop", sa.Double(), nullable=False),
        sa.Column("target", sa.Double(), nullable=False),
        sa.Column("gross_pnl", sa.Double(), nullable=False),
        sa.Column("fees", sa.Double(), nullable=False),
        sa.Column("funding", sa.Double(), nullable=False),
        sa.Column("net_pnl", sa.Double(), nullable=False),
        sa.Column("exit_reason", sa.String(length=16), nullable=False),
        sa.Column("r_multiple", sa.Double(), nullable=False),
        sa.Column("bars_held", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trades_run_id", "trades", ["run_id"], unique=False)
    op.create_index("ix_trades_symbol_entry_ts", "trades", ["symbol", "entry_ts"], unique=False)
    op.create_table(
        "risk_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("signal_id", sa.Integer(), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved", sa.Boolean(), nullable=False),
        sa.Column("qty", sa.Double(), nullable=False),
        sa.Column("rule", sa.String(length=64), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.ForeignKeyConstraint(
            ["signal_id"],
            ["signals.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_decisions_run_id", "risk_decisions", ["run_id"], unique=False)
    op.create_index("ix_risk_decisions_signal_id", "risk_decisions", ["signal_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_risk_decisions_signal_id", table_name="risk_decisions")
    op.drop_index("ix_risk_decisions_run_id", table_name="risk_decisions")
    op.drop_table("risk_decisions")
    op.drop_index("ix_trades_symbol_entry_ts", table_name="trades")
    op.drop_index("ix_trades_run_id", table_name="trades")
    op.drop_table("trades")
    op.drop_index("ix_signals_symbol_ts", table_name="signals")
    op.drop_index("ix_signals_run_id", table_name="signals")
    op.drop_table("signals")
    op.drop_index("ix_risk_events_run_id", table_name="risk_events")
    op.drop_table("risk_events")
    op.drop_table("regimes")
    op.drop_table("open_positions")
    op.drop_index("ix_fills_symbol_ts", table_name="fills")
    op.drop_index("ix_fills_run_id", table_name="fills")
    op.drop_table("fills")
    op.drop_table("equity_snapshots")
    op.drop_table("runs")
    op.drop_table("funding_rates")
    op.drop_table("candles")
