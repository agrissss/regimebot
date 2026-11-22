"""trades: stop, target, R-multiple and regime may be NULL (a daily_trend trade has none)

Expand-only: relaxing NOT NULL keeps every existing row valid and every running writer working.
The downgrade fails while NULL rows exist (delete the daily_trend runs' trades first).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-22 15:00:00
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_COLUMNS = {
    "initial_stop": sa.Double(),
    "target": sa.Double(),
    "r_multiple": sa.Double(),
    "regime": sa.String(length=32),
}


def upgrade() -> None:
    with op.batch_alter_table("trades") as batch:
        for name, type_ in _COLUMNS.items():
            batch.alter_column(name, existing_type=type_, nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("trades") as batch:
        for name, type_ in _COLUMNS.items():
            batch.alter_column(name, existing_type=type_, nullable=False)
