"""fills: the bid and ask a fill was priced from

Expand-only: two nullable columns, so existing rows (NULL: not recorded) and running code that
does not write them keep working.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-21 18:30:00
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("fills", sa.Column("bid", sa.Double(), nullable=True))
    op.add_column("fills", sa.Column("ask", sa.Double(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("fills") as batch:
        batch.drop_column("ask")
        batch.drop_column("bid")
