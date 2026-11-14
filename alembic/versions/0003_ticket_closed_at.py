"""Track when a ticket was closed for retention cleanup."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0003_ticket_closed_at"
down_revision = "0002_gemini_embedding_model"
branch_labels = None
depends_on = None


def has_closed_at():
    return any(column["name"] == "closed_at"
               for column in inspect(op.get_bind()).get_columns("tickets"))


def upgrade():
    # 0001 uses current metadata.create_all, so a fresh install may already have it.
    if not has_closed_at():
        op.add_column("tickets", sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(sa.text(
        "UPDATE tickets SET closed_at = updated_at "
        "WHERE status = 'closed' AND closed_at IS NULL"
    ))


def downgrade():
    if has_closed_at():
        op.drop_column("tickets", "closed_at")
