"""Persist the Telegram processing schedule."""

import sqlalchemy as sa
from alembic import op

revision = "0005_telegram_processing"
down_revision = "0004_telegram_reply_notification"
branch_labels = None
depends_on = None


def upgrade():
    # The initial migration creates tables from current model metadata.
    columns = sa.inspect(op.get_bind()).get_columns("messages")
    if not any(column["name"] == "processing_due_at" for column in columns):
        op.add_column("messages", sa.Column("processing_due_at", sa.DateTime(timezone=True)))
        op.create_index("ix_messages_processing_due_at", "messages", ["processing_due_at"])


def downgrade():
    op.drop_index("ix_messages_processing_due_at", table_name="messages")
    op.drop_column("messages", "processing_due_at")
