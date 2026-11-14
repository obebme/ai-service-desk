"""Persist Telegram reply notification delivery state."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0004_telegram_reply_notification"
down_revision = "0003_ticket_closed_at"
branch_labels = None
depends_on = None


def existing_message_columns():
    return {column["name"] for column in inspect(op.get_bind()).get_columns("messages")}


def upgrade():
    # Fresh installs may already have these columns from metadata.create_all.
    existing = existing_message_columns()
    if "telegram_chat_id" not in existing:
        op.add_column(
            "messages",
            sa.Column("telegram_chat_id", sa.String(120), nullable=True),
        )
    if "telegram_delivery_status" not in existing:
        op.add_column(
            "messages",
            sa.Column("telegram_delivery_status", sa.String(20), nullable=True),
        )
    if "telegram_delivery_updated_at" not in existing:
        op.add_column(
            "messages",
            sa.Column("telegram_delivery_updated_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade():
    existing = existing_message_columns()
    if "telegram_delivery_updated_at" in existing:
        op.drop_column("messages", "telegram_delivery_updated_at")
    if "telegram_delivery_status" in existing:
        op.drop_column("messages", "telegram_delivery_status")
    if "telegram_chat_id" in existing:
        op.drop_column("messages", "telegram_chat_id")
