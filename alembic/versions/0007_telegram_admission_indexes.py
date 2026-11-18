"""Index recent Telegram updates and tickets by requester."""

from alembic import op

revision = "0007_telegram_admission_indexes"
down_revision = "0006_ticket_assignee"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_tickets_requester_created_at", "tickets", ["requester_id", "created_at"])
    op.create_index(
        "ix_messages_requester_channel_created_at", "messages", ["requester_id", "channel", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_messages_requester_channel_created_at", table_name="messages")
    op.drop_index("ix_tickets_requester_created_at", table_name="tickets")
