"""Track the specialist assigned to each ticket."""
import sqlalchemy as sa
from alembic import op

revision = "0006_ticket_assignee"
down_revision = "0005_telegram_processing"
branch_labels = None
depends_on = None


def upgrade():
    columns = sa.inspect(op.get_bind()).get_columns("tickets")
    if not any(column["name"] == "assigned_agent" for column in columns):
        op.add_column("tickets", sa.Column("assigned_agent", sa.String(80)))


def downgrade():
    op.drop_column("tickets", "assigned_agent")
