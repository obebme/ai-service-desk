"""Track the embedding model used by each knowledge chunk."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "0002_gemini_embedding_model"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def has_embedding_model():
    return any(column["name"] == "embedding_model"
               for column in inspect(op.get_bind()).get_columns("knowledge_chunks"))


def upgrade():
    if not has_embedding_model():
        op.add_column("knowledge_chunks", sa.Column(
            "embedding_model", sa.String(120), nullable=False, server_default="nomic-embed-text"
        ))
    op.alter_column("knowledge_chunks", "embedding_model",
                    existing_type=sa.String(120), server_default="gemini-embedding-001")


def downgrade():
    if has_embedding_model():
        op.drop_column("knowledge_chunks", "embedding_model")
