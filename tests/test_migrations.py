import os
import subprocess
import sys
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url


def test_upgrade_from_initial_schema_preserves_closed_ticket():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL migration tests")

    schema = f"test_migrations_{uuid4().hex}"
    admin_engine = create_engine(database_url)
    url = make_url(database_url).update_query_dict({"options": f"-csearch_path={schema},public"})
    engine = create_engine(url)
    environment = {
        **os.environ,
        "DATABASE_URL": url.render_as_string(hide_password=False),
        "EMBEDDING_DIMENSIONS": "768",
    }

    with admin_engine.begin() as connection:
        connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        # Keep Alembic from discovering public.alembic_version through search_path.
        connection.exec_driver_sql(
            f'CREATE TABLE "{schema}".alembic_version (version_num VARCHAR(32) PRIMARY KEY)'
        )

    try:
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "0001_initial"],
                       env=environment, check=True)
        inspector = inspect(engine)
        assert {"closed_at", "assigned_agent"}.isdisjoint(
            column["name"] for column in inspector.get_columns("tickets")
        )
        assert {"telegram_chat_id", "telegram_delivery_status", "telegram_delivery_updated_at",
                "processing_due_at"}.isdisjoint(
            column["name"] for column in inspector.get_columns("messages")
        )
        assert "embedding_model" not in {column["name"] for column in inspector.get_columns("knowledge_chunks")}

        requester, ticket, message = uuid4(), uuid4(), uuid4()
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO users (id, display_name, role, created_at)
                VALUES (:id, 'Historical user', 'user', now())
            """), {"id": requester})
            connection.execute(text("""
                INSERT INTO tickets (id, ticket_number, requester_id, category, subcategory,
                                     priority, status, subject, description, created_at, updated_at)
                VALUES (:id, 'SD-HISTORY', :requester, 'other', 'other', 'medium', 'closed',
                        'Historical ticket', 'Preserve this ticket', now(), now())
            """), {"id": ticket, "requester": requester})
            connection.execute(text("""
                INSERT INTO messages (id, requester_id, ticket_id, channel, sender_type, body, created_at)
                VALUES (:id, :requester, :ticket, 'api', 'user', 'Historical message', now())
            """), {"id": message, "requester": requester, "ticket": ticket})

        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"],
                       env=environment, check=True)
        subprocess.run([sys.executable, "-m", "alembic", "current", "--check-heads"],
                       env=environment, check=True)
        with engine.connect() as connection:
            assert connection.execute(text("""
                SELECT description, closed_at = updated_at, assigned_agent
                FROM tickets WHERE id = :id
            """), {"id": ticket}).one() == ("Preserve this ticket", True, None)
            assert connection.execute(text("""
                SELECT ticket_id, body, telegram_delivery_status, processing_due_at
                FROM messages WHERE id = :id
            """), {"id": message}).one() == (ticket, "Historical message", None, None)
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        admin_engine.dispose()
