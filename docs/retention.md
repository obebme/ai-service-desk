# Data retention

The retention job keeps data for 365 days. Run `python -m scripts.cleanup_retention`
to report eligible row counts without changing data. After reviewing the output,
run `python -m scripts.cleanup_retention --execute` from the daily scheduler.
The default batch size is 500 rows per transaction; `--batch-size` accepts 1–5000.

The job removes only tickets with `status=closed` and `closed_at` older than the
cutoff, and removes their messages first. It also removes old knowledge documents
(their chunks are removed by the existing database cascade), audit rows, and
messages with no ticket. Messages attached to non-expired tickets are retained.
Open tickets are never selected. Each batch is idempotent and uses a transaction;
dry-run is the default.

Closing a ticket sets `closed_at`; reopening clears it. Closing it again starts a
new retention period. Apply Alembic migrations through the normal deployment
process before using the cleanup command; this implementation does not run them.
Configure and monitor a daily host/operator scheduler separately.
