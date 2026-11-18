"""Create the service configuration from the tracked template."""

import os
from pathlib import Path
import secrets


ROOT = Path(__file__).resolve().parents[1]
TOKENS = {
    "GENERATE_A_UNIQUE_PASSWORD",
    "replace-with-telegram-webhook-secret",
    "replace-with-long-agent-token",
    "replace-with-long-admin-token",
    "replace-with-a-long-random-session-signing-secret",
    "replace-with-a-long-random-key",
    "replace-with-a-random-workflow-ingress-token",
    "replace-with-a-different-random-workflow-api-token",
}


def main():
    contents = (ROOT / ".env.example").read_text(encoding="utf-8")
    for placeholder in TOKENS:
        contents = contents.replace(placeholder, secrets.token_urlsafe(32))

    try:
        descriptor = os.open(ROOT / ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SystemExit(".env already exists. Edit it directly to change your configuration.")
    with os.fdopen(descriptor, "w", encoding="utf-8") as configuration:
        configuration.write(contents)
    print("Created .env. Add your provider credentials before starting the integrations.")


if __name__ == "__main__":
    main()
