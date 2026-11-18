"""Register the Compose Telegram webhook using the configured bot."""

import argparse
import os
from urllib.parse import urlsplit

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Public HTTPS URL ending in /integrations/telegram/webhook")
    url = parser.parse_args().url
    token = os.getenv("TELEGRAM_TOKEN", "")
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    address = urlsplit(url)
    if address.scheme != "https" or not address.hostname or address.path != "/integrations/telegram/webhook":
        raise SystemExit("Configure the public HTTPS Telegram webhook in Compose.")
    if not token or not secret:
        raise SystemExit("Set TELEGRAM_TOKEN and TELEGRAM_WEBHOOK_SECRET in .env.")

    try:
        probe = httpx.post(url, json={}, timeout=10)
        if probe.status_code != 401:
            raise SystemExit(f"Public webhook returned HTTP {probe.status_code}; expected HTTP 401.")
        response = httpx.post(
            f"https://api.telegram.org/bot{token}/setWebhook",
            json={"url": url, "secret_token": secret, "allowed_updates": ["message", "callback_query"]},
            timeout=10,
        )
        if response.status_code != 200 or response.json().get("ok") is not True:
            raise SystemExit(f"Telegram rejected webhook registration (HTTP {response.status_code}).")
    except (httpx.HTTPError, ValueError):
        raise SystemExit("Could not reach the public webhook or Telegram Bot API.") from None
    print("Telegram webhook registered.")


if __name__ == "__main__":
    main()
