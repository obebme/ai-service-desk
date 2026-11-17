"""Forward only the Telegram webhook from the tunnel origin."""

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WEBHOOK_PATH = "/integrations/telegram/webhook"
UPSTREAM = os.getenv("TELEGRAM_INGRESS_UPSTREAM", "http://127.0.0.1:8000")


class WebhookIngress(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_error(404)

    def do_POST(self):
        if self.path != WEBHOOK_PATH:
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if not 0 < size <= 262144:
            self.send_error(413)
            return
        request = Request(
            f"{UPSTREAM}{WEBHOOK_PATH}",
            data=self.rfile.read(size),
            headers={
                "Content-Type": "application/json",
                "X-Telegram-Bot-Api-Secret-Token": self.headers.get(
                    "X-Telegram-Bot-Api-Secret-Token", ""
                ),
            },
        )
        try:
            with urlopen(request, timeout=10) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        except OSError:
            status = 502
        self.send_response(status)
        self.end_headers()


if __name__ == "__main__":
    host = os.getenv("TELEGRAM_INGRESS_BIND", "127.0.0.1")
    with ThreadingHTTPServer((host, 18081), WebhookIngress) as server:
        server.serve_forever()
