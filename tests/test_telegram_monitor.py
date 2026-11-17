import httpx

from scripts import monitor_telegram as monitor


def test_webhook_checks_registration_and_rejects_offline_tunnel():
    webhook = "https://example.ngrok-free.dev/integrations/telegram/webhook"

    def probe(registered, status):
        def respond(request):
            if request.url.host == "api.telegram.org":
                return httpx.Response(200, json={"ok": True, "result": {"url": registered}})
            assert request.method == "POST"
            assert "x-telegram-bot-api-secret-token" not in request.headers
            return httpx.Response(status)

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            return monitor.check_webhook(client, "https://api.telegram.org/botTEST", webhook)

    assert probe(webhook, 401) is None
    assert probe(webhook, 404) == "webhook HTTP 404"
    assert probe("", 401) == "Telegram webhook registration mismatch"


def test_queue_thresholds_include_count_and_oldest_age():
    assert monitor.queue_problem(0, 0, 10, 120) is None
    assert monitor.queue_problem(9, 119, 10, 120) is None
    assert monitor.queue_problem(10, 0, 10, 120) == "queue: 10 pending; oldest 0s"
    assert monitor.queue_problem(1, 120, 10, 120) == "queue: 1 pending; oldest 120s"


def test_alert_retries_errors_and_sends_recovery_once(monkeypatch):
    outgoing = []

    def send(client, bot, chat_id, message):
        outgoing.append(message)
        return len(outgoing) != 1

    monkeypatch.setattr(monitor, "send_alert", send)
    previous = frozenset()
    problem = {"webhook": "webhook HTTP 404"}
    previous = monitor.notify(None, "", "", previous, problem)
    assert not previous
    previous = monitor.notify(None, "", "", previous, problem)
    assert previous == {"webhook"}
    assert monitor.notify(None, "", "", previous, {"webhook": "webhook HTTP 502"}) == previous
    previous = monitor.notify(None, "", "", previous, {})
    assert not previous
    assert not monitor.notify(None, "", "", previous, {})
    assert outgoing == [
        "Service Desk: webhook HTTP 404",
        "Service Desk: webhook HTTP 404",
        "Service Desk: monitoring checks recovered",
    ]
