# n8n workflows

## Incoming Request

`workflows/incoming-request.json` accepts an authenticated message ID from FastAPI and asks the backend to process it. Import this workflow in n8n, then create two **Header Auth** credentials:

1. **Service Desk ingress token**: header `X-Workflow-Token`, value matching `N8N_INGRESS_TOKEN` in the API `.env`.
2. **Service Desk workflow token**: header `X-Workflow-Token`, value matching `WORKFLOW_TOKEN` in the API `.env`.

Set `N8N_WEBHOOK_URL=http://n8n:5678/webhook/service-desk-incoming`, `N8N_INGRESS_TOKEN`, and `WORKFLOW_TOKEN` in `.env`; use different long random values for the two tokens. Set `TELEGRAM_TOKEN` to the BotFather token and `TELEGRAM_WEBHOOK_SECRET` to a separate webhook secret. Keep all credentials in the deployment secret store or `.env` (never in workflow JSON or source control).

Start the stack with `docker compose --profile automation up -d --build`. Import the workflow, create and assign the two credentials, then activate it. Only then register Telegram's webhook to the public HTTPS URL ending in `/integrations/telegram/webhook`, setting its `secret_token` to exactly the `TELEGRAM_WEBHOOK_SECRET` value. Telegram must send `X-Telegram-Bot-Api-Secret-Token`; the API rejects a missing or incorrect value. Keep the n8n editor port restricted to a trusted network.

Processing, RAG sufficiency checks, deterministic department routing, ticket creation and fallback behavior remain in FastAPI. That keeps retries idempotent and prevents workflow edits from bypassing backend authorization. Add further notification and scheduled workflows after their API event contracts are defined.

The background worker waits for the workflow result. Telegram receives HTTP 200 after the update and its processing schedule are committed to PostgreSQL. A connection error, timeout, or non-2xx n8n response triggers direct processing of the same message. If direct processing fails, the worker schedules a retry. FastAPI stores new text updates by chat ID and message ID. Both paths use the same row lock, reuse the saved result, and skip Telegram delivery once an attempt has been recorded. Configure Telegram to send `message` and `callback_query` updates. Private-chat button callbacks and photos captioned `/add TICKET_NUMBER` enter the same queue. Unsupported updates are acknowledged and ignored. Leave `N8N_WEBHOOK_URL` empty to use direct FastAPI processing from the outset.
