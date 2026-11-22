from pydantic import PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://servicedesk:servicedesk@localhost:5432/servicedesk"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"
    telegram_token: str = ""
    telegram_webhook_secret: str = ""
    telegram_queue_limit: PositiveInt = 100
    telegram_workers: PositiveInt = 1
    n8n_webhook_url: str = ""
    n8n_ingress_token: str = ""
    workflow_token: str = ""
    user_api_tokens: dict[str, str] = {}
    agent_api_token: str = ""
    panel_agent_tokens: dict[str, str] = {}
    admin_api_token: str = ""
    session_signing_secret: str | None = None
    embedding_dimensions: int = 768
    confidence_threshold: float = 0.65

    def validate_user_api_tokens(self) -> None:
        tokens = self.user_api_tokens.values()
        agents = self.panel_agent_tokens
        privileged_tokens = {self.agent_api_token, self.admin_api_token}
        if (any(not name.strip() or len(name) > 80 for name in agents)
                or any(not token.strip() or not token.isascii() for token in agents.values())
                or len(set(agents.values())) != len(agents)
                or set(agents.values()) & privileged_tokens):
            raise RuntimeError("PANEL_AGENT_TOKENS requires unique names and personal tokens")
        privileged_tokens.update(agents.values())
        if (any(not token.strip() or not token.isascii() for token in tokens)
                or len(set(tokens)) != len(tokens)
                or set(tokens) & privileged_tokens):
            raise RuntimeError("USER_API_TOKENS requires a separate non-empty ASCII token per user")

    def validate_session_signing_secret(self) -> str:
        if not self.session_signing_secret or not self.session_signing_secret.strip():
            raise RuntimeError(
                "SESSION_SIGNING_SECRET is required; set it in the environment before starting the service"
            )
        if len(self.session_signing_secret.encode("utf-8")) < 32:
            raise RuntimeError(
                "SESSION_SIGNING_SECRET must contain at least 32 bytes; generate a random secret before starting the service"
            )
        return self.session_signing_secret


settings = Settings()
