import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AnalysisCategory = Literal["network", "account", "hardware", "software", "security", "other"]
AnalysisSubcategory = Literal[
    "vpn", "internet", "authentication", "printer", "software", "security", "other"
]


def _normalize_analysis_label(value: object, aliases: dict[str, str]) -> object:
    if not isinstance(value, str):
        return value
    normalized = " ".join(value.strip().lower().split())
    return aliases.get(normalized, normalized)


class Analysis(BaseModel):
    category: AnalysisCategory = "other"
    subcategory: AnalysisSubcategory = "other"
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    summary: str
    sentiment: Literal["negative", "neutral", "positive"] = "neutral"
    needs_human: bool = False
    confidence: float = Field(ge=0, le=1)
    suggested_action: str

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> object:
        return _normalize_analysis_label(value, {"access management": "account"})

    @field_validator("subcategory", mode="before")
    @classmethod
    def normalize_subcategory(cls, value: object) -> object:
        return _normalize_analysis_label(value, {"vpn connection": "vpn"})

    @field_validator("priority", "sentiment", mode="before")
    @classmethod
    def normalize_enum_labels(cls, value: object) -> object:
        return _normalize_analysis_label(value, {})


class TicketCreate(BaseModel):
    subject: str = Field(min_length=3, max_length=240)
    description: str = Field(min_length=1, max_length=20000)
    category: str = Field(default="other", max_length=60)
    subcategory: str = Field(default="other", max_length=80)
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class TicketPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["open", "in_progress", "waiting_for_user", "resolved", "closed"] | None = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    resolution: str | None = Field(default=None, max_length=20000)

    @model_validator(mode="after")
    def validate_patch(self):
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided")
        for field in ("status", "priority"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class MessageCreate(BaseModel):
    body: str = Field(min_length=1, max_length=20000)


class MessageView(BaseModel):
    id: uuid.UUID
    ticket_id: uuid.UUID | None
    sender_type: Literal["user", "agent"]
    body: str
    created_at: datetime
    model_config = {"from_attributes": True}


class WorkflowMessageRequest(BaseModel):
    message_id: uuid.UUID


class TicketView(BaseModel):
    id: uuid.UUID
    ticket_number: str
    requester_id: uuid.UUID
    category: str
    subcategory: str
    priority: str
    status: str
    subject: str
    description: str
    resolution: str | None
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}
