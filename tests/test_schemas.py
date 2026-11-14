import pytest
from pydantic import ValidationError

from app.schemas import Analysis, TicketCreate, TicketPatch


def test_analysis_rejects_confidence_outside_zero_to_one():
    with pytest.raises(ValidationError):
        Analysis(summary="VPN issue", confidence=1.2, suggested_action="escalate")


def test_ticket_create_rejects_unknown_priority():
    with pytest.raises(ValidationError):
        TicketCreate(subject="Printer offline", description="Cannot print", priority="urgent")


def test_ticket_patch_rejects_unknown_status():
    with pytest.raises(ValidationError):
        TicketPatch(status="pending")


@pytest.mark.parametrize("payload", [
    {},
    {"status": None},
    {"priority": None},
    {"resolution": None, "status": None},
    {"unexpected": "value"},
    {"status": "pending"},
    {"priority": "urgent"},
])
def test_ticket_patch_rejects_empty_null_or_invalid_fields(payload):
    with pytest.raises(ValidationError):
        TicketPatch(**payload)


def test_ticket_patch_accepts_one_or_more_valid_changes():
    patch = TicketPatch(status="closed", priority="critical", resolution=None)
    assert patch.model_dump(exclude_unset=True) == {
        "status": "closed",
        "priority": "critical",
        "resolution": None,
    }


@pytest.mark.parametrize("status", [
    "open", "in_progress", "waiting_for_user", "resolved", "closed",
])
def test_ticket_patch_accepts_supported_statuses(status):
    assert TicketPatch(status=status).status == status


@pytest.mark.parametrize(("field", "length"), [("category", 61), ("subcategory", 81)])
def test_ticket_create_rejects_values_too_long_for_database(field, length):
    with pytest.raises(ValidationError):
        TicketCreate(subject="valid subject", description="description", **{field: "x" * length})
