import pytest

from src.phi_filter import RedactionResult, redact


@pytest.mark.parametrize(
    "text, kind, secret",
    [
        ("Patient MRN 12345678 cannot log in", "mrn", "12345678"),
        ("Patient MRN: 12345678 cannot log in", "mrn", "12345678"),
        ("SSN on file is 123-45-6789", "ssn", "123-45-6789"),
        ("Lookup failed for DOB 01/02/1980 today", "dob", "01/02/1980"),
        ("Call me back at +1 555-123-4567 please", "phone", "555-123-4567"),
        ("Call me back at (555) 123-4567 please", "phone", "123-4567"),
        ("Send the report to jane@example.org now", "email", "jane@example.org"),
    ],
)
def test_redacts_each_identifier_kind(text, kind, secret):
    result = redact(text)
    assert isinstance(result, RedactionResult)
    assert secret not in result.text
    assert f"[REDACTED:{kind}]" in result.text
    assert result.found == [kind]


def test_redacts_full_phone_including_country_code():
    result = redact("Call +1 555-123-4567.")
    assert result.text == "Call [REDACTED:phone]."


def test_redacts_mixed_text_and_reports_kinds_in_order_of_appearance():
    text = (
        "MRN 12345678, DOB 01/02/1980, email jane@example.org, "
        "phone +1 555-123-4567, SSN 123-45-6789"
    )
    result = redact(text)
    assert result.text == (
        "[REDACTED:mrn], [REDACTED:dob], email [REDACTED:email], "
        "phone [REDACTED:phone], SSN [REDACTED:ssn]"
    )
    assert result.found == ["mrn", "dob", "email", "phone", "ssn"]


def test_found_lists_each_kind_once():
    result = redact("a@example.org and b@example.org")
    assert result.text == "[REDACTED:email] and [REDACTED:email]"
    assert result.found == ["email"]


@pytest.mark.parametrize(
    "text",
    [
        "Windows Update fails with error 0x80070005 after reboot",
        "Error code 0x800F0922 during install",
        "Cannot reach the gateway at 192.168.10.254 or 10.0.0.1",
        "Following up on ticket TKT-20240115-0042",
        "See ticket #48213 and INC0012345",
        "Outlook build 16.0.17531.20140 crashes",
        "Printer on floor 3, room 214, is jammed",
    ],
)
def test_leaves_it_identifiers_untouched(text):
    result = redact(text)
    assert result.text == text
    assert result.found == []


def test_empty_text():
    assert redact("") == RedactionResult(text="", found=[])
