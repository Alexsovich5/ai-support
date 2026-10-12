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


# --- bypass classes: input is normalised before matching -------------------

@pytest.mark.parametrize(
    "text, secret",
    [
        ("SSN １２３-４５-６７８９ on file", "６７８９"),  # full-width digits
        ("SSN 123​-45-​6789 on file", "6789"),  # zero-width split
        ("SSN 123‐45‐6789 on file", "6789"),  # unicode hyphens
        ("call 5 5 5 1 2 3 4 5 6 7 now", "4 5 6 7"),  # spaced phone digits
        ("SSN 123 45 6789 on file", "6789"),  # spaced SSN
        ("ssn is 123456789", "123456789"),  # undelimited 9 digits
        ("MRN 1 2 3 4 5 6 7 8 locked", "5 6 7 8"),  # spaced MRN
        ("ssn%3A%20123%2D45%2D6789", "6789"),  # percent-encoded
        ("SSN 123&#45;45&#45;6789", "6789"),  # HTML entities
    ],
)
def test_normalised_bypass_forms_are_redacted(text, secret):
    result = redact(text)
    assert secret not in result.text
    assert "6789" not in result.text and "4567" not in result.text and "5678" not in result.text
    assert result.found


@pytest.mark.parametrize(
    "text",
    [
        "DOB 1980-02-01 on the form",
        "born on March 2, 1980 per the badge",
        "date of birth: 2 March 1980",
        "D.O.B. 01.02.80",
        "birthdate 02/01/1980",
    ],
)
def test_dates_of_birth_in_several_formats(text):
    result = redact(text)
    assert "1980" not in result.text and ".80" not in result.text
    assert "dob" in result.found


def test_url_userinfo_is_redacted():
    result = redact("Proxy is https://bob:hunter2@intranet.example/path")
    assert "hunter2" not in result.text and "bob:" not in result.text
    assert "intranet.example/path" in result.text


def test_street_address_and_labelled_patient_name_are_redacted():
    result = redact("patient John Smith lives at 42 Elm Street and cannot log in")
    assert "John Smith" not in result.text
    assert "42 Elm Street" not in result.text
    assert {"name", "address"} <= set(result.found)


def test_patient_portal_product_name_is_not_a_patient_name():
    text = "Patient Portal login fails for the front desk"
    assert redact(text).text == text


def test_phi_inside_embedded_json_is_redacted():
    text = 'payload {"patient": {"ssn": "123456789", "patient_name": "Jane Roe", "dob": "1980-02-01"}}'
    result = redact(text)
    for secret in ("123456789", "Jane Roe", "1980-02-01"):
        assert secret not in result.text


def test_phi_inside_base64_fragment_is_redacted():
    import base64

    blob = base64.b64encode(b"patient SSN 123-45-6789 locked out").decode()
    result = redact(f"attachment {blob} please check")
    assert blob not in result.text
    assert "attachment" in result.text and "please check" in result.text
    assert result.found


def test_redact_structure_walks_nested_keys_and_values():
    from src.phi_filter import redact_structure

    data = {
        "ticket": "VPN drops",
        "patient": {"mrn": "A99887766", "notes": ["call 555-123-4567"], "name": "Jane Roe"},
        "patient_name": "John Smith",
    }
    cleaned, found = redact_structure(data)
    flat = repr(cleaned)
    for secret in ("A99887766", "555-123-4567", "Jane Roe", "John Smith"):
        assert secret not in flat
    assert cleaned["ticket"] == "VPN drops"
    assert set(cleaned) == set(data)
    assert found


# --- fail closed --------------------------------------------------------------

def test_filter_error_withholds_the_text(monkeypatch):
    import src.phi_filter as phi

    def boom(_text):
        raise RuntimeError("regex engine failure")

    monkeypatch.setattr(phi, "_redact_normalised", boom)
    result = phi.redact("SSN 123-45-6789")
    assert result.text == phi.WITHHELD_TEXT
    assert "withheld" in result.found


def test_text_over_the_size_cap_is_withheld():
    import src.phi_filter as phi

    result = phi.redact("SSN 123-45-6789 " + "a" * phi.MAX_REDACT_CHARS)
    assert result.text == phi.WITHHELD_TEXT
    assert "withheld" in result.found


@pytest.mark.parametrize(
    "unit",
    ["a.", "/a", "1-", "a@", "a@a.", "1 ", "​1", "%2", "&#4", "patient A ", "42 Aa "],
)
def test_adversarial_input_finishes_in_bounded_time(unit):
    import time

    import src.phi_filter as phi

    near_cap = unit * ((phi.MAX_REDACT_CHARS - 10) // len(unit))
    started = time.perf_counter()
    phi.redact(near_cap)
    phi.redact(unit * (1_000_000 // len(unit)))  # 1 MB: over the cap
    assert time.perf_counter() - started < 3.0
