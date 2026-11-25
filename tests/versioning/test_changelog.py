"""Tests for versioning.changelog."""
from versioning.changelog import section

TEXT = "## [1.1.0] - 2026-12-01\n- api: x\n\n## [1.0.0] - 2026-11-25\n- y"


def test_section():
    assert section(TEXT, "1.1.0") == "- api: x"
    assert section(TEXT, "1.0.0") == "- y"
    assert section(TEXT, "2.0.0") is None
