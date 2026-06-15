"""Orthanc routing-rule matching.

Replaces the in-endpoint ``globals()`` reflection. ``filter_rules.json`` maps sets
of DICOM tag values to an ExaminationType + ModelJob; matching keeps the existing
file format. (The instance buffering/debounce lives in the router + worker, not
here.)
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("api")


def load_rules(rules_path: Path) -> list[dict]:
    return json.loads(Path(rules_path).read_text()).get("Rules", [])


def match_rule(tags: dict, rules: list[dict]) -> dict | None:
    """Return the first rule whose DICOM-tag conditions all match ``tags``, else None.

    ``tags`` is the simplified-JSON dict forwarded by the Orthanc plugin; each rule
    lists ``[tag, expected_value]`` pairs that must all be present and equal.
    """
    for rule in rules:
        if all(tags.get(tag) == value for tag, value in rule.get("DICOM Tags", [])):
            return rule
    return None
