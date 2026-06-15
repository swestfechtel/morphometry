"""Unit tests for the API schemas and settings (no app/db/queue required)."""
import math

import pytest
from pydantic import ValidationError

from api.schemas.docker_io import ErrorsModel, LandmarksModel, ResultsModel
from api.schemas.examination import ExaminationUpdate
from api.schemas.uploads import OrthancInstanceMeta
from api.settings import Settings


def test_results_model_valid_and_murphy_optional():
    r = ResultsModel.model_validate_json(
        '{"femoral_torsion_left": 1, "femoral_torsion_right": 2,'
        ' "tibial_torsion_left": 5, "tibial_torsion_right": 6}'
    )
    assert r.femoral_torsion_left == 1
    assert r.femoral_torsion_left_murphy is None


def test_results_model_rejects_missing_required():
    with pytest.raises(ValidationError):
        ResultsModel.model_validate_json('{"femoral_torsion_left": 1}')


def test_landmarks_sanitize_replaces_nan():
    lm = LandmarksModel.model_validate({"femur": {"Lee": {"left": {"p": [float("nan"), 1.0]}}}})
    out = lm.sanitized()
    assert out["femur"]["Lee"]["left"]["p"] == [0, 1.0]


def test_errors_model_defaults_empty():
    assert ErrorsModel.model_validate({}).errors == []


def test_examination_update_whitelist():
    upd = ExaminationUpdate.model_validate({"status": "processed", "landmarks": {"a": 1}, "ignored": "x"})
    assert upd.status.value == "processed"
    assert not hasattr(upd, "ignored")


def test_orthanc_meta_tag_lookup():
    meta = OrthancInstanceMeta.model_validate({"AccessionNumber": "ACC1", "0008,1030": "MRT Beinachsenmessung"})
    assert meta.accession_number == "ACC1"
    assert meta.tag("0008,1030") == "MRT Beinachsenmessung"


def test_settings_csv_split_and_defaults():
    s = Settings(api_keys="a, b ,c", cors_allow_origins="http://x", storage_dir="/tmp/s")
    assert s.api_keys == ["a", "b", "c"]
    assert s.cors_allow_origins == ["http://x"]
    assert s.auth_enabled is True
    assert s.resolved_database_url == "sqlite:////tmp/s/api.db"
    assert s.incoming_dir.name == "_incoming"
