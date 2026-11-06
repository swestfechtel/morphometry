"""The torsion container's QC flag strings survive the API's two schemas.

``morphometry/docker/compute_torsion.py`` has emitted ``*_murphy_qc_flags`` since ``822181b`` and
gained the Lee and condylar families since, but every one of them was **computed and thrown away**
on the API path: ``api/schemas/docker_io.py::ResultsModel`` has ``extra="ignore"`` and did not
declare the fields, so ``api/tasks/torsion.py``'s ``results.model_dump()`` persisted only the six
angles. CLAUDE.md recorded plumbing them through as needing "a schema and DB change"; it needs
**no database change at all**, because ``Examination.torsion_values`` is a JSON column
(``api/db/models.py:37``) and the task persists the dump wholesale, so declaring the fields is
sufficient to store them.

These tests pin the three properties that make that safe:

1. the eight flag fields survive ``ResultsModel`` instead of being dropped;
2. the result splats into ``TorsionValues`` without a validation error -- the reason the fields
   default to ``""`` and not ``None``, since ``TorsionValues`` declares them as ``str``;
3. a results dict from a container old enough not to emit the keys still validates, so an
   examination processed before this change is readable rather than a 500.

Property 3 carries a limitation worth stating rather than hiding: an absent flag and a sound
landmark both come out ``""``, so a client cannot tell "the QC did not run" from "the QC found
nothing". That is a consequence of the container recording no provenance for its own QC, not of
this change, and ``TorsionValues``' docstring says so.
"""
import pytest

from api.schemas.docker_io import ResultsModel
from api.schemas.examination import TorsionValues
from api.serializers import _sanitize

#: Every QC flag key the container writes, in the container's own PATIENT-side naming. Note this
#: crosses over relative to ``tests/golden/``, whose keys are IMAGE side.
QC_FLAG_KEYS = (
    'femoral_torsion_left_murphy_qc_flags',
    'femoral_torsion_right_murphy_qc_flags',
    'femoral_torsion_left_lee_qc_flags',
    'femoral_torsion_right_lee_qc_flags',
    'femur_condylar_line_left_qc_flags',
    'femur_condylar_line_right_qc_flags',
    'tibia_condylar_line_left_qc_flags',
    'tibia_condylar_line_right_qc_flags',
)

#: The six angles every ``results.json`` carries, which ``ResultsModel`` requires.
ANGLES = {
    'femoral_torsion_left': 11.0,
    'femoral_torsion_right': 12.0,
    'femoral_torsion_left_murphy': 30.0,
    'femoral_torsion_right_murphy': 31.0,
    'tibial_torsion_left': 37.0,
    'tibial_torsion_right': 38.0,
}


def test_the_qc_flag_fields_survive_the_results_model():
    """A non-empty flag string reaches the persisted dump instead of being dropped.

    The regression this exists for: with the fields undeclared and ``extra="ignore"``, every one
    of these was silently discarded between the container and the database.
    """
    raw = dict(ANGLES)
    raw.update({k: f'flag_for_{i}:1.23' for i, k in enumerate(QC_FLAG_KEYS)})

    dumped = ResultsModel.model_validate(raw).model_dump()

    for i, key in enumerate(QC_FLAG_KEYS):
        assert dumped[key] == f'flag_for_{i}:1.23', f'{key} did not survive ResultsModel'


def test_the_dump_splats_into_the_response_schema():
    """``api/serializers.py:42`` does ``TorsionValues(**_sanitize(row.torsion_values))``.

    So the two schemas have to agree on the field types. ``TorsionValues`` declares the flags as
    ``str``, which is why ``ResultsModel`` defaults them to ``""`` rather than ``None`` -- a
    ``None`` here would raise on every examination whose container did not emit the key.
    """
    raw = dict(ANGLES)
    raw['femoral_torsion_left_murphy_qc_flags'] = 'murphy_axis_short:6.37'

    values = TorsionValues(**_sanitize(ResultsModel.model_validate(raw).model_dump()))

    assert values.femoral_torsion_left_murphy_qc_flags == 'murphy_axis_short:6.37'
    assert values.tibia_condylar_line_right_qc_flags == ''
    assert values.femoral_torsion_left == pytest.approx(11.0)


def test_a_results_dict_without_any_flag_keys_still_validates():
    """An examination processed by a container older than these keys must stay readable.

    It comes out with every flag ``''``, which is indistinguishable from a sound landmark -- the
    limitation the module docstring records, and the reason this test asserts the *absence* of an
    exception rather than a meaningful value.
    """
    values = TorsionValues(**_sanitize(ResultsModel.model_validate(dict(ANGLES)).model_dump()))

    assert all(getattr(values, key) == '' for key in QC_FLAG_KEYS)


def test_the_flag_fields_are_not_required_of_the_container():
    """``ResultsModel`` must not start rejecting a container that omits them.

    The torsion image is rebuilt separately from the API, so the two can be out of step in either
    direction; a missing flag key must never turn a successful run into a validation failure.
    """
    model = ResultsModel.model_validate(dict(ANGLES))

    assert model.femoral_torsion_left_murphy_qc_flags == ''
    assert model.femoral_torsion_left == pytest.approx(11.0)
