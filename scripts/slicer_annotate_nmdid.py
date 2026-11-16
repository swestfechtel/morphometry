"""3D Slicer helper for annotating the held-out NMDID legs (lesser trochanter + posterior tibial plateau).

Run inside 3D Slicer's Python console (View -> Python Console)::

    exec(open('/home/simon/Work/morphometry/scripts/slicer_annotate_nmdid.py').read())
    start()            # opens the first case not yet annotated
    next_case()        # saves the current case, opens the next one
    prev_case()        # saves, goes back one
    save()             # saves without moving
    goto(7)            # saves, opens annotation-order position 7
    status()           # how many legs are complete
    start(repeat=True) # the repeat pass (>= 1 week later): 5 set-A cases, fresh points, saved apart

Each case opens with the CT in a bone window, the pipeline segmentation as an **outline** only, and
two markups nodes, ``<case>_R`` (red) and ``<case>_L`` (blue) for the **patient's** right and left
leg, each holding the five labelled points still unplaced:

* ``LT_superior``  -- most proximal point of the lesser trochanter (upper border leaving the shaft/neck)
* ``LT_tip``       -- the lesser trochanter's apex (most prominent posteromedial point)
* ``LT_inferior``  -- most distal point, where its lower border merges into the shaft
* ``tib_post_med`` -- most posterior point of the medial tibial condyle      } same axial slice, just
* ``tib_post_lat`` -- most posterior point of the lateral tibial condyle     } below the joint surface

To place them: Markups module, select the leg's node, then click each point's "place" button in the
control-point table (or enter place mode, which fills the next unplaced point). Do not rotate or
reformat the slice views. A point that cannot be placed stays unplaced -- write why in its
Description column. Do not look at pipeline landmarks or values for these cases.

Files are written as ``<annotations>/<case>_<R|L>.mrk.json`` (repeat pass: ``<annotations>/repeat/``),
in Slicer's default coordinate system (LPS in the file); re-opening a case restores its saved points,
so an interrupted session simply continues. Paths are those of the local sshfs mount.
"""
import csv
import os

import slicer
import vtk

NMDID = '/home/simon/sshfs/hpcproject/workspace_simon/nmdid'
SETS_CSV = '/home/simon/Work/morphometry/docs/evidence/nmdid_annotation/annotation_sets.csv'
ANNOTATIONS = os.path.join(NMDID, 'annotations')

LABELS = ('LT_superior', 'LT_tip', 'LT_inferior', 'tib_post_med', 'tib_post_lat')
SIDES = {'R': (0.9, 0.2, 0.2), 'L': (0.2, 0.4, 0.95)}  # patient side -> point colour
BONE_WINDOW, BONE_LEVEL = 1800, 400

_state = {'rows': [], 'index': 0, 'repeat': False, 'case': None}


def _out_dir() -> str:
    """Return the output directory of the current pass (created on demand).

    :return: The annotations directory, or its ``repeat/`` subdirectory on the repeat pass.
    """
    path = os.path.join(ANNOTATIONS, 'repeat') if _state['repeat'] else ANNOTATIONS
    os.makedirs(path, exist_ok=True)
    return path


def _markups_path(case: str, side: str) -> str:
    """Return the file a leg's points are saved to.

    :param case: The six-digit case id.
    :param side: Patient side, 'R' or 'L'.
    :return: The ``.mrk.json`` path.
    """
    return os.path.join(_out_dir(), f'{case}_{side}.mrk.json')


def _read_rows(repeat: bool) -> list:
    """Read the annotation order from the committed sets CSV.

    :param repeat: If True, keep only the cases marked for the repeat pass.
    :return: The rows (dicts with ``order``, ``case``, ``set``, ``repeat``) in annotation order.
    """
    with open(SETS_CSV) as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: int(r['order']))
    return [r for r in rows if r['repeat'] == '1'] if repeat else rows


def _new_leg_node(case: str, side: str):
    """Create a leg's markups node with the five labelled points, all unplaced.

    :param case: The six-digit case id.
    :param side: Patient side, 'R' or 'L'.
    :return: The vtkMRMLMarkupsFiducialNode.
    """
    node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', f'{case}_{side}')
    for label in LABELS:
        i = node.AddControlPoint(vtk.vtkVector3d(0, 0, 0), label)
        node.UnsetNthControlPointPosition(i)
    return node


def _leg_node(case: str, side: str):
    """Load a leg's saved points if they exist, else create fresh unplaced ones; style the node.

    :param case: The six-digit case id.
    :param side: Patient side, 'R' or 'L'.
    :return: The markups node.
    """
    path = _markups_path(case, side)
    if os.path.exists(path):
        node = slicer.util.loadMarkups(path)
        node.SetName(f'{case}_{side}')
        missing = [l for l in LABELS if l not in
                   [node.GetNthControlPointLabel(i) for i in range(node.GetNumberOfControlPoints())]]
        for label in missing:  # a point deleted by accident comes back unplaced
            node.UnsetNthControlPointPosition(node.AddControlPoint(vtk.vtkVector3d(0, 0, 0), label))
    else:
        node = _new_leg_node(case, side)
    display = node.GetDisplayNode()
    display.SetSelectedColor(*SIDES[side])
    display.SetColor(*SIDES[side])
    display.SetTextScale(2.5)
    node.SetLocked(False)
    return node


def _load(index: int) -> None:
    """Clear the scene and open the case at ``index`` of the current pass.

    :param index: Position in the current pass's row list (0-based).
    :return: None.
    """
    rows = _state['rows']
    index = max(0, min(index, len(rows) - 1))
    _state['index'] = index
    case = rows[index]['case']
    _state['case'] = case
    slicer.mrmlScene.Clear(0)

    ct = slicer.util.loadVolume(os.path.join(NMDID, 'nifti', f'case-{case}.nii.gz'))
    ct.SetName(f'case-{case}')
    vol_display = ct.GetDisplayNode()
    vol_display.SetAutoWindowLevel(False)
    vol_display.SetWindowLevel(BONE_WINDOW, BONE_LEVEL)

    labelmap = slicer.util.loadLabelVolume(os.path.join(NMDID, 'segmentations', f'case-{case}_seg.nii.gz'))
    seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', f'case-{case}_seg')
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, seg)
    slicer.mrmlScene.RemoveNode(labelmap)
    seg_display = seg.GetDisplayNode()
    seg_display.SetVisibility2DFill(False)
    seg_display.SetVisibility2DOutline(True)
    seg_display.SetVisibility3D(False)

    for side in SIDES:
        _leg_node(case, side)

    slicer.util.setSliceViewerLayers(background=ct, fit=True)
    try:
        slicer.util.selectModule('Markups')
    except RuntimeError:  # no main window (headless test run)
        pass
    print(f'[{index + 1}/{len(rows)}] case {case}' + ('  (repeat pass)' if _state['repeat'] else ''))


def _leg_complete(case: str, side: str) -> bool:
    """Whether a saved leg file has all five points placed or explained.

    :param case: The six-digit case id.
    :param side: Patient side, 'R' or 'L'.
    :return: True if the file exists and every label is placed or carries a description.
    """
    import json
    path = _markups_path(case, side)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        points = json.load(fh)['markups'][0]['controlPoints']
    done = {p['label'] for p in points if p.get('positionStatus') == 'defined' or p.get('description')}
    return set(LABELS) <= done


def save() -> None:
    """Save both legs of the current case and report unplaced points.

    :return: None.
    """
    case = _state['case']
    if case is None:
        print('no case open; call start()')
        return
    for side in SIDES:
        node = slicer.util.getNode(f'{case}_{side}')
        slicer.util.saveNode(node, _markups_path(case, side))
        unplaced = [node.GetNthControlPointLabel(i) for i in range(node.GetNumberOfControlPoints())
                    if node.GetNthControlPointPositionStatus(i) != slicer.vtkMRMLMarkupsNode.PositionDefined
                    and not node.GetNthControlPointDescription(i)]
        print(f'saved {case}_{side}' + (f'  -- unplaced without a note: {", ".join(unplaced)}' if unplaced else ''))


def start(repeat: bool = False) -> None:
    """Begin (or resume) a pass at the first case whose two legs are not both complete.

    :param repeat: If True, run the repeat pass (5 set-A cases, saved under ``repeat/``).
    :return: None.
    """
    _state['repeat'] = repeat
    _state['rows'] = _read_rows(repeat)
    first = next((i for i, r in enumerate(_state['rows'])
                  if not all(_leg_complete(r['case'], s) for s in SIDES)), len(_state['rows']) - 1)
    _load(first)


def next_case() -> None:
    """Save the current case and open the next one in the annotation order."""
    save()
    if _state['index'] + 1 >= len(_state['rows']):
        print('last case of this pass')
        return
    _load(_state['index'] + 1)


def prev_case() -> None:
    """Save the current case and open the previous one."""
    save()
    _load(_state['index'] - 1)


def goto(order: int) -> None:
    """Save the current case and open the case at a 1-based position of the current pass.

    :param order: 1-based position in the current pass.
    """
    if _state['case'] is not None:
        save()
    _load(order - 1)


def status() -> None:
    """Print how many legs of the current pass are complete (placed or explained)."""
    rows = _state['rows'] or _read_rows(_state['repeat'])
    done = sum(_leg_complete(r['case'], s) for r in rows for s in SIDES)
    print(f'{done} of {2 * len(rows)} legs complete' + (' (repeat pass)' if _state['repeat'] else ''))
