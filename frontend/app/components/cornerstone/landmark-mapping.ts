// Pure helpers mapping between the stored torsion landmark tree (voxel indices,
// sub-volume-relative z) and flat combined-volume voxel coordinates. No
// Cornerstone/DOM dependency, so this is the unit-testable isolation point for:
//   1. the hip/knee/ankle z-offset handling (preserving the old SVG component's
//      `add_arrays(p, [0,0,offset])` semantics), and
//   2. any axis permutation/flip between the stored numpy-LPI [x,y,z] order and
//      the NIfTI loader's vtk image-index order (fix it ONLY in to/fromLoaderIndex).
import { Point3D, TorsionLandmarks } from '@/app/types';
import {
  femoralProximalAngle,
  femoralDistalAngle,
  tibialProximalAngle,
  tibialDistalAngle,
} from '@/app/utils';

export interface Offsets {
  kneeOffset: number;   // combined-volume z where the knee sub-volume starts
  ankleOffset: number;  // combined-volume z where the ankle sub-volume starts
  zMax: number;         // total z of the combined volume (exclusive upper bound)
}

export interface LandmarkRef {
  path: string[];   // e.g. ['femur', 'Lee', 'left', 'hip_start']
  voxel: Point3D;   // combined-volume voxel index (offset applied)
}

function isPoint3D(v: unknown): v is Point3D {
  return Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === 'number');
}

/** Combined-volume z-range [lo, hi) the point named `key` must stay within. */
export function zRangeForKey(key: string, o: Offsets): [number, number] {
  if (key.startsWith('ankle')) return [o.ankleOffset, o.zMax];
  if (key.startsWith('knee')) return [o.kneeOffset, o.ankleOffset];
  return [0, o.kneeOffset]; // hip_*
}

/** z added to a stored (sub-volume-relative) point to place it in combined space. */
export function zOffsetForKey(key: string, o: Offsets): number {
  return zRangeForKey(key, o)[0];
}

/**
 * Adapter between the stored numpy-LPI voxel order and the NIfTI loader's vtk
 * image index order. Identity by default; if the voxel↔world verification shows
 * a flip/permutation, change ONLY these two functions.
 */
export function toLoaderIndex(v: Point3D): Point3D {
  return [v[0], v[1], v[2]];
}
export function fromLoaderIndex(v: Point3D): Point3D {
  return [v[0], v[1], v[2]];
}

export function getByPath(root: TorsionLandmarks, path: string[]): Point3D | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return path.reduce<any>((acc, k) => (acc ? acc[k] : null), root) as Point3D | null;
}

function setByPath(root: TorsionLandmarks, path: string[], value: Point3D): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let node: any = root;
  for (let i = 0; i < path.length - 1; i++) node = node[path[i]];
  node[path[path.length - 1]] = value;
}

/** Flatten the landmark tree into combined-volume voxel refs (loader index order). */
export function listLandmarks(landmarks: TorsionLandmarks, o: Offsets): LandmarkRef[] {
  const out: LandmarkRef[] = [];
  const walk = (node: unknown, path: string[]) => {
    if (isPoint3D(node)) {
      const key = path[path.length - 1];
      const combined: Point3D = [node[0], node[1], node[2] + zOffsetForKey(key, o)];
      out.push({ path, voxel: toLoaderIndex(combined) });
      return;
    }
    if (node && typeof node === 'object') {
      for (const k of Object.keys(node as Record<string, unknown>)) {
        walk((node as Record<string, unknown>)[k], [...path, k]);
      }
    }
  };
  walk(landmarks, []);
  return out;
}

/**
 * A torsion reference line: a start/end landmark pair that shares an axial slice and
 * defines one measurement axis, plus the labels used to annotate it in the viewer.
 */
/** Which signed torsion component a reference line represents (see {@link referenceLineAngle}). */
export type AngleKind = 'femoralProximal' | 'femoralDistal' | 'tibialProximal' | 'tibialDistal';

export interface ReferenceLineSpec {
  // Landmark path(s) each endpoint maps to. Usually one, but the femoral condylar line
  // is shared by both methods (Lee & Murphy have identical knee points), so editing it
  // must write both — hence a list.
  startPaths: string[][];
  endPaths: string[][];
  startLabel: string;    // anatomical name of the start point, e.g. 'Femoral head centre'
  endLabel: string;      // anatomical name of the end point
  colorKey: 'neck' | 'condyle' | 'tibia' | 'malleolus'; // groups lines by axis type for colouring
  side: 'left' | 'right';
  angleKind: AngleKind;  // proximal/distal component this line contributes to the torsion
}

/**
 * The signed axial-plane angle (degrees) of a reference line, i.e. the proximal or
 * distal component of its torsion, read from a landmark tree. Returns null if the
 * endpoints are missing. Uses the line's first path (all paths share the same point).
 *
 * :param tree: the stored torsion landmark tree.
 * :param spec: the reference line to measure.
 * :return: the signed angle in degrees, or null.
 */
export function referenceLineAngle(tree: TorsionLandmarks, spec: ReferenceLineSpec): number | null {
  const s = getByPath(tree, spec.startPaths[0]);
  const e = getByPath(tree, spec.endPaths[0]);
  if (!s || !e) return null;
  switch (spec.angleKind) {
    case 'femoralProximal': return femoralProximalAngle(s, e, spec.side);
    case 'femoralDistal': return femoralDistalAngle(s, e, spec.side);
    case 'tibialProximal': return tibialProximalAngle(s, e, spec.side);
    case 'tibialDistal': return tibialDistalAngle(s, e, spec.side);
  }
}

/**
 * Localized anatomical labels drawn on the reference lines. Resolved from the message
 * catalog by the viewport hook (which has the translator) and passed in here, so this
 * pure module stays free of any React/i18n dependency.
 */
export interface LandmarkLabels {
  femoralHeadCentre: string;
  femoralNeckCentre: string;
  posteriorCondyleMedial: string;
  posteriorCondyleLateral: string;
  posteriorTibialCondyleMed: string;
  posteriorTibialCondyleLat: string;
  medialMalleolus: string;
  lateralMalleolus: string;
}

/**
 * Enumerate the reference lines present in a landmark tree, with anatomical labels.
 *
 * Femur (both sides): the femoral neck axis (hip_start→hip_end), one per method (Lee &
 * Murphy differ only in the proximal neck landmark), and the posterior condylar axis
 * (knee_start→knee_end), drawn ONCE per side because the distal landmarks are identical
 * across methods — editing it updates every present method's knee points. Tibia (both
 * sides): the posterior tibial axis (knee_start→knee_end) and the transmalleolar axis
 * (ankle_start→ankle_end). Missing methods/sides are skipped.
 *
 * The knee condyle endpoints (femoral condylar + tibial condylar) are medial/lateral
 * depending on image side: on the left half `knee_start` is lateral, on the right half
 * `knee_start` is medial. (Ankle malleoli are named by fixed start/end for now.)
 *
 * :param landmarks: the stored torsion landmark tree.
 * :return: one {@link ReferenceLineSpec} per axis actually present in the tree.
 */
export function buildReferenceLines(landmarks: TorsionLandmarks, labels: LandmarkLabels): ReferenceLineSpec[] {
  const specs: ReferenceLineSpec[] = [];
  // Which knee endpoint is medial vs lateral flips with the image side.
  const kneeLabels = (side: string, medial: string, lateral: string) =>
    side === 'left' ? { start: lateral, end: medial } : { start: medial, end: lateral };

  const femur = (landmarks?.femur ?? {}) as unknown as Record<string, Record<string, unknown>>;
  const femurMethods = ['Lee', 'Murphy'];
  for (const side of ['left', 'right']) {
    const methodsPresent = femurMethods.filter((m) => femur[m]?.[side]);
    if (!methodsPresent.length) continue;
    // Femoral neck axis: one line per method (the neck landmarks are method-specific).
    for (const method of methodsPresent) {
      specs.push({
        startPaths: [['femur', method, side, 'hip_start']],
        endPaths: [['femur', method, side, 'hip_end']],
        startLabel: labels.femoralHeadCentre, endLabel: labels.femoralNeckCentre,
        colorKey: 'neck', side: side as 'left' | 'right', angleKind: 'femoralProximal',
      });
    }
    // Posterior condylar axis: ONE line, editing writes the knee points of all methods.
    // Order methods so a non-degenerate knee point drives the geometry — a method whose
    // try-block failed leaves its knee at [0, 0, 0] (fallback), which must not be picked
    // as the line's position (startPaths[0]) if the other method has a real point.
    const kneeOk = (m: string): boolean => {
      const ks = getByPath(landmarks, ['femur', m, side, 'knee_start']);
      return !!ks && (ks[0] !== 0 || ks[1] !== 0 || ks[2] !== 0);
    };
    const kneeOrder = [...methodsPresent.filter(kneeOk), ...methodsPresent.filter((m) => !kneeOk(m))];
    const cond = kneeLabels(side, labels.posteriorCondyleMedial, labels.posteriorCondyleLateral);
    specs.push({
      startPaths: kneeOrder.map((m) => ['femur', m, side, 'knee_start']),
      endPaths: kneeOrder.map((m) => ['femur', m, side, 'knee_end']),
      startLabel: cond.start, endLabel: cond.end,
      colorKey: 'condyle', side: side as 'left' | 'right', angleKind: 'femoralDistal',
    });
  }
  const tibia = (landmarks?.tibia ?? {}) as unknown as Record<string, unknown>;
  for (const side of ['left', 'right']) {
    if (!tibia[side]) continue;
    const base = ['tibia', side];
    const cond = kneeLabels(side, labels.posteriorTibialCondyleMed, labels.posteriorTibialCondyleLat);
    specs.push({
      startPaths: [[...base, 'knee_start']], endPaths: [[...base, 'knee_end']],
      startLabel: cond.start, endLabel: cond.end,
      colorKey: 'tibia', side: side as 'left' | 'right', angleKind: 'tibialProximal',
    });
    specs.push({
      startPaths: [[...base, 'ankle_start']], endPaths: [[...base, 'ankle_end']],
      startLabel: labels.medialMalleolus, endLabel: labels.lateralMalleolus,
      colorKey: 'malleolus', side: side as 'left' | 'right', angleKind: 'tibialDistal',
    });
  }
  return specs;
}

/**
 * Write an edited combined-volume voxel (loader index order) back into a cloned
 * landmark tree: undo the axis adapter, clamp z into the point's sub-volume,
 * remove the offset, and round to integer indices.
 */
export function applyEditedVoxel(
  landmarks: TorsionLandmarks,
  path: string[],
  loaderVoxel: Point3D,
  o: Offsets,
): TorsionLandmarks {
  const combined = fromLoaderIndex(loaderVoxel);
  const key = path[path.length - 1];
  const [lo, hi] = zRangeForKey(key, o);
  const z = Math.min(hi - 1, Math.max(lo, Math.round(combined[2])));
  const stored: Point3D = [Math.round(combined[0]), Math.round(combined[1]), z - lo];
  const clone = structuredClone(landmarks);
  setByPath(clone, path, stored);
  return clone;
}
