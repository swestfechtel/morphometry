'use client';

// A custom Cornerstone3D annotation tool for the torsion reference lines.
//
// Each torsion measurement is defined by two landmarks that share an axial slice
// (e.g. the femoral-head centre and the femoral-neck centre define the femoral neck
// axis). This tool renders that pair as ONE annotation: a thick, bright connecting
// line with two large, always-visible endpoint handles, an anatomical label next to
// each endpoint, and the axis name near the middle.
//
// It subclasses the stock `LengthTool` purely to inherit its proven interaction
// machinery (hit-testing, handle selection, drag, ANNOTATION_MODIFIED events). Only
// the rendering is overridden — no length is computed or shown. Editing a handle
// fires ANNOTATION_MODIFIED, which `use-cornerstone-viewport` maps back to the two
// landmark voxels (both endpoints share the slice, so both are written on each edit).
import { LengthTool, drawing, annotation as csAnnotation } from '@cornerstonejs/tools';

const { drawLine, drawHandles, drawTextBox } = drawing;
const isAnnotationVisible = csAnnotation.visibility.isAnnotationVisible;
const getAnnotations = csAnnotation.state.getAnnotations;

/** Extra fields we stash on `annotation.data` to drive rendering. */
export interface ReferenceLineData {
  color?: string;
  startLabel?: string;
  endLabel?: string;
  angleLabel?: string; // this line's signed proximal/distal angle, e.g. 'Proximal: 12.3°'
}

// Styling for a clearly visible overlay on greyscale MRI.
const LINE_WIDTH = 2.5;
const HANDLE_RADIUS = 5;
const ENDPOINT_FONT_PX = 13;
const ANGLE_FONT_PX = 13;
const LABEL_FONT_FAMILY = 'Helvetica, Arial, sans-serif';

/** Midpoint of two 2D canvas points. */
function midpoint(a: number[], b: number[]): [number, number] {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

/** A label to draw, with its desired anchor; positions are resolved to avoid overlap. */
interface LabelRequest {
  annotationUID: string;
  uid: string;      // node id within the annotation ('startLabel' | 'endLabel' | 'angleLabel')
  text: string;
  x: number;        // desired left edge (canvas px)
  y: number;        // desired vertical anchor (canvas px)
  color: string;
  fontPx: number;
  order: number;    // lower is placed first (endpoints before angle captions)
}

interface LabelRect { l: number; t: number; r: number; b: number; }

function rectsOverlap(a: LabelRect, b: LabelRect): boolean {
  return a.l < b.r && a.r > b.l && a.t < b.b && a.b > b.t;
}

/** Rough label width from character count (SVG text isn't measured before drawing). */
function estLabelWidth(text: string, fontPx: number): number {
  return text.length * fontPx * 0.55 + 4;
}

/**
 * Resolve a label's draw position so it doesn't overlap already-placed labels: keep its
 * x, and search small vertical offsets (0, ±1, ±2 … line-heights) for the nearest free
 * slot. Records the chosen rectangle in `placed`. This keeps labels near their anchor
 * while stopping the endpoint/angle captions from piling on top of each other.
 */
function placeLabel(req: LabelRequest, placed: LabelRect[]): { x: number; y: number } {
  const w = estLabelWidth(req.text, req.fontPx);
  const h = req.fontPx * 1.3;
  const step = h + 2;
  const offsets = [0];
  for (let k = 1; k <= 15; k++) offsets.push(k * step, -k * step);
  for (const off of offsets) {
    const t = req.y - req.fontPx * 0.8 + off;
    const rect: LabelRect = { l: req.x, t, r: req.x + w, b: t + h };
    if (!placed.some((p) => rectsOverlap(p, rect))) {
      placed.push(rect);
      return { x: req.x, y: req.y + off };
    }
  }
  const t = req.y - req.fontPx * 0.8;
  placed.push({ l: req.x, t, r: req.x + w, b: t + h });
  return { x: req.x, y: req.y };
}

/* eslint-disable @typescript-eslint/no-explicit-any */

export class ReferenceLineTool extends LengthTool {
  static toolName = 'TorsionReferenceLine';

  constructor(props: any = {}, defaultProps: any = undefined) {
    super(props, defaultProps);

    // Override only the drawing; all interaction/hit-testing comes from LengthTool.
    this.renderAnnotation = (enabledElement: any, svgDrawingHelper: any): boolean => {
      let rendered = false;
      const { viewport } = enabledElement;
      const { element } = viewport;

      let annotations = getAnnotations(this.getToolName(), element);
      if (!annotations?.length) return rendered;
      // Only annotations attached to the current slice (stack matches by referencedImageId).
      annotations = this.filterInteractableAnnotationsForElement(element, annotations);
      if (!annotations?.length) return rendered;

      // First pass: draw the lines + handles and collect the label requests.
      const labels: LabelRequest[] = [];
      for (const ann of annotations as any[]) {
        const { annotationUID, data } = ann;
        const points = data?.handles?.points;
        if (!points || points.length < 2) continue;
        if (!isAnnotationVisible(annotationUID)) continue;

        const d = data as ReferenceLineData & { handles: { points: number[][] } };
        const color = d.color ?? 'rgb(255, 90, 90)';
        const c0 = viewport.worldToCanvas(points[0]);
        const c1 = viewport.worldToCanvas(points[1]);

        // The connecting reference line.
        drawLine(svgDrawingHelper, annotationUID, 'line', c0 as any, c1 as any, {
          color,
          lineWidth: LINE_WIDTH,
          shadow: true,
        });

        // Large, solid, always-visible endpoint handles (not just on hover).
        drawHandles(svgDrawingHelper, annotationUID, 'handles', [c0, c1] as any, {
          color,
          handleRadius: `${HANDLE_RADIUS}`,
          fill: color,
        });

        if (d.startLabel) {
          labels.push({ annotationUID, uid: 'startLabel', text: d.startLabel, x: c0[0] + 8, y: c0[1], color, fontPx: ENDPOINT_FONT_PX, order: 0 });
        }
        if (d.endLabel) {
          labels.push({ annotationUID, uid: 'endLabel', text: d.endLabel, x: c1[0] + 8, y: c1[1], color, fontPx: ENDPOINT_FONT_PX, order: 0 });
        }
        if (d.angleLabel) {
          const mid = midpoint(c0, c1);
          labels.push({ annotationUID, uid: 'angleLabel', text: d.angleLabel, x: mid[0] + 8, y: mid[1] + 12, color, fontPx: ANGLE_FONT_PX, order: 1 });
        }
        rendered = true;
      }

      // Second pass: place labels so they don't overlap each other, then draw them.
      // (renderAnnotation sees ALL of this tool's annotations at once, so we can resolve
      // collisions across different reference lines — the main source of unreadable labels.)
      labels.sort((a, b) => a.order - b.order);
      const placed: LabelRect[] = [];
      for (const L of labels) {
        const pos = placeLabel(L, placed);
        drawTextBox(svgDrawingHelper, L.annotationUID, L.uid, [L.text], [pos.x, pos.y] as any, {
          color: L.color, fontSize: `${L.fontPx}px`, fontFamily: LABEL_FONT_FAMILY, padding: 0,
        });
      }
      return rendered;
    };
  }
}
