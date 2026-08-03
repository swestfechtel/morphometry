'use client';

// The Cornerstone3D viewer lifecycle for a torsion examination — a single AXIAL
// native-slice viewer.
//
// The torsion MRI is highly anisotropic (≈0.6 mm in-plane, ≈9 mm through-plane) and
// is really a 2D slice stack, not an isotropic 3D volume. So the view is one
// **StackViewport** rendering the native axial slices (no reslicing, no volume
// mapper), which renders reliably regardless of acquisition orientation. (An
// earlier 3-plane volume-MPR variant is stashed under `stashed/` — it rendered
// black on exactly axis-aligned volumes and couldn't show landmarks on tilted
// ones.)
//
// The image is served as a NIfTI whose per-slice imageIds also drive the mask
// labelmap overlay. Landmarks are seeded as draggable Probe annotations placed with
// each slice's own geometry (see stackSliceWorld); edits flow back to React state
// (live torsion recompute) and a save callback. All Cornerstone state is torn down
// on unmount.
import { useEffect, useRef } from 'react';
import {
  RenderingEngine,
  Enums,
  metaData,
  cache,
  eventTarget,
  type Types,
} from '@cornerstonejs/core';
import {
  addTool,
  ToolGroupManager,
  StackScrollTool,
  PanTool,
  ZoomTool,
  WindowLevelTool,
  annotation as csAnnotation,
  Enums as csToolsEnums,
} from '@cornerstonejs/tools';
import { createNiftiImageIdsAndCacheMetadata } from '@cornerstonejs/nifti-volume-loader';

import { TorsionLandmarks } from '@/app/types';
import { ensureCornerstoneInit } from './cs-init';
import { imageVolumeUrl } from './cs-volume-url';
import { ReferenceLineTool } from './reference-line-tool';
import {
  listLandmarks,
  applyEditedVoxel,
  fromLoaderIndex,
  buildReferenceLines,
  referenceLineAngle,
  zRangeForKey,
  Offsets,
  ReferenceLineSpec,
} from './landmark-mapping';

const { ViewportType } = Enums;

const AXIAL_VIEWPORT = 'CS-AX';

// Per-axis colours for the reference lines (bright against greyscale MRI).
const LINE_COLORS = {
  neck: 'rgb(255, 90, 90)',       // femoral neck axis — red
  condyle: 'rgb(90, 170, 255)',   // condylar axis — blue
  tibia: 'rgb(120, 230, 140)',    // posterior tibial axis — green
  malleolus: 'rgb(240, 200, 90)', // transmalleolar axis — amber
} as const;

/** The on-slice angle caption for a reference line, e.g. 'Proximal: 12.3°' (or undefined). */
function angleLabelFor(tree: TorsionLandmarks, spec: ReferenceLineSpec): string | undefined {
  const a = referenceLineAngle(tree, spec);
  if (a == null) return undefined;
  const role = spec.angleKind.endsWith('Proximal') ? 'Proximal' : 'Distal';
  return `${role}: ${a.toFixed(1)}°`;
}

// NOTE: the segmentation labelmap overlay is not implemented on this stack viewer.
// Stack labelmaps need per-slice *derived* images populated from the mask (see the
// "No derived image found" path in @cornerstonejs/tools); the volume-labelmap
// version lives in the stashed MPR variant. `showSegmentation` is accepted but
// currently inert — the parent hides its toggle. Re-add here as a follow-up.

/**
 * A robust display window from actual intensities: the 1st–99th percentile of a
 * strided sample. The NIfTI loader forces a fixed CT window (400/40) that renders
 * MRI black; a plain min→max window is fragile too, because a few bright outlier
 * voxels widen it until real tissue collapses to near-black. Percentiles ignore
 * those outliers. Returns null if the data is unusable.
 */
function computeVoiRange(data: ArrayLike<number> | undefined): { lower: number; upper: number } | null {
  if (!data || !data.length) return null;
  const stride = Math.max(1, Math.floor(data.length / 500_000)); // cap the sort cost
  const sample: number[] = [];
  for (let i = 0; i < data.length; i += stride) {
    const val = data[i];
    if (Number.isFinite(val)) sample.push(val);
  }
  if (sample.length <= 4) return null;
  sample.sort((a, b) => a - b);
  const lower = sample[Math.floor(sample.length * 0.01)];
  const upper = sample[Math.floor(sample.length * 0.99)];
  if (Number.isFinite(lower) && Number.isFinite(upper) && upper > lower) return { lower, upper };
  return null;
}

interface PlaneModule {
  imagePositionPatient?: number[];
  imageOrientationPatient?: number[];
  rowPixelSpacing?: number;
  columnPixelSpacing?: number;
}

/**
 * World position of in-plane voxel (i, j) on stack slice `imageId`, from the
 * slice's OWN geometry (imagePlaneModule). The NIfTI loader's per-slice frame
 * differs from the vtk volume's index→world, so a probe placed via a volume
 * transform projects onto a tilted acquisition slice shifted (it sits far in the
 * slice-normal direction, and projecting along the tilt skews the in-plane spot).
 * Placing it at the slice's own world puts it exactly on voxel (i, j).
 */
function stackSliceWorld(imageId: string, i: number, j: number): Types.Point3 | null {
  const p = metaData.get('imagePlaneModule', imageId) as PlaneModule | undefined;
  const ipp = p?.imagePositionPatient, iop = p?.imageOrientationPatient;
  const dI = p?.columnPixelSpacing, dJ = p?.rowPixelSpacing;
  if (!ipp || !iop || dI == null || dJ == null) return null;
  const [rx, ry, rz, cx, cy, cz] = iop;
  return [
    ipp[0] + i * dI * rx + j * dJ * cx,
    ipp[1] + i * dI * ry + j * dJ * cy,
    ipp[2] + i * dI * rz + j * dJ * cz,
  ];
}

/** Inverse of {@link stackSliceWorld}: the in-plane voxel (i, j) of a world point on `imageId`. */
function stackVoxelFromWorld(imageId: string, world: Types.Point3): { i: number; j: number } | null {
  const p = metaData.get('imagePlaneModule', imageId) as PlaneModule | undefined;
  const ipp = p?.imagePositionPatient, iop = p?.imageOrientationPatient;
  const dI = p?.columnPixelSpacing, dJ = p?.rowPixelSpacing;
  if (!ipp || !iop || dI == null || dJ == null) return null;
  const [rx, ry, rz, cx, cy, cz] = iop;
  const dx = world[0] - ipp[0], dy = world[1] - ipp[1], dz = world[2] - ipp[2];
  return { i: (dx * rx + dy * ry + dz * rz) / dI, j: (dx * cx + dy * cy + dz * cz) / dJ };
}

export interface UseViewportArgs {
  accession: string;
  landmarks: TorsionLandmarks;
  offsets: Offsets;
  showSegmentation: boolean;
  /** Called on every landmark edit with the updated tree (drives live recompute). */
  onLandmarksChange: (next: TorsionLandmarks) => void;
  /** Called when a drag finishes, to arm the PATCH save. */
  onDragEnd: (next: TorsionLandmarks) => void;
}

export function useCornerstoneViewport(args: UseViewportArgs) {
  // One axial viewport; the extra refs keep the component's div layout stable.
  const refs = [useRef<HTMLDivElement>(null), useRef<HTMLDivElement>(null), useRef<HTMLDivElement>(null)];

  // Stable per-mount ids (avoid Date.now/Math.random for SSR-safety; accession is unique enough).
  const engineId = `torsion-engine-${args.accession}`;
  const toolGroupId = `torsion-tg-${args.accession}`;

  // Latest landmarks/callbacks for the annotation-modified handler, without re-running setup.
  const landmarksRef = useRef(args.landmarks);
  landmarksRef.current = args.landmarks;
  const cbRef = useRef({ onChange: args.onLandmarksChange, onDragEnd: args.onDragEnd, offsets: args.offsets });
  cbRef.current = { onChange: args.onLandmarksChange, onDragEnd: args.onDragEnd, offsets: args.offsets };

  // annotationUID -> the reference line's spec, to route endpoint edits back into the
  // tree (startPaths/endPaths) and recompute its angle label. Both endpoints share the slice.
  const annotationPaths = useRef<Map<string, ReferenceLineSpec>>(new Map());

  // --- setup / teardown (runs once per accession) --------------------------
  useEffect(() => {
    let engine: RenderingEngine | null = null;
    let destroyed = false;
    let resizeObserver: ResizeObserver | null = null;
    let onWheel: ((e: WheelEvent) => void) | null = null;
    const element = refs[0].current;

    const onAnnotationModified = (evt: Event) => {
      // @ts-expect-error CustomEvent detail shape from Cornerstone
      const ann = evt.detail?.annotation;
      if (!ann?.annotationUID) return;
      const spec = annotationPaths.current.get(ann.annotationUID);
      const points = ann.data?.handles?.points as Types.Point3[] | undefined;
      const refId = ann.metadata?.referencedImageId as string | undefined;
      if (!spec || !points || points.length < 2 || !refId) return;
      // Map both dragged world points back to combined-volume voxels using the SAME
      // stack-slice geometry the line was placed with (in-plane i,j), keeping the
      // line's slice fixed (its two endpoints stay co-planar). Frame is the
      // combined-volume z, parsed from the referenced imageId (`?frame=N`). We rewrite
      // BOTH endpoints on every edit; the un-dragged one maps back to the same voxel.
      const frame = refId.includes('?frame=') ? Number(refId.split('?frame=')[1]) : NaN;
      const ijStart = stackVoxelFromWorld(refId, points[0]);
      const ijEnd = stackVoxelFromWorld(refId, points[1]);
      if (!ijStart || !ijEnd || !Number.isFinite(frame)) return;
      const startVoxel: Types.Point3 = [Math.round(ijStart.i), Math.round(ijStart.j), frame];
      const endVoxel: Types.Point3 = [Math.round(ijEnd.i), Math.round(ijEnd.j), frame];
      // Write every landmark path this endpoint maps to (the femoral condylar line is
      // shared by both methods, so a single drag updates each method's knee point).
      let next = landmarksRef.current;
      for (const p of spec.startPaths) next = applyEditedVoxel(next, p, startVoxel, cbRef.current.offsets);
      for (const p of spec.endPaths) next = applyEditedVoxel(next, p, endVoxel, cbRef.current.offsets);
      // Refresh this line's on-slice angle caption live (the total torsion updates via React).
      ann.data.angleLabel = angleLabelFor(next, spec);
      cbRef.current.onChange(next);
    };
    const onMouseUp = () => cbRef.current.onDragEnd(landmarksRef.current);

    (async () => {
      await ensureCornerstoneInit();
      if (!element) return;

      const imageIds = await createNiftiImageIdsAndCacheMetadata({ url: imageVolumeUrl(args.accession) });
      if (destroyed) return;

      engine = new RenderingEngine(engineId);
      engine.setViewports([{ viewportId: AXIAL_VIEWPORT, type: ViewportType.STACK, element }]);
      const axialVp = engine.getViewport(AXIAL_VIEWPORT) as Types.IStackViewport;
      await axialVp.setStack([...imageIds], Math.floor(imageIds.length / 2));

      // The NIfTI loader hardcodes the window (400/40, a CT window), rendering MRI
      // black. Derive a VOI from the current (middle) slice's actual intensities.
      const currentImage = cache.getImage(axialVp.getCurrentImageId());
      const voiRange = computeVoiRange(currentImage?.getPixelData?.() as ArrayLike<number> | undefined);
      if (voiRange) { try { axialVp.setProperties({ voiRange }); } catch { /* not ready */ } }
      axialVp.render();

      // --- tools: scroll/pan/zoom/window-level + reference-line landmarks --
      [StackScrollTool, PanTool, ZoomTool, ReferenceLineTool, WindowLevelTool].forEach(addTool);
      const tg = ToolGroupManager.createToolGroup(toolGroupId);
      if (!tg) return;
      tg.addViewport(AXIAL_VIEWPORT, engineId);
      tg.addTool(StackScrollTool.toolName);
      tg.addTool(PanTool.toolName);
      tg.addTool(ZoomTool.toolName);
      tg.addTool(ReferenceLineTool.toolName);
      tg.addTool(WindowLevelTool.toolName);
      tg.setToolActive(StackScrollTool.toolName, { bindings: [{ mouseButton: csToolsEnums.MouseBindings.Wheel }] });
      tg.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: csToolsEnums.MouseBindings.Auxiliary }] });
      tg.setToolActive(ZoomTool.toolName, { bindings: [{ mouseButton: csToolsEnums.MouseBindings.Secondary }] });
      // Window/level (contrast/brightness) on plain left-drag; a press on a reference-
      // line endpoint handle is hit-tested first, so landmark dragging still works.
      tg.setToolActive(WindowLevelTool.toolName, {
        bindings: [{ mouseButton: csToolsEnums.MouseBindings.Primary }],
      });
      // Passive: existing endpoint handles stay draggable, but clicks don't add lines.
      tg.setToolPassive(ReferenceLineTool.toolName);

      // --- seed reference-line annotations from voxel coords ---------------
      const frameOfReferenceUID = axialVp.getFrameOfReferenceUID?.();
      const viewPlaneNormal = axialVp.getCamera?.().viewPlaneNormal;
      // Look up each landmark's combined-volume voxel (offsets applied) by path.
      const voxelByPath = new Map(
        listLandmarks(args.landmarks, args.offsets).map((r) => [r.path.join('.'), r.voxel] as const),
      );
      for (const spec of buildReferenceLines(args.landmarks)) {
        // All paths of an endpoint share the same voxel (identical across methods); read the first.
        const startVox = voxelByPath.get(spec.startPaths[0].join('.'));
        const endVox = voxelByPath.get(spec.endPaths[0].join('.'));
        if (!startVox || !endVox) continue;
        const s = fromLoaderIndex(startVox);
        const e = fromLoaderIndex(endVox);
        // Both endpoints share the slice; render the whole line on the start's slice
        // (frame = combined-volume z) and place both from THAT slice's own geometry so
        // the line stays in-plane even if the stored z of the two points drifts slightly.
        const frame = Math.round(s[2]);
        const referencedImageId = imageIds[frame];
        if (!referencedImageId) continue;
        const wStart = stackSliceWorld(referencedImageId, s[0], s[1]);
        const wEnd = stackSliceWorld(referencedImageId, e[0], e[1]);
        if (!wStart || !wEnd) continue;
        const ann = {
          // isVisible/isLocked are required for the handles to be moveable: the
          // mouse-down handler's filterToolsWithMoveableHandles skips any annotation
          // whose isVisible is not truthy (addAnnotation does NOT default them), which
          // is what otherwise lets the left-drag fall through to window/level.
          isVisible: true,
          isLocked: false,
          highlighted: false,
          invalidated: false,
          metadata: {
            toolName: ReferenceLineTool.toolName,
            FrameOfReferenceUID: frameOfReferenceUID,
            referencedImageId,
            viewPlaneNormal,
          },
          data: {
            handles: { points: [wStart, wEnd], activeHandleIndex: null },
            cachedStats: {},
            color: LINE_COLORS[spec.colorKey],
            startLabel: spec.startLabel,
            endLabel: spec.endLabel,
            angleLabel: angleLabelFor(args.landmarks, spec),
          },
        };
        const uid = csAnnotation.state.addAnnotation(ann as never, frameOfReferenceUID as never);
        if (uid) annotationPaths.current.set(uid, spec);
      }

      // ANNOTATION_MODIFIED is dispatched on the GLOBAL Cornerstone eventTarget (not the
      // viewport element), so listen there; the handler filters to our annotations by UID.
      eventTarget.addEventListener(csToolsEnums.Events.ANNOTATION_MODIFIED as unknown as string, onAnnotationModified);
      element.addEventListener('mouseup', onMouseUp);
      axialVp.render();

      // The layout sizes the viewport to the screen (see cornerstone-torsion-viewer),
      // but Cornerstone's canvas doesn't follow element size changes on its own. Keep
      // the render resolution in sync (and preserve the camera) whenever the element
      // resizes — window resize, dark-mode reflow, etc.
      resizeObserver = new ResizeObserver(() => {
        try { engine?.resize(true, true); } catch { /* engine torn down */ }
      });
      resizeObserver.observe(element);

      // Move a landmark ACROSS slices: while a reference-line handle is being dragged,
      // Cornerstone blocks the wheel (StackScroll) because `isInteractingWithTool` is set,
      // so scrolling can't change the slice. We intercept the wheel ourselves during a drag
      // and relocate the whole line to the adjacent slice — both endpoints move together to
      // stay co-planar (the measurement is defined in one axial plane), keeping their
      // in-plane position; the dragged handle then continues tracking the mouse on the new
      // slice. Clamped to the line's own sub-volume (hip/knee/ankle) z-range. When no drag
      // is active we do nothing and let Cornerstone scroll normally.
      onWheel = (e: WheelEvent) => {
        const inst = tg.getToolInstance(ReferenceLineTool.toolName) as unknown as { editData?: { annotation?: unknown } } | undefined;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const editAnn = inst?.editData?.annotation as any;
        if (!editAnn) return; // not dragging a reference line → normal scroll
        const spec = annotationPaths.current.get(editAnn.annotationUID);
        const pts = editAnn.data?.handles?.points as Types.Point3[] | undefined;
        if (!spec || !pts || pts.length < 2) return;
        e.preventDefault();

        const cur = axialVp.getCurrentImageIdIndex();
        // Keep the line inside its anatomical sub-volume so the stored landmark and the
        // rendered slice stay consistent (applyEditedVoxel clamps the stored z the same way).
        const key = spec.startPaths[0][spec.startPaths[0].length - 1];
        const [lo, hi] = zRangeForKey(key, cbRef.current.offsets);
        const target = cur + (e.deltaY > 0 ? 1 : -1);
        const next = Math.max(lo, Math.min(hi - 1, Math.max(0, Math.min(imageIds.length - 1, target))));
        if (next === cur) return;

        const oldRefId = imageIds[cur];
        const newRefId = imageIds[next];
        if (!oldRefId || !newRefId) return;

        // Re-place both endpoints on the new slice at their current in-plane (i, j).
        const ij: { i: number; j: number }[] = [];
        for (let k = 0; k < 2; k++) {
          const p = stackVoxelFromWorld(oldRefId, pts[k]);
          if (!p) return;
          const w = stackSliceWorld(newRefId, p.i, p.j);
          if (!w) return;
          ij.push(p);
          pts[k] = w;
        }
        editAnn.metadata.referencedImageId = newRefId;

        // Update the landmark tree: both endpoints' frame → new slice, in-plane preserved.
        let tree = landmarksRef.current;
        for (const p of spec.startPaths) tree = applyEditedVoxel(tree, p, [Math.round(ij[0].i), Math.round(ij[0].j), next], cbRef.current.offsets);
        for (const p of spec.endPaths) tree = applyEditedVoxel(tree, p, [Math.round(ij[1].i), Math.round(ij[1].j), next], cbRef.current.offsets);
        editAnn.data.angleLabel = angleLabelFor(tree, spec);
        cbRef.current.onChange(tree);

        // Show the new slice (also re-renders the relocated line there).
        axialVp.setImageIdIndex(next);
      };
      element.addEventListener('wheel', onWheel, { passive: false });
    })();

    return () => {
      destroyed = true;
      resizeObserver?.disconnect();
      if (onWheel) element?.removeEventListener('wheel', onWheel);
      eventTarget.removeEventListener(csToolsEnums.Events.ANNOTATION_MODIFIED as unknown as string, onAnnotationModified);
      element?.removeEventListener('mouseup', onMouseUp);
      annotationPaths.current.forEach((_p, uid) => csAnnotation.state.removeAnnotation(uid));
      annotationPaths.current.clear();
      ToolGroupManager.destroyToolGroup(toolGroupId);
      engine?.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [args.accession]);

  return { refs };
}
