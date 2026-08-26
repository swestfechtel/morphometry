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
import { useTranslations } from 'next-intl';
import {
  RenderingEngine,
  Enums,
  metaData,
  cache,
  eventTarget,
  imageLoader,
  getRenderingEngine,
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
  segmentation as csSegmentation,
  Enums as csToolsEnums,
} from '@cornerstonejs/tools';
import { createNiftiImageIdsAndCacheMetadata } from '@cornerstonejs/nifti-volume-loader';

import { TorsionLandmarks } from '@/app/types';
import { ensureCornerstoneInit } from './cs-init';
import { imageVolumeUrl, maskVolumeUrl } from './cs-volume-url';
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
  type LandmarkLabels,
} from './landmark-mapping';

const { ViewportType } = Enums;
const LABELMAP = csToolsEnums.SegmentationRepresentations.Labelmap;

const AXIAL_VIEWPORT = 'CS-AX';

// Per-axis colours for the reference lines (bright against greyscale MRI).
const LINE_COLORS = {
  neck: 'rgb(255, 90, 90)',       // femoral neck axis — red
  condyle: 'rgb(90, 170, 255)',   // condylar axis — blue
  tibia: 'rgb(120, 230, 140)',    // posterior tibial axis — green
  malleolus: 'rgb(240, 200, 90)', // transmalleolar axis — amber
} as const;

// Overlay colours (RGBA) for the mask labels: hip=1, knee=2, ankle=3 (see
// api/domain/masks.combine_region_masks). Distinct, saturated, semi-transparent.
const SEGMENT_COLORS: Record<number, Types.Color> = {
  1: [255, 220, 60, 255],  // hip — amber
  2: [180, 80, 220, 255],  // knee — violet
  3: [60, 200, 220, 255],  // ankle — cyan
};

/** Formats an on-slice angle caption from the (localized) role + value, e.g. 'Proximal: 12.3°'. */
type FormatAngle = (role: 'proximal' | 'distal', value: string) => string;

/** The on-slice angle caption for a reference line, e.g. 'Proximal: 12.3°' (or undefined). */
function angleLabelFor(tree: TorsionLandmarks, spec: ReferenceLineSpec, formatAngle: FormatAngle): string | undefined {
  const a = referenceLineAngle(tree, spec);
  if (a == null) return undefined;
  const role = spec.angleKind.endsWith('Proximal') ? 'proximal' : 'distal';
  return formatAngle(role, a.toFixed(1));
}

// Segmentation labelmap overlay for the axial stack.
//
// A StackViewport labelmap can't consume the mask NIfTI directly — it needs a
// per-slice *derived* labelmap image (Uint8) tied to each source frame's geometry
// (otherwise Cornerstone raises "No derived image found" and paints nothing). So
// `setupSegmentation` creates one blank labelmap image per source frame with
// `createAndCacheDerivedLabelmapImages`, then fills each from the aligned mask
// (combined.nii.gz: hip=1, knee=2, ankle=3), loaded through the SAME per-slice
// loader as the image so the pixel layout matches frame-for-frame. `showSegmentation`
// toggles the representation's visibility.

/**
 * Build and attach the mask labelmap overlay to the axial stack viewport.
 *
 * The segmentation id is deterministic (`torsion-seg-{accession}`) and computed by
 * the caller so teardown can always remove it, even if this async build is still
 * in flight — see the teardown in {@link useCornerstoneViewport}.
 *
 * @param segmentationId   Deterministic id to register the labelmap under.
 * @param accession        Examination accession (drives the mask volume URL).
 * @param imageIds         The image stack's per-slice imageIds; derived labelmap images are tied to these.
 * @param viewport         The axial StackViewport to render the labelmap into.
 * @param getVisible       Reads the live "show overlay" state; applied at attach time (not captured early).
 * @param onImagesCreated  Called with the derived labelmap image ids as soon as they are cached, so the
 *                         caller can purge them on teardown even if this build is later cancelled.
 * @param isCancelled      Returns true once the viewer has been torn down, to abort mid-load.
 * @return True if the labelmap was attached; false if the mask was unavailable or the build was cancelled.
 */
async function setupSegmentation(
  segmentationId: string,
  accession: string,
  imageIds: string[],
  viewport: Types.IStackViewport,
  getVisible: () => boolean,
  onImagesCreated: (labelmapImageIds: string[]) => void,
  isCancelled: () => boolean,
): Promise<boolean> {
  try {
    // Mask slices, ordered/loaded identically to the image stack (same NIfTI loader),
    // so mask frame k overlays image frame k with matching in-plane layout.
    const maskImageIds = await createNiftiImageIdsAndCacheMetadata({ url: maskVolumeUrl(accession) });
    if (isCancelled()) return false;

    // One blank Uint8 labelmap image per source frame, tied to that frame's geometry.
    // These are cached under fresh `derived:<uuid>` ids the moment they're created, so
    // hand them back immediately — teardown must purge them even if we bail out below.
    const labelmapImages = imageLoader.createAndCacheDerivedLabelmapImages(imageIds);
    onImagesCreated(labelmapImages.map((im) => im.imageId));
    const n = Math.min(imageIds.length, maskImageIds.length);
    if (imageIds.length !== maskImageIds.length) {
      // Expected to be equal (combined.nii.gz is built aligned to display.nii.gz). A
      // mismatch means some slices go un-overlaid (blank labelmap) rather than mispainted.
      console.warn(`Mask/image slice count mismatch (image=${imageIds.length}, mask=${maskImageIds.length}); overlaying ${n}.`);
    }
    await Promise.all(
      Array.from({ length: n }, async (_unused, k) => {
        const maskImage = await imageLoader.loadAndCacheImage(maskImageIds[k]);
        const src = maskImage.getPixelData();
        const dst = labelmapImages[k].getPixelData();
        // Copy label values (round in case the mask is served as float 1.0/2.0/3.0).
        for (let idx = 0; idx < dst.length; idx++) {
          const v = src[idx];
          dst[idx] = v > 0 ? Math.round(v) : 0;
        }
      }),
    );
    // From here to the return there is NO await, so teardown (a synchronous React
    // cleanup) cannot interleave: either it already ran (caught here) and we add
    // nothing, or it runs after we return and removes us by the deterministic id.
    if (isCancelled()) return false;

    csSegmentation.addSegmentations([
      { segmentationId, representation: { type: LABELMAP, data: { imageIds: labelmapImages.map((im) => im.imageId) } } },
    ]);
    csSegmentation.addLabelmapRepresentationToViewport(viewport.id, [{ segmentationId, type: LABELMAP }]);
    for (const [index, color] of Object.entries(SEGMENT_COLORS)) {
      try { csSegmentation.config.color.setSegmentIndexColor(viewport.id, segmentationId, Number(index), color); }
      catch { /* colour LUT not ready — best-effort */ }
    }
    // Read the toggle at apply-time (not captured at call-time): the user may have
    // flipped it while the mask was loading. Any concurrent toggle-effect fired a
    // no-op while unattached, so this is the authoritative final state.
    csSegmentation.config.visibility.setSegmentationRepresentationVisibility(
      viewport.id, { segmentationId, type: LABELMAP }, getVisible(),
    );
    viewport.render();
    return true;
  } catch (err) {
    console.warn('Segmentation labelmap not available', err);
    return false;
  }
}

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

  // Localized labels drawn on the canvas. Kept in a ref (updated each render) so the
  // setup effect reads the current language without listing it as a dependency; the
  // viewer is also remounted on locale change (key={locale}) so labels rebuild fresh.
  const tLandmarks = useTranslations('landmarks');
  const labelBundleRef = useRef<{ labels: LandmarkLabels; formatAngle: FormatAngle }>(null!);
  labelBundleRef.current = {
    labels: {
      femoralHeadCentre: tLandmarks('femoralHeadCentre'),
      femoralNeckCentre: tLandmarks('femoralNeckCentre'),
      posteriorCondyleMedial: tLandmarks('posteriorCondyleMedial'),
      posteriorCondyleLateral: tLandmarks('posteriorCondyleLateral'),
      posteriorTibialCondyleMed: tLandmarks('posteriorTibialCondyleMed'),
      posteriorTibialCondyleLat: tLandmarks('posteriorTibialCondyleLat'),
      medialMalleolus: tLandmarks('medialMalleolus'),
      lateralMalleolus: tLandmarks('lateralMalleolus'),
    },
    formatAngle: (role, value) =>
      role === 'proximal' ? tLandmarks('angleProximal', { value }) : tLandmarks('angleDistal', { value }),
  };

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

  // Deterministic labelmap segmentation id, so teardown/toggle can address it without
  // waiting for the async attach to finish. The latest toggle value is mirrored into a
  // ref for the async setup (which applies it once the overlay is ready).
  const segmentationId = `torsion-seg-${args.accession}`;
  const showSegRef = useRef(args.showSegmentation);
  showSegRef.current = args.showSegmentation;
  // Derived labelmap image ids (fresh `derived:<uuid>` each mount) to purge on teardown,
  // so they don't accumulate in the global cornerstone image cache across remounts.
  const labelmapImageIdsRef = useRef<string[]>([]);

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
      ann.data.angleLabel = angleLabelFor(next, spec, labelBundleRef.current.formatAngle);
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
      for (const spec of buildReferenceLines(args.landmarks, labelBundleRef.current.labels)) {
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
            angleLabel: angleLabelFor(args.landmarks, spec, labelBundleRef.current.formatAngle),
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

      // Attach the mask labelmap overlay (async: streams + fills the mask). Landmarks
      // and interaction are already live above, so the overlay simply appears when ready.
      // Fire-and-forget: teardown removes the labelmap (and its derived images) regardless
      // of whether this promise has resolved.
      void setupSegmentation(
        segmentationId, args.accession, [...imageIds], axialVp,
        () => showSegRef.current,                                 // #2: read visibility at apply-time
        (ids) => { labelmapImageIdsRef.current = ids; },          // #4: record for cache purge
        () => destroyed,
      );

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
        editAnn.data.angleLabel = angleLabelFor(tree, spec, labelBundleRef.current.formatAngle);
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
      // Always attempt removal by the deterministic id — the async setupSegmentation
      // may have attached it after this cleanup started (or be about to), and
      // removeSegmentation is a safe no-op when the id was never added.
      try { csSegmentation.removeSegmentation(segmentationId); } catch { /* not added */ }
      // Purge the per-slice derived labelmap images: removeSegmentation does NOT touch
      // the image cache, and these carry fresh `derived:<uuid>` ids every mount, so
      // without this they accumulate (doubled under React StrictMode's mount/remount).
      for (const id of labelmapImageIdsRef.current) {
        if (cache.getImageLoadObject(id)) {
          try { cache.removeImageLoadObject(id); } catch { /* already gone */ }
        }
      }
      labelmapImageIdsRef.current = [];
      ToolGroupManager.destroyToolGroup(toolGroupId);
      engine?.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [args.accession]);

  // --- segmentation visibility toggle --------------------------------------
  // Show/hide the labelmap without rebuilding it. Guarded because the overlay is
  // attached asynchronously — until then there is no representation to toggle, so
  // this no-ops (caught) and setupSegmentation applies the current value (via
  // showSegRef) once it finishes.
  useEffect(() => {
    try {
      csSegmentation.config.visibility.setSegmentationRepresentationVisibility(
        AXIAL_VIEWPORT, { segmentationId, type: LABELMAP }, args.showSegmentation,
      );
      getRenderingEngine(engineId)?.getViewport(AXIAL_VIEWPORT)?.render();
    } catch { /* representation not ready yet */ }
  }, [args.showSegmentation, engineId, segmentationId]);

  return { refs };
}
