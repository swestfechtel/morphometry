'use client';

// A single AXIAL native-slice viewport rendering the torsion NIfTI with a labelmap
// overlay and draggable landmark probes. Client-only; import this via next/dynamic
// with { ssr: false } so no Cornerstone code runs during server rendering.
// (A stashed 3-plane MPR variant lives under `stashed/`.)
import { TorsionExamination, TorsionFemurLandmarks, TorsionLandmarks, TorsionMethodSides } from '@/app/types';
import { computeFemoralTorsion, computeTibialTorsion } from '@/app/utils';
import { useCornerstoneViewport } from './use-cornerstone-viewport';
import { Offsets } from './landmark-mapping';

interface Props {
  examination: TorsionExamination;
  showSegmentation: boolean;
  // Live landmark state (updated on every edit) — drives the on-viewer torsion readout.
  landmarks: TorsionLandmarks;
  hasChanges: boolean;         // whether there are unsaved landmark edits
  onSave: () => void;          // persist the pending edits (PATCH)
  saveChangesCallback: (landmarks: TorsionLandmarks) => void;
  setLandmarksCallback: (landmarks: TorsionLandmarks) => void;
}

/** One "L: x°  R: y°" row of torsion values, only rendered for sides that exist. */
function TorsionRow({ label, left, right }: { label: string; left: string | null; right: string | null }) {
  return (
    <div className="flex justify-between gap-3 tabular-nums">
      <span className="text-white/70">{label}</span>
      <span>
        {left != null && <span className="ml-2">L:&nbsp;{left}°</span>}
        {right != null && <span className="ml-2">R:&nbsp;{right}°</span>}
      </span>
    </div>
  );
}

/** Top-right overlay with the femoral (per method) and tibial torsion, from live landmarks. */
function TorsionReadout({ landmarks }: { landmarks: TorsionLandmarks }) {
  const femurTorsion = (method: string, side: 'left' | 'right'): string | null => {
    const m = landmarks?.femur?.[method as keyof TorsionFemurLandmarks] as TorsionMethodSides | undefined;
    const s = m?.[side];
    if (!s) return null;
    return computeFemoralTorsion(s.hip_start, s.hip_end, s.knee_start, s.knee_end, side);
  };
  const tibiaTorsion = (side: 'left' | 'right'): string | null => {
    const s = landmarks?.tibia?.[side];
    if (!s) return null;
    return computeTibialTorsion(s.knee_start, s.knee_end, s.ankle_start, s.ankle_end, side);
  };

  return (
    <div className="rounded bg-black/60 px-3 py-2 text-xs text-white select-none shadow">
      <div className="mb-1 font-semibold text-white/90">Torsion</div>
      <TorsionRow label="Femur (Lee)" left={femurTorsion('Lee', 'left')} right={femurTorsion('Lee', 'right')} />
      <TorsionRow label="Femur (Murphy)" left={femurTorsion('Murphy', 'left')} right={femurTorsion('Murphy', 'right')} />
      <TorsionRow label="Tibia" left={tibiaTorsion('left')} right={tibiaTorsion('right')} />
    </div>
  );
}

export function CornerstoneTorsionViewer({
  examination, showSegmentation, landmarks, hasChanges, onSave, saveChangesCallback, setLandmarksCallback,
}: Props) {
  const offsets: Offsets = {
    kneeOffset: examination.knee_offset,
    ankleOffset: examination.ankle_offset,
    // total combined-volume z; shape is [x, y, z]. Fall back generously if absent.
    zMax: examination.shape?.[2] ?? examination.ankle_offset + 10_000,
  };

  const { refs } = useCornerstoneViewport({
    accession: examination.accession_number,
    landmarks: examination.landmarks,
    offsets,
    showSegmentation,
    onLandmarksChange: setLandmarksCallback,
    onDragEnd: saveChangesCallback,
  });

  return (
    <div className="flex flex-col gap-1 items-center">
      {/* Square viewport sized to fit the screen: it grows to the column width but is
          capped so its height never exceeds the viewport (minus navbar + margins),
          so the whole image is visible without page scrolling. The Cornerstone hook
          re-renders at the new resolution via a ResizeObserver. */}
      <div
        className="relative bg-black w-full"
        style={{ aspectRatio: '1 / 1', maxWidth: 'calc(100vh - 9rem)', maxHeight: 'calc(100vh - 9rem)' }}
      >
        <div ref={refs[0]} className="w-full h-full" style={{ minHeight: 0 }} onContextMenu={(e) => e.preventDefault()} />
        <span className="absolute top-1 left-2 text-xs text-white/70 pointer-events-none select-none">
          Axial
        </span>
        {/* Top-right overlay: torsion readout with the Save button directly below it. The
            container ignores pointer events so it never blocks dragging; the button re-enables them. */}
        <div className="absolute top-1 right-1 z-10 flex flex-col items-end gap-2 pointer-events-none">
          <TorsionReadout landmarks={landmarks} />
          {hasChanges && (
            <button
              onClick={onSave}
              className="pointer-events-auto px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs shadow"
            >
              Save changes
            </button>
          )}
        </div>
      </div>
      <div className="text-center text-xs text-white/40 select-none px-2">
        scroll: slice · left-drag: contrast/brightness · right-drag: zoom · middle-drag: pan
      </div>
    </div>
  );
}
