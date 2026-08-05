'use client';

// Phase 2 of the upload flow: pick the correct DICOM series from a pending
// examination and start processing. Supports two protocols:
//   - whole_leg: one series contains hip+knee+ankle and is auto-split (pick one card)
//   - regions:   three separate series, each assigned to a region (hip/knee/ankle)
// On confirm it POSTs /upload/torsion/select and polls the returned job, navigating
// to the viewer once the pipeline finishes.
import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiFetch } from '@/app/auth';
import { PollingComponent } from '@/app/components/polling-component';
import { seriesPreviewUrl } from '@/app/components/cornerstone/cs-volume-url';
import { SeriesInfo } from '@/app/types';

type Mode = 'whole_leg' | 'regions';
type Region = 'hip' | 'knee' | 'ankle';
const REGIONS: Region[] = ['hip', 'knee', 'ankle'];

/** A scrollable multi-slice preview of one series (wheel to page through slices). */
function SeriesPreview({ examinationId, series }: { examinationId: string; series: SeriesInfo }) {
  const count = series.preview_count;
  const ref = useRef<HTMLDivElement>(null);
  const [idx, setIdx] = useState(Math.floor(count / 2));

  useEffect(() => {
    const el = ref.current;
    if (!el || count <= 1) return;
    // Native non-passive listener so we can preventDefault (stop the page scrolling
    // while paging slices); React's onWheel is passive and would warn.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setIdx((i) => (i + (e.deltaY > 0 ? 1 : -1) + count) % count);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [count]);

  if (count === 0) {
    return (
      <div className="flex aspect-square w-full items-center justify-center bg-black text-xs text-white/50">
        no preview
      </div>
    );
  }
  return (
    <div ref={ref} className="relative aspect-square w-full bg-black" title="scroll to page through slices">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={seriesPreviewUrl(examinationId, series.uid, idx)}
        alt={`preview ${idx + 1}`}
        className="h-full w-full object-contain select-none"
        draggable={false}
      />
      {count > 1 && (
        <span className="absolute bottom-1 right-2 text-xs text-white/70 select-none pointer-events-none">
          {idx + 1}/{count}
        </span>
      )}
    </div>
  );
}

interface CardProps {
  examinationId: string;
  series: SeriesInfo;
  mode: Mode;
  selected: boolean;                        // whole_leg: this card is chosen
  assignedRegion: Region | null;            // regions: region this series is assigned to
  onSelectWhole: () => void;
  onAssignRegion: (region: Region) => void;
}

function SeriesCard({ examinationId, series, mode, selected, assignedRegion, onSelectWhole, onAssignRegion }: CardProps) {
  const highlight = mode === 'whole_leg' ? selected : assignedRegion != null;
  return (
    <div
      className={`rounded border ${highlight ? 'border-blue-500 ring-2 ring-blue-500' : 'border-gray-300 dark:border-gray-700'} bg-white dark:bg-gray-800 overflow-hidden`}
    >
      <div
        className={mode === 'whole_leg' ? 'cursor-pointer' : ''}
        onClick={mode === 'whole_leg' ? onSelectWhole : undefined}
      >
        <SeriesPreview examinationId={examinationId} series={series} />
      </div>
      <div className="p-2 text-sm text-gray-800 dark:text-gray-100">
        <p className="font-medium truncate" title={series.description ?? ''}>
          {series.description || '(no description)'}
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {[series.modality, `${series.instances} img`, series.rows && series.cols ? `${series.rows}×${series.cols}` : null]
            .filter(Boolean)
            .join(' · ')}
        </p>
        {mode === 'regions' && (
          <div className="mt-2 flex gap-1">
            {REGIONS.map((region) => (
              <button
                key={region}
                type="button"
                onClick={() => onAssignRegion(region)}
                className={`flex-1 rounded px-1 py-1 text-xs capitalize ${
                  assignedRegion === region
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                {region}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function SeriesPicker(
  { examinationId, series, onUploadAnother }:
  { examinationId: string; series: SeriesInfo[]; onUploadAnother?: () => void },
) {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>('whole_leg');
  const [selectedWhole, setSelectedWhole] = useState<string | null>(null);
  const [regions, setRegions] = useState<Record<Region, string | null>>({ hip: null, knee: null, ankle: null });
  const [jobId, setJobId] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [failed, setFailed] = useState(false);
  const [done, setDone] = useState(false);

  const assignRegion = (uid: string, region: Region) => {
    setRegions((prev) => {
      const next: Record<Region, string | null> = { ...prev };
      REGIONS.forEach((r) => { if (next[r] === uid) next[r] = null; }); // this series leaves its old region
      if (prev[region] !== uid) next[region] = uid;                     // toggle/assign to the clicked region
      return next;
    });
  };

  const canSubmit =
    mode === 'whole_leg'
      ? selectedWhole != null
      : REGIONS.every((r) => regions[r] != null);

  const submit = async () => {
    setError(null);
    setSubmitting(true);
    const payload =
      mode === 'whole_leg'
        ? { examination_id: examinationId, mode, series_uid: selectedWhole }
        : { examination_id: examinationId, mode, hip: regions.hip, knee: regions.knee, ankle: regions.ankle };
    try {
      const resp = await apiFetch('/upload/torsion/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (resp.status === 202) {
        const data = await resp.json();
        setJobId(data.job_id);
      } else {
        const data = await resp.json().catch(() => ({}));
        setError(data.detail || `Failed to start processing (HTTP ${resp.status}).`);
      }
    } catch {
      setError('Network error while starting processing.');
    } finally {
      setSubmitting(false);
    }
  };

  // Processing runs on the queue (one GPU worker serializes jobs), so once a series
  // is submitted the user can immediately upload/queue more — no need to wait here.
  const uploadAnother = onUploadAnother ?? (() => router.push('/upload'));

  if (jobId !== undefined) {
    return (
      <div className="flex flex-col items-center gap-4 p-8 text-gray-800 dark:text-gray-100">
        <PollingComponent job_id={jobId} callback={() => setDone(true)} onError={() => setFailed(true)} />
        {!done && !failed && (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Queued for processing — it runs in the background, so you can upload more examinations now.
          </p>
        )}
        <div className="flex flex-wrap items-center justify-center gap-3">
          {done && (
            <a
              href={`/examinations/${examinationId}`}
              className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500"
            >
              View result
            </a>
          )}
          <button
            type="button"
            onClick={uploadAnother}
            className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500"
          >
            Upload another examination
          </button>
          <a
            href="/examinations"
            className="rounded border border-gray-300 dark:border-gray-600 px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            View all examinations
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-white">Select the series to process</h2>
        <div className="inline-flex overflow-hidden rounded border border-gray-300 dark:border-gray-700">
          {(['whole_leg', 'regions'] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 text-sm ${
                mode === m
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200'
              }`}
            >
              {m === 'whole_leg' ? 'Single whole-leg series' : 'Separate hip / knee / ankle'}
            </button>
          ))}
        </div>
      </div>

      <p className="mb-3 text-sm text-gray-600 dark:text-gray-300">
        {mode === 'whole_leg'
          ? 'Pick the one series that contains the whole leg — it is split into hip, knee and ankle automatically.'
          : 'Assign one series to each of hip, knee and ankle.'}
      </p>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5">
        {series.map((s) => {
          const assignedRegion = REGIONS.find((r) => regions[r] === s.uid) ?? null;
          return (
            <SeriesCard
              key={s.uid}
              examinationId={examinationId}
              series={s}
              mode={mode}
              selected={selectedWhole === s.uid}
              assignedRegion={assignedRegion}
              onSelectWhole={() => setSelectedWhole(s.uid)}
              onAssignRegion={(region) => assignRegion(s.uid, region)}
            />
          );
        })}
      </div>

      {error && <p className="mt-4 text-sm text-red-500">{error}</p>}

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={submit}
          disabled={!canSubmit || submitting}
          className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? 'Starting…' : 'Process selected series'}
        </button>
        {mode === 'regions' && (
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {REGIONS.map((r) => `${r}: ${regions[r] ? '✓' : '—'}`).join('   ')}
          </span>
        )}
      </div>
    </div>
  );
}
