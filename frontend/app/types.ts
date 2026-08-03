// Augment React types to allow non-standard folder-picker attributes on <input>
declare module 'react' {
    interface InputHTMLAttributes<T> {
        directory?: string;
        webkitdirectory?: string;
    }
}

// Examination status and type
export type ExaminationStatus =
    | 'pending_selection'   // uploaded; awaiting the user's series pick
    | 'unprocessed'
    | 'segmented'
    | 'running'
    | 'processed'
    | 'failed';
export type ExaminationType = 'torsion';

// --- Series selection (two-phase upload) ---

/** One candidate DICOM series offered for selection on a pending upload. */
export interface SeriesInfo {
    uid: string;
    description: string | null;
    modality: string | null;
    instances: number;
    rows: number | null;
    cols: number | null;
    preview_count: number;
}

/** Detail payload for a pending_selection examination (GET /examinations/{id}). */
export interface PendingExamination extends BaseExamination {
    type: 'pending';
    series: SeriesInfo[];
}

// Shared base for all examination list entries
export interface BaseExamination {
    accession_number: string;
    patient_name: string;
    study_description: string;
    study_date: string;
    study_time: string;
    status: ExaminationStatus;
    // Id of a queued/running job for this examination (null when nothing is in flight).
    active_job_id?: string | null;
}

// --- Torsion (MRI) types ---

export type Point3D = [number, number, number];

export interface TorsionBonePoints {
    hip_start: Point3D;
    hip_end: Point3D;
    knee_start: Point3D;
    knee_end: Point3D;
}

export interface TorsionTibiaPoints {
    knee_start: Point3D;
    knee_end: Point3D;
    ankle_start: Point3D;
    ankle_end: Point3D;
}

export interface TorsionMethodSides {
    left: TorsionBonePoints;
    right: TorsionBonePoints;
}

export interface TorsionFemurLandmarks {
    Lee: TorsionMethodSides;
    Murphy: TorsionMethodSides;
}

export interface TorsionTibiaLandmarks {
    left: TorsionTibiaPoints;
    right: TorsionTibiaPoints;
}

export interface TorsionLandmarks {
    femur: TorsionFemurLandmarks;
    tibia: TorsionTibiaLandmarks;
}

export interface TorsionExamination extends BaseExamination {
    // Legacy base64 PNG slice lists — no longer served once the Cornerstone
    // viewer is in use (kept optional for backward compatibility).
    image?: string[];
    segmentation?: string[];
    landmarks: TorsionLandmarks;
    shape: number[];        // [x, y, z] of the combined volume
    knee_offset: number;
    ankle_offset: number;
}

// --- Job polling ---

// `queued`/`running`/`finished`/`failed` come from the backend JobState; `error` is
// a synthetic status the poller sets for transport failures (non-OK HTTP / network).
export type JobStatus = 'queued' | 'running' | 'finished' | 'failed' | 'error';

export interface JobData {
    status: JobStatus;
    error?: string;     // backend job error detail (JobStatus.error)
    message?: string;   // poller-set transport error text
}