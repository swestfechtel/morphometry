// Augment React types to allow non-standard folder-picker attributes on <input>
declare module 'react' {
    interface InputHTMLAttributes<T> {
        directory?: string;
        webkitdirectory?: string;
    }
}

// Examination status and type
export type ExaminationStatus = 'unprocessed' | 'segmented' | 'running' | 'processed';
export type ExaminationType = 'torsion' | 'xray';

// Shared base for all examination list entries
export interface BaseExamination {
    accession_number: string;
    patient_name: string;
    study_description: string;
    study_date: string;
    study_time: string;
    status: ExaminationStatus;
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

// --- X-ray types ---

export type Point2D = [number, number];

export interface XrayAxis {
    start: Point2D;
    end: Point2D;
}

export type XrayLandmarks = Record<string, XrayAxis>;

export interface XrayExamination extends BaseExamination {
    image: string;
    landmarks: XrayLandmarks;
}

// --- Job polling ---

export interface JobData {
    status: 'running' | 'finished' | 'error';
    message?: string;
}