'use client';

import { useState } from "react";
import dynamic from "next/dynamic";
import { apiFetch } from "@/app/auth";
import { TorsionExamination, TorsionLandmarks } from "@/app/types";

// Cornerstone is client-only (DOM + web workers); load it with ssr disabled so
// nothing runs during server rendering.
const CornerstoneTorsionViewer = dynamic(
    () => import("@/app/components/cornerstone/cornerstone-torsion-viewer").then((m) => m.CornerstoneTorsionViewer),
    { ssr: false, loading: () => <div className="text-white">Loading viewer…</div> },
);

export function TorsionExaminationComponent({ examination }: { examination: TorsionExamination }) {
    const [landmarks, setLandmarks] = useState<TorsionLandmarks>(examination.landmarks);
    const [changes, setChanges] = useState<TorsionLandmarks | null>(null);
    // Segmentation overlay is not yet implemented on the axial stack viewer; the
    // toggle is hidden below. Kept as constant state so the prop plumbing stays intact.
    const [showSegmentation] = useState(true);

    const update = async () => {
        if (!changes) return;
        const to_update = { 'landmarks': changes };
        await apiFetch('/examinations/' + examination.accession_number, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(to_update),
        });
        setChanges(null);
    };

    // Examination metadata shown to the left of the viewer (replaces the page header).
    const details: { label: string; value: string }[] = [
        { label: 'Patient name', value: examination.patient_name },
        { label: 'Study date', value: examination.study_date },
        { label: 'Study time', value: examination.study_time },
        { label: 'Study description', value: examination.study_description },
        { label: 'Accession number', value: examination.accession_number },
        { label: 'Status', value: examination.status },
    ];

    return (
        <div className="flex gap-4 px-4 py-4 items-start">
            <aside className="w-64 shrink-0">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-white mb-3">Examination Details</h2>
                <dl className="flex flex-col gap-2">
                    {details.map(({ label, value }) => (
                        <div key={label} className="flex flex-col">
                            <dt className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</dt>
                            <dd className="text-sm text-gray-800 dark:text-gray-100 break-words">{value || '—'}</dd>
                        </div>
                    ))}
                </dl>
            </aside>
            <div className="flex-1 min-w-0">
                {/* Torsion values and the Save button live in the viewer's top-right overlay; the
                    proximal/distal angle of each reference line is drawn on its own slice. */}
                <CornerstoneTorsionViewer
                    examination={examination}
                    showSegmentation={showSegmentation}
                    landmarks={landmarks}
                    hasChanges={changes != null}
                    onSave={update}
                    saveChangesCallback={setChanges}
                    setLandmarksCallback={setLandmarks}
                />
            </div>
        </div>
    );
}