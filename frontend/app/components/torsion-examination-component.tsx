'use client';

import { useState } from "react";
import dynamic from "next/dynamic";
import { useLocale, useTranslations } from "next-intl";
import { apiFetch } from "@/app/auth";
import { TorsionExamination, TorsionLandmarks } from "@/app/types";

// Small client component for the dynamic-import loading state, so the placeholder
// text can be translated (the dynamic() options object can't call hooks itself).
function ViewerLoading() {
    const t = useTranslations("viewer");
    return <div className="text-white">{t("loading")}</div>;
}

// Cornerstone is client-only (DOM + web workers); load it with ssr disabled so
// nothing runs during server rendering.
const CornerstoneTorsionViewer = dynamic(
    () => import("@/app/components/cornerstone/cornerstone-torsion-viewer").then((m) => m.CornerstoneTorsionViewer),
    { ssr: false, loading: () => <ViewerLoading /> },
);

export function TorsionExaminationComponent({ examination }: { examination: TorsionExamination }) {
    const t = useTranslations();
    // Remount the Cornerstone viewer on locale change so its on-canvas landmark labels
    // (built during setup) are rebuilt in the new language.
    const locale = useLocale();
    const [landmarks, setLandmarks] = useState<TorsionLandmarks>(examination.landmarks);
    const [changes, setChanges] = useState<TorsionLandmarks | null>(null);
    // Whether the mask labelmap overlay (hip/knee/ankle) is painted on the slices.
    const [showSegmentation, setShowSegmentation] = useState(true);

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
        { label: t('examination.labels.patientName'), value: examination.patient_name },
        { label: t('examination.labels.studyDate'), value: examination.study_date },
        { label: t('examination.labels.studyTime'), value: examination.study_time },
        { label: t('examination.labels.studyDescription'), value: examination.study_description },
        { label: t('examination.labels.accessionNumber'), value: examination.accession_number },
        { label: t('examination.labels.status'), value: examination.status },
    ];

    return (
        <div className="flex gap-4 px-4 py-4 items-start">
            <aside className="w-64 shrink-0">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-white mb-3">{t('viewer.detailsTitle')}</h2>
                <dl className="flex flex-col gap-2">
                    {details.map(({ label, value }) => (
                        <div key={label} className="flex flex-col">
                            <dt className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</dt>
                            <dd className="text-sm text-gray-800 dark:text-gray-100 break-words">{value || '—'}</dd>
                        </div>
                    ))}
                </dl>

                {/* Segmentation overlay toggle (hip/knee/ankle mask painted on the slices). */}
                <label className="mt-4 flex items-center gap-2 cursor-pointer select-none">
                    <input
                        type="checkbox"
                        checked={showSegmentation}
                        onChange={(e) => setShowSegmentation(e.target.checked)}
                        className="h-4 w-4 accent-blue-600"
                    />
                    <span className="text-sm text-gray-800 dark:text-gray-100">{t('viewer.showSegmentation')}</span>
                </label>
            </aside>
            <div className="flex-1 min-w-0">
                {/* Torsion values and the Save button live in the viewer's top-right overlay; the
                    proximal/distal angle of each reference line is drawn on its own slice. */}
                <CornerstoneTorsionViewer
                    key={locale}
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