'use client';

import { useState } from 'react';
import { apiFetch } from '@/app/auth';
import { SeriesPicker } from '@/app/components/series-picker';
import { PendingExamination } from '@/app/types';

export default function UploadPage() {
    const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
    const [message, setMessage] = useState('');
    const [busy, setBusy] = useState(false);
    // When a directory has been enumerated, its candidate series are shown for
    // selection; picking one starts processing (see SeriesPicker).
    const [pending, setPending] = useState<PendingExamination | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setSelectedFiles(e.target.files);
        }
    };

    // Torsion: enumerate the series in the selected examination directory.
    async function enumerateTorsionSeries(files: FileList) {
        const formData = new FormData();
        for (const file of files) formData.append('files', file);
        const response = await apiFetch('/upload/torsion/series', { method: 'POST', body: formData });
        if (response.status === 201) {
            setPending((await response.json()) as PendingExamination);
        } else if (response.status === 400) {
            setMessage('This examination already exists on the server.');
        } else {
            const data = await response.json().catch(() => ({}));
            setMessage('Error uploading examination: ' + (data.detail ?? response.statusText));
        }
    }

    async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();
        setMessage('');
        if (!selectedFiles) {
            setMessage('Please select an examination directory first.');
            return;
        }
        setBusy(true);
        try {
            await enumerateTorsionSeries(selectedFiles);
        } catch (error) {
            console.error('Error uploading:', error);
            setMessage('Network error occurred.');
        } finally {
            setBusy(false);
        }
    }

    // Once a torsion directory is enumerated, replace the form with the series picker.
    // "Upload another" resets back to the form so more exams can be queued without
    // waiting for the previous one to finish processing.
    if (pending) {
        return (
            <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
                <SeriesPicker
                    examinationId={pending.accession_number}
                    series={pending.series}
                    onUploadAnother={() => { setPending(null); setMessage(''); setSelectedFiles(null); }}
                />
            </div>
        );
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
            <form
                onSubmit={handleSubmit}
                className="bg-white dark:bg-gray-800 p-8 rounded shadow-md w-full max-w-md"
            >
                <h2 className="text-2xl font-bold mb-2 text-gray-800 dark:text-gray-100">
                    Upload Examination
                </h2>
                <p className="mb-4 text-sm text-gray-600 dark:text-gray-300">
                    Select the whole torsion examination folder exported from the PACS. The
                    server lists the DICOM series it contains so you can pick the correct one.
                </p>
                <div className="mb-4">
                    <input
                        type="file"
                        directory=""
                        webkitdirectory=""
                        onChange={handleFileChange}
                        className="w-full text-gray-800 dark:text-gray-100 hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent rounded-md border border-gray-300 dark:border-gray-700 p-2"
                        multiple
                    />
                </div>
                <button
                    type="submit"
                    disabled={busy}
                    className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {busy ? 'Uploading…' : 'Upload & list series'}
                </button>
                {message && (
                    <p className="mt-4 text-center text-gray-800 dark:text-gray-100">
                        {message}
                    </p>
                )}
            </form>
        </div>
    );
}
