'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import server_config from "@/app/server_config";

export default function UploadPage() {
    const [selectedFile, setSelectedFile] = useState<FileList | null>(null);
    const [examinationType, setExaminationType] = useState('Torsion');
    const [message, setMessage] = useState('');
    const router = useRouter();

    // Handle file input change
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setSelectedFile(e.target.files);
        }
    };

    // Submit the form to the API endpoint
    async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();

        if (!selectedFile) {
            setMessage('Please select a file first.');
            return;
        }

        // Create FormData to send the file
        const formData = new FormData();

        for (const file of selectedFile) {
            formData.append('files', file);
            formData.append('examination_type', examinationType);
        }

        try {
            const response = await fetch(server_config.model_api + '/upload/', {
                method: 'POST',
                body: formData,
            });

            if (response.status == 201) {
                const data = await response.json();
                setMessage('File sent successfully!');

                // router.push(`/results/${data.examination_id}`);

            } else if (response.status == 400) {
                setMessage('Examination already exists on server.');
            }
            else {
                const data = await response.json();
                setMessage('Error sending file: ' + data.error);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            setMessage('Network error occurred.');
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
            <form
                onSubmit={handleSubmit}
                className="bg-white dark:bg-gray-800 p-8 rounded shadow-md w-full max-w-md"
            >
                <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-gray-100">
                    Upload Files
                </h2>
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
                <div className="mb-4">
                    <label className="block mb-2 text-gray-800 dark:text-gray-100" htmlFor="examinationType">
                        Examination Type
                    </label>
                    <select
                        id="examinationType"
                        value={examinationType}
                        onChange={e => setExaminationType(e.target.value)}
                        className="w-full p-2 border border-gray-300 dark:border-gray-700 rounded text-gray-800 dark:text-gray-100 bg-white dark:bg-gray-800"
                    >
                        <option value="default" disabled>Select a type</option>
                        <option value="torsion" className="text-gray-800 dark:text-gray-100">Torsion</option>
                        <option value="x_ray_foot_ap" className="text-gray-800 dark:text-gray-100">X-ray foot AP</option>
                    </select>
                </div>
                <button
                    type="submit"
                    className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded transition duration-300"
                >
                    Send Files
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