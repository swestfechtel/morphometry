'use client';

import { useState } from "react";
import server_config from "@/app/server_config";
import { ImageWithLandmarks } from "@/app/components/xray-image-component";
import { subtract_arrays, angleBetweenVectors2D } from "@/app/utils";
import { XrayExamination, XrayLandmarks } from "@/app/types";

export function XrayExaminationComponent({ examination }: { examination: XrayExamination }) {
    const [landmarks, setLandmarks] = useState<XrayLandmarks>(examination.landmarks);
    const [changes, setChanges] = useState<XrayLandmarks | null>(null);

    const update = async () => {
        if (!changes) return;
        await fetch(server_config.model_api + '/examinations/' + examination.accession_number, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(changes),
        });
        setChanges(null);
    };

    return (
        <div>
            <div className="grid grid-rows-1 grid-cols-2 mx-auto px-4 shadow items-start">
                <div className="mx-4 my-4 col-span-1 row-span-1">
                    <ImageWithLandmarks examination={examination} saveChangesCallback={setChanges} setLandmarksCallback={setLandmarks} />
                </div>
                <div className="mx-4 my-4 col-span-1 grid grid-cols-1 gap-2">
                    {changes &&
                    <button
                        onClick={() => update()}
                        className="px-2 py-2 bg-blue-500 text-white rounded w-auto h-auto self-start justify-self-start"
                    >
                        Save changes
                    </button>
                    }
                    {landmarks &&
                    <p className="px-2 py-2 bg-blue-500 text-white rounded w-auto h-auto self-start justify-self-start">Hallux-Valgus Angle: {
                        angleBetweenVectors2D(
                            subtract_arrays(landmarks['longitudinal_firstmetatarsal_axis']['end'], landmarks['longitudinal_firstmetatarsal_axis']['start']),
                            subtract_arrays(landmarks['longitudinal_phalanx_axis']['end'], landmarks['longitudinal_phalanx_axis']['start'])
                        ).toFixed(1)}
                    </p>
                    }
                </div>
            </div>
        </div>
    );
}