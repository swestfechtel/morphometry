'use client';

import { useState } from "react";
import server_config from "@/app/server_config";
import { ImageWithLandmarks } from "@/app/components/torsion-image-component";
import { computeFemoralTorsion, computeTibialTorsion } from "@/app/utils";
import { TorsionExamination, TorsionFemurLandmarks, TorsionLandmarks, TorsionMethodSides } from "@/app/types";

export function TorsionExaminationComponent({ examination }: { examination: TorsionExamination }) {
    const [landmarks, setLandmarks] = useState<TorsionLandmarks>(examination.landmarks);
    const [changes, setChanges] = useState<TorsionLandmarks | null>(null);
    const [showSegmentation, toggleShowSegmentation] = useState(true);

    const update = async () => {
        if (!changes) return;
        const to_update = { 'landmarks': changes };
        await fetch(server_config.model_api + '/examinations/' + examination.accession_number, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(to_update),
        });
        setChanges(null);
    };

    const getFemurMethodLandmarks = (method: string): TorsionMethodSides | null => {
        const femur = landmarks?.femur;
        if (!femur) return null;
        return femur[method as keyof TorsionFemurLandmarks] ?? null;
    };

    const renderFemurTorsion = (method: string) => {
        const femurMethod = getFemurMethodLandmarks(method);
        if (!femurMethod?.left || !femurMethod?.right) return null;
        return (
            <>
                <p className="text-white">Femur Torsion Left ({method}): {computeFemoralTorsion(femurMethod['left']['hip_start'],
                    femurMethod['left']['hip_end'],
                    femurMethod['left']['knee_start'],
                    femurMethod['left']['knee_end'], 'left')}</p>
                <p className="text-white">Femur Torsion Right ({method}): {computeFemoralTorsion(femurMethod['right']['hip_start'],
                    femurMethod['right']['hip_end'],
                    femurMethod['right']['knee_start'],
                    femurMethod['right']['knee_end'], 'right')}</p>
            </>
        );
    };

    return (
        <div>
            <div className="grid grid-rows-1 grid-cols-2 mx-auto px-4 shadow items-start">
                <div className="mx-4 my-4 col-span-1 row-span-1">
                    <ImageWithLandmarks examination={examination} showSegmentation={showSegmentation} saveChangesCallback={setChanges} setLandmarksCallback={setLandmarks} />
                </div>
                <div className="mx-4 my-4 col-span-1 grid grid-cols-1 gap-2">
                    <button
                        onClick={() => toggleShowSegmentation(!showSegmentation)}
                        className="px-2 py-2 bg-blue-500 text-white rounded w-auto h-auto self-start justify-self-start"
                    >
                        Toggle Segmentation
                    </button>
                    {
                        changes &&
                        <button
                            onClick={() => update()}
                            className="px-2 py-2 bg-blue-500 text-white rounded w-auto h-auto self-start justify-self-start"
                        >
                            Save changes
                        </button>
                    }
                    {landmarks &&
                    <div className="px-2 py-2 bg-blue-500 text-white rounded w-auto h-auto self-start justify-self-start">
                        {renderFemurTorsion('Lee')}
                        {renderFemurTorsion('Murphy')}
                        <p className="text-white">Tibia Torsion Left (Ellipse/Ulm): {computeTibialTorsion(landmarks['tibia']['left']['knee_start'],
                            landmarks['tibia']['left']['knee_end'],
                            landmarks['tibia']['left']['ankle_start'],
                            landmarks['tibia']['left']['ankle_end'], 'left')}</p>
                        <p className="text-white">Tibia Torsion Right (Ellipse/Ulm): {computeTibialTorsion(landmarks['tibia']['right']['knee_start'],
                            landmarks['tibia']['right']['knee_end'],
                            landmarks['tibia']['right']['ankle_start'],
                            landmarks['tibia']['right']['ankle_end'], 'right')}</p>
                    </div>
                    }
                </div>
            </div>
        </div>
    );
}