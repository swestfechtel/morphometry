'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { add_arrays } from "@/app/utils";
import { TorsionExamination, TorsionLandmarks, TorsionBonePoints, TorsionTibiaPoints } from "@/app/types";

interface LandmarkGroup {
    bone: string;
    side: string;
    method?: string;
    coords: TorsionBonePoints | TorsionTibiaPoints;
}

interface Props {
    examination: TorsionExamination;
    showSegmentation: boolean;
    saveChangesCallback: (landmarks: TorsionLandmarks) => void;
    setLandmarksCallback: (landmarks: TorsionLandmarks) => void;
}

export function ImageWithLandmarks({ examination, showSegmentation, saveChangesCallback, setLandmarksCallback }: Props) {
    const [landmarks, setLandmarks] = useState<TorsionLandmarks>(examination.landmarks);
    const [dragging, setDragging] = useState<string[] | null>(null);
    const [currentImageIndex, setCurrentImageIndex] = useState(0);

    const [xFactor, setXFactor] = useState(0);
    const [yFactor, setYFactor] = useState(0);
    const [xFactorReverse, setXFactorReverse] = useState(0);
    const [yFactorReverse, setYFactorReverse] = useState(0);

    const containerRef = useRef<HTMLDivElement>(null);

    const images = useMemo(
        () => examination.image.map(layer => 'data:image/png;base64,' + layer),
        [examination.image]
    );
    const imagesWithMask = useMemo(
        () => examination.segmentation.map(layer => 'data:image/png;base64,' + layer),
        [examination.segmentation]
    );

    const handleMouseDown = (landmark: string[]) => (e: React.MouseEvent) => {
        setDragging(landmark);
        e.stopPropagation();
        e.preventDefault();
    };

    const handleMouseUp = () => {
        if (dragging === null) return;
        saveChangesCallback(landmarks);
        setDragging(null);
    };

    // Traverse nested landmark object by path array, returning the coordinate array
    const getLandmarkByPath = (root: TorsionLandmarks, path: string[]): number[] | null => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return path.reduce<any>((acc, key) => (acc ? acc[key] : null), root) as number[] | null;
    };

    const buildLandmarkGroups = (): LandmarkGroup[] => {
        const groups: LandmarkGroup[] = [];
        if (!landmarks) return groups;

        for (const [bone, boneData] of Object.entries(landmarks)) {
            if (!boneData) continue;

            if (bone === 'femur') {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const femurData = boneData as any;
                const hasMethods = femurData?.Lee || femurData?.Murphy;

                if (hasMethods) {
                    for (const [methodKey, methodData] of Object.entries(femurData)) {
                        if (!methodData) continue;
                        for (const [sideKey, coords] of Object.entries(methodData as Record<string, TorsionBonePoints>)) {
                            groups.push({ bone, method: methodKey, side: sideKey, coords });
                        }
                    }
                } else {
                    for (const [sideKey, coords] of Object.entries(femurData as Record<string, TorsionBonePoints>)) {
                        groups.push({ bone, side: sideKey, coords });
                    }
                }
            } else {
                for (const [sideKey, coords] of Object.entries(boneData as Record<string, TorsionTibiaPoints>)) {
                    groups.push({ bone, side: sideKey, coords });
                }
            }
        }

        return groups;
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (dragging === null) return;
        if (!containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const newLandmarks = structuredClone(landmarks);
        const landmark = getLandmarkByPath(newLandmarks, dragging);
        if (!landmark) return;
        landmark[0] = x * xFactorReverse;
        landmark[1] = y * yFactorReverse;
        setLandmarks(newLandmarks);
        setLandmarksCallback(newLandmarks);
    };

    const handleWheel = (e: WheelEvent) => {
        e.preventDefault();
        if (e.deltaY > 0) {
            setCurrentImageIndex((prevIndex) => (prevIndex + 1) % images.length);
        } else if (e.deltaY < 0) {
            setCurrentImageIndex((prevIndex) =>
                (prevIndex - 1 + images.length) % images.length
            );
        }
    };

    const handleResize = (entries: ResizeObserverEntry[]) => {
        const rect = entries[0].contentRect;
        setXFactor(rect.width / examination.shape[0]);
        setYFactor(rect.height / examination.shape[1]);
        setXFactorReverse(examination.shape[0] / rect.width);
        setYFactorReverse(examination.shape[1] / rect.height);
    };

    useEffect(() => {
        if (!containerRef.current) return;
        const container = containerRef.current;

        const resizeObserver = new window.ResizeObserver(handleResize);
        resizeObserver.observe(container);
        container.addEventListener('wheel', handleWheel, { passive: false });

        // Initial scale calculation
        const rect = container.getBoundingClientRect();
        setXFactor(rect.width / examination.shape[0]);
        setYFactor(rect.height / examination.shape[1]);
        setXFactorReverse(examination.shape[0] / rect.width);
        setYFactorReverse(examination.shape[1] / rect.height);

        return () => {
            resizeObserver.disconnect();
            container.removeEventListener('wheel', handleWheel);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div
            ref={containerRef}
            className="relative resize overflow-auto"
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
        >
            <img
                src={showSegmentation ? images[currentImageIndex] : imagesWithMask[currentImageIndex]}
                alt={`Image ${currentImageIndex + 1}`}
                className="w-full h-full"
            />

            {buildLandmarkGroups().map(({ bone, side, method, coords }) => {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const c = coords as any;
                let startFirst, endFirst, startSecond, endSecond;
                if (bone === 'femur') {
                    startFirst = c['hip_start'];
                    endFirst = c['hip_end'];
                    startSecond = add_arrays(c['knee_start'], [0, 0, examination.knee_offset]);
                    endSecond = add_arrays(c['knee_end'], [0, 0, examination.knee_offset]);
                } else {
                    startFirst = add_arrays(c['knee_start'], [0, 0, examination.knee_offset]);
                    endFirst = add_arrays(c['knee_end'], [0, 0, examination.knee_offset]);
                    startSecond = add_arrays(c['ankle_start'], [0, 0, examination.ankle_offset]);
                    endSecond = add_arrays(c['ankle_end'], [0, 0, examination.ankle_offset]);
                }

                const basePath = method ? [bone, method, side] : [bone, side];
                const keyPrefix = `${bone}-${method ?? 'default'}-${side}`;

                return (
                    <div key={keyPrefix}>
                        <svg
                            className="absolute top-0 left-0 w-full h-full"
                            style={{ pointerEvents: "none" }}
                            visibility={currentImageIndex === startFirst[2] ? 'visible' : 'hidden'}
                        >
                            <line
                                x1={startFirst[0] * xFactor} y1={startFirst[1] * yFactor}
                                x2={endFirst[0] * xFactor} y2={endFirst[1] * yFactor}
                                stroke="red" strokeWidth="2"
                            />
                            <circle
                                cx={startFirst[0] * xFactor} cy={startFirst[1] * yFactor}
                                r="8" fill="blue" stroke="white" strokeWidth="2"
                                style={{ pointerEvents: "auto", cursor: "pointer" }}
                                onMouseDown={handleMouseDown([...basePath, bone === 'femur' ? 'hip_start' : 'knee_start'])}
                            />
                            <circle
                                cx={endFirst[0] * xFactor} cy={endFirst[1] * yFactor}
                                r="8" fill="green" stroke="white" strokeWidth="2"
                                style={{ pointerEvents: "auto", cursor: "pointer" }}
                                onMouseDown={handleMouseDown([...basePath, bone === 'femur' ? 'hip_end' : 'knee_end'])}
                            />
                        </svg>
                        <svg
                            className="absolute top-0 left-0 w-full h-full"
                            style={{ pointerEvents: "none" }}
                            visibility={currentImageIndex === startSecond[2] ? 'visible' : 'hidden'}
                        >
                            <line
                                x1={startSecond[0] * xFactor} y1={startSecond[1] * yFactor}
                                x2={endSecond[0] * xFactor} y2={endSecond[1] * yFactor}
                                stroke="red" strokeWidth="2"
                            />
                            <circle
                                cx={startSecond[0] * xFactor} cy={startSecond[1] * yFactor}
                                r="8" fill="blue" stroke="white" strokeWidth="2"
                                style={{ pointerEvents: "auto", cursor: "pointer" }}
                                onMouseDown={handleMouseDown([...basePath, bone === 'femur' ? 'knee_start' : 'ankle_start'])}
                            />
                            <circle
                                cx={endSecond[0] * xFactor} cy={endSecond[1] * yFactor}
                                r="8" fill="green" stroke="white" strokeWidth="2"
                                style={{ pointerEvents: "auto", cursor: "pointer" }}
                                onMouseDown={handleMouseDown([...basePath, bone === 'femur' ? 'knee_end' : 'ankle_end'])}
                            />
                        </svg>
                    </div>
                );
            })}

            {/* Resize icon in the bottom right corner */}
            <div
                className="absolute right-0 bottom-0 z-10 flex items-end justify-end pointer-events-none"
                style={{ width: 32, height: 32 }}
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                     className="bi bi-arrows-angle-expand" viewBox="0 0 16 16">
                    <path fillRule="evenodd"
                          d="M5.828 10.172a.5.5 0 0 0-.707 0l-4.096 4.096V11.5a.5.5 0 0 0-1 0v3.975a.5.5 0 0 0 .5.5H4.5a.5.5 0 0 0 0-1H1.732l4.096-4.096a.5.5 0 0 0 0-.707m4.344-4.344a.5.5 0 0 0 .707 0l4.096-4.096V4.5a.5.5 0 1 0 1 0V.525a.5.5 0 0 0-.5-.5H11.5a.5.5 0 0 0 0 1h2.768l-4.096 4.096a.5.5 0 0 0 0 .707"/>
                </svg>
            </div>
        </div>
    );
}