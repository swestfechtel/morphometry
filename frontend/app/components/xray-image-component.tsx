'use client';

import { useState, useRef } from 'react';
import { XrayExamination, XrayLandmarks } from "@/app/types";

interface Props {
    examination: XrayExamination;
    saveChangesCallback: (landmarks: XrayLandmarks) => void;
    setLandmarksCallback: (landmarks: XrayLandmarks) => void;
}

export function ImageWithLandmarks({ examination, saveChangesCallback, setLandmarksCallback }: Props) {
    const [landmarks, setLandmarks] = useState<XrayLandmarks>(examination.landmarks);
    const [dragging, setDragging] = useState<[string, 'start' | 'end'] | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const handleMouseDown = (landmark: [string, 'start' | 'end']) => (e: React.MouseEvent) => {
        setDragging(landmark);
        e.stopPropagation();
        e.preventDefault();
    };

    const handleMouseUp = () => {
        if (dragging === null) return;
        saveChangesCallback(landmarks);
        setDragging(null);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (dragging === null) return;
        if (!containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const newLandmarks = structuredClone(landmarks);
        const landmark = newLandmarks[dragging[0]][dragging[1]];
        landmark[0] = x;
        landmark[1] = y;
        setLandmarks(newLandmarks);
        setLandmarksCallback(newLandmarks);
    };

    return (
        <div
            ref={containerRef}
            className="relative"
            style={{ maxHeight: '100vh' }}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
        >
            <img
                src={'data:image/png;base64,' + examination.image}
                className="block object-contain"
                style={{ maxHeight: '100vh' }}
                alt="x-ray"
            />
            {Object.entries(landmarks).map(([axis_name, coords]) => {
                const [startX, startY] = coords['start'];
                const [endX, endY] = coords['end'];
                return (
                    <svg className="absolute top-0 left-0 w-full h-full" style={{ pointerEvents: "none" }} visibility={"visible"} key={axis_name}>
                        <circle cx={startX} cy={startY} r={4} fill="blue" stroke="white" strokeWidth="1" style={{ pointerEvents: "auto", cursor: "pointer" }} onMouseDown={handleMouseDown([axis_name, 'start'])} />
                        <circle cx={endX} cy={endY} r={4} fill="red" stroke="white" strokeWidth="1" style={{ pointerEvents: "auto", cursor: "pointer" }} onMouseDown={handleMouseDown([axis_name, 'end'])} />
                        <line x1={startX} y1={startY} x2={endX} y2={endY} stroke="white" strokeWidth="2" />
                    </svg>
                );
            })}
        </div>
    );
}