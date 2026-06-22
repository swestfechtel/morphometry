'use client';

import layers_b64 from "@/app/mad_ui_test.json";
import { useState, useEffect, useRef } from "react";
import { subtract_arrays, vectorLength } from "@/app/utils";

export function MadComponent() {
    const images = layers_b64.layers.map(layer => "data:image/png;base64," + layer);

    const [currentImageIndex, setCurrentImageIndex] = useState(0);
    const [firstLine, setFirstLine] = useState([[0, 0, 0], [100, 0, 100]]);
    const [secondLine, setSecondLine] = useState([[0, 0, 0], [100, 0, 100]]);
    const [firstLineLength, setFirstLineLength] = useState(vectorLength(subtract_arrays(firstLine[1], firstLine[0])));
    const [secondLineLength, setSecondLineLength] = useState(vectorLength(subtract_arrays(secondLine[1], secondLine[0])));
    const [dragging, setDragging] = useState<string | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const handleMouseDown = (landmark: string) => (e: React.MouseEvent) => {
        setDragging(landmark);
        e.stopPropagation();
        e.preventDefault();
    };

    const handleMouseUp = () => {
        if (dragging === null) return;
        setDragging(null);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (dragging === null) return;
        if (!containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        switch (dragging) {
            case 'firstLineStart':
                setFirstLine([[x, currentImageIndex, y], firstLine[1]]);
                break;
            case 'firstLineEnd':
                setFirstLine([firstLine[0], [x, currentImageIndex, y]]);
                break;
            case 'secondLineStart':
                setSecondLine([[x, currentImageIndex, y], secondLine[1]]);
                break;
            case 'secondLineEnd':
                setSecondLine([secondLine[0], [x, currentImageIndex, y]]);
                break;
        }

        setFirstLineLength(vectorLength(subtract_arrays(firstLine[1], firstLine[0])));
        setSecondLineLength(vectorLength(subtract_arrays(secondLine[1], secondLine[0])));
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

    useEffect(() => {
        if (!containerRef.current) return;
        const container = containerRef.current;
        container.addEventListener("wheel", handleWheel, { passive: false });
        return () => {
            container.removeEventListener("wheel", handleWheel);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div ref={containerRef} className="relative"
             onMouseMove={handleMouseMove}
             onMouseUp={handleMouseUp}
             onMouseLeave={handleMouseUp}
        >
            <img src={images[currentImageIndex]} alt="" className="" />
            <div className="px-2 py-2 bg-blue-500 text-white rounded absolute top-0 left-0">
                <p className="text-white">Red Line Length: {firstLineLength}</p>
                <p className="text-white">Blue Line Length: {secondLineLength}</p>
            </div>
            <svg className="absolute top-0 left-0 w-full h-full"
                 style={{ pointerEvents: 'none' }}
            >
                <line
                    x1={firstLine[0][0]} y1={firstLine[0][2]}
                    x2={firstLine[1][0]} y2={firstLine[1][2]}
                    stroke="red" strokeWidth="2"
                />
                <circle
                    cx={firstLine[0][0]} cy={firstLine[0][2]}
                    r="5" fill="red"
                    style={{ pointerEvents: "auto", cursor: "pointer" }}
                    onMouseDown={handleMouseDown('firstLineStart')}
                />
                <circle
                    cx={firstLine[1][0]} cy={firstLine[1][2]}
                    r="5" fill="red"
                    style={{ pointerEvents: "auto", cursor: "pointer" }}
                    onMouseDown={handleMouseDown('firstLineEnd')}
                />
            </svg>
            <svg className="absolute top-0 left-0 w-full h-full"
                 style={{ pointerEvents: 'none' }}
            >
                <line
                    x1={secondLine[0][0]} y1={secondLine[0][2]}
                    x2={secondLine[1][0]} y2={secondLine[1][2]}
                    stroke="blue" strokeWidth="2"
                />
                <circle
                    cx={secondLine[0][0]} cy={secondLine[0][2]}
                    r="5" fill="blue"
                    style={{ pointerEvents: "auto", cursor: "pointer" }}
                    onMouseDown={handleMouseDown('secondLineStart')}
                />
                <circle
                    cx={secondLine[1][0]} cy={secondLine[1][2]}
                    r="5" fill="blue"
                    style={{ pointerEvents: "auto", cursor: "pointer" }}
                    onMouseDown={handleMouseDown('secondLineEnd')}
                />
            </svg>
        </div>
    );
}