import { useState, useEffect, useRef } from "react";
import server_config from "@/app/server_config";
import { JobData } from "@/app/types";

function usePolling(apiEndpoint: string, interval: number, onData: (data: JobData) => void) {
    const onDataRef = useRef(onData);
    onDataRef.current = onData;

    useEffect(() => {
        let isMounted = true;

        const fetchData = async () => {
            try {
                const response = await fetch(apiEndpoint);
                if (!isMounted) return;
                if (response.ok) {
                    const data = await response.json() as JobData;
                    if (data.status === 'finished' || data.status === 'error') {
                        clearInterval(intervalId);
                    }
                    onDataRef.current(data);
                } else {
                    clearInterval(intervalId);
                    onDataRef.current({ status: 'error', message: response.statusText });
                }
            } catch (error) {
                clearInterval(intervalId);
                onDataRef.current({ status: 'error', message: String(error) });
            }
        };

        const intervalId = setInterval(fetchData, interval);

        return () => {
            isMounted = false;
            clearInterval(intervalId);
        };
    }, [apiEndpoint, interval]);
}

export function PollingComponent({ job_id, callback }: { job_id: string; callback: (jobId: string) => void }) {
    const [data, setData] = useState<JobData | null>(null);
    const callbackRef = useRef(callback);
    callbackRef.current = callback;

    usePolling(server_config.model_api + '/jobs/' + job_id, 5000, setData);

    useEffect(() => {
        if (data?.status === 'finished') {
            callbackRef.current(job_id);
        }
    }, [data?.status, job_id]);

    return (
        <div className="ml-2">
            {data && data.status === 'running' &&
                <span
                    className="inline-flex items-center rounded bg-blue-50 px-4 py-2 font-semibold text-blue-700 ring-1 ring-blue-700/10 ring-inset">
                Processing...
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                     fill="currentColor" className="bi bi-arrow-clockwise spin ml-2"
                     viewBox="0 0 16 16">
                    <path fillRule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2z"/>
                    <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466"/>
                </svg>
            </span>
            }
            {data && data.status === 'finished' &&
                <span
                    className="inline-flex items-center rounded bg-green-50 px-4 py-2 font-semibold text-green-700 ring-1 ring-green-700/10 ring-inset">
                Processing finished!
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                     fill="currentColor" className="bi bi-check-circle ml-2" viewBox="0 0 16 16">
                    <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16"/>
                    <path d="m10.97 4.97-.02.022-3.473 4.425-2.093-2.094a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-1.071-1.05"/>
                </svg>
            </span>
            }
            {data && data.status === 'error' &&
                <span
                    className="inline-flex items-center rounded bg-red-50 px-4 py-2 font-semibold text-red-700 ring-1 ring-red-700/10 ring-inset">
                Error processing data: {data.message}
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                         className="bi bi-explicit" viewBox="0 0 16 16">
                  <path d="M6.826 10.88H10.5V12h-5V4.002h5v1.12H6.826V7.4h3.457v1.073H6.826z"/>
                  <path
                      d="M2.5 0A2.5 2.5 0 0 0 0 2.5v11A2.5 2.5 0 0 0 2.5 16h11a2.5 2.5 0 0 0 2.5-2.5v-11A2.5 2.5 0 0 0 13.5 0zM1 2.5A1.5 1.5 0 0 1 2.5 1h11A1.5 1.5 0 0 1 15 2.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 1 13.5z"/>
                </svg>
            </span>
            }
        </div>
    );
}