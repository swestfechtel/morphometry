'use client';

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { PollingComponent } from "@/app/components/polling-component";
import server_config from "@/app/server_config";
import { BaseExamination } from "@/app/types";

function wait(seconds: number) {
    return new Promise(resolve => setTimeout(resolve, seconds * 1000));
}

export function ListComponent({ examinations, description }: { examinations: BaseExamination[]; description: string }) {
    const [accessionNumberFilter, setAccessionNumberFilter] = useState("");
    const [patientNameFilter, setPatientNameFilter] = useState("");
    const [studyDescriptionFilter, setStudyDescriptionFilter] = useState("");
    const [studyDateFilter, setStudyDateFilter] = useState("");
    const [studyTimeFilter, setStudyTimeFilter] = useState("");
    const [statusFilter, setStatusFilter] = useState("");

    const filteredExaminations = examinations.filter((examination) => {
        return (
            (examination.accession_number.toLowerCase().includes(accessionNumberFilter.toLowerCase()) || accessionNumberFilter === "") &&
            (examination.patient_name.toLowerCase().includes(patientNameFilter.toLowerCase()) || patientNameFilter === "") &&
            (examination.study_description.toLowerCase().includes(studyDescriptionFilter.toLowerCase()) || studyDescriptionFilter === "") &&
            (examination.study_date.includes(studyDateFilter) || studyDateFilter === "") &&
            (examination.study_time.includes(studyTimeFilter) || studyTimeFilter === "") &&
            (examination.status.toLowerCase().includes(statusFilter.toLowerCase()) || statusFilter === "")
        );
    });

    return (
      <div>
          <p className="text-gray-800 dark:text-white">
              Below is a list of all <u>{description}</u> examinations previously sent to the system.
          </p>
          <table
              className="min-w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 mt-6">
              <thead>
              <tr>
                  <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-left text-lg font-semibold text-gray-800 dark:text-white">Accession Number</th>
                  <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-left text-lg font-semibold text-gray-800 dark:text-white">Patient Name</th>
                  <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-left text-lg font-semibold text-gray-800 dark:text-white">Description</th>
                  <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-left text-lg font-semibold text-gray-800 dark:text-white">Examination Date</th>
                  <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-left text-lg font-semibold text-gray-800 dark:text-white">Examination Time</th>
                  <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-left text-lg font-semibold text-gray-800 dark:text-white">Status</th>
                  <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-left text-lg font-semibold text-gray-800 dark:text-white">Actions</th>
              </tr>
              </thead>
              <tbody>
              <tr key="filter" className="">
                  <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">
                      <input type="text" placeholder="Filter by Accession Number" value={accessionNumberFilter} onChange={(e) => setAccessionNumberFilter(e.target.value)} className="border border-gray-300 dark:border-gray-700 rounded p-2"/>
                  </td>
                  <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">
                      <input type="text" placeholder="Filter by Patient Name" value={patientNameFilter} onChange={(e) => setPatientNameFilter(e.target.value)} className="border border-gray-300 dark:border-gray-700 rounded p-2"/>
                  </td>
                  <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">
                      <input type="text" placeholder="Filter by Description" value={studyDescriptionFilter} onChange={(e) => setStudyDescriptionFilter(e.target.value)} className="border border-gray-300 dark:border-gray-700 rounded p-2"/>
                  </td>
                  <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">
                      <input type="text" placeholder="Filter by Date" value={studyDateFilter} onChange={(e) => setStudyDateFilter(e.target.value)} className="border border-gray-300 dark:border-gray-700 rounded p-2"/>
                  </td>
                  <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">
                      <input type="text" placeholder="Filter by Time" value={studyTimeFilter} onChange={(e) => setStudyTimeFilter(e.target.value)} className="border border-gray-300 dark:border-gray-700 rounded p-2"/>
                  </td>
                  <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">
                      <input type="text" placeholder="Filter by Status" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="border border-gray-300 dark:border-gray-700 rounded p-2"/>
                  </td>
                  <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white"></td>
              </tr>
              {filteredExaminations.map((examination) => (
                  <ListElementComponent examination={examination} key={examination.accession_number}/>
              ))}
              </tbody>
          </table>
      </div>
    );
}

function ListElementComponent({ examination }: { examination: BaseExamination }) {
    const [jobId, setJobId] = useState<string | undefined>(undefined);
    const router = useRouter();

    // Start polling if the examination comes back from the server already in 'running' state
    useEffect(() => {
        if (examination.status === 'running') {
            setJobId(examination.accession_number);
        }
    }, [examination.status, examination.accession_number]);

    const segment = async () => {
        const tmp = await fetch(server_config.model_api + '/model/segmentation/' + examination.accession_number, { method: 'POST' });
        const json = await tmp.json();
        setJobId(json.job_id);
    };

    const process = async () => {
        const tmp = await fetch(server_config.model_api + '/model/torsion/' + examination.accession_number, { method: 'POST' });
        const json = await tmp.json();
        setJobId(json.job_id);
    };

    const deleteExamination = async () => {
        const tmp = await fetch(server_config.model_api + '/examinations/' + examination.accession_number, { method: 'DELETE' });
        if (tmp.status === 205) {
            router.refresh();
        }
    };

    const pollingCallback = async (_jobId: string) => {
        await wait(1);
        setJobId(undefined);
        router.refresh();
    };

    return (
        <tr key={examination.accession_number} className="hover:bg-gray-100 dark:hover:bg-gray-700">
            <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">{examination.accession_number}</td>
            <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">{examination.patient_name}</td>
            <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">{examination.study_description}</td>
            <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">{examination.study_date}</td>
            <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">{examination.study_time}</td>
            <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700 text-lg text-gray-800 dark:text-white">
                {jobId !== undefined ?
                    <PollingComponent job_id={jobId} callback={pollingCallback}/>
                :
                    examination.status
                }
            </td>
            <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-700">
                {examination.status === 'processed' &&
                <a href={`/examinations/${examination.accession_number}`} className="text-blue-500 hover:underline">View Examination</a>
                }
                {(examination.status === 'unprocessed' || examination.status === 'segmented') && jobId === undefined &&
                <a href="#" onClick={process} className="text-blue-500 hover:underline">Start Processing</a>
                }
                <br/>
                <a href="#" onClick={deleteExamination} className="text-red-500 hover:underline">Delete</a>
            </td>
        </tr>
    );
}