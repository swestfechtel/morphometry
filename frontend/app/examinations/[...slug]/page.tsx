import server_config from "@/app/server_config";
import Link from "next/link";
import { redirect } from "next/navigation";
import { ListComponent } from "@/app/components/list-component";
import FilterComponent from "@/app/components/filter-component";
import { TorsionExaminationComponent } from '@/app/components/torsion-examination-component';
import { SeriesPicker } from "@/app/components/series-picker";
import { serverAuthHeaders } from "@/app/server-auth";
import { BaseExamination } from "@/app/types";

export default async function page({params,}: {params: Promise<{slug: string}>}){
    const { slug } = await params;

    if (slug[0] === 'mr-torsion') {
        const result = await fetch(server_config.model_api + '/examinations/', { headers: await serverAuthHeaders() });
        if (result.status === 401) redirect('/login?next=/examinations/' + slug[0]);
        const result_json = await result.json();
        const examinations = result_json.examinations;

        return (
            <div className="min-h-screen bg-white dark:bg-gray-800">
                {/* Header */}
                <header className="bg-white dark:bg-gray-700 shadow mb-2 my-auto px-4 py-4">
                    <h1 className="text-4xl font-bold text-gray-800 dark:text-white">Examinations</h1>
                </header>
                {/* Main Content */}
                <div className="mx-auto px-4 shadow">
                    <FilterComponent state={slug[0]}/>
                    <section id="content" className="py-12">
                        <ListComponent examinations={examinations} description="Torsionsbemaßung (MRT)"/>
                    </section>
                </div>
            </div>
        );
    }
    else {

        const result = await fetch(server_config.model_api + '/examinations/' + slug[0], { method: 'GET', headers: await serverAuthHeaders() });
        if (result.status === 401) redirect('/login?next=/examinations/' + slug[0]);
        const examination = await result.json();

        return (
            <div>
                {/* The torsion view renders its own details panel beside the viewer, so the
                    header is only shown for other examination types. */}
                { examination.type !== 'torsion' &&
                <div className="bg-white dark:bg-gray-700 shadow sticky top-4 z-40 w-full mx-auto px-4 py-6">
                    <h1 className="text-4xl font-bold text-gray-800 dark:text-white">Examination Details</h1>
                    <p className="text-lg text-gray-700 dark:text-gray-200 mt-2 px-2 py-1">
                    <span
                        className="inline-flex items-center rounded-md bg-gray-50 px-2 py-1 text-m font-medium text-gray-600 ring-1 ring-gray-500/10 ring-inset mx-1">
                        Patient name: {examination.patient_name}
                    </span>
                        <span
                            className="inline-flex items-center rounded-md bg-gray-50 px-2 py-1 text-m font-medium text-gray-600 ring-1 ring-gray-500/10 ring-inset mx-1">
                        Study date: {examination.study_date}
                    </span>
                        <span
                            className="inline-flex items-center rounded-md bg-gray-50 px-2 py-1 text-m font-medium text-gray-600 ring-1 ring-gray-500/10 ring-inset mx-1">
                        Study time: {examination.study_time}
                    </span>
                        <span
                            className="inline-flex items-center rounded-md bg-gray-50 px-2 py-1 text-m font-medium text-gray-600 ring-1 ring-gray-500/10 ring-inset mx-1">
                        Study description: {examination.study_description}
                    </span>
                        <span
                            className="inline-flex items-center rounded-md bg-gray-50 px-2 py-1 text-m font-medium text-gray-600 ring-1 ring-gray-500/10 ring-inset mx-1">
                        Accession number: {examination.accession_number}
                    </span>
                    </p>
                </div>
                }
                <div className="min-h-screen bg-white dark:bg-gray-800">
                    { examination.type === 'pending' &&
                        <SeriesPicker examinationId={examination.accession_number} series={examination.series} />
                    }
                    { examination.type === 'torsion' &&
                    <TorsionExaminationComponent examination={examination}/>
                    }
                </div>
            </div>
        );
    }
}
