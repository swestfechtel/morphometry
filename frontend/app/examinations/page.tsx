'use server';

import {ListComponent} from "@/app/components/list-component";
import FilterComponent from "@/app/components/filter-component";
import server_config from "@/app/server_config";

export default async function page() {
    const result = await fetch(server_config.model_api + '/examinations/');
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
                <FilterComponent state="none"/>
                <section id="content" className="py-12">
                    <ListComponent examinations={examinations} description=""/>
                </section>
            </div>
        </div>
    );
}