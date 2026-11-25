import { redirect } from "next/navigation";
import { getTranslations } from "next-intl/server";
import {ListComponent} from "@/app/components/list-component";
import FilterComponent from "@/app/components/filter-component";
import VersionFilter from "@/app/components/version-filter";
import server_config from "@/app/server_config";
import { serverAuthHeaders } from "@/app/server-auth";
import { BaseExamination, Release } from "@/app/types";

/** First value of a Next.js search parameter (which may be repeated), or "". */
function first(value: string | string[] | undefined): string {
    return (Array.isArray(value) ? value[0] : value) ?? "";
}

export default async function page({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
    const t = await getTranslations("examinations");
    const sp = await searchParams;
    const version = first(sp.version);
    const stage = first(sp.stage);

    const query = new URLSearchParams();
    if (version) query.set("version", version);
    if (version && stage) query.set("stage", stage);
    const qs = query.toString();

    const headers = await serverAuthHeaders();
    const result = await fetch(server_config.model_api + '/examinations/' + (qs ? `?${qs}` : ''), { headers });
    // Keep the filter across the login round trip.
    if (result.status === 401) redirect('/login?next=' + encodeURIComponent('/examinations' + (qs ? `?${qs}` : '')));

    let examinations: BaseExamination[] = [];
    let error: string | null = null;
    if (result.ok) {
        examinations = (await result.json()).examinations;
    } else {
        // e.g. 404 for an unknown version: the API's {detail, code} body carries the message.
        const body = await result.json().catch(() => null);
        error = (typeof body?.detail === "string" && body.detail) || `HTTP ${result.status}`;
    }

    // /releases is open and optional: if it is unreachable the dropdown just offers fewer options.
    let releases: Release[] = [];
    try {
        const r = await fetch(server_config.model_api + '/releases', { headers });
        if (r.ok) releases = (await r.json()).releases ?? [];
    } catch { /* ignore */ }

    return (
        <div className="min-h-screen bg-white dark:bg-gray-800">
            {/* Header */}
            <header className="bg-white dark:bg-gray-700 shadow mb-2 my-auto px-4 py-4">
                <h1 className="text-4xl font-bold text-gray-800 dark:text-white">{t("title")}</h1>
            </header>
            {/* Main Content */}
            <div className="mx-auto px-4 shadow">
                <FilterComponent state="none"/>
                <VersionFilter releases={releases} version={version} stage={stage}/>
                {error && (
                    <p role="alert" className="mt-4 rounded border border-red-300 bg-red-50 p-3 text-red-800 dark:bg-red-900 dark:text-red-100">
                        {t("versionError", { message: error })}
                    </p>
                )}
                <section id="content" className="py-12">
                    <ListComponent examinations={examinations} description=""/>
                </section>
            </div>
        </div>
    );
}
