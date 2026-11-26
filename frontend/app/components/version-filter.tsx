'use client';

import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { Release } from "@/app/types";

const selectStyling = "border border-gray-300 dark:border-gray-700 rounded p-2 text-lg text-gray-800 dark:text-white bg-white dark:bg-gray-800";

/**
 * Release-version search controls for the examinations list.
 *
 * Renders a version <select> (all / each release newest first / unreleased / none) and a
 * stage <select> (any / torsion / segmentation). Changing either navigates to
 * `/examinations?version=...&stage=...`, dropping empty parameters, so the filter lives in
 * the URL (survives reload, shareable) and the server component re-fetches.
 *
 * @param releases - Releases from GET /releases, newest first (the order is kept as given).
 * @param version - Currently selected `version` search parameter ("" for none).
 * @param stage - Currently selected `stage` search parameter ("" for any).
 */
export default function VersionFilter({ releases, version, stage }: { releases: Release[]; version: string; stage: string }) {
    const t = useTranslations("examinations");
    const router = useRouter();

    const navigate = (nextVersion: string, nextStage: string) => {
        const params = new URLSearchParams();
        if (nextVersion) params.set("version", nextVersion);
        // A stage without a version filters nothing, so only keep it alongside a version.
        if (nextVersion && nextStage && nextStage !== "any") params.set("stage", nextStage);
        const qs = params.toString();
        router.push(qs ? `/examinations?${qs}` : "/examinations");
    };

    // Keep a version typed into the URL by hand selectable, so the <select> never lies.
    const known = new Set(["", "unreleased", "none", ...releases.map((r) => r.version)]);

    return (
        <div className="flex flex-wrap items-center gap-4 py-3 border-b border-gray-300 dark:border-gray-700">
            <label className="flex items-center gap-2 text-gray-800 dark:text-white">
                <span>{t("filterVersion")}</span>
                <select aria-label={t("filterVersion")} className={selectStyling} value={version}
                        onChange={(e) => navigate(e.target.value, stage)}>
                    <option value="">{t("versionAll")}</option>
                    {releases.map((r) => (
                        <option key={r.version} value={r.version}>{r.version}</option>
                    ))}
                    <option value="unreleased">{t("versionUnreleased")}</option>
                    <option value="none">{t("versionNone")}</option>
                    {!known.has(version) && <option value={version}>{version}</option>}
                </select>
            </label>
            <label className="flex items-center gap-2 text-gray-800 dark:text-white">
                <span>{t("filterStage")}</span>
                <select aria-label={t("filterStage")} className={selectStyling} value={stage || "any"}
                        disabled={!version} onChange={(e) => navigate(version, e.target.value)}>
                    <option value="any">{t("stageAny")}</option>
                    <option value="torsion">{t("stageTorsion")}</option>
                    <option value="segmentation">{t("stageSegmentation")}</option>
                </select>
            </label>
        </div>
    );
}
