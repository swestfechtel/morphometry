import Link from "next/link";
import { useTranslations } from "next-intl";

export default function FilterComponent({ state }: { state?: string }) {
    const t = useTranslations("examinations");
    const activeStyling = "py-2 px-4 text-lg font-semibold text-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 bg-amber-200";
    const inactiveStyling = "py-2 px-4 text-lg font-semibold text-gray-800 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700";

    return (
        <div className="flex border-b border-gray-300 dark:border-gray-700">
            <Link className={state === 'mr-torsion' ? activeStyling : inactiveStyling} href="/examinations/mr-torsion">
                {t("filterTorsionMr")}
            </Link>
            <Link className={inactiveStyling} href="/examinations/">
                {t("filterClear")}
            </Link>
        </div>
    );
}
