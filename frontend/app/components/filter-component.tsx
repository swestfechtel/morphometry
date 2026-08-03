import Link from "next/link";

export default function FilterComponent({ state }: { state?: string }) {
    const activeStyling = "py-2 px-4 text-lg font-semibold text-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 bg-amber-200";
    const inactiveStyling = "py-2 px-4 text-lg font-semibold text-gray-800 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700";

    return (
        <div className="flex border-b border-gray-300 dark:border-gray-700">
            <Link className={state === 'mr-torsion' ? activeStyling : inactiveStyling} href="/examinations/mr-torsion">
                Torsionsbemaßung (MRT)
            </Link>
            <Link className={inactiveStyling} href="/examinations/">
                Clear Filter
            </Link>
        </div>
    );
}