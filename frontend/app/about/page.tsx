import Link from "next/link";
import { useTranslations } from "next-intl";

/**
 * About page.
 *
 * Static server component explaining what the tool measures, the processing
 * pipeline behind it, the technology stack, and the (important) clinical-use
 * disclaimer. Linked from the navbar (`app/layout.tsx`). Styled to match the
 * app's dark-mode-default Tailwind conventions.
 */
export default function About() {
  const t = useTranslations("about");
  const pipelineSteps = ["ingestion", "segmentation", "torsion", "review"] as const;
  // Rich-text tag renderers reused across the emphasized sentences.
  const bold = (chunks: React.ReactNode) => (
    <span className="font-semibold text-gray-900 dark:text-white">{chunks}</span>
  );
  const medium = (chunks: React.ReactNode) => <span className="font-medium">{chunks}</span>;

  return (
    <div className="min-h-screen bg-white dark:bg-gray-800">
      <header className="bg-gray-100 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-4 py-16">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
            {t("title")}
          </h1>
          <p className="mt-4 max-w-3xl text-lg text-gray-600 dark:text-gray-300">
            {t.rich("intro", { b: (c) => <span className="font-semibold">{c}</span> })}
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4">
        {/* What it measures */}
        <section className="py-12 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {t("measures.title")}
          </h2>
          <p className="mt-3 max-w-3xl text-gray-700 dark:text-gray-200">
            {t("measures.intro")}
          </p>
          <ul className="mt-4 max-w-3xl space-y-3 text-gray-700 dark:text-gray-200">
            <li>{t.rich("measures.femoral", { b: bold, m: medium })}</li>
            <li>{t.rich("measures.tibial", { b: bold, m: medium })}</li>
          </ul>
          <p className="mt-4 max-w-3xl text-gray-700 dark:text-gray-200">
            {t("measures.note")}
          </p>
        </section>

        {/* Pipeline */}
        <section className="py-12 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {t("pipeline.title")}
          </h2>
          <p className="mt-3 max-w-3xl text-gray-700 dark:text-gray-200">
            {t("pipeline.intro")}
          </p>
          <ol className="mt-6 max-w-3xl space-y-4">
            {pipelineSteps.map((key, i) => (
              <li key={key} className="flex gap-4">
                <div className="mt-0.5 flex h-8 w-8 flex-none items-center justify-center rounded-full bg-blue-600 text-sm font-bold text-white">
                  {i + 1}
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">
                    {t(`pipeline.steps.${key}.title`)}
                  </h3>
                  <p className="mt-1 text-gray-700 dark:text-gray-200">
                    {t(`pipeline.steps.${key}.body`)}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        {/* Technology */}
        <section className="py-12 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {t("technology.title")}
          </h2>
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white">
                {t("technology.analysis.title")}
              </h3>
              <p className="mt-2 text-sm text-gray-700 dark:text-gray-200">
                {t("technology.analysis.body")}
              </p>
            </div>
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white">
                {t("technology.web.title")}
              </h3>
              <p className="mt-2 text-sm text-gray-700 dark:text-gray-200">
                {t("technology.web.body")}
              </p>
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <section className="py-12">
          <div className="rounded-lg border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 p-6">
            <h2 className="text-xl font-bold text-red-700 dark:text-red-400">
              {t("disclaimer.title")}
            </h2>
            <p className="mt-2 max-w-3xl text-red-800 dark:text-red-200">
              {t.rich("disclaimer.body", { b: (c) => <span className="font-semibold">{c}</span> })}
            </p>
          </div>
          <div className="mt-8 flex flex-wrap gap-4">
            <Link
              href="/upload"
              className="rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white shadow transition-colors hover:bg-blue-700"
            >
              {t("uploadCta")}
            </Link>
            <Link
              href="/examinations/"
              className="rounded-lg border border-gray-300 dark:border-gray-600 px-6 py-3 font-semibold text-gray-800 dark:text-gray-100 transition-colors hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              {t("viewCta")}
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
