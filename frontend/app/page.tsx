import Link from "next/link";
import { useTranslations } from "next-intl";

/**
 * Landing page.
 *
 * A static server component describing the morphometry torsion tool: what it
 * does, the upload → segmentation → torsion → review workflow, and the primary
 * calls to action (view existing examinations / upload a new one). Styled to
 * match the app's dark-mode-default Tailwind conventions (see `app/layout.tsx`).
 */
export default function Home() {
  const t = useTranslations("landing");
  const steps = ["upload", "select", "compute", "review"] as const;
  const features = ["femoral", "tibial", "interactive"] as const;

  return (
    <div className="min-h-screen bg-white dark:bg-gray-800">
      {/* Hero */}
      <header className="bg-gray-100 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-4 py-20 text-center">
          <p className="text-sm font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400">
            {t("eyebrow")}
          </p>
          <h1 className="mt-3 text-4xl md:text-5xl font-bold text-gray-900 dark:text-white">
            {t("title")}
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-gray-600 dark:text-gray-300">
            {t("subtitle")}
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
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
          <p className="mt-6 text-sm font-medium text-red-600 dark:text-red-400">
            {t("disclaimer")}
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4">
        {/* How it works */}
        <section className="py-16 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            {t("howItWorks.title")}
          </h2>
          <p className="mt-2 max-w-2xl text-gray-600 dark:text-gray-300">
            {t("howItWorks.intro")}
          </p>
          <ol className="mt-8 grid gap-6 md:grid-cols-4">
            {steps.map((key, i) => (
              <li
                key={key}
                className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 p-6"
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-600 font-bold text-white">
                  {i + 1}
                </div>
                <h3 className="mt-4 text-lg font-semibold text-gray-900 dark:text-white">
                  {t(`howItWorks.steps.${key}.title`)}
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                  {t(`howItWorks.steps.${key}.body`)}
                </p>
              </li>
            ))}
          </ol>
        </section>

        {/* Features */}
        <section className="py-16 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            {t("whatYouGet.title")}
          </h2>
          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {features.map((key) => (
              <div
                key={key}
                className="rounded-lg border border-gray-200 dark:border-gray-700 p-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {t(`whatYouGet.features.${key}.title`)}
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                  {t(`whatYouGet.features.${key}.body`)}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Call to action */}
        <section className="py-16 text-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {t("cta.title")}
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-gray-600 dark:text-gray-300">
            {t("cta.body")}
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/upload"
              className="rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white shadow transition-colors hover:bg-blue-700"
            >
              {t("cta.uploadCta")}
            </Link>
            <Link
              href="/about"
              className="rounded-lg border border-gray-300 dark:border-gray-600 px-6 py-3 font-semibold text-gray-800 dark:text-gray-100 transition-colors hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              {t("cta.learnMore")}
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
