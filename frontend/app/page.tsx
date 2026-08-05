import Link from "next/link";

/**
 * Landing page.
 *
 * A static server component describing the morphometry torsion tool: what it
 * does, the upload → segmentation → torsion → review workflow, and the primary
 * calls to action (view existing examinations / upload a new one). Styled to
 * match the app's dark-mode-default Tailwind conventions (see `app/layout.tsx`).
 */
export default function Home() {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-800">
      {/* Hero */}
      <header className="bg-gray-100 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-4 py-20 text-center">
          <p className="text-sm font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400">
            Lower-limb morphometry
          </p>
          <h1 className="mt-3 text-4xl md:text-5xl font-bold text-gray-900 dark:text-white">
            Automated femoral &amp; tibial torsion
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-gray-600 dark:text-gray-300">
            Upload a lower-limb MRI examination and get segmentation, torsion
            angles, and editable measurement landmarks — computed automatically
            and reviewable in the browser.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/upload"
              className="rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white shadow transition-colors hover:bg-blue-700"
            >
              Upload an examination
            </Link>
            <Link
              href="/examinations/"
              className="rounded-lg border border-gray-300 dark:border-gray-600 px-6 py-3 font-semibold text-gray-800 dark:text-gray-100 transition-colors hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              View examinations
            </Link>
          </div>
          <p className="mt-6 text-sm font-medium text-red-600 dark:text-red-400">
            Research tool — not intended for clinical use.
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4">
        {/* How it works */}
        <section className="py-16 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            How it works
          </h2>
          <p className="mt-2 max-w-2xl text-gray-600 dark:text-gray-300">
            From a raw DICOM export to reviewed torsion angles in a few steps.
          </p>
          <ol className="mt-8 grid gap-6 md:grid-cols-4">
            {[
              {
                step: "1",
                title: "Upload",
                body: "Select the whole examination folder. The DICOM series are enumerated and previewed — no manual sorting.",
              },
              {
                step: "2",
                title: "Select series",
                body: "Pick the whole-leg series (auto-split into hip, knee and ankle) or assign three separate region series.",
              },
              {
                step: "3",
                title: "Compute",
                body: "A GPU pipeline segments the bones and computes femoral and tibial torsion — queued and run in the background.",
              },
              {
                step: "4",
                title: "Review",
                body: "Inspect the volume, drag the measurement landmarks, and watch the angles update live before saving.",
              },
            ].map(({ step, title, body }) => (
              <li
                key={step}
                className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 p-6"
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-600 font-bold text-white">
                  {step}
                </div>
                <h3 className="mt-4 text-lg font-semibold text-gray-900 dark:text-white">
                  {title}
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                  {body}
                </p>
              </li>
            ))}
          </ol>
        </section>

        {/* Features */}
        <section className="py-16 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            What you get
          </h2>
          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {[
              {
                title: "Femoral torsion",
                body: "Both the Lee and Murphy landmark methods, per side, so you can compare definitions on the same scan.",
              },
              {
                title: "Tibial torsion",
                body: "Proximal and distal reference axes measured on their native slices, with the signed angle shown per line.",
              },
              {
                title: "Interactive review",
                body: "Reference-line landmarks are draggable — including across slices — and every edit is saved back to the examination.",
              },
            ].map(({ title, body }) => (
              <div
                key={title}
                className="rounded-lg border border-gray-200 dark:border-gray-700 p-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {title}
                </h3>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                  {body}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Call to action */}
        <section className="py-16 text-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Ready to start?
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-gray-600 dark:text-gray-300">
            Upload an examination to run the pipeline, or read more about the
            measurements and how they are computed.
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/upload"
              className="rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white shadow transition-colors hover:bg-blue-700"
            >
              Upload an examination
            </Link>
            <Link
              href="/about"
              className="rounded-lg border border-gray-300 dark:border-gray-600 px-6 py-3 font-semibold text-gray-800 dark:text-gray-100 transition-colors hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              Learn more
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
