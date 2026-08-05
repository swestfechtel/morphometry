import Link from "next/link";

/**
 * About page.
 *
 * Static server component explaining what the tool measures, the processing
 * pipeline behind it, the technology stack, and the (important) clinical-use
 * disclaimer. Linked from the navbar (`app/layout.tsx`). Styled to match the
 * app's dark-mode-default Tailwind conventions.
 */
export default function About() {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-800">
      <header className="bg-gray-100 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-4 py-16">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
            About
          </h1>
          <p className="mt-4 max-w-3xl text-lg text-gray-600 dark:text-gray-300">
            A web front end for computing and reviewing lower-limb torsion
            measurements from MRI, built on the open-source{" "}
            <span className="font-semibold">morphometry</span> analysis library.
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4">
        {/* What it measures */}
        <section className="py-12 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            What it measures
          </h2>
          <p className="mt-3 max-w-3xl text-gray-700 dark:text-gray-200">
            The tool derives rotational alignment of the lower limb from a
            segmentation of the femur, tibia and fibula:
          </p>
          <ul className="mt-4 max-w-3xl space-y-3 text-gray-700 dark:text-gray-200">
            <li>
              <span className="font-semibold text-gray-900 dark:text-white">
                Femoral torsion
              </span>{" "}
              — the angle between the femoral neck axis and the posterior
              condylar axis, computed with both the{" "}
              <span className="font-medium">Lee</span> and{" "}
              <span className="font-medium">Murphy</span> landmark definitions
              for each side.
            </li>
            <li>
              <span className="font-semibold text-gray-900 dark:text-white">
                Tibial torsion
              </span>{" "}
              — the angle between the proximal tibial axis and the trans-malleolar
              axis at the ankle, per side.
            </li>
          </ul>
          <p className="mt-4 max-w-3xl text-gray-700 dark:text-gray-200">
            Each measurement is defined by a pair of reference lines on their
            native acquisition slices. Because both bones are highly anisotropic
            in these scans (fine in-plane, coarse through-plane), the viewer
            renders native axial slices rather than a resliced volume, and the
            landmarks are placed on the slice they were measured on.
          </p>
        </section>

        {/* Pipeline */}
        <section className="py-12 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            The pipeline
          </h2>
          <p className="mt-3 max-w-3xl text-gray-700 dark:text-gray-200">
            An upload is turned into reviewable measurements by a sequence of
            background steps:
          </p>
          <ol className="mt-6 max-w-3xl space-y-4">
            {[
              {
                title: "Ingestion",
                body: "The uploaded examination folder is scanned for DICOM series, which are grouped, cleaned of non-image files, and previewed so the correct series can be chosen.",
              },
              {
                title: "Segmentation",
                body: "A deep-learning model (nnU-Net) segments the hip, knee and ankle regions into bone labels on a GPU. Jobs are serialised through a single worker so the GPU is never oversubscribed.",
              },
              {
                title: "Torsion computation",
                body: "The morphometry library derives the anatomical landmarks and reference axes from the segmentation masks and computes the femoral and tibial torsion angles.",
              },
              {
                title: "Review & correction",
                body: "Results are presented in the interactive viewer. Landmarks can be dragged — within a slice or across slices — and the angles recompute live; corrections are saved back to the examination.",
              },
            ].map(({ title, body }, i) => (
              <li key={title} className="flex gap-4">
                <div className="mt-0.5 flex h-8 w-8 flex-none items-center justify-center rounded-full bg-blue-600 text-sm font-bold text-white">
                  {i + 1}
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">
                    {title}
                  </h3>
                  <p className="mt-1 text-gray-700 dark:text-gray-200">
                    {body}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        {/* Technology */}
        <section className="py-12 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Technology
          </h2>
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white">
                Analysis &amp; API
              </h3>
              <p className="mt-2 text-sm text-gray-700 dark:text-gray-200">
                A Python morphometry library computes the measurements from
                segmentation masks. A FastAPI service handles DICOM ingestion,
                stores examinations, and dispatches segmentation and torsion jobs
                to a Redis-backed worker running containerised models.
              </p>
            </div>
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white">
                Web interface
              </h3>
              <p className="mt-2 text-sm text-gray-700 dark:text-gray-200">
                This front end is a Next.js / React application. The examination
                viewer renders the NIfTI volumes directly in the browser with
                Cornerstone3D and draws the draggable measurement landmarks on
                top.
              </p>
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <section className="py-12">
          <div className="rounded-lg border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 p-6">
            <h2 className="text-xl font-bold text-red-700 dark:text-red-400">
              Disclaimer
            </h2>
            <p className="mt-2 max-w-3xl text-red-800 dark:text-red-200">
              This is a research and development tool. It is{" "}
              <span className="font-semibold">not a medical device</span> and is{" "}
              <span className="font-semibold">not intended for clinical use</span>,
              diagnosis, or treatment decisions. Automated measurements can be
              wrong and must always be verified by a qualified reader.
            </p>
          </div>
          <div className="mt-8 flex flex-wrap gap-4">
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
        </section>
      </main>
    </div>
  );
}
