import Image from "next/image";

export default function Home() {
  return (
  <div className="min-h-screen bg-white dark:bg-gray-800">
    {/* Header */}
    <header className="bg-white dark:bg-gray-700 shadow">
      <div className="container mx-auto px-4 py-6">
        <h1 className="text-4xl font-bold text-gray-800 dark:text-white">My Landing Page</h1>
      </div>
    </header>

    {/* Main Content */}
    <main className="container mx-auto px-4 shadow">
      {/* Introduction Section */}
      <section id="introduction" className="py-12">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">Introduction</h2>
        <p className="text-lg text-gray-700 dark:text-gray-200">
          Welcome to our landing page! This page provides a brief introduction to our project and its key features. Our goal is to create a user-friendly single-page experience that guides you through what we offer.
        </p>
      </section>

      {/* Quick Start Section */}
      <section id="quick-start" className="py-12 border-t">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">Quick Start</h2>
        <p className="text-lg text-gray-700 dark:text-gray-200 mb-6">
          Get started quickly by following these easy steps:
        </p>
        <ul className="list-decimal list-inside text-lg text-gray-700 dark:text-gray-200 space-y-2">
          <li>Clone the repository from GitHub.</li>
          <li>Install the required dependencies.</li>
          <li>Run the development server using <code>npm run dev</code> or <code>yarn dev</code>.</li>
          <li>Open your browser and navigate to <code>http://localhost:3000</code> to see your landing page.</li>
        </ul>
      </section>

      {/* More Information Section */}
      <section id="more-information" className="py-12 border-t">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">More Information</h2>
        <p className="text-lg text-gray-700 dark:text-gray-200">
          For more details, please refer to our documentation or contact our team. Additional resources are available on our GitHub page and community forums. We are committed to providing the best user experience and continuous improvements.
        </p>
      </section>
    </main>
  </div>
  );
}
