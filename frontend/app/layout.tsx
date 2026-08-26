import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { NextIntlClientProvider } from "next-intl";
import { getLocale } from "next-intl/server";
import { AppChrome } from "@/app/components/app-chrome";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Morphometry",
  description: "Lower-limb torsion analysis — research tool, not for clinical use.",
};

// Server component: it resolves the active locale (cookie-based, see i18n/) so it can
// set <html lang> correctly on the server (no flash) and hand the message catalog to
// the client tree. NextIntlClientProvider without props inherits both the locale and
// the messages from the request config (i18n/request.ts). All interactive chrome
// (navbar, dark-mode + language toggles) lives in the client AppChrome below.
export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const locale = await getLocale();
  return (
    <html lang={locale}>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <NextIntlClientProvider>
          <AppChrome>{children}</AppChrome>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
