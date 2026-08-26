// Locale configuration for the app's English/German i18n (next-intl, cookie-based
// — no URL locale routing). The language is chosen in the navbar and persisted in a
// cookie (see i18n/locale.ts); these constants are the single source of truth for
// which locales exist and which is the default.

export const locales = ["en", "de"] as const;

export type Locale = (typeof locales)[number];

export const defaultLocale: Locale = "en";

// Human-readable names shown in the navbar language selector (each in its own tongue).
export const localeNames: Record<Locale, string> = {
  en: "English",
  de: "Deutsch",
};

/** Narrow an arbitrary string to a supported Locale, falling back to the default. */
export function asLocale(value: string | undefined | null): Locale {
  return locales.includes(value as Locale) ? (value as Locale) : defaultLocale;
}
