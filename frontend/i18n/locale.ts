"use server";

// Server actions for reading/writing the user's chosen locale. The locale lives in a
// cookie (not the URL) so the navbar language selector can switch it without changing
// routes, and both server components and client components resolve the same value.
import { cookies } from "next/headers";
import { asLocale, type Locale } from "./config";

// next-intl's conventional cookie name; it reads this automatically in some helpers,
// and we read/write it explicitly here.
const LOCALE_COOKIE = "NEXT_LOCALE";
// One year — the choice should stick across sessions.
const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365;

/** The current locale from the cookie, narrowed to a supported one (default: en). */
export async function getUserLocale(): Promise<Locale> {
  const store = await cookies();
  return asLocale(store.get(LOCALE_COOKIE)?.value);
}

/** Persist the chosen locale in the cookie. Called by the navbar language selector. */
export async function setUserLocale(locale: Locale): Promise<void> {
  const store = await cookies();
  store.set(LOCALE_COOKIE, locale, {
    path: "/",
    maxAge: ONE_YEAR_SECONDS,
    sameSite: "lax",
  });
}
