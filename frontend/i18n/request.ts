// Request-scoped next-intl configuration (wired up in next.config.ts via
// createNextIntlPlugin). On every request it resolves the active locale from the
// cookie and loads that locale's message catalog, which both server components
// (getTranslations) and the client provider (NextIntlClientProvider) consume.
import { getRequestConfig } from "next-intl/server";
import { getUserLocale } from "./locale";

export default getRequestConfig(async () => {
  const locale = await getUserLocale();
  return {
    locale,
    messages: (await import(`../messages/${locale}.json`)).default,
  };
});
