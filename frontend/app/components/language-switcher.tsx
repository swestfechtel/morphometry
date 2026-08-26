'use client';

import { useTransition } from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { locales, localeNames, type Locale } from '@/i18n/config';
import { setUserLocale } from '@/i18n/locale';

// Navbar language selector (sits next to the dark-mode toggle). Writes the chosen
// locale to the cookie via a server action, then refreshes so both server components
// (re-rendered with the new catalog) and client components (via the provider) update.
export function LanguageSwitcher() {
    const t = useTranslations('language');
    const locale = useLocale();
    const router = useRouter();
    const [isPending, startTransition] = useTransition();

    function onChange(event: React.ChangeEvent<HTMLSelectElement>) {
        const next = event.target.value as Locale;
        startTransition(async () => {
            await setUserLocale(next);
            router.refresh();
        });
    }

    return (
        <select
            aria-label={t('label')}
            value={locale}
            onChange={onChange}
            disabled={isPending}
            className="rounded border border-gray-600 bg-gray-800 px-2 py-1 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
            {locales.map((l) => (
                <option key={l} value={l}>
                    {localeNames[l]}
                </option>
            ))}
        </select>
    );
}
