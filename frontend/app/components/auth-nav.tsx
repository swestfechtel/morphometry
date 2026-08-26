'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getUsername, clearSession, AUTH_CHANGED_EVENT } from '@/app/auth';

// Navbar auth control. The username lives in a cookie the server can't see at render
// time, so it's read after mount (a useEffect) to avoid an SSR hydration mismatch.
// This component sits in the root layout, which never unmounts across client-side
// navigation, so it also subscribes to AUTH_CHANGED_EVENT to re-read the cookie the
// moment login/logout happens (a mount-only read would show a stale "Sign in").
export function AuthNav() {
    const [username, setUsername] = useState<string | null>(null);
    const router = useRouter();

    useEffect(() => {
        const sync = () => setUsername(getUsername() ?? null);
        sync();
        window.addEventListener(AUTH_CHANGED_EVENT, sync);
        return () => window.removeEventListener(AUTH_CHANGED_EVENT, sync);
    }, []);

    const logout = () => {
        clearSession();
        router.push('/login');
    };

    if (!username) {
        return <Link href="/login" className="text-white hover:text-gray-300">Sign in</Link>;
    }
    return (
        <div className="flex items-center gap-3 text-white">
            <span className="text-sm" title="Signed in">{username}</span>
            <button onClick={logout} className="text-sm underline hover:no-underline">Sign out</button>
        </div>
    );
}
