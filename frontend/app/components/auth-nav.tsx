'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getUsername, clearSession } from '@/app/auth';

// Navbar auth control. The username lives in a cookie the server can't see at render
// time, so it's read after mount (a useEffect) to avoid an SSR hydration mismatch.
export function AuthNav() {
    const [username, setUsername] = useState<string | null>(null);
    const router = useRouter();

    useEffect(() => { setUsername(getUsername() ?? null); }, []);

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
