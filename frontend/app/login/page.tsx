'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import server_config from '@/app/server_config';
import { setSession } from '@/app/auth';

export default function LoginPage() {
    const router = useRouter();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();
        setError(null);
        setBusy(true);
        try {
            const resp = await fetch(server_config.model_api + '/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            if (resp.ok) {
                const data = await resp.json();
                setSession(data.access_token, data.username, data.expires_in);
                const next = new URLSearchParams(window.location.search).get('next');
                router.push(next || '/examinations');
            } else if (resp.status === 401) {
                setError('Invalid username or password.');
            } else {
                setError(`Login failed (HTTP ${resp.status}).`);
            }
        } catch {
            setError('Network error — is the API reachable?');
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
            <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 p-8 rounded shadow-md w-full max-w-sm">
                <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-gray-100">Sign in</h2>
                <label className="block mb-1 text-sm text-gray-700 dark:text-gray-300" htmlFor="username">Username</label>
                <input
                    id="username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    autoComplete="username"
                    className="mb-4 w-full rounded border border-gray-300 dark:border-gray-700 p-2 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
                />
                <label className="block mb-1 text-sm text-gray-700 dark:text-gray-300" htmlFor="password">Password</label>
                <input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="current-password"
                    className="mb-4 w-full rounded border border-gray-300 dark:border-gray-700 p-2 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
                />
                {error && <p className="mb-3 text-sm text-red-500">{error}</p>}
                <button
                    type="submit"
                    disabled={busy || !username || !password}
                    className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 px-4 rounded transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {busy ? 'Signing in…' : 'Sign in'}
                </button>
            </form>
        </div>
    );
}
