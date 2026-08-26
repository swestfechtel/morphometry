'use client';

// Client-side session handling. The signed login token is kept in a cookie (not
// localStorage) so that server components — which fetch the API during SSR — can
// read and forward it too (see app/server-auth.ts). The cookie is JS-readable so
// client fetches and media URLs (<img>, the Cornerstone loader) can attach it.
import server_config from '@/app/server_config';

const TOKEN_COOKIE = 'morph_token';
const USER_COOKIE = 'morph_user';

// Optional machine API key (kept working alongside user login). Inlined at build.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

// Fired on the window whenever the session changes (login/logout). The root layout
// mounts once and never unmounts across client-side navigation, so components that
// display auth state (e.g. AuthNav) can't rely on a re-render to re-read the cookie —
// they subscribe to this event instead and update immediately.
export const AUTH_CHANGED_EVENT = 'morph-auth-changed';

function notifyAuthChanged(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
  }
}

function readCookie(name: string): string | undefined {
  if (typeof document === 'undefined') return undefined;
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]+)`));
  return match ? decodeURIComponent(match[1]) : undefined;
}

export function getToken(): string | undefined {
  return readCookie(TOKEN_COOKIE);
}

export function getUsername(): string | undefined {
  return readCookie(USER_COOKIE);
}

/** Persist the login session in cookies after a successful /auth/login. */
export function setSession(token: string, username: string, maxAgeSeconds: number): void {
  const secure = typeof location !== 'undefined' && location.protocol === 'https:' ? '; secure' : '';
  const attrs = `path=/; max-age=${maxAgeSeconds}; samesite=lax${secure}`;
  document.cookie = `${TOKEN_COOKIE}=${encodeURIComponent(token)}; ${attrs}`;
  document.cookie = `${USER_COOKIE}=${encodeURIComponent(username)}; ${attrs}`;
  notifyAuthChanged();
}

export function clearSession(): void {
  document.cookie = `${TOKEN_COOKIE}=; path=/; max-age=0`;
  document.cookie = `${USER_COOKIE}=; path=/; max-age=0`;
  notifyAuthChanged();
}

/** Auth headers for JSON API calls: Bearer token and/or X-API-Key when present. */
export function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  if (API_KEY) headers['X-API-Key'] = API_KEY;
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

/** Auth query string for media URLs (<img>/volume loader can't always set headers). */
export function authQuery(): string {
  const parts: string[] = [];
  const token = getToken();
  if (token) parts.push(`token=${encodeURIComponent(token)}`);
  if (API_KEY) parts.push(`api_key=${encodeURIComponent(API_KEY)}`);
  return parts.join('&');
}

/**
 * fetch() against the API with auth headers attached. On 401 it clears the session
 * and redirects to /login (so an expired/invalid token bounces the user to sign in).
 */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const resp = await fetch(server_config.model_api + path, {
    ...init,
    headers: { ...(init.headers ?? {}), ...authHeaders() },
  });
  if (resp.status === 401 && typeof window !== 'undefined') {
    clearSession();
    window.location.href = '/login';
  }
  return resp;
}
