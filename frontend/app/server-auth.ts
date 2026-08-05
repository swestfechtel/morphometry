import { cookies } from 'next/headers';

// Server-side counterpart of app/auth.ts: server components fetch the API during
// SSR, so they read the login token from the request cookies and forward it as a
// Bearer header (plus the optional machine API key).
export async function serverAuthHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {};
  const apiKey = process.env.NEXT_PUBLIC_API_KEY;
  if (apiKey) headers['X-API-Key'] = apiKey;
  const token = (await cookies()).get('morph_token')?.value;
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}
