'use client';

// Builds the API volume URLs and the single auth-token source used by the
// Cornerstone NIfTI loader. The loader fetches URLs itself, so auth is attached
// two ways (whichever the deployment uses): the `X-API-Key` header via the
// loader's `beforeSend` hook (see cs-init), and an `?api_key=` query param as a
// fallback. Today auth is disabled in dev, so both are effectively no-ops.
//
// The URL paths MUST end in `.nii.gz`: the loader decides whether to gunzip the
// response solely from `new URL(url).pathname.endsWith('.gz')` (Content-Type is
// ignored). Without the suffix it parses raw gzip bytes as a NIfTI header and
// fails with "Array buffer allocation failed". The `?api_key=` query param is not
// part of `pathname`, so appending it after the suffix is safe.
import server_config from '@/app/server_config';

// Optional client-visible API key. Set NEXT_PUBLIC_API_KEY only if the API has
// auth enabled (MORPH_API_API_KEYS). Left undefined in dev.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

export function authHeaders(): Record<string, string> {
  return API_KEY ? { 'X-API-Key': API_KEY } : {};
}

function withKey(url: string): string {
  return API_KEY ? `${url}?api_key=${encodeURIComponent(API_KEY)}` : url;
}

export function imageVolumeUrl(accession: string): string {
  return withKey(`${server_config.model_api}/examinations/${accession}/volume/image.nii.gz`);
}

export function maskVolumeUrl(accession: string): string {
  return withKey(`${server_config.model_api}/examinations/${accession}/volume/mask.nii.gz`);
}

// A candidate series' preview slice PNG. Loaded via <img>, so like the volumes it
// relies on the `?api_key=` query-param fallback when auth is enabled.
export function seriesPreviewUrl(accession: string, seriesUid: string, index: number): string {
  return withKey(
    `${server_config.model_api}/examinations/${accession}/series/${encodeURIComponent(seriesUid)}/preview/${index}.png`,
  );
}
