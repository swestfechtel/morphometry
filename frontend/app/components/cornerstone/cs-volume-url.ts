'use client';

// Builds the API media URLs used by the Cornerstone NIfTI loader and <img> tags.
// These fetch the URL themselves, so auth is attached two ways (see app/auth.ts):
// the `Authorization`/`X-API-Key` header via the loader's `beforeSend` hook (see
// cs-init) — re-exported here as `authHeaders` — and a `?token=`/`?api_key=` query
// string as a fallback for requests that can't set a header (<img>).
//
// The URL paths MUST end in `.nii.gz`: the loader decides whether to gunzip the
// response solely from `new URL(url).pathname.endsWith('.gz')` (Content-Type is
// ignored). Without the suffix it parses raw gzip bytes as a NIfTI header and
// fails with "Array buffer allocation failed". The query string is not part of
// `pathname`, so appending it after the suffix is safe.
import server_config from '@/app/server_config';
import { authHeaders, authQuery } from '@/app/auth';

export { authHeaders };

function withKey(url: string): string {
  const query = authQuery();
  return query ? `${url}?${query}` : url;
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
