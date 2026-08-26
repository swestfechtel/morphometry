// API base URL used by all fetches.
//
// It is resolved *per access* (via a getter) because the correct value differs
// between server-side rendering and the browser:
//
//   * Server components fetch during SSR from the Node process on this machine, so
//     they should hit the API on loopback (`http://localhost:8000`) — always
//     reachable, and browser address-space rules don't apply server-side.
//   * Client code (login, apiFetch, the Cornerstone volume/preview URLs) runs in
//     the browser, where the request must stay within the page's own address
//     space. Chrome's Private Network Access policy forbids a page served from a
//     LAN/public IP (e.g. http://134.130.11.71:3000) from fetching a *more-private*
//     `localhost` address, so such a page must call the API at its OWN hostname
//     (http://134.130.11.71:8000). A page on http://localhost:3000 likewise calls
//     http://localhost:8000. Neither then crosses into a more-private space.
//
// Resolution order:
//   1. NEXT_PUBLIC_MODEL_API — explicit override (inlined at build time). Set this
//      only when the API lives on a different host than the one serving the UI;
//      leave it unset to get the same-hostname derivation below (recommended for
//      the localhost + LAN-IP dual-access setup).
//   2. In the browser: same protocol + hostname as the current page, on the API
//      port (NEXT_PUBLIC_MODEL_API_PORT, default 8000).
//   3. On the server (or if `window` is somehow unavailable): http://localhost:<port>.
//
// NEXT_PUBLIC_* values are inlined at build time, so rebuild after changing them.

const API_PORT = process.env.NEXT_PUBLIC_MODEL_API_PORT ?? "8000";

function resolveApiBase(): string {
  const override = process.env.NEXT_PUBLIC_MODEL_API;
  if (override) return override;
  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:${API_PORT}`;
  }
  return `http://localhost:${API_PORT}`;
}

const server_config = {
  get model_api(): string {
    return resolveApiBase();
  },
};

export default server_config;
