import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

// next-intl without i18n routing: the locale comes from a cookie (see i18n/locale.ts),
// not the URL. The plugin wires up ./i18n/request.ts as the request-scoped config.
const withNextIntl = createNextIntlPlugin("./i18n/request.ts");

const nextConfig: NextConfig = {
  // Cornerstone3D ships ESM workers/WASM that must be transpiled by the bundler.
  transpilePackages: [
    "@cornerstonejs/core",
    "@cornerstonejs/tools",
    "@cornerstonejs/nifti-volume-loader",
  ],
  webpack: (config) => {
    // Let webpack emit Cornerstone's .wasm assets, and stub node-only modules
    // that some Cornerstone dependencies reference in unused code paths.
    config.module.rules.push({ test: /\.wasm$/, type: "asset/resource" });
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false };
    return config;
  },
};

export default withNextIntl(nextConfig);
