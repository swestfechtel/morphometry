import type { NextConfig } from "next";

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

export default nextConfig;
