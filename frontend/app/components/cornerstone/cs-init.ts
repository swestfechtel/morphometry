'use client';

// Idempotent Cornerstone3D initialisation. Safe to call from every viewer mount;
// the underlying init runs exactly once. Client-only — never import from a server
// component (it touches the DOM / web workers).
import { init as coreInit, imageLoader } from '@cornerstonejs/core';
import { init as toolsInit } from '@cornerstonejs/tools';
import {
  cornerstoneNiftiImageLoader,
  init as niftiInit,
} from '@cornerstonejs/nifti-volume-loader';

import { authHeaders } from './cs-volume-url';

let initPromise: Promise<void> | null = null;

export function ensureCornerstoneInit(): Promise<void> {
  if (!initPromise) {
    initPromise = (async () => {
      await coreInit();
      await toolsInit();
      // Register the 'nifti:' image loader so volumeLoader can stream .nii.gz.
      imageLoader.registerImageLoader('nifti', cornerstoneNiftiImageLoader);
      // Attach the API key to the loader's volume fetches (no-op when auth is off).
      niftiInit({
        beforeSend: (_xhr, defaultHeaders) => ({ ...defaultHeaders, ...authHeaders() }),
      });
    })();
  }
  return initPromise;
}
