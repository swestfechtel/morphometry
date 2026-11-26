# Changelog

One shared semantic version (`VERSION`) covers all four images. Entries are prefixed
`nnunet:`, `torsion:`, `api:` or `frontend:`.

## [1.0.0] - 2026-11-25

- nnunet: ResEncUNetXL 5-fold ensemble for hip/knee/ankle segmentation (datasets 004, 006, 007), same checkpoints and the exact package set of the pre-versioning image `sha256:70229af9…` (now pinned in `api/docker/nnunet_torsion/requirements.txt`); rebuilt slim (79.5 GB → 25.6 GB) by dropping the duplicate checkpoint layer, the unused `checkpoint_best.pth` files and the pip cache. Checked on stored examination 0003590329 (all three regions): old vs slim masks differ by 0–2 voxels per region (of ~16 M), within the pre-versioning image's own run-to-run variation (1–2 voxels; nnUNet on GPU is not bit-deterministic); all 24 model files byte-identical to `checkpoints.sha256`.
- torsion: morphometry library as of this release (mirror-equivariant torsion/CCD, QC gates, 3-D NaKo CCD), packaged with pinned dependencies.
- api: FastAPI service with DICOM/Orthanc ingest, two-phase torsion upload, RQ worker dispatch and username/password plus API-key auth.
- frontend: Next.js 15 web UI for the API.

### Pre-versioning history

Images built before versions existed, kept locally as `swestfechtel/torsion:pre-<sha>`:

- torsion: `pre-1bd8980`
- torsion: `pre-2d1fc91`
- torsion: `pre-51ce760`
- torsion: `pre-6d7be91`
- torsion: `pre-7ccaf44`
- torsion: `pre-822181b`
- torsion: `pre-pin-2026-06-17`
- nnunet: superseded Docker Hub manifest `sha256:69b6e0e0400daf13221c8ab6b24f472b827c882b6cc0f49ef3d033a2b0b34cff`
- nnunet: local image `sha256:70229af92e5a589e7f3ff615c539f42a09a748535b72bfb917a6d95ea67a7f9c` (79.5 GB), the image the API ran before 1.0.0; not archived on Docker Hub, superseded by the slim 1.0.0 build (same software and checkpoints)
