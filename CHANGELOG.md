# Changelog

One shared semantic version (`VERSION`) covers all four images. Entries are prefixed
`nnunet:`, `torsion:`, `api:` or `frontend:`.

## [1.0.0] - 2026-11-25

- nnunet: ResEncUNetXL 5-fold ensemble for hip/knee/ankle segmentation (datasets 004, 006, 007), unchanged since the pre-versioning image.
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
