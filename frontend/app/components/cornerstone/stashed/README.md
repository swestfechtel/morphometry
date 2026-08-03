# Stashed viewer variants

Retired-but-preserved implementations, kept as `.txt` so they are **not** compiled,
typechecked, or bundled. Restore by copying the body back into the active module.

- **`use-cornerstone-viewport.mpr.tsx.txt`** — the 3-viewport variant: axial as a
  native-slice `StackViewport` **plus** sagittal/coronal **volume MPR** reslices
  (crosshair-free) with the segmentation labelmap on the MPR panes. Stashed
  2026-07-29 when we dropped MPR in favour of a single axial stack (the MPR panes
  render black on exactly axis-aligned volumes and don't show landmarks — see the
  header comment in the active `use-cornerstone-viewport.ts`). Reintroduce this
  when isotropic data or a fixed vtk volume-mapper path makes MPR worthwhile.
