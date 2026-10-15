# Knee cartilage subregions: aligning with Wirth & Eckstein 2008

Reference: W. Wirth and F. Eckstein, "A Technique for Regional Analysis of Femorotibial
Cartilage Thickness Based on Quantitative Magnetic Resonance Imaging", *IEEE TMI* 27(6),
737–744, 2008 (`Wirth_Eckstein_2008.pdf`, repo root, untracked).

Status (2026-10-15): analysis, steps 0, 1 and 2 done. Step 1 (shared bone-surface geometry)
is `morphometry/cartilage/surface.py`; step 2 (tibia) is `morphometry/cartilage/tibia_regions.py`,
wired into `Tibia` (`morphometry/cartilage/knee.py`). The femur (step 3) and the area-based
thickness (step 4) have not started.

## 1. Where the current code departs from the paper

### General
| | Paper | Code |
|---|---|---|
| Surface the regions are defined on | Total area of subchondral bone (tAB) as a triangle mesh, including denuded bone. Regions are defined there and transferred to the cartilage surface (AC) by distance. | Cartilage voxels only; the bone labels are never used, denuded areas are invisible. |
| What the 20% / 33% shares measure | Surface **area** | Voxel **count** (cartilage volume) |
| Units | mm | Voxel indices; anisotropic voxels (0.75 × 0.31 × 0.31 mm on `dfg_knees`) ignored |
| Frame | The bone's own principal axes | Image axes, so not invariant to the knee's position in the scanner |
| Boundaries | Triangles split exactly along the cut | Whole voxels assigned; fragments removed afterwards |

### Tibia
| | Paper | Code |
|---|---|---|
| Plate frame | Centre of gravity S and inertia-tensor eigenvectors b1, b2, b3 of the plate's tAB; b3 = plate normal | Mean position only (one-cluster KMeans) |
| Central region (cMT/cLT) | Elliptical cylinder x²/λ1² + y²/λ2² = 1 along b3 in the principal frame; axes reflect the plate's A-P vs M-L size; scaled to exactly 20% of tAB **area** | A 3D sphere in voxel-index space holding 20% of cartilage **voxels** (since 60425aa; before, a constant 10 voxels, 6–9%) |
| Peripheral split | Two planes through S containing b3, normals `n = e_z × R(±45°) · tc`, tc = line joining the two plate centres | The diagonals of the plate's axis-aligned bounding box in the image's axial plane: through the box centre, not S; angle set by the box's aspect ratio |
| Naming | i/e from distance to the other plate; medial/lateral and a/p from distance to a fibula landmark | Hard-coded by image quadrant |
| Plate names | Medial (MT) / lateral (LT) | Patient left / right: the code's `LT` is *left* tibia, the paper's `LT` is *lateral* tibia |

### Femur
| | Paper | Code |
|---|---|---|
| A-P extent of the weight-bearing condyle (cMF/cLF) | Bony landmarks, independent of the tibia (refs [9]–[12]). In the Chondrometrics convention: from the trochlear notch (a line parallel to the femoral shaft through its most posterior point) to 60% of the distance to the posterior end of the condyles (some later studies use 75%) | The A-P extent of the tibial central region, i.e. tibia- and flexion-dependent; `_restrict_to_contact_cluster` exists only to patch the flexion effect |
| Internal / central / external | Per coronal slice (triangle strip), points at ⅓ and ⅔ of the **arc length** along the strip's anterior and posterior edge sequences; cut planes with normal d × e through them; ≈33% of tAB **area** each; i/e from distance to the other condyle | Two straight cuts at fixed image-x positions, adjusted in 1-voxel steps until each third holds 33 ± 2% of the **voxels** (max 30 steps) |
| Trochlea / posterior condyle (`alf/arf/plf/prf`) | Not in the paper | An in-house extension |

### Thickness (defined in the paper too)
| | Paper | Code |
|---|---|---|
| Mean thickness | ThC.cAB.aMe = mean of (AC → cAB) and (cAB → AC) nearest Euclidean distances, area-weighted. ThC.tAB.aMe counts denuded bone as 0 mm | Normal ray-cast (`mesh`) or one-direction nearest neighbour (`knn`) from per-(x, y)-column voxel surfaces, separately per subregion |
| Other statistics | aMax, aMav (top 1%), SD, aMin, aMiv (bottom 1%) | Per-point map only |

## 2. Step 0: inputs (resolved / open)

- **Bone labels (resolved).** `dfg_knees` (96 segmentations): 1 = femur bone, 2 = tibia
  bone, 3 = femoral cartilage, 4 = tibial cartilage, 5 = patella (anterior to the femur).
  Label 6 is a stray label (18–908 voxels, in 6 of 96 files). A real tAB is possible.
- **No fibula label (resolved).** Medial/lateral cannot come from a fibula landmark as in
  the paper. Front/back comes from the image's A-P axis (reliable after LPI
  reorientation). Medial/lateral needs the knee's laterality (**decided**): a
  `laterality='left' | 'right' | None` argument, plus support for a per-subject table
  (e.g. CSV `subject,laterality`) in the scripts. With `None` the regions keep
  patient-left/right names (`LT`/`RT`), so nothing downstream breaks; with a value they
  are named medial/lateral (`MT`/`LT`, `cMF`/`cLF`).
- **cMF/cLF A-P limits (decided).** Trochlear notch → 60% of the distance
  to the posterior condylar end (default), with the fraction a parameter so 75% is
  available. The trochlear-notch landmark is defined parallel to the femoral shaft, which
  on a knee FOV has to be approximated from the distal femur.
- **No legacy switch (decided).** The new subdivision replaces the old one outright; old
  numbers are reproducible only from git history (last old-scheme commit: the one before
  the rewrite lands). Rows from the two schemes must not be pooled.

## 3. Implementation plan

**1. Shared geometry (`morphometry/cartilage/`).**
- Bone surface meshes in mm (marching cubes on labels 1/2, smoothed, decimated).
- cAB = bone triangles with cartilage on them (distance to the cartilage mask below a
  tolerance). tAB = cAB plus enclosed denuded holes (hole filling on the bone surface,
  bounded so it never reaches into non-articular bone).
- All area bookkeeping by triangle area; a cut-mesh helper that splits triangles along a
  plane or an implicit surface.

*Implemented* (`morphometry/cartilage/surface.py`, tests `tests/unit/test_cartilage_surface.py`):
`bone_surface_mesh`, `cartilage_coverage`, `total_subchondral_area` (+ `uncovered_components`),
`triangle_areas` / `triangle_centroids`, `split_mesh_by_scalar` / `split_mesh_by_plane`, and the
geometry helpers `index_to_world_affine`, `label_array_xyz`, `voxel_spacing`,
`indices_to_world`. Deviations from the plan above:
- No decimation: a knee bone mesh is 240-360k triangles and the whole chain (mesh, cAB, tAB)
  takes ~1 s per bone, so there is no need to trade area accuracy for size.
- FOV cuts: the shafts are extended past the array border by replicated slices and the
  smoothed mesh is then cut exactly at the border, leaving it open there (no cap). The hole
  rule uses that open boundary, plus a size bound (`max_hole_fraction` 0.5 of cAB) for bones
  the FOV does not cut (the femur of `K5_90_belastet` is closed).
- The splitter is numpy, not VTK `clip_scalar`: VTK interpolates the cut in float32 (area
  error ~1e-8 relative); the numpy version conserves area to double precision.
- Coverage tolerance 0.6 x voxel diagonal (0.52 mm), an empirical knee point (see the
  `cartilage_coverage` docstring); the distance rule gives ~1-1.6% more tAB than a normal
  probe would. On the five knees checked (K1_0_unbelastet, K1_0_belastet, K5_90_belastet,
  K3_60_unbelastet, K8_30_belastet) every enclosed hole was a coverage pinhole (<= 0.4 mm²,
  37-364 per bone, <= 20 mm² in total); other knees have more (K10_90_unbelastet femur 552
  uncovered components, K12_60_belastet femur 688, one of them 4 mm²).
- World coordinates are RAS for both image backends (SimpleITK's LPS is converted).
  Tibial tAB 1876-2370 mm² (both plates), femoral tAB (whole femoral cartilage incl.
  trochlea) 4472-6051 mm².
- Small covered islands away from the articular surface are kept (0.1 mm² specks on
  K1_0_belastet); assigning patches to plates/condyles is left to steps 2-3.

**2. Tibia.**
- Split the tAB into two plates (connected components / the gap at the intercondylar
  eminence).
- Hole filling is not plate-aware (step 1's `total_subchondral_area`): if the medial and
  lateral cartilage touched anteriorly and posteriorly, the intercondylar area would be
  enclosed and filled into tAB. Fill holes per plate (split cAB first, then fill each plate's
  holes) and/or cap the absolute hole size.
- Per plate: area-weighted centre S and principal axes. Central region: normalised
  elliptical radius ρ = √((x/a)² + (y/b)²) in the plate plane, with a : b from the plate's
  in-plane principal spreads; the cutoff ρ* is the area-weighted 20% quantile (exact, no
  iteration), boundary triangles cut at ρ = ρ*.
- Peripheral split: two planes through S containing b3, at ±45° to tc projected into the
  plate plane. Internal = the wedge facing the other plate, external opposite,
  anterior/posterior from the image A-P axis.
- Central share is a parameter (the paper's Table IV uses 10–50%).

*Implemented* (`morphometry/cartilage/tibia_regions.py::subdivide_tibia`, wired into `Tibia`;
tests `tests/unit/test_tibia_regions.py`). Decisions and deviations:
- Plates: the two largest edge-connected cAB components (`find_plates`), the smaller at least
  0.2 x the larger; otherwise a fallback area-weighted 2-means split of the largest component
  (cuts any cartilage bridge at the 2-means boundary). On the 96 `dfg_knees` the fallback never
  fired (smallest plate ratio 0.61). Patient left = the plate with the smaller RAS world x.
  Other covered islands are treated as uncovered: enclosed by a plate they come back through
  its hole fill, otherwise they are dropped (median 0.15 mm², max 0.66 mm² per knee on the 94
  usable knees; 15 mm² on the broken K7_90_belastet).
- Hole filling per plate: `total_subchondral_area` with only that plate's cAB as covered. The
  other plate is then uncovered bone joined to the open non-articular remainder, so the
  intercondylar area is never enclosed; no absolute size cap was needed. Filled area per plate:
  median 0.4 / 2.3 mm² (patient left / right), max 26 mm².
- Moments: exact area-weighted triangle second moments (`area_moments`), not the paper's
  vertex moments (which weight by vertex density). b3 = least-spread axis, oriented superior;
  b1 towards the other plate; b2 = b3 x b1. (Inertia-tensor and covariance eigenvectors
  coincide; only the ranking convention differs.)
- Ellipse aspect: semi-axes in the ratio of the in-plane standard deviations
  sqrt(var1) : sqrt(var2) -- exact diameter ratio for a uniform elliptical plate. The paper does
  not say how its lambda1/lambda2 come from the tensor; squaring (variances) or swapping
  (moments of inertia about b1/b2) are the other readings and were not used.
- The 20% is not an area-weighted quantile (rho is interpolated linearly along edges by the
  splitter); it is found by bisection on the actually cut mesh, to 1e-5 of the plate tAB (the
  paper: ~1e-3).
- Anterior/posterior from world +y projected into the plate plane (no fibula); raises if the
  perpendicular to tc is more than 60 deg off A-P (min |cos| on dfg_knees 0.81, i.e. <= 36 deg).
- Names: keys stay patient-side (`clt` ... `prt`); laterality only adds the medial/lateral
  mapping (`anatomical_tibia_names`, `rename_tibia_regions`, `Tibia.anatomical_names`,
  script `--laterality-table`), and `split_tibia_thickness_by_compartment` refuses `*MT` keys.
- Transfer to the cartilage voxels by nearest tAB triangle *centroid* (world mm). The voxel
  maps and the `(x, y)`-keyed thickness dicts are unchanged (step 4 replaces them). QC:
  `Tibia.transfer_qc` holds the voxel-to-tAB distance statistics; more than 1% of the voxels
  beyond 6 mm logs a WARNING (clean knees: max 4.8 mm; K7_90_belastet: 49%, max 45 mm, the
  only one flagged).
- Real data (94 usable knees; K11_90_belastet fails -- broken segmentation, 1 mm voxels with
  the tibia label filling the volume; K7_90_belastet is excluded -- its tibial cartilage is a
  quarter of the unloaded scan's): shares of plate tAB, patient left / right, mean (SD):
  c 20.0/20.0, e 16.9 (0.8)/15.9 (0.7), i 15.6 (1.4)/14.1 (1.2), a 23.8 (1.1)/25.0 (0.9),
  p 23.8 (1.0)/25.0 (0.9) (paper Table I, MT/LT: e 16.0/17.7, i 15.6/17.5, a 24.2/22.6,
  p 24.1/22.2). Loaded vs unloaded (46 pairs): median |delta share| 0.14 pp, max 1.3; within a
  subject/load across flexion angles (24 series): median SD 0.21 pp.

Open issues for steps 3-4: the femoral CWBZ still takes its A-P window from cLT/cRT's voxel
extent (so it moved with the new central region); the thickness is still the voxel-column
method per region; the laterality of the `dfg_knees` knees is not recorded anywhere yet (the
patient-right plate is the larger in 79/94, which would make them mostly left knees if the
medial plate is the larger one -- to be confirmed, not assumed).

K7/K8/K11 orientation (flagged, not fixed): three of the 96 `dfg_knees` scans do not land
in LPI after `transform_coordinate_system` -- K7_90_belastet (raw IAR -> LSP), K8_90_belastet
and K11_90_belastet (raw SPR -> LIA); the other 93 (raw PIR/LIP) do. A tilt of the slice
axes against the scanner frame is not the problem: it comes from the knee flexion and is
harmless (index axis 2 vs world S-I, median 4.6/11.1/24.1/38.7 deg at 0/30/60/90 deg flexion;
9 healthy scans exceed 40 deg, K10_90_belastet 57.5 deg on one axis, all land in LPI and
measure fine). What is wrong in the flagged scans is that the *anatomy* is rotated in the
index frame: the tibial plate normal is within 17.0 deg of index axis 2 in all 93 LPI scans
but 25.7 deg (K11), 75.5 deg (K7) and 83.0 deg (K8) off it, and the femoral-minus-tibial
cartilage centroid along axis 2 is -28 to -12.2 mm in the 93 vs -2.6 (K7), +1.6 (K8) and
+35.3 mm (K11). The non-LPI label is how that shows up: with two voxel axes near a tie
between two anatomical axes (K7's first axis: 0.706 S-I vs 0.703 A-P), nibabel's greedy,
voxel-axis-ordered `io_orientation` resolves the tie differently once the axes are
permuted. The reorientation is left unchanged (the user attributes these scans to a scanner
misconfiguration); `Image.orientation_ok` / `achieved_axcodes` record it, and both knee
scripts write `non_lpi_orientation:<codes>` (plus `tibia_transfer_far:<fraction>`, K7: 0.494)
into a `qc_flags` column. K11_90_belastet is a separate defect that the orientation flag
catches only by coincidence: 1 mm isotropic with 344 slices and a 512 mm FOV (every other
scan is 0.31 x 0.31 x 0.75 mm, 172 slices), its tibial cartilage is 9.4k voxels scattered
over a 258 x 318 x 213 mm box, and the tibia subdivision fails ('plates overlap'). Until
resolved, results of these three scans (patient sides, superior/inferior surfaces,
subregions) are not trustworthy.

**3. Femur.**
- A femoral frame from the bone: distal shaft axis (as far as the FOV allows) and the
  posterior condylar line, so flexed knees need no contact-cluster patch.
- Landmarks: trochlear notch (most posterior point of the trochlear groove, on a line
  parallel to the shaft); posterior end of each condyle. cMF/cLF = notch → 60% (param).
- Thirds: slice cMF/cLF's tAB with coronal planes in the femoral frame, cut each slice's
  contour at ⅓ and ⅔ of its arc length, connect the cut points between neighbouring slices
  into cut surfaces (the paper's V1, V2), split triangles. Internal = the third nearer the
  other condyle.
- Keep the trochlea and posterior condyle as a documented extension (trochlea = anterior
  of the notch; posterior = beyond the 60% mark).

**4. Transfer and thickness.**
- Each AC triangle takes the region of its nearest tAB triangle.
- ThC.cAB.aMe (both directions, averaged, area-weighted), ThC.tAB.aMe (denuded = 0),
  aMax, aMav, SD, aMin, aMiv per region.
- Keep the `{region: {(x, y): mm}}` maps (or an adapter): `find_thinnest_area`,
  `split_tibia_thickness_by_compartment` (`*LT` / `*RT` suffixes), the plotting code and
  `scripts/compute_knee_cartilage_thickness.py` / `analyse_thinnest_point_flexion.py`
  depend on them. Renaming to medial/lateral must not silently break the left/right split.

**5. Tests.**
- Phantoms: a flat elliptical plate gives a central region of exactly 20% of the area;
  peripheral wedges are exactly 90° about S; a cylindrical condyle gives arc-length thirds
  of 33.3%.
- Invariance: rotate the volume (e.g. 15° axial), resample to other voxel sizes, mirror
  left-right: the region shares must not change (mirror: left and right swap exactly).
- Real data (`dfg_knees`): tables like the paper's Tables I/II (cT ≈ 20.0%, thirds ≈ 33.3%,
  peripheral sizes), old-vs-new thickness per region, a knee-cartilage golden.
