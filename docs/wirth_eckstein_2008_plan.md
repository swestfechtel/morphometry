# Knee cartilage subregions: aligning with Wirth & Eckstein 2008

Reference: W. Wirth and F. Eckstein, "A Technique for Regional Analysis of Femorotibial
Cartilage Thickness Based on Quantitative Magnetic Resonance Imaging", *IEEE TMI* 27(6),
737–744, 2008 (`Wirth_Eckstein_2008.pdf`, repo root, untracked).

Status (2026-10-19): analysis and steps 0-5 done. Step 1 (shared bone-surface geometry) is
`morphometry/cartilage/surface.py`; step 2 (tibia) is `morphometry/cartilage/tibia_regions.py`
and step 3 (femur) `morphometry/cartilage/femur_regions.py`; step 4 (the area-weighted surface
thickness and the shared in-plane frame) is `morphometry/cartilage/thickness.py`. All four are
wired into `Tibia` / `Femur` (`morphometry/cartilage/knee.py`). The voxel-column thickness
("mesh" / "knn") is gone; `method=` rejects every value. Step 5 (tests) is the
`needs_dfg_knees` fixture, the 14 knee-cartilage goldens in
`tests/characterization/test_knee_cartilage.py` and the cohort report
`scripts/report_knee_cartilage_cohort.py`, which re-derives every cohort number quoted below.

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
| A-P extent of the weight-bearing condyle (cMF/cLF) | Bony landmarks, independent of the tibia (refs [9]–[12]). In the Chondrometrics convention: from the trochlear notch (a line parallel to the femoral shaft through its most posterior point) to 60% of the distance to the posterior end of the condyles (some later studies use 75%) | *(before step 3)* The A-P extent of the tibial central region, i.e. tibia- and flexion-dependent; `_restrict_to_contact_cluster` existed only to patch the flexion effect |
| Internal / central / external | Per coronal slice (triangle strip), points at ⅓ and ⅔ of the **arc length** along the strip's anterior and posterior edge sequences; cut planes with normal d × e through them; ≈33% of tAB **area** each; i/e from distance to the other condyle | *(before step 3)* Two straight cuts at fixed image-x positions, adjusted in 1-voxel steps until each third held 33 ± 2% of the **voxels** (max 30 steps) |
| Trochlea / posterior condyle (`alf/arf/plf/prf`) | Not in the paper | An in-house extension |

### Thickness (defined in the paper too)
| | Paper | Code |
|---|---|---|
| Mean thickness | ThC.cAB.aMe = mean of (AC → cAB) and (cAB → AC) nearest Euclidean distances, area-weighted. ThC.tAB.aMe counts denuded bone as 0 mm | *(before step 4)* Normal ray-cast (`mesh`) or one-direction nearest neighbour (`knn`) from per-(x, y)-column voxel surfaces, separately per subregion |
| Other statistics | aMax, aMav (top 1%), SD, aMin, aMiv (bottom 1%) | *(before step 4)* Per-point map only |

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
  trochlea) 4472-6051 mm² — the knees step 1 was developed on. Step 5's cohort run
  (`scripts/report_knee_cartilage_cohort.py`, all 93 clean knees) gives wider ranges:
  tibial tAB 2179 ± 307 mm² (1802-2801), femoral 5225 ± 668 (4266-6445). Note the tibial
  figure is not the whole tibial tAB but the two plates' (step 2 fills holes per plate and
  drops covered islands outside both plates).
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
  subject/load across flexion angles (24 series): median SD 0.21 pp. All of these reproduce
  from `scripts/report_knee_cartilage_cohort.py` (step 5) to the decimal shown, on its default
  93-scan clean set as well as on this 94-scan one
  (`--include-flagged --exclude K7_90_belastet K11_90_belastet`).

Open issues after step 4, in order of importance:

1. **The central femoral thirds read ~0.5 mm thicker than the literature.** `cclf` / `ccrf`
   come out at 3.15 / 3.05 mm (SD 0.42 / 0.50) over the 93 clean knees, where published
   healthy central weight-bearing femoral cartilage is ~2.3-2.7 mm; every other subregion sits
   inside the usual ~1.3-2.5 mm, and the tibial central regions (2.30 / 2.26 mm) are right.
   It is **not** the measurement: cross-checked against a completely independent quantity --
   the volume of the cartilage voxels assigned to a region divided by its tAB area -- `aMe`
   agrees to 0.1-0.5 mm on every subregion of three spot-checked knees, with the volume ratio
   the higher of the two, which is what a convex surface gives. So the central femoral
   cartilage really is that thick **in these label maps** (MONAI predictions, not reader
   segmentations); whether it is in the knees is another question.
2. **cAB reaches past the cartilage edge, which is what makes the peripheral minima small.**
   `cartilage_coverage` is a distance test with a 0.52 mm tolerance, so a fraction of a
   millimetre of bone *beside* rather than *under* the cartilage counts as cAB, and such a
   triangle's nearest AC point is the edge of the AC sheet at roughly its own height.
   The evidence for that is the **cAB** minimum, not the tAB one: over the 93 clean knees the
   peripheral subregions' mean ThC.cAB.aMin has a median of 0.156 mm (min 0.003, max 0.702;
   only 2.0% of the 1488 scan x region values below 0.05 mm), i.e. half an in-plane voxel of
   0.31 mm -- consistent with a triangle reading the height of the AC rim beside it, and far
   below the 1.3-2.5 mm of real peripheral cartilage. **The tAB minima are not evidence**:
   74.5% of those 1488 values are exactly 0.0 and `alf` / `arf` / `prf` are 0.0 in every scan,
   which is the denuded-as-0.0 mm *convention* (`thickness.py`: one denuded triangle in a
   region sets its tAB minimum to zero), not a measurement of thin cartilage. The step-4
   bullet above quotes the tAB figures 0.00-0.15 mm for that reason; read with the cAB twins
   (1.59-2.41 mm central, 0.11-0.23 peripheral) they say the same thing about the mechanism
   without resting on the convention. The resemblance to the paper's remark about peripheral
   minima is still a coincidence of mechanism: the paper's cartilage reconstruction tapers to
   zero at the plate edge, ours does not. A normal probe for cAB (the documented alternative in
   `surface.cartilage_coverage`) would change the region shares as well as the thickness, so it
   was not touched here.
3. The laterality of the `dfg_knees` knees is still not recorded anywhere (the patient-right
   plate is the larger in 79 of the 94 knees whose tibia subdivides -- step 5 re-derived that
   count -- which would make them mostly left knees if the medial plate is the larger one; to
   be confirmed, not assumed), so the cohort tables above are patient-side and cannot be read
   as medial/lateral. Nothing in the code or the data can settle it: it needs the acquisition
   records. Until then the paper's Table I / Table II values can only be compared with ours
   as a set, not column by column, which is how `scripts/report_knee_cartilage_cohort.py`
   prints them (its own rows carry no `paper` value; the paper's are separate rows).
4. **Flexion spread is dominated by the labels, not by the geometry.** The extension zones
   still have the loosest *area* across a flexion series (CV 6.5-7.8%, step 3) because how far
   the trochlear cartilage is labelled varies between acquisitions; in *thickness* the loosest
   are the tibial anterior and central regions (`art` CV 15.6%, `alt` 9.8%, `crt` 9.7%, median
   over the 24 flexion series, against 4.1-8.4% for the femoral thirds — step 5's cohort report
   reproduces the three tibial figures exactly and corrects the femoral range, which was quoted
   as 2-5%. Only **21** of the 24 series hold all four angles: the three flagged
   `*_90_belastet` scans are dropped, so their series have three. Over those 21 alone the
   femoral thirds give 4.04-8.45% and `crt` 10.24%, so the series set is worth stating), and at
   least part
   of that is the tibial label itself shrinking with flexion -- K1 loses 33% of its tibial
   cartilage voxels from 0 to 90 deg. Whether the cartilage really thins under flexion or the
   segmentation loses it cannot be told from these label maps.
5. `Femur.extract_subregions` (and the tibia's) still run `_remove_subregion_fragments` on the
   transferred voxels, a safety net inherited from the voxel scheme. It no longer touches any
   measurement -- the thickness is computed on the surfaces -- but it does silently drop
   cartilage from the voxel clouds `plot_segments` draws, so it is still worth removing.
6. Smaller ones: the AC rim angle and probe depths are not tuned. They are now parameters of
   `articular_surface_mesh` and plumbed through `measure_thickness`, so a cohort sweep is
   possible; on a sharp-edged flat 2 mm phantom they move only the rim band and never the
   interior (angle 30/45/60/75/89 deg -> whole-plate cMe 2.095/2.083/2.021/1.802/1.788,
   interior 2.098 throughout; probe depths (0.25,1.0)/(0.5,1.0)/(1.0,1.0) -> 1.834/2.021/2.075),
   and on K2_0_belastet's tibia 45/60/75 deg give 2.417/2.387/2.353 mm, but neither has been
   swept on the cohort. `aMav` / `aMiv` use the paper's 1% without a cross-check, and the
   dilution of the AC->cAB direction by the denuded area fraction is our reading of a step the
   paper does not spell out.

**Closed since this list was written** (nothing else on it has moved):

- *(was 7)* `notebooks/cartilage_thickness.ipynb` called the voxel-column API and had been
  broken since step 3. Step 5 rewrote it against the current API (`Tibia` / `Femur`,
  `plot_regions` / `plot_knee_segments` / `plot_knee_thickness`, `find_thinnest_area`,
  `scan_qc_flags`) and ran it end to end on its original Duesseldorf scan
  (`PYTHONPATH=<repo> PYVISTA_OFF_SCREEN=true python -m nbconvert --to notebook --execute`,
  run from the repo root with the venv's own interpreter -- the notebook's `kernelspec` is the
  generic `python3`, so without `PYTHONPATH` the kernel starts in `notebooks/` and cannot
  import `morphometry`, and without `PYVISTA_OFF_SCREEN` the plotting cells block). It is still an
  exploratory notebook with no tests and no callers, so it can break again silently.
- *(not on the list, but the step-5 gap)* there was no committed reference value for any knee
  cartilage number. There are 14 golden files now; see step 5 below.

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
scripts write `non_lpi_orientation:<codes>` (plus `tibia_transfer_far:<fraction>`, K7: 0.494,
and `femur_transfer_far`, K7: 0.124, K11: 0.022 -- step 5's cohort run lists the full flag
strings of all three)
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

*Implemented* (`morphometry/cartilage/femur_regions.py::subdivide_femur`, wired into `Femur`;
tests `tests/unit/test_femur_regions.py`). Decisions and deviations:
- Frame: shaft axis = principal direction of the centroids of 3 mm cross-sections of the femur
  in a band −5 to +25 mm around the top of the articular surface (`estimate_shaft_axis`),
  seeded by bone-centroid − cartilage-centroid, 3 refit passes, proximal bins dropped while
  they hold < 0.75 of the next bin's voxels (a cross-section cut by the FOV has a biased
  centroid). The band is deliberately *metaphyseal* — it starts 5 mm inside the condylar flare,
  not in the shaft: a knee FOV holds only 25–40 mm of femur above the cartilage and its
  proximal end is cut differently in different acquisitions of the same knee. Expect a few
  degrees of difference from an axis fitted to a whole femur; what matters here is that it is
  the same few degrees in every scan of one knee.
  **Calibration caveat.** The band was *selected* by flexion reproducibility on these same 96
  scans, so the invariance figures below are not an unbiased estimate of it, and it is not even
  the argmax of that objective. Over the 24 series (cF-area CV averaged over the two condyles, median/p90/max;
  A-P-extent CV median/max): (−5,25) 2.41/4.40/6.66 and 1.47/2.86; (−5,35) 2.54/3.45/4.48 and 0.14/2.22;
  (0,30) 2.77/3.80/4.16 and 1.63/3.75; (5,35) 2.53/3.90/5.25 and 1.61/8.57 plus one scan it
  cannot fit at all. (−5,35) is at least as defensible; the default only wins the cF-CV median.
  The band barely moves the thirds' *shares* (max difference between any two of those four
  bands, per scan: median 0.24 pp, max 1.58) but does move the shaft axis (median 1.9°, max
  7.5° between (−5,25) and (−5,35)) and the cF area / A-P extent (median 0.3%, max 4.2%) — the
  very quantities the reproducibility figures are about. It is a parameter of
  `subdivide_femur` (`shaft_band_mm`, `shaft_bin_mm`, `shaft_min_bin_ratio`), not a hidden
  module default.
- M-L axis = the posterior condylar line (`posterior_condylar_axis`, iterated from world +x);
  A-P = `cross(proximal, ml)`. No world direction is consulted for anterior — it follows from
  the right-handed frame, which is why a 90°-flexed knee needs no special case. A consequence
  of the M-L axis *being* the posterior condylar line: both condyles' posterior ends share one
  A-P coordinate, so the two 60% borders coincide.
- Trochlear notch from the **bone**, not the cartilage (`notch_depth_profile`,
  `find_trochlear_notch`): per 1 mm coronal slice, the most distal bone coordinate per 1 mm M-L
  bin; the valley depth is the drop from the lower flanking maximum to a bin between them, an
  empty bin (the open fossa) counting as infinitely deep; walk anteriorly from the deepest
  slice until the valley is shallower than 8 mm. The fossa is 20–30 mm deep, the trochlear
  groove 3–6, and the profile falls ~1.5 mm per mm of A-P travel there, so ±2 mm of threshold
  move the border by ~1 mm. The cartilage-gap version of the same walk was tried first and
  dropped: how far the trochlear cartilage is labelled across the groove is not reproducible
  between acquisitions of one knee (on K11_unbelastet it put the notch 13 mm apart at 0° and
  90°, where the bone profile agrees to 1 mm).
- Thirds as a continuous arc-length field (`arc_length_coordinate`) rather than the paper's
  per-triangle-strip construction: cross-sections 0.5 mm apart, each contour parametrised by
  arc length from its internal end, interpolated onto the vertices, cut exactly at 1/3 and 2/3.
  The triangulation then does not have to be strip-ordered, and a cylindrical condyle gives
  exactly a third of the area per third.
- The articular surface wraps proximally round the posterior condyle, so a coronal plane in the
  posterior part of the window cuts it twice. Inside the window the two sheets are separate
  patches; cF is the largest and the rest is counted with the posterior condyle zone. Measured
  by disabling `_largest_patch` with everything else fixed: without it 9 of the 93 clean scans
  have a third more than 1.2 pp off 33.3%, worst 40.5% (K7_90_unbelastet). The discarded patch
  is a sliver in a healthy subdivision (per condyle median 0.00%, p90 0.21%, max 14.90% of the
  A-P window on K4_90_belastet's right condyle); above `MAX_WRAP_FRACTION` (25%) it warns.
- Extension zones kept: trochlea (`alf`/`arf`) anterior of the notch plane, posterior condyle
  (`plf`/`prf`) behind the 60% plane, split by the same plane that separates the condyles (the
  MAD-filtered line of intercondylar-fossa centres, |slope| clamped to 0.5).
- Transfer to the cartilage voxels by nearest tAB triangle centroid, with the tibia's QC
  (`Femur.transfer_qc`, WARNING over 1% of voxels beyond 6 mm, `femur_transfer_far` flag).
- Names: keys stay patient-side (`iclf` … `prf`); `anatomical_femur_names` /
  `rename_femur_regions` add the paper's `icMF` … `ecLF`, plus `aMF`/`pMF`/`aLF`/`pLF` for the
  two extension zones (**our names**, the paper has none).
- Real data (96 `dfg_knees`, all subdivide; K7/K8/K11_90 stay flagged and are excluded from the
  statistics, n = 93): shares of cF tAB, patient-left / patient-right condyle, mean (SD):
  i 33.4 (0.30)/33.5 (0.29), c 33.2 (0.14)/33.2 (0.13), e 33.4 (0.21)/33.3 (0.20) (paper
  Table II, cMF/cLF: ic 33.5/33.4, cc 33.2/33.2, ec 33.3/33.4). cF tAB 548 ± 101 / 542 ± 82 mm²,
  notch-to-posterior extent 30.1 ± 3.4 mm, trochlea 1209/1103 mm², posterior condyle
  882/942 mm², whole femoral tAB 5225 mm² (4266–6445). Across the four flexion angles of one
  subject+load (24 series): CV of cF tAB 1.9–3.0% median against a 1.7% floor from the whole
  femoral tAB (i.e. the segmentation's own variability), SD of each third's share 0.07–0.22 pp;
  the extension zones are looser (6.5–7.8%) because how far the trochlea is labelled is not
  flexion-invariant. Loaded vs unloaded (45 pairs): cF tAB median |Δ| 0.93/1.22%. (Step 5's
  report reproduces all of these. Two were quoted differently before: the flexion SD of each
  third's share as 0.05–0.23 pp (it is a plain SD of the four shares, in percentage points, not
  normalised by anything) and the loaded-vs-unloaded cF tAB difference as 1.0/1.3% (that one is
  the relative difference `100 × |a − b| / mean(a, b)`).)
- Validation: the femur-tibia sagittal angle (the femoral frame against a tibial shaft axis
  fitted the same way) tracks the nominal flexion — median −32.9 / −13.8 / +19.2 / +43.7° at
  0/30/60/90°, Spearman ρ 0.956, strictly increasing in all 21 complete series, fit
  `0.86 × nominal − 34.9°`. The offset is the tibial axis (proximal metaphysis, line-fit
  residual 3.1 mm against the femur's 1.6), not the femoral frame. At nominal 0° the femoral
  axis's *sagittal* tilt against the scanner's superior axis — the only component that reaches
  the subregions, since a coronal tilt rotates the axis within the prox–ml plane and leaves
  `cross(prox, ml)` unchanged — is +1.22° ± 3.43 (n = 24); the full 3-D angle to world +z is
  9.80° ± 4.12, the remainder a coronal tilt of −6.04° ± 8.15.
- Thickness (still step 4's voxel-column method) now picks its column axis per subregion from
  the femoral frame rather than hard-coding it: superior-inferior everywhere except the two
  posterior-condyle zones, which use the image axis nearest the *femoral* A-P axis. That is the
  image A-P axis in an extended knee (what the old code assumed) but the image S-I axis at 90°
  of flexion, where the old assumption measured across the whole condyle (`left_posterior_zone`
  4.88 mm mean / 11.8 max on K9_90_belastet → 2.29 / 3.61).
  **This supersedes every existing `analyse_thinnest_point_flexion.py` run.** The posterior
  zones hold 8.8 / 14.4 / 48.0 / 76.3% of the tibiofemoral overlap the thinnest-point search
  runs on at 0 / 30 / 60 / 90°; their old values were implausibly large, so they never won the
  argmin and now can — which is the hypothesis the script tests. Over the 93 clean scans, with
  the subregions fixed and only the depth axis switched back, the thinnest point is unchanged
  at 0° (0/24) and 30° (0/24), moves on 13/24 at 60° and 20/21 at 90° (max 118 voxels,
  |Δ ap_fraction| median 0.300 / max 0.944 at 90°), and the sign of the per-series Spearman ρ
  flips on **9 of 24 series** (e.g. K9_belastet −0.20 → +0.80, K10_unbelastet +0.60 → −0.40).
  Old and new thinnest-point locations, `ap_fraction`s and trends must not be pooled or
  compared.

**4. Transfer and thickness.**
- Each AC triangle takes the region of its nearest tAB triangle.
- ThC.cAB.aMe (both directions, averaged, area-weighted), ThC.tAB.aMe (denuded = 0),
  aMax, aMav, SD, aMin, aMiv per region.
- Keep the `{region: {(x, y): mm}}` maps (or an adapter): `find_thinnest_area`,
  `split_tibia_thickness_by_compartment` (`*LT` / `*RT` suffixes), the plotting code and
  `scripts/compute_knee_cartilage_thickness.py` / `analyse_thinnest_point_flexion.py`
  depend on them. Renaming to medial/lateral must not silently break the left/right split.

*Implemented* (`morphometry/cartilage/thickness.py::measure_thickness`, wired into
`Tibia.calculate_thickness` / `Femur.calculate_thickness`; tests
`tests/unit/test_cartilage_thickness.py`). Decisions and deviations:
- **The articular surface AC** (`surface.articular_surface_mesh`) is the outer surface of the
  **union** of the bone and its cartilage label, so it holds no bone-cartilage interface at all
  and the two sides of a layer never have to be told apart by distance (which fails exactly
  where the cartilage is thin). The cartilage *rim*, and the band smoothing leaves at the foot
  of the rim, are excluded by two tests against the bone surface beneath each triangle: its
  normal must be within 60 deg of the bone normal there, and the voxel half a voxel down the
  **bone's** normal must carry the cartilage label. Keeping the rim reads 1.63 mm on the flat
  2.0 mm plate phantom and keeping only its foot 1.67 mm (excluded: 1.95); the error is
  proportional to a subregion's perimeter-to-area ratio, which is what distinguishes the
  peripheral subregions from the central one, so it is not a constant offset.
- Distances are exact point-to-triangle (`surface.surface_distances`, VTK's cell locator), not
  nearest-centroid, which would run up to ~0.1 mm short on a 0.1 mm^2 triangle mesh.
- **The two directions are pooled into one area-weighted sample**, each carrying half the
  weight, so the pooled mean is exactly `(cMe + bMe) / 2 = aMe` and aSD / aMax / aMav / aMin /
  aMiv describe one distribution. For the tAB variants the denuded triangles enter the cAB->AC
  side at 0.0 mm and the AC->cAB side -- which has no denuded triangles of its own -- is diluted
  by the same denuded area fraction through one 0.0 mm entry, so `tab_aMe = aMe * cAB/tAB`
  exactly, as it does for the cMe direction by construction. The paper does not say how it
  dilutes the AC side; this is our reading.
- aMav / aMiv split the triangle that straddles the 1% cut by area, so they are continuous in
  the weights.
- **The shared in-plane frame is the tibial plateau plane** (`KneeFrame`,
  `knee_frame_from_tibia`): origin at the midpoint of the two plates' centres of gravity,
  `e3` the mean plate normal, `e1` the line from the patient-left to the patient-right plate,
  `e2 = e3 x e1`. Maps are keyed by the integer `grid_mm` (0.5 mm) cell of each triangle
  centroid projected into it. The collapsed direction is the direction the joint is compressed
  along at *every* flexion angle, so the tibiofemoral overlap survives 0-90 deg; a femoral frame
  does not (phantom: common footprint > 1500 mm^2 at every angle in the tibial frame, < 300 mm^2
  at 90 deg in a co-rotating femoral one). `Femur.calculate_thickness(tibia)` therefore takes
  the tibia again -- for the frame only, not for the subregions. Without one it falls back to a
  femoral frame marked `source='femur'`, and `find_thinnest_area` refuses to overlap maps from
  different frames instead of pairing up unrelated positions.
- The per-position map is the **cAB->AC direction only** (denuded = 0 mm), area-averaged per
  cell: a bone-anchored map, which is what `find_thinnest_area` and the heatmap want. The
  two-directional statistics are reported alongside, not as a map.
- **Only the joint-facing sheet enters the maps** (`thickness._joint_facing_sheet`). Projecting
  a curved surface onto the shared plane folds it onto itself: on the 96 `dfg_knees`, cells
  whose contributing femoral centroids span more than 5 mm are 10/7/13/22% of the femoral cells
  at 0/30/60/90 deg of flexion (26-47% of the tAB *area*) and 14/5/2/18% of the cells the
  thinnest-point search uses, with the largest spreads 30-60 mm. Averaging those mixes the
  trochlea with the posterior condyle and -- worse -- puts the cell's recorded articular
  position *inside the bone* between the two sheets, which is where the thinnest-point marker
  and the femoral arc coordinate read it from. Keeping the sheet nearest the other bone fixes
  that; the regional statistics are per triangle and do not move at all. On the cohort it
  changes the thinnest point itself very little (median |delta arc| 0.000, moved by more than
  0.05 on 44 of 279 compartment results), so it is a correctness fix rather than a result.
- **The overlap is where the two *projections* meet**, which is close to but not the same as
  where the joint touches: the trochlea projects onto the plateau too, and supplies the femoral
  value at the thinnest point in 16 of 93 whole-knee and 31 of 93 patient-left searches.
- Both statistic families are reported and must not be mixed: the bare names (`aMe`, `aSD`,
  `aMax`, `aMav`, `aMin`, `aMiv`) are ThC.**cAB** and the `tab_` ones ThC.**tAB**, the same with
  denuded bone counted as 0.0 mm. Before this, `aMe` was cAB while `aSD`/`aMin`/... came from
  the tAB distribution, so `aMe +- aSD` mixed families and `aMin` was exactly 0 for any region
  with any denuded bone.
- `find_thinnest_area` restricts the neighbourhood-centre search **and the refinement disc**
  to cells whose whole `k x k` window is inside the overlap footprint. The thinnest *area* must
  be fully measured, and the footprint boundary is exactly where the cartilage runs out and the
  thickness falls towards zero, so a partially-observed window wins the argmin on nearly every
  knee. The refinement escaping that rule was the dominant rim artefact: it took the femoral
  thickness at the returned point from a median 1.51 mm to 0.62 mm (below 1 mm in 67% of
  results instead of 11%) over the 93 clean knees. `edge_margin_mm` and `min_thickness_mm`
  exist for the residual and are **off by default** -- see the long note on
  `knee.DEFAULT_EDGE_MARGIN_MM` for why the data cannot choose a value and why any value biases
  a flexion trend towards the null.
- **Caveat wherever a per-compartment p-value is quoted.** The always-on full-window
  restriction removes a median 27.3% of the candidate cells (IQR 25.0-30.3) and moves the
  chosen cell in **279 of 279** compartment results, by a median |delta arc| of 0.233 (p90
  0.729, max 0.997). It is not directionally biased and the headline survives without it --
  switching it off gives whole knee 20/24 p=0.0008, patient-left 21/24 p=0.0001, patient-right
  18/24 p=0.0113, against 20/24, 22/24 and 21/24 with it -- but *which* compartment looks most
  significant is not stable under it. Quote the per-compartment sign tests with that caveat or
  not at all.
- `measure_thickness` returns a `ThicknessMaps`, a `dict` subclass, so every consumer still
  sees `{region: {(u, v): mm}}` while `.statistics`, `.frame`, `.positions` and
  `.triangle_thickness` ride along. `plot_thickness` now colours the tAB mesh itself (the
  paper's Fig. 4(b) / 5(b) view) instead of reconstructing a surface from voxels.
- Invariance on *real* data, not just phantoms: resampling K2_0_belastet from its native
  0.75 x 0.3125 x 0.3125 mm grid to 0.5 mm isotropic (nearest-neighbour) moves the tibial
  subregion aMe by at most 0.073 mm (<= 2%), and measuring the same scan through the SimpleITK
  backend instead of nibabel reproduces every subregion bit for bit.
- Real data (96 `dfg_knees`, of which **95 measure** -- K11_90_belastet's tibia raises 'the two
  tibial plates overlap after hole filling' and its femur returns a NaN `cclf`; ~20 s per knee,
  7.1 s tibia and 12.9 s femur. K7_90_belastet and K8_90_belastet are QC-flagged, K11_90_belastet
  is excluded by its failure, so n = 93): ThC.cAB.aMe mean (SD) mm -- tibia c 2.30 (0.48) / 2.26 (0.53), i 1.90/1.89,
  e 1.55/1.55, a 1.69/1.79, p 1.91/1.65 (patient-left / patient-right plate); femur cc
  3.15 (0.42) / 3.05 (0.50), ic 2.25/2.35, ec 2.17/1.99, trochlea 2.26/2.31, posterior condyle
  2.46/2.38. Whole-bone aMe **1.80-1.87 mm (tibia) / 2.34-2.41 mm (femur)** at 0/30/60/90 deg
  (tibia 1.798/1.853/1.868/1.825, femur 2.408/2.398/2.367/2.337). These were quoted as
  1.81-1.87 / 2.33-2.40, from a cAB-area-weighted mean of the subregion `aMe`s; the union's
  `aMe` is exact only if each *direction* keeps its own weights (`cMe` by cAB, `bMe` by AC
  area), which is what `scripts/report_knee_cartilage_cohort.py` now does -- worth up to
  0.025 mm (1.3%) on a whole bone, and enough to move three of these four bounds.
  Mean ThC.**tAB**.aMin is 1.2-2.2 mm in the four central regions and 0.00-0.15 mm everywhere
  else, which reproduces the paper's remark -- here through cAB reaching a fraction of a
  millimetre sideways past the cartilage edge, not through a tapering cartilage reconstruction.
  (Those two figures are the tAB family, i.e. 0 wherever a region holds any denuded bone, even
  though every other statistic in this bullet is cAB; step 5's run confirms them --
  1.22-2.23 mm central, 0.000-0.149 peripheral -- and gives the cAB twins as 1.59-2.41 mm
  central and 0.11-0.23 mm peripheral, which are *not* near zero.) Loaded vs unloaded:
  median |delta aMe| 0.085 mm (p90 0.278, max 1.13, n = 900 pairs); across the four flexion
  angles of one subject+load: median SD 0.117 mm (p90 0.296, max 0.769), median CV 5.7% --
  spread, not precision (the paper's test-retest rms SD is 19-84 um; these are not repeats).
- **The thinnest-point flexion analysis was re-run on the cohort** with the new thickness, the
  shared frame, the joint-facing sheet, the refinement fix and the femoral-frame `ap_fraction`
  (`scripts/analyse_thinnest_point_flexion.py` over all 96 scans). It supersedes every earlier
  run, including the ones made earlier in this same step. Exactly 3 of the 288
  (scan x compartment) rows are NaN, all three of them the broken K11_90_belastet; every one of
  the 24 (subject, load) series is testable in every compartment. Two arms, the default
  (`--edge-margin 0`, i.e. the full-window restriction only) and an aggressive 2.5 mm margin:

  | | whole knee | patient-left | patient-right |
  |---|---|---|---|
  | rho median (IQR), margin 0 | +0.80 (+0.25..+0.80) | +0.90 (+0.80..+1.00) | +0.80 (+0.40..+1.00) |
  | rho median (IQR), margin 2.5 | +0.80 (+0.75..+1.00) | +1.00 (+0.80..+1.00) | +0.80 (+0.40..+1.00) |
  | slope/deg median, margin 0 / 2.5 | +0.0042 / +0.0048 | +0.0066 / +0.0067 | +0.0046 / +0.0030 |
  | sign test, margin 0 | 20/24, p = 0.0008 | 22/24, p = 0.00003 | 21/24, p = 0.0001 |
  | sign test, margin 2.5 | 20/24, p = 0.0008 | 23/24, p = 0.000003 | 19/24, p = 0.0033 |

  The point itself moves at 2.5 mm (272 of 279 compartment results pick a different cell) and
  the artefact indicators improve -- the femoral thickness at the point rises from a median
  1.52 to 1.68 mm, the share below 1 mm falls 14.0 -> 9.7%, and the share whose arc position
  sits in the posterior tenth falls 15.4 -> 9.3% -- while the trend itself barely does.

  These are separate uncorrected tests; the two loads of one subject are not independent; the
  per-compartment numbers are not robust to the search restrictions (see the caveat above);
  and 'left'/'right' is the *patient* side, not medial/lateral -- the knees' laterality is
  still unrecorded (open issue 3), so the per-side result cannot be read as medial or lateral.
  **Reported, not interpreted.**
- **Do not pool thickness rows produced before and after this change.** The old `knn` method
  measured one-directional nearest-neighbour distances between *voxel centres* on the two voxel
  surfaces, i.e. about one voxel short: on K1_0_unbelastet the new `aMe` is higher in all 20
  subregions (median +0.40 mm, max +1.26 -- `clt` 1.800 -> 2.487, `cclf` 1.976 -> 3.236), and on
  K1_90_unbelastet (median |delta| 0.364) the two trochlea zones instead come *down* from their
  implausible voxel-column values (`left_anterior_zone` 4.002 -> 2.613, `right_anterior_zone`
  3.827 -> 2.686), because the old column axis crossed the whole trochlea. The per-position
  maps are not comparable at all: different keys in a different frame.

**5. Tests.**
- Phantoms: a flat elliptical plate gives a central region of exactly 20% of the area;
  peripheral wedges are exactly 90° about S; a cylindrical condyle gives arc-length thirds
  of 33.3%.
- Invariance: rotate the volume (e.g. 15° axial), resample to other voxel sizes, mirror
  left-right: the region shares must not change (mirror: left and right swap exactly).
- Real data (`dfg_knees`): tables like the paper's Tables I/II (cT ≈ 20.0%, thirds ≈ 33.3%,
  peripheral sizes), old-vs-new thickness per region, a knee-cartilage golden.

*Implemented.* The phantom and invariance tests are in `tests/unit/test_cartilage_surface.py`,
`tests/unit/test_tibia_regions.py`, `tests/unit/test_femur_regions.py` and
`tests/unit/test_cartilage_thickness.py` (thickness: a 2 mm layer on a flat plate and on a
cylindrical condyle read to within a few per cent, a wedge reads its volume-over-area mean,
denuded bone dilutes `tab_aMe` by exactly its area fraction, the statistics on a known
weighted distribution, area weighting unchanged by refining half a mesh, invariance under
pose / voxel size / oblique sampling / SimpleITK / mirroring, and the shared frame keeping the
tibiofemoral overlap at 0-90 deg of flexion where a femoral frame loses it). Step 5 added the
three real-data pieces that were missing: a data fixture, the goldens and a cohort report.

**A `dfg_knees` fixture and marker**, in the house pattern of `needs_augsburg` / `needs_nako` /
`needs_ct`: the marker `needs_dfg_knees` (registered in `pyproject.toml`), the directory in
`tests/data_paths.py` (`DFG_KNEES_DIR`, env override `MORPH_DFG_KNEES_DIR`, scan names through
`dfg_knee_path`, the two golden scans as `DFG_KNEES_GOLDEN_SCAN{,_FLEXED}`) and the fixtures
`dfg_knees_dir` / `dfg_knee_file` / `dfg_knee` in `tests/conftest.py`, which **skip** when the
data is absent. The local skip in `test_cartilage_surface.py` (env `MORPH_DFG_KNEES_SAMPLE`,
now gone) was migrated to it. `dfg_knee` loads a scan the way the knee scripts do -- LPI,
**no** `remove_outliers()`, the bones clean their own cartilage masks -- so the goldens pin
what production produces; `conftest`'s generic `_load_seg` would not. Checked on a machine
without the data: `MORPH_DFG_KNEES_DIR=/nonexistent pytest -m "not needs_ct"` gives
1260 passed, 16 skipped.

**Knee cartilage goldens** (`tests/characterization/test_knee_cartilage.py`, 14 files in
`tests/golden/`, compared at the harness's default tolerances rtol 1e-4 / atol 1e-3, captured
with `MORPH_UPDATE_GOLDEN=1`). Everything is pinned on **`K1_0_unbelastet`** -- a clean,
unloaded, extended knee with no QC flag, and the scan this document already quotes its
old-vs-new thickness comparison on (`clt` 1.800 -> 2.487 mm, `cclf` 1.976 -> 3.236 mm, both
reproduced by the goldens):

- `knee_tibia_region_shares` (10), `knee_tibia_region_areas` (10 + the area dropped outside
  both plates) and `knee_tibia_plate_frames` (52: per plate S, the three principal axes, the
  variances, the internal and anterior directions, rho*, the semi-axes and the tAB / cAB
  areas). The two in-plane variances are 90.5 vs 43.9 mm² (patient-left) and 133.8 vs 61.1
  (patient-right), so the eigenvector order is not a near tie.
- `knee_femur_region_shares` (6 thirds), `knee_femur_region_areas` (10, incl. trochlea and
  posterior condyle) and `knee_femur_frame` (30: the three axes, the trochlear-notch point and
  its A-P coordinate, the condyle divider, the shaft-fit residual and bin count, the whole
  femoral tAB, and per condyle the posterior and 60% borders, cF tAB / cAB and the trochlea and
  posterior-zone areas).
- `knee_tibia_thickness_statistics` and `knee_femur_thickness_statistics` (200 each): all 20
  statistics of `thickness.STATISTIC_KEYS` for each of the 10 subregions -- both ThC families,
  the cAB / tAB / AC areas and the denuded fraction.
- `knee_thinnest_point` (30): the whole knee and both patient-side compartments, each with the
  thinnest cell, the neighbourhood centre, the smoothed and raw combined thickness, the two
  per-bone contributions, the kernel size and the point's arc fraction along the femoral
  articular surface.

A second, cheaper set is pinned on **`K1_90_unbelastet`**, the 90° scan of the same subject and
load: `knee90_region_shares` (16), `knee90_region_areas` (20), `knee90_femur_frame` (30),
`knee90_thickness_mean` (20 × `aMe`) and `knee90_thinnest_point` (24 -- the thinnest cells and
their thicknesses, but **not** the neighbourhood centres, see below). It costs ~15 s and
earns it: at 90° of flexion the femoral frame, the posterior-condyle zones,
`_restrict_to_contact_cluster` and the shared in-plane frame all do work an extended knee never
exercises (the posterior zones hold 76% of the tibiofemoral overlap there against 9% at 0°),
and the axis switch recorded in step 3 moved exactly those numbers. Both scans are measured
once per module run and cached (~15 s each on a 32-core workstation; the whole module is ~31 s).

Left out of the goldens deliberately, each for a stated reason:

- `n_tab_triangles` / `n_ac_triangles`: integer mesh counts with no tolerance to give -- a
  `scikit-image` or VTK upgrade would fail them without any measurement having changed, and the
  areas already cover what they would catch.
- `CondyleFrame.n_slices`, the same argument: it counts how many 0.5 mm cross-sections fit in
  the weight-bearing A-P window, i.e. quantises a continuous length, and the window has only
  0.049 mm of headroom at 0° (0.139 at 90°) -- a twentieth of a millimetre, a sixth of an
  in-plane voxel, flips 31 to 30 for nothing. Over the cohort it takes 13 distinct values
  (29-43), so it is genuinely mobile. `FemoralFrame.shaft_bins` *is* pinned: it is 10 on all 96
  scans.
- the **neighbourhood centre** of the flexed scan's thinnest-point search. On
  `K1_90_unbelastet` the smoothed minimum is attained *bitwise identically* at six cells
  spanning 2.5 mm (both the whole knee and the patient-right compartment; the patient-left one
  is a two-way tie 3.5e-3 mm away), so which one is reported is decided by the C-order grid
  traversal, not by the data. The refined `point` inside the disc around it is stable and stays
  pinned, as does the 0° scan's centre (single-cell minima there, runner-up +0.073 / +0.18 mm).
- the transferred region voxel clouds: they move with the fragment-removal safety net (open
  issue 5) and no measurement reads them.

Sensitivity, measured rather than assumed, by perturbing a scratch copy of the package:

| perturbation | result |
|---|---|
| `surface.default_coverage_tolerance` 0.6 -> 0.6006 × the voxel diagonal (+0.1%) | 8 of 15 tests fail (both bones' areas and frames, both thickness blocks) |
| the same, 0.6 -> 0.62 (+3%) | 11 of 15 fail (every area, frame, thickness and thinnest-point golden) |
| `Tibia(central_fraction=)` 0.2 -> 0.205 and `Femur(weightbearing_fraction=)` 0.6 -> 0.61 | 12 of 15 fail, this time including the three share goldens |

The share goldens surviving the first perturbation is the point of them: the central region
holds 20% and the thirds 33.3% of their own tAB *by construction*, whatever the coverage rule
does to the areas. The two thinnest-point goldens survive the second: that search runs on the
union of the regions, not on their boundaries.

**A cohort report script**, `scripts/report_knee_cartilage_cohort.py`, so the numbers quoted in
this document and in `CLAUDE.md` are re-derivable instead of trusted. House style of
`compute_knee_cartilage_thickness.py`: `-i` takes files and/or directories, `-o` a directory,
`--workers` fans out over the CPUs, every scan is measured in a try/except so one bad scan
cannot abort the cohort, and the scan's `scan_qc_flags` string rides along in a `qc_flags`
column. By default the statistics cover the scans that measured *and* carry no QC flag (93 of
the 96 `dfg_knees`); `--include-flagged` and `--exclude <scan> ...` reproduce an older set
(the tibial table above was computed over 94 scans:
`--include-flagged --exclude K7_90_belastet K11_90_belastet`). It writes four files:
`scan_metrics.csv` (long: one row per scan × bone × region, plus the aggregates `plate_left` /
`plate_right` / `cf_left` / `cf_right` / `whole`), `scan_geometry.csv` (long: one row per scan ×
geometry or QC scalar), `cohort_summary.csv` (long: one row per statistic -- `table, scope,
region, unit, n, mean, sd, min, max, p50, p90, paper`, every number in the report among them)
and `report.md` (the Table I / Table II analogues against the paper's published values, the
area and thickness tables, the reproducibility tables and a QC section). Reproducibility is
reported as **spread, not precision**, with that stated in the output: loaded vs unloaded is
one pair per subject and angle (45), the flexion spread the SD across the angles of one subject
and load (24 series), and the knee is repositioned and reloaded between them -- the paper's
19-84 µm test-retest rms SD is a different quantity. The pooled rows cover the 20 subregions
only (pooling the aggregates too would double-count them), which is the convention the figures
in this document use -- 45 pairs × 20 subregions = the 900 `aMe` differences quoted in step 4.
Its aggregation half is unit-tested without any data in
`tests/unit/test_knee_cohort_report_script.py`.

**What the full run found** (all 96 scans, 14 workers, ~3.5 minutes). Everything in this
document and in `CLAUDE.md` reproduced, to the decimal quoted, except six items, all now
corrected in place above: the whole-bone `aMe` per flexion angle (1.81-1.87 / 2.33-2.40 ->
**1.80-1.87 / 2.34-2.41**, because an aggregate's `aMe` has to weight each direction by its
own area -- see the step-4 bullet), the femoral thirds' thickness CV across a flexion series
(quoted 2-5%, actually 4.1-8.4%), the loaded-vs-unloaded cF tAB difference
(1.0/1.3% -> 0.93/1.22%), the flexion SD of each third's share (0.05-0.23 pp -> 0.07-0.22),
step 1's tAB ranges (measured on a handful of knees, not the cohort) and step 4's `aMin`
sentence (correct, but it is the **tAB** family in a bullet whose other names are cAB, and the
tAB zeros are a convention rather than evidence -- see open issue 2). Per-scan timings are machine-dependent
and not a discrepancy: this workstation does a subdivision in 1.2 s (tibia) / 1.9 s (femur) and
a whole knee including both thickness measurements in ~15 s single-process, against the ~3 s
and ~20 s recorded earlier. Confirmed unchanged, among others: the tibial Table I shares and
their SDs, the femoral thirds, cF tAB 548 ± 101 / 542 ± 82 mm², the notch-to-posterior extent
30.1 ± 3.4 mm, the trochlea and posterior-condyle areas, the whole femoral tAB 5225 (4266-6445),
every subregion's `aMe` mean and SD, the 900-pair
loaded-vs-unloaded `aMe` median 0.085 mm / p90 0.278 / max 1.13, the flexion `aMe` SD median
0.117 mm and CV 5.7%, the extension zones' 6.5-7.8% area CV, the QC counts and distances
(K7_90_belastet 49% of its tibial cartilage beyond 6 mm, max 45 mm; its tibial cartilage 26% of
the unloaded scan's; 15 mm² of dropped islands against a cohort median of 0.15), the smallest
plate ratio 0.614 (`K1_90_belastet`, on the covered areas `find_plates` compares; 0.636 on the
filled tAB), the minimum anterior-direction |cos| 0.805, the patient-right plate being the
larger in 79 of 94, and K1 losing 33% of its tibial cartilage voxels from 0 to 90°.
