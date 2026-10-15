# Knee cartilage subregions: aligning with Wirth & Eckstein 2008

Reference: W. Wirth and F. Eckstein, "A Technique for Regional Analysis of Femorotibial
Cartilage Thickness Based on Quantitative Magnetic Resonance Imaging", *IEEE TMI* 27(6),
737–744, 2008 (`Wirth_Eckstein_2008.pdf`, repo root, untracked).

Status (2026-10-15): analysis and step 0 done. The `calculate_ellipse` bug is fixed (60425aa); the
rewrite itself has not started. Code: `morphometry/cartilage/knee.py`.

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

**2. Tibia.**
- Split the tAB into two plates (connected components / the gap at the intercondylar
  eminence).
- Per plate: area-weighted centre S and principal axes. Central region: normalised
  elliptical radius ρ = √((x/a)² + (y/b)²) in the plate plane, with a : b from the plate's
  in-plane principal spreads; the cutoff ρ* is the area-weighted 20% quantile (exact, no
  iteration), boundary triangles cut at ρ = ρ*.
- Peripheral split: two planes through S containing b3, at ±45° to tc projected into the
  plate plane. Internal = the wedge facing the other plate, external opposite,
  anterior/posterior from the image A-P axis.
- Central share is a parameter (the paper's Table IV uses 10–50%).

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
