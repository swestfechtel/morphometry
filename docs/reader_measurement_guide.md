# Reader Measurement Guide

This document describes, step by step, how each morphometric parameter in this
project is measured by the algorithm. It is intended for human readers who will
perform manual measurements of the same parameters, so that the manual readings
can be made as close as possible to the algorithmic definitions and the two can
be compared fairly.

For every parameter you will find:

- **What it is** — the anatomic definition the algorithm implements.
- **Which image / plane** — the slice and viewing plane the algorithm works on.
- **Landmarks** — how each landmark is located, in anatomic terms.
- **Reference lines & computation** — how the lines are constructed and how the
  final number is derived.
- **Sign / range conventions** — when relevant.

> The descriptions translate the algorithm's voxel/array operations into
> anatomic instructions. Where the algorithm makes an arbitrary implementation
> choice (e.g. *"the slice with the largest cross-sectional bone area"*), the
> instruction is given verbatim so a reader can reproduce it.

---

## 0. Conventions you need before you start

### 0.1 Orientation

Before any measurement, every mask is reoriented to a standard **LPI** frame:

| Axis | Anatomic direction (increasing index) |
|------|----------------------------------------|
| 1st  | toward patient **Left** (medial–lateral / sagittal coordinate) |
| 2nd  | toward **Posterior** (anterior–posterior / coronal coordinate) |
| 3rd  | toward **Inferior** (superior–inferior / "slice" or "layer" index) |

- A **transverse (axial) slice** is a single value of the 3rd axis — this is the
  plane in which all torsion and rotation angles are measured.
- A **coronal slice** is a single value of the 2nd axis — this is the plane in
  which CEA, HKA, JLCA and mechanical-axis deviation are measured.

### 0.2 "Side" means *image* side, not patient side

The algorithms split a whole pelvis/leg mask down the midline into a left
*image* half and a right *image* half. Because the first axis increases toward
the patient's left, the **left image half is the patient's right leg** and vice
versa. When you measure manually, just measure each leg on its own; the
left/right bookkeeping is only an internal detail. Each leg is measured
independently.

### 0.3 Angle and distance arithmetic

- All angles are the unsigned angle between two vectors (`arccos` of the
  normalized dot product), in **degrees**.
- All physical distances (lengths, offsets, deviations) are computed in
  **millimetres** in patient (world) coordinates, i.e. voxel spacing and slice
  thickness are accounted for.

---

## 1. Femoral torsion

Femoral torsion is the angle, **in the transverse plane**, between the axis of
the femoral neck (proximal reference line) and the posterior condylar line of
the distal femur (distal reference line). The project implements **two methods**
for the proximal reference line — **Lee** and **Murphy** — and reports both.
The distal reference line is identical for both.

### 1.1 Proximal reference line

Common first step (both methods):

1. **Femoral head centre.** Fit a sphere to the contour of the femoral head, and take its centre and radius (r).

**Method "Lee" — femoral neck centre:**

2. Going inferiorly from the head, find the first transverse slice on which (a)
   the femur forms a *single* connected outline and (b) a circle of radius `2·r`
   centred on the head centre still intersects bone. This is the neck slice.
3. On that slice, keep only the bone outline lying in a thin ring around the
   head centre (between ≈0.9·*r* and ≈1.1·*r*, the inner/outer bounds refined
   from the position of the antero-posterior neck "notch"), then further keep
   only outline points close to the neck's own centre of mass.
4. **Fit a straight line through the head centre and these neck-outline points**
   (if the outline is split anterior/posterior, fit each side and average the
   slopes). This line is the **femoral neck axis**. Its lateral-to-medial
   direction defines the proximal reference line.

**Method "Murphy" — femoral neck base:**

2. **Lesser trochanter.** Find the transversal slice where the lesser trochanter's protrusion is most prominent.
3. The **femoral neck base** is the centre of mass of the femoral shaft
   cross-section on that lesser-trochanter slice. The proximal reference line
   runs from the femoral head centre to this neck-base point.

### 1.2 Distal reference line (posterior condylar line)

1. Choose the transverse slice with the **largest convex cross-sectional bone
   area** of the distal femur (the slice through the bulk of both condyles).
2. Identify the **intercondylar notch** (the antero-posterior gap between the
   condyles) on that slice.
3. The reference line is the **tangent to the two posterior condyles** — i.e.
   the line resting against the most posterior aspect of the medial and lateral
   condyles (the classic posterior condylar line). The algorithm anchors the
   line at the most posterior bone point and rotates it until it just touches
   both condyles.

### 1.3 Computation

- Measure the angle of the proximal reference line to the medial–lateral axis,
  and the angle of the posterior condylar line to the medial–lateral axis, **in
  the transverse plane**.
- Both partial angles are folded into the 0–90° range. They are **added** when
  the neck and the condylar line tilt in opposite antero-posterior directions
  and **subtracted** when they tilt the same way. The result is the net
  transverse-plane angle between femoral neck axis and posterior condylar line.

---

## 2. Tibial torsion

The angle, **in the transverse plane**, between the posterior condylar line of
the proximal tibia and the trans-malleolar (tibia–fibula) line at the distal
tibia.

### 2.1 Proximal reference line (tibial plateau)

Identical construction to the femoral posterior condylar line (§1.2) but applied
to the **tibia at the knee**: on the tibial slice with the largest convex bone
area, the line tangent to the two posterior aspects of the tibial plateau.

### 2.2 Distal reference line (ankle)

1. Find the transverse slice with the **largest tibial diameter** at the ankle
   (equivalent ellipse diameter).
2. On that slice, compute the **centre of mass of the tibia** and the **centre
   of mass of the fibula**.
3. The distal reference line connects these two centroids.

### 2.3 Computation

Each line's angle to the medial–lateral axis is measured in the transverse
plane and folded to 0–90°; the two are added or subtracted depending on whether
they tilt to the same or opposite antero-posterior side, giving the net
transverse-plane angle between the proximal tibial condylar line and the distal
tibia–fibula line.

---

## 3. Knee rotation angle

The angle, **in the transverse plane**, between the **posterior condylar line of
the distal femur** and the **posterior condylar line of the proximal tibia** at
the knee.

- The **femoral** line is built exactly as in §1.2.
- The **tibial** line is built exactly as in §2.1.
- Both lines are taken at the knee, and the transverse-plane angle between them
  is reported (folded to 0–90° and combined by tilt direction, as above). A
  result of exactly 180° is reset to 0°.

This is effectively the rotational mismatch between the femoral and tibial
posterior condylar axes.

---

## 4. CCD angle (caput–collum–diaphyseal / neck–shaft angle)

The angle between the **femoral neck axis** and the **femoral shaft axis**. Two
values are reported: the true **3D angle** and its **projection onto the coronal
plane**.

### 4.1 Landmarks

1. **Femoral head centre** — sphere fit to the superior femoral head surface
   (as in §1.1, step 1); gives centre and radius *r*.
2. **Femoral neck centre** — collect the femoral bone voxels lying in a hollow
   spherical shell around the head centre (between *r* and 1.2·*r*) that are
   both **distal and lateral** to the head centre; their centre of mass is the
   neck centre. The **neck axis** is head centre → neck centre.
3. **Femoral shaft axis** —
   - If a knee image is available: from the centre of mass of the most distal
     slice of the proximal-femur mask to the centre of mass of the most
     proximal slice of the distal-femur (knee) mask.
   - Otherwise: from the distal-slice centroid of the proximal-femur mask up to
     the tip of the greater trochanter region (first slice where the proximal
     femur splits into two components; centroid of the smaller component).

### 4.2 Computation

- **CCD (3D):** angle between the neck vector and the shaft vector in 3D; if the
  raw angle is acute it is taken as `180° − angle` so the reported neck–shaft
  angle is the obtuse one (typical ~120–135°).
- **CCD (projected):** the same two vectors with their antero-posterior
  component set to zero (projected onto the coronal plane), then the angle
  between them, again reported as the obtuse value.

**For a reader:** the projected CCD corresponds to the conventional coronal
neck–shaft angle; draw the neck axis (head centre → neck centre) and the shaft
axis (mid-diaphyseal line) on the coronal view and measure the medial angle
between them.

---

## 5. Leg-length parameters (bone lengths)

Length of the **femur**, **tibia**, or **whole leg**, measured in millimetres as
the straight-line (Euclidean) distance between two centroids.

1. **Proximal point** — centre of mass of the bone on its **most proximal
   (superior) slice**.
2. **Distal point** —
   - Femur / whole leg: centre of mass on the **most distal (inferior) slice**.
   - Tibia: centre of mass on the **distal articulating surface** slice — i.e.
     the most distal slice before the cross-sectional area changes abruptly
     (the tibial plafond, just above the ankle joint), not the medial malleolus
     tip.
3. The length is the distance between these two centroids in patient
   coordinates (mm), accounting for slice spacing.

**For a reader:** locate the centroid of the proximal-most and distal-most bone
cross-sections and measure the 3D distance between them. For the tibia, use the
distal tibial joint surface rather than the malleolus.

---

## 6. Acetabular version (acetabular anteversion)

The transverse-plane angle describing how anteverted the acetabular opening is,
measured on a single transverse slice.

### 6.1 Slice and landmarks

1. **Femoral head centres** of both hips are located (sphere fit, §1.1). The
   measurement slice is the transverse slice **midway between the two femoral
   head centres**.
2. On that slice, for each acetabulum:
   - **Posterior rim point (p1)** — the most lateral bone point of the
     acetabulum within its **posterior** portion (the anterior third of the
     acetabulum, anterior to the femoral-head-centre level, is excluded).
   - **Anterior rim point (p2)** — the most lateral bone point of the acetabulum
     within its **anterior** portion (the posterior part is excluded).
3. A reference line **G** connects the two femoral head centres (the
   inter-acetabular / horizontal reference).

### 6.2 Computation

- Construct the line **perpendicular to G** passing through the posterior rim
  point p1 (this perpendicular points in the antero-posterior direction).
- The acetabular version is the angle between the **acetabular rim line**
  (posterior rim p1 → anterior rim p2) and that **perpendicular reference**.

**For a reader:** on the chosen transverse slice, draw the line connecting both
femoral head centres, drop a perpendicular at the acetabulum, draw the line
joining the anterior and posterior bony acetabular rims, and report the angle
between the rim line and the perpendicular.

---

## 7. Center-edge (CE) angle (lateral CE angle of Wiberg)

The coronal-plane angle between a vertical reference through the femoral head
centre and the line to the lateral edge of the acetabular roof.

### 7.1 Landmarks

1. **Femoral head centres** of both hips (sphere fit, §1.1), giving centre and
   radius for each.
2. **Reference line G** connecting the two femoral head centres (horizontal
   pelvic reference). The **vertical reference** is the line through the femoral
   head centre **perpendicular to G**, directed proximally (superiorly).
3. **Lateral acetabular edge point** — the **most lateral** bony point of the
   acetabular roof situated **above** the femoral head (the search starts ~1.1–
   1.5 head-radii above the head centre and takes the most lateral, then most
   superior, acetabular voxel).

### 7.2 Computation

The CE angle is the angle between:

- the **vertical reference** (proximal direction, perpendicular to the
  inter-head line G), and
- the line from the **femoral head centre to the lateral acetabular edge
  point**.

This is measured in the coronal plane. (An optional mode projects all landmarks
to a single coronal plane first; the default does not.)

**For a reader:** on the coronal view, mark the femoral head centre, draw the
horizontal line through both head centres, raise the vertical from the head
centre, draw the line from the head centre to the lateral edge of the
acetabular sourcil, and measure the angle between vertical and that line.

---

## 8. Femoral offset

The perpendicular distance (mm) from the **femoral head centre** to the
**femoral shaft axis**.

1. **Femoral head centre** — sphere fit (§1.1).
2. **Femoral shaft axis** — same construction as in §4.1, step 3.
3. Drop a perpendicular from the head centre onto the shaft axis; the femoral
   offset is the length of that perpendicular, in millimetres, in patient
   coordinates.

(A projected variant zeroes the antero-posterior component so the measurement is
made purely in the coronal plane; the default uses the full 3D perpendicular
distance.)

**For a reader:** measure the shortest distance from the centre of the femoral
head to the central long axis of the femoral shaft.

---

## 9. Hip–knee–ankle (HKA) angle

The coronal-plane angle of the mechanical axis: the angle at the knee between
the femoral-head→knee segment and the knee→ankle segment.

### 9.1 Landmarks (whole-leg image)

1. **Hip point** — femoral head centre (sphere fit, §1.1).
2. **Knee point** — centre of mass of the tibia in the knee region (the proximal
   tibia at the joint).
3. **Ankle point** — centre of mass of the **distal tibial articulating
   surface** (the plafond slice, found as in §5 — the most distal slice before
   the abrupt area change).

### 9.2 Computation

- Vector 1: hip point → knee point.
- Vector 2: knee point → ankle point.
- Both vectors are **projected onto the coronal plane** (antero-posterior
  component set to zero), and the angle between them is reported.

A straight mechanical axis gives ~0°; varus/valgus increases it. (Note: this is
the deviation from a straight line, not the 180°-based HKA convention — i.e. a
perfectly straight leg reads 0°.)

**For a reader:** mark femoral head centre, knee centre, and ankle (plafond)
centre on the coronal view and measure the angle the leg makes at the knee.

---

## 10. Joint-line convergence angle (JLCA)

The coronal-plane angle between the distal femoral joint line and the proximal
tibial joint line.

### 10.1 Femoral joint line

1. Identify the femoral condyles by scanning transverse slices: the **two
   distal-most condylar centres**. Specifically, take the most distal slice on
   which the femur is two separate condyle components and use the **smaller
   condyle's** centroid as one point, and the most distal slice on which it is a
   single component for the other point.
2. The femoral joint line connects these two condylar points (projected to the
   coronal plane).

### 10.2 Tibial joint line

1. **Intercondylar eminence** — centroid of the first (most proximal) tibial
   slice; this divides the plateau into medial and lateral halves.
2. Restrict to points **lateral of the eminence by more than ⅔ of the
   eminence-to-lateral-edge distance** and **medial of the eminence by more than
   ⅔ of the eminence-to-medial-edge distance** (i.e. the outer parts of each
   plateau).
3. In each of those regions, take the **most proximal (highest) point** — the
   medial and lateral plateau apices.
4. The tibial joint line connects these two points (projected to the coronal
   plane).

### 10.3 Computation

The JLCA is the angle between the femoral joint line and the tibial joint line
in the coronal plane (folded to ≤90°).

**For a reader:** on the coronal view, draw the line tangent to the two distal
femoral condyles and the line tangent to the two tibial plateau surfaces, and
measure the angle at which they converge.

---

## 11. Mechanical-axis deviation (MAD)

The perpendicular distance (mm) from the **knee centre** to the **mechanical
axis (Mikulicz line)**, measured in the coronal plane.

### 11.1 Landmarks (whole-leg image)

1. **Femoral head centre** (sphere fit, §1.1).
2. **Ankle centre** — centroid of the distal tibial articulating surface
   (plafond slice, as in §5 / §9).
3. **Mikulicz line** — the line connecting the femoral head centre and the ankle
   centre.
4. **Knee centre** — centre of mass of the tibia in the knee region.

### 11.2 Computation

- All three landmarks are projected onto the coronal plane (antero-posterior
  component zeroed).
- MAD is the **perpendicular distance from the knee centre to the Mikulicz
  line**.
- **Sign:** the value is **negative for medial deviation** (knee centre medial
  to the mechanical axis) and **positive for lateral deviation**.

**For a reader:** draw the line from the femoral head centre to the centre of
the ankle, then measure the perpendicular distance from the centre of the knee
to that line; record medial deviation as negative and lateral as positive.

---

## Appendix — Summary of measurement planes

| Parameter | Plane measured in | Primary landmarks |
|-----------|-------------------|-------------------|
| Femoral torsion | Transverse | Femoral neck axis vs. posterior femoral condylar line |
| Tibial torsion | Transverse | Proximal tibial condylar line vs. distal tibia–fibula line |
| Knee rotation angle | Transverse | Posterior femoral vs. posterior tibial condylar lines |
| CCD angle | 3D + coronal projection | Femoral neck axis vs. femoral shaft axis |
| Leg length | 3D distance | Proximal vs. distal bone centroids |
| Acetabular version | Transverse | Acetabular rim line vs. perpendicular to inter-head line |
| Center-edge angle | Coronal | Vertical through head centre vs. line to lateral acetabular edge |
| Femoral offset | 3D distance (or coronal) | Femoral head centre vs. femoral shaft axis |
| Hip–knee–ankle angle | Coronal | Hip→knee vs. knee→ankle vectors |
| Joint-line convergence angle | Coronal | Femoral condylar line vs. tibial plateau line |
| Mechanical-axis deviation | Coronal | Knee centre vs. Mikulicz (hip–ankle) line |