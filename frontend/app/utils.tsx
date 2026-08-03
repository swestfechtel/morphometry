/**
 * Computes the Hadamard (element-wise) product of two arrays.
 * @param a First array of numbers
 * @param b Second array of numbers
 * @returns Array containing the element-wise products
 */
export function hadamard_product(a: number[], b: number[]) {
    const result = [];
    for (let i = 0; i < a.length; i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

/**
 * Adds two arrays element-wise.
 * @param a First array of numbers
 * @param b Second array of numbers
 * @returns Array containing the element-wise sums
 */
export function add_arrays(a: number[], b: number[]) {
    const result = [];
    for (let i = 0; i < a.length; i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * Subtracts the second array from the first array element-wise.
 * @param a First array of numbers
 * @param b Second array of numbers
 * @returns Array containing the element-wise differences
 */
export function subtract_arrays(a: number[], b: number[]) {
    const result = [];
    for (let i = 0; i < a.length; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

/**
 * Calculates the angle in degrees between two 3D vectors.
 * @param a First 3D vector [x, y, z]
 * @param b Second 3D vector [x, y, z]
 * @returns Angle in degrees between vectors a and b
 */
export function angleBetweenVectors(a: number[], b: number[]): number {
    if (a.length !== 3 || b.length !== 3) {
        throw new Error("Both vectors must be three-dimensional.");
    }

    // Calculate the dot product
    const dotProduct = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    // Calculate the magnitudes of the vectors
    const magnitudeA = Math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2);
    const magnitudeB = Math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2);

    // Ensure magnitudes are not zero to avoid division by zero
    if (magnitudeA === 0 || magnitudeB === 0) {
        // throw new Error("Vectors must not have zero magnitude.");
        return 0;
    }

    // Calculate the angle in radians
    const angleInRadians = Math.acos(dotProduct / (magnitudeA * magnitudeB));

    // Convert radians to degrees
    return (angleInRadians * 180) / Math.PI;
}

/**
 * Signed angle (degrees) of the femoral neck (proximal) reference line, measured in the
 * axial plane relative to the medio-lateral axis. This is the proximal component of the
 * femoral torsion; the total torsion is this minus {@link femoralDistalAngle}.
 * @param proximal_start Femoral head centre [x, y, z]
 * @param proximal_end Femoral neck centre [x, y, z]
 * @param side 'left' or 'right' (image side, not patient)
 * @returns Signed proximal angle in degrees.
 */
export function femoralProximalAngle(proximal_start: number[], proximal_end: number[], side: string): number {
    const proximal_vector = subtract_arrays(proximal_end, proximal_start);
    let proximal_angle = angleBetweenVectors(proximal_vector, side == 'right' ? [-1, 0, 0] : [1, 0, 0]);
    if (proximal_angle > 90)
        proximal_angle = 180 - proximal_angle;
    if (proximal_end[1] - proximal_start[1] < 0)
        proximal_angle = -proximal_angle;
    return proximal_angle;
}

/**
 * Signed angle (degrees) of the posterior condylar (distal) femoral reference line in
 * the axial plane. Distal component of the femoral torsion.
 * @param distal_start One posterior condyle point [x, y, z]
 * @param distal_end The other posterior condyle point [x, y, z]
 * @param side 'left' or 'right' (image side, not patient)
 * @returns Signed distal angle in degrees.
 */
export function femoralDistalAngle(distal_start: number[], distal_end: number[], side: string): number {
    let knee_start = distal_start;
    let knee_end = distal_end;
    if (knee_start[0] < knee_end[0]) {
        const tmp = knee_start;
        knee_start = knee_end;
        knee_end = tmp;
    }
    const distal_vector = subtract_arrays(knee_end, knee_start);
    let distal_angle = angleBetweenVectors(distal_vector, [-1, 0, 0]);
    if (distal_angle > 90)
        distal_angle = 180 - distal_angle;
    const distal_orientation = distal_end[1] - distal_start[1];
    if (side == 'right') {
        if (distal_orientation < 0)
            distal_angle = -distal_angle;
    }
    else {
        if (distal_orientation > 0)
            distal_angle = -distal_angle;
    }
    return distal_angle;
}

/**
 * Computes the femoral torsion angle based on proximal and distal landmarks.
 * @param proximal_start Start point of the proximal vector [x, y, z]
 * @param proximal_end End point of the proximal vector [x, y, z]
 * @param distal_start Start point of the distal vector [x, y, z]
 * @param distal_end End point of the distal vector [x, y, z]
 * @param side 'left' or 'right' indicating the side of the body
 * @returns Femoral torsion angle as a string with one decimal place
 */
export function computeFemoralTorsion(proximal_start: number[], proximal_end: number[], distal_start: number[], distal_end: number[], side: string): string {
    const proximal_angle = femoralProximalAngle(proximal_start, proximal_end, side);
    const distal_angle = femoralDistalAngle(distal_start, distal_end, side);
    return (proximal_angle - distal_angle).toFixed(1);
}

/**
 * Signed angle (degrees) of the posterior tibial condylar (proximal) reference line in
 * the axial plane. Proximal component of the tibial torsion.
 * @param proximal_start One posterior tibial condyle point [x, y, z]
 * @param proximal_end The other posterior tibial condyle point [x, y, z]
 * @param side 'left' or 'right' (image side, not patient)
 * @returns Signed proximal angle in degrees.
 */
export function tibialProximalAngle(proximal_start: number[], proximal_end: number[], side: string): number {
    let knee_start = proximal_start;
    let knee_end = proximal_end;
    if (knee_start[0] < knee_end[0]) {
        const tmp = knee_start;
        knee_start = knee_end;
        knee_end = tmp;
    }
    const proximal_vector = subtract_arrays(knee_end, knee_start);
    let proximal_angle = angleBetweenVectors(proximal_vector, [-1, 0, 0]);
    if (proximal_angle > 90)
        proximal_angle = 180 - proximal_angle;
    const proximal_orientation = proximal_end[1] - proximal_start[1];
    if (side == 'right') {
        if (proximal_orientation < 0)
            proximal_angle = -proximal_angle;
    }
    else {
        if (proximal_orientation > 0)
            proximal_angle = -proximal_angle;
    }
    return proximal_angle;
}

/**
 * Signed angle (degrees) of the transmalleolar (distal) tibial reference line in the
 * axial plane. Distal component of the tibial torsion.
 * @param distal_start Medial malleolus [x, y, z]
 * @param distal_end Lateral malleolus [x, y, z]
 * @param side 'left' or 'right' (image side, not patient)
 * @returns Signed distal angle in degrees.
 */
export function tibialDistalAngle(distal_start: number[], distal_end: number[], side: string): number {
    const distal_vector = subtract_arrays(distal_end, distal_start);
    let distal_angle = angleBetweenVectors(distal_vector, side == 'right' ? [-1, 0, 0] : [1, 0, 0]);
    if (distal_angle > 90)
        distal_angle = 180 - distal_angle;
    const distal_orientation = distal_end[1] - distal_start[1];
    if (distal_orientation < 0)
        distal_angle = -distal_angle;
    return distal_angle;
}

/**
 * Computes the tibial torsion angle based on proximal and distal landmarks.
 * @param proximal_start Start point of the proximal vector [x, y, z]
 * @param proximal_end End point of the proximal vector [x, y, z]
 * @param distal_start Start point of the distal vector [x, y, z]
 * @param distal_end End point of the distal vector [x, y, z]
 * @param side 'left' or 'right' indicating the side of the body
 * @returns Tibial torsion angle as a string with one decimal place
 */
export function computeTibialTorsion(proximal_start: number[], proximal_end: number[], distal_start: number[], distal_end: number[], side: string): string {
    const proximal_angle = tibialProximalAngle(proximal_start, proximal_end, side);
    const distal_angle = tibialDistalAngle(distal_start, distal_end, side);
    return (distal_angle - proximal_angle).toFixed(1);
}