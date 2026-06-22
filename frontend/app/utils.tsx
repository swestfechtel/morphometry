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
 * Calculates the angle in degrees between two 2D vectors.
 * @param a First 2D vector [x, y]
 * @param b Second 2D vector [x, y]
 * @returns Angle in degrees between vectors a and b
 */
export function angleBetweenVectors2D(a: number[], b: number[]): number {
    if (a.length !== 2 || b.length !== 2) {
        throw new Error("Both vectors must be two-dimensional.");
    }

    // Calculate the dot product
    const dotProduct = a[0] * b[0] + a[1] * b[1];

    // Calculate the magnitudes of the vectors
    const magnitudeA = Math.sqrt(a[0] ** 2 + a[1] ** 2);
    const magnitudeB = Math.sqrt(b[0] ** 2 + b[1] ** 2);

    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }

    // Clamp value to avoid NaN due to floating point errors
    const cosTheta = Math.max(-1, Math.min(1, dotProduct / (magnitudeA * magnitudeB)));
    const angleInRadians = Math.acos(cosTheta);

    return (angleInRadians * 180) / Math.PI;
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
    const proximal_vector = subtract_arrays(proximal_end, proximal_start);

    let knee_start = distal_start;
    let knee_end = distal_end;

    if (knee_start[0] < knee_end[0]) {
        let tmp = knee_start;
        knee_start = knee_end;
        knee_end = tmp;
    }

    const distal_vector = subtract_arrays(knee_end, knee_start);

    let proximal_angle = angleBetweenVectors(proximal_vector, side == 'right' ? [-1, 0, 0] : [1, 0, 0]);
    let distal_angle = angleBetweenVectors(distal_vector, [-1, 0, 0]);

    if (proximal_angle > 90)
        proximal_angle = 180 - proximal_angle;

    if (distal_angle > 90)
        distal_angle = 180 - distal_angle;

    let proximal_orientation = proximal_end[1] - proximal_start[1];
    if (proximal_orientation < 0)
        proximal_angle = -proximal_angle;

    let distal_orientation = distal_end[1] - distal_start[1];
    if (side == 'right') {
        if (distal_orientation < 0)
            distal_angle = -distal_angle;
    }
    else {
        if (distal_orientation > 0)
            distal_angle = -distal_angle;
    }

    return (proximal_angle - distal_angle).toFixed(1);
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
    let knee_start = proximal_start;
    let knee_end = proximal_end;

    if (knee_start[0] < knee_end[0]) {
        let tmp = knee_start;
        knee_start = knee_end;
        knee_end = tmp;
    }

    const proximal_vector = subtract_arrays(knee_end, knee_start);
    const distal_vector = subtract_arrays(distal_end, distal_start);

    let proximal_angle = angleBetweenVectors(proximal_vector, [-1, 0, 0])
    let distal_angle = angleBetweenVectors(distal_vector, side == 'right' ? [-1, 0, 0] : [1, 0, 0]);

    if (proximal_angle > 90)
        proximal_angle = 180 - proximal_angle;

    if (distal_angle > 90)
        distal_angle = 180 - distal_angle;

    let proximal_orientation = proximal_end[1] - proximal_start[1];

    if (side == 'right') {
        if (proximal_orientation < 0)
            proximal_angle = -proximal_angle;
    }
    else {
        if (proximal_orientation > 0)
            proximal_angle = -proximal_angle;
    }

    let distal_orientation = distal_end[1] - distal_start[1];
    if (distal_orientation < 0) {
        distal_angle = -distal_angle;
    }

    return (distal_angle - proximal_angle).toFixed(1);
}

export function computeHalluxValgusAngle(landmarks: number[][]): string {
    const longitudinal_first_metatarsal_axis = subtract_arrays(landmarks[16], landmarks[17]);
    const longitudinal_phalanx_axis = subtract_arrays(landmarks[32], landmarks[33]);

    const angle = angleBetweenVectors2D(longitudinal_first_metatarsal_axis, longitudinal_phalanx_axis);
    return angle.toFixed(1);
}

export function vectorLength(vector: number[]): number {
    return Math.sqrt(vector.reduce((sum, component) => sum + component * component, 0));
}