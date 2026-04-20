import pydicom
import os
import numpy as np
from pathlib import Path
import SimpleITK as sitk


def combine_dicom_series(torso_dir, lower_ext_dir, output_dir, z_tolerance=0.5):
    """
    Combine two DICOM series, removing overlapping slices.
    
    z_tolerance: slices within this distance (mm) are considered duplicates
    """
    
    def load_series(directory):
        slices = []
        for f in Path(directory).iterdir():
            ds = pydicom.dcmread(f)
            z_pos = float(ds.ImagePositionPatient[2])
            slices.append({'z': z_pos, 'ds': ds, 'path': f})
        return slices
    
    torso_slices = load_series(torso_dir)
    lower_slices = load_series(lower_ext_dir)
    
    # Find overlapping region and decide which series to prioritize
    torso_z = [s['z'] for s in torso_slices]
    lower_z = [s['z'] for s in lower_slices]
    
    torso_min, torso_max = min(torso_z), max(torso_z)
    lower_min, lower_max = min(lower_z), max(lower_z)
    
    print(f"Torso z-range: {torso_min:.1f} to {torso_max:.1f} mm")
    print(f"Lower ext z-range: {lower_min:.1f} to {lower_max:.1f} mm")
    
    # Combine all slices
    all_slices = torso_slices + lower_slices
    all_slices.sort(key=lambda x: x['z'])
    
    # Remove duplicates: keep first occurrence (or choose by series preference)
    filtered_slices = []
    for s in all_slices:
        if not filtered_slices:
            filtered_slices.append(s)
            continue
        
        # Check if this slice is too close to the previous one
        if abs(s['z'] - filtered_slices[-1]['z']) > z_tolerance:
            filtered_slices.append(s)
        else:
            # Duplicate detected - keep the one from preferred series
            # Here we keep the existing one (first encountered)
            print(f"Removing duplicate slice at z={s['z']:.1f} mm")
    
    print(f"\nOriginal: {len(all_slices)} slices")
    print(f"After dedup: {len(filtered_slices)} slices")
    
    # Save combined series
    new_series_uid = pydicom.uid.generate_uid()
    os.makedirs(output_dir, exist_ok=True)
    
    for i, s in enumerate(filtered_slices):
        ds = s['ds']
        ds.SeriesInstanceUID = new_series_uid
        ds.InstanceNumber = i + 1
        ds.SeriesDescription = "Combined Torso + Lower Extremities"
        
        output_path = Path(output_dir) / f"slice_{i:04d}.dcm"
        ds.save_as(output_path)
    
    return filtered_slices
    
    
def combine_dicom_series_with_alignment(torso_dir, lower_ext_dir, output_dir, z_tolerance=0.5):
    """
    Combine DICOM series respecting ImagePositionPatient for proper alignment.
    """
    
    def load_series(directory):
        slices = []
        for f in Path(directory).iterdir():
            ds = pydicom.dcmread(f)
            ipp = [float(x) for x in ds.ImagePositionPatient]  # [x, y, z]
            slices.append({
                'x': ipp[0],
                'y': ipp[1],  # This is typically the coronal (anterior-posterior) axis
                'z': ipp[2],
                'ds': ds,
                'path': f
            })
        return slices
    
    torso_slices = load_series(torso_dir)
    lower_slices = load_series(lower_ext_dir)
    
    # Analyze the offset between series
    torso_y = np.mean([s['y'] for s in torso_slices])
    lower_y = np.mean([s['y'] for s in lower_slices])
    y_offset = torso_y - lower_y
    
    torso_x = np.mean([s['x'] for s in torso_slices])
    lower_x = np.mean([s['x'] for s in lower_slices])
    x_offset = torso_x - lower_x
    
    torso_pixel_spacing = torso_slices[0]['ds'].PixelSpacing
    torso_frame_of_reference_uid = torso_slices[0]['ds'].FrameOfReferenceUID
    
    print(f"Torso mean IPP (x, y): ({torso_x:.2f}, {torso_y:.2f})")
    print(f"Lower mean IPP (x, y): ({lower_x:.2f}, {lower_y:.2f})")
    print(f"Offset (x, y): ({x_offset:.2f}, {y_offset:.2f}) mm")
    
    # Option 1: Trust the IPP values (3D Slicer's approach)
    # Just ensure they're preserved - no modification needed
    # The viewer should place slices correctly in 3D space
    
    # Option 2: Align to a common reference (e.g., align lower to torso)
    # This modifies the IPP of one series to match the other
    
    align_to_torso = True  # Set to False to keep original IPP
    
    if align_to_torso and (abs(x_offset) > 0.1 or abs(y_offset) > 0.1):
        print(f"\nAligning lower extremities to torso reference frame...")
        for s in lower_slices:
            s['ds'].ImagePositionPatient[0] = str(s['x'] + x_offset)
            s['ds'].ImagePositionPatient[1] = str(s['y'] + y_offset)
            s['x'] += x_offset
            s['y'] += y_offset
            s['ds'].PixelSpacing = torso_pixel_spacing
            s['ds'].FrameOfReferenceUID = torso_frame_of_reference_uid
    
    # Combine and sort by z
    all_slices = torso_slices + lower_slices
    all_slices.sort(key=lambda x: x['z'])
    
    # Remove duplicates
    filtered_slices = []
    for s in all_slices:
        if not filtered_slices or abs(s['z'] - filtered_slices[-1]['z']) > z_tolerance:
            filtered_slices.append(s)
        else:
            print(f"Removing duplicate at z={s['z']:.1f}")
    
    # Save
    new_series_uid = pydicom.uid.generate_uid()
    os.makedirs(output_dir, exist_ok=True)
    
    for i, s in enumerate(filtered_slices):
        ds = s['ds']
        ds.SeriesInstanceUID = new_series_uid
        ds.InstanceNumber = i + 1
        ds.SeriesDescription = "Combined Torso + Lower Extremities (Aligned)"
        
        output_path = Path(output_dir) / f"slice_{i:04d}.dcm"
        ds.save_as(output_path)
    
    print(f"\nSaved {len(filtered_slices)} slices to {output_dir}")
    return filtered_slices


def combine_dicom_series_with_geometry_preservation(torso_dir, lower_ext_dir, output_dir, z_tolerance=0.5):
    """
    Combine DICOM series by resampling both onto a common uniform grid.

    This approach properly handles x/y offsets between series by:
    1. Loading both series as SimpleITK volumes with proper geometry
    2. Defining a common reference grid (using torso x/y alignment, encompassing both z-ranges)
    3. Resampling BOTH volumes onto this common grid
    4. Merging the resampled volumes using maximum intensity
    5. Writing the combined volume as a regular DICOM series

    Args:
        torso_dir: Path to torso DICOM series directory
        lower_ext_dir: Path to lower extremity DICOM series directory
        output_dir: Path to save combined series
        z_tolerance: Not used (kept for API compatibility)
    """

    print("Loading DICOM series with SimpleITK...")

    # Load series as SimpleITK images
    reader = sitk.ImageSeriesReader()

    # Load torso series
    torso_files = reader.GetGDCMSeriesFileNames(torso_dir)
    reader.SetFileNames(torso_files)
    torso_volume = reader.Execute()

    # Load lower extremity series
    lower_files = reader.GetGDCMSeriesFileNames(lower_ext_dir)
    reader.SetFileNames(lower_files)
    lower_volume = reader.Execute()

    # Get geometry information
    torso_origin = np.array(torso_volume.GetOrigin())
    lower_origin = np.array(lower_volume.GetOrigin())
    torso_spacing = np.array(torso_volume.GetSpacing())
    lower_spacing = np.array(lower_volume.GetSpacing())
    torso_size = np.array(torso_volume.GetSize())
    lower_size = np.array(lower_volume.GetSize())

    offset = lower_origin - torso_origin

    print(f"\nGeometry Analysis:")
    print(f"Torso origin (x,y,z): {torso_origin}")
    print(f"Lower origin (x,y,z): {lower_origin}")
    print(f"Offset (x,y,z): {offset} mm")
    print(f"Torso spacing: {torso_spacing}")
    print(f"Lower spacing: {lower_spacing}")
    print(f"Torso size: {torso_size}")
    print(f"Lower size: {lower_size}")

    # Define common reference grid - use torso's x/y alignment and encompass both z-ranges
    reference_spacing = torso_spacing
    reference_direction = torso_volume.GetDirection()
    reference_origin = torso_origin.copy()

    # Calculate z-extent to include both volumes
    torso_z_max = torso_origin[2] + (torso_size[2] - 1) * torso_spacing[2]
    lower_z_max = lower_origin[2] + (lower_size[2] - 1) * lower_spacing[2]
    combined_z_min = min(torso_origin[2], lower_origin[2])
    combined_z_max = max(torso_z_max, lower_z_max)

    # Set reference origin to start at the minimum z
    reference_origin[2] = combined_z_min

    # Calculate combined size
    reference_size = torso_size.copy()
    reference_size[2] = int(np.ceil((combined_z_max - combined_z_min) / reference_spacing[2])) + 1

    print(f"\nReference Grid:")
    print(f"Origin: {reference_origin}")
    print(f"Spacing: {reference_spacing}")
    print(f"Size: {reference_size}")

    # Resample BOTH volumes onto the reference grid
    print("\nResampling both volumes onto common grid...")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_spacing.tolist())
    resampler.SetSize(reference_size.astype(int).tolist())
    resampler.SetOutputOrigin(reference_origin.tolist())
    resampler.SetOutputDirection(reference_direction)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)  # Air HU for CT

    torso_resampled = resampler.Execute(torso_volume)
    lower_resampled = resampler.Execute(lower_volume)

    print(f"Resampled torso: {torso_resampled.GetSize()}")
    print(f"Resampled lower: {lower_resampled.GetSize()}")

    # Combine by taking maximum intensity (handles overlap and avoids air gaps)
    combined_volume = sitk.Maximum(torso_resampled, lower_resampled)

    print(f"\nCombined volume: {combined_volume.GetSize()}")
    print(f"Origin: {combined_volume.GetOrigin()}")
    print(f"Spacing: {combined_volume.GetSpacing()}")

    # Write combined volume as DICOM series using pydicom
    print(f"\nWriting DICOM series to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Read reference metadata
    reference_ds = pydicom.dcmread(torso_files[0])

    # Generate UIDs
    series_uid = pydicom.uid.generate_uid()
    study_uid = reference_ds.StudyInstanceUID if hasattr(reference_ds, 'StudyInstanceUID') else pydicom.uid.generate_uid()

    print(f"Study Instance UID: {study_uid}")
    print(f"Series Instance UID: {series_uid}")

    # Convert to numpy array for slice-by-slice writing
    combined_array = sitk.GetArrayFromImage(combined_volume)  # Shape: (slices, rows, cols)

    # Write each slice
    for i in range(combined_volume.GetSize()[2]):
        # Start with a copy of reference dataset
        ds = pydicom.dcmread(torso_files[0])

        # Update pixel data
        slice_array = combined_array[i, :, :]

        # Preserve the original pixel representation and rescale parameters
        # IMPORTANT: The combined_array from SimpleITK is already in HU (real values)
        # We need to convert back to stored pixel values using RescaleSlope and RescaleIntercept
        # Formula: StoredValue = (HU - RescaleIntercept) / RescaleSlope

        rescale_intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
        rescale_slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0

        # Convert HU values back to stored pixel values
        stored_array = (slice_array - rescale_intercept) / rescale_slope

        # Round and clip to the appropriate data type range
        if ds.pixel_array.dtype == np.int16:
            stored_array = np.clip(stored_array, -32768, 32767).astype(np.int16)
        elif ds.pixel_array.dtype == np.uint16:
            stored_array = np.clip(stored_array, 0, 65535).astype(np.uint16)
        else:
            stored_array = stored_array.astype(ds.pixel_array.dtype)

        ds.PixelData = stored_array.tobytes()

        # Update dimensions
        ds.Rows, ds.Columns = slice_array.shape

        # Ensure RescaleSlope and RescaleIntercept are preserved
        if not hasattr(ds, 'RescaleIntercept'):
            ds.RescaleIntercept = 0
        if not hasattr(ds, 'RescaleSlope'):
            ds.RescaleSlope = 1

        # Set UIDs (same for all slices except SOP Instance UID)
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = pydicom.uid.generate_uid()

        # Set instance number
        ds.InstanceNumber = i + 1

        # Set series description
        ds.SeriesDescription = "Combined Torso + Lower Extremities (Resampled)"

        # Set geometry from combined volume
        ipp = combined_volume.TransformIndexToPhysicalPoint((0, 0, i))
        ds.ImagePositionPatient = [float(ipp[0]), float(ipp[1]), float(ipp[2])]

        direction = combined_volume.GetDirection()
        ds.ImageOrientationPatient = [float(direction[0]), float(direction[1]), float(direction[2]),
                                       float(direction[3]), float(direction[4]), float(direction[5])]

        spacing = combined_volume.GetSpacing()
        ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
        ds.SliceThickness = float(spacing[2])

        ds.SliceLocation = float(ipp[2])

        # Save
        output_path = os.path.join(output_dir, f'slice_{i:04d}.dcm')
        ds.save_as(output_path)

    print(f"Successfully wrote {combined_volume.GetSize()[2]} DICOM slices")
    print(f"All slices share StudyInstanceUID: {study_uid}")
    print(f"All slices share SeriesInstanceUID: {series_uid}")

    return combined_volume


if __name__ == '__main__':
    combine_dicom_series_with_geometry_preservation('/home/simon/Data/NMDID/case-100010/omi/incomingdir/case-100010/STANDARD_HEAD-NECK-U-EXT/BONE_TORSO_3_X_3',
                         '/home/simon/Data/NMDID/case-100010/omi/incomingdir/case-100010/STANDARD_HEAD-NECK-U-EXT/BONE_L-EXT_3X3',
                         './test_combined')
