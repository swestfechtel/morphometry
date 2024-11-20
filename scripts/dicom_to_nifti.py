import argparse
import SimpleITK as sitk


def read_dicom(filename: str) -> sitk.Image:
    """
    Read image in dicom format.

    :param filename: Path to the dicom file.
    :return: SimpleITK image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dicom_names)
    return reader.Execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a dicom image series to nifti format.')
    parser.add_argument('--i', type=str, help='Path to the directory containing the dicom series.')
    parser.add_argument('--o', type=str, help='Path to write the nifti file to.')
    args = parser.parse_args()
    input_path = args.i
    output_path = args.o
    image = read_dicom(input_path)
    sitk.WriteImage(image, output_path)
