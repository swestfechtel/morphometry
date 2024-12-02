import argparse
import SimpleITK as sitk


def read_dicom_series(directory: str) -> sitk.Image:
    """
    Read image from dicom directory (i.e. series consisting of multiple files).

    :param directory: Path to the dicom directory.
    :return: SimpleITK image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def read_dicom_image(filename: str) -> sitk.Image:
    """
    Read image from a single dicom file.

    :param filename: Path to the dicom image.
    :return: SimpleITK image
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    return reader.Execute()


def dicom_to_nifti(input_path: str, output_path: str, single_file: bool):
    """
    Convert a dicom image series to nifti format.

    :param input_path: Path to the directory containing the dicom series.
    :param output_path: Path to write the nifti file to.
    :param single_file: Whether the input is a single dicom file.
    """
    image = read_dicom_image(input_path) if single_file else read_dicom_series(input_path)
    sitk.WriteImage(image, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a dicom image series to nifti format.')
    parser.add_argument('-i', type=str, help='Path to the directory containing the dicom series.')
    parser.add_argument('-o', type=str, help='Path to write the nifti file to.')
    parser.add_argument('-s', '--single_file', action='store_true', help='Whether the input is a single dicom file.')
    args = parser.parse_args()
    dicom_to_nifti(args.i, args.o, args.single_file)
