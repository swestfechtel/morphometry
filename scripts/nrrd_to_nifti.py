import vtk
import argparse
from typing import Tuple


def read_nrrd(filename: str) -> Tuple[vtk.vtkImageData:, vtk.vtkInformation]:
    """
    Read image in nrrd format.

    :param filename: Path to the nrrd file.
    :return: vtk image and image information
    """
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info


def write_nifti(image: vtk.vtkImageData,filename: str, info: vtk.vtkInformation) -> None:
    """
    Write vtk image to nifti format.

    :param image: A vtk image object to be written.
    :param filename: Path to write the nifti file to.
    :param info: Image information.
    """
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a nrrd image to nifti format.')
    parser.add_argument('--i', type=str, help='Path to the nrrd file.')
    parser.add_argument('--o', type=str, help='Path to write the nifti file to.')
    args = parser.parse_args()
    input_path = args.i
    output_path = args.o
    image, info = read_nrrd(input_path)
    write_nifti(image, output_path, info)
