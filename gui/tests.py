import nnunet_config
from examination import Examination
import SimpleITK as sitk
import multiprocessing
import numpy as np


def test_1():
    exam = Examination()
    exam.read_dicom_series('./uploads/tmpp_h2mp4f')
    exam.read_dicom_metadata('./uploads/tmpp_h2mp4f')
    exam.split_series()
    exam.compute_segmentations()
    """
    sitk.WriteImage(exam.hip, '/home/simon/Downloads/hip.nii.gz')
    sitk.WriteImage(exam.knee, '/home/simon/Downloads/knee.nii.gz')
    sitk.WriteImage(exam.ankle, '/home/simon/Downloads/ankle.nii.gz')
    exam.write_segmentation('hip', '/home/simon/Downloads/hip_seg.nii.gz')
    exam.write_segmentation('knee', '/home/simon/Downloads/knee_seg.nii.gz')
    exam.write_segmentation('ankle', '/home/simon/Downloads/ankle_seg.nii.gz')
    """
    exam.compute_torsional_alignment()


def test_2_f(x):
    return x ** 2


def test_2():
    pool = multiprocessing.Pool(4)
    results = pool.map(test_2_f, range(10))
    print(results)
    pool.close()
    pool.join()


def test_3(array: np.ndarray):
    array = array * 2


def test_4():
    exam = Examination()
    exam.read_segmentation('/home/simon/Downloads/hip_seg.nii.gz', 'hip')
    exam.read_segmentation('/home/simon/Downloads/knee_seg.nii.gz', 'knee')
    exam.read_segmentation('/home/simon/Downloads/ankle_seg.nii.gz', 'ankle')
    exam.compute_torsional_alignment()
    exam.combine_masks()


if __name__ == '__main__':
    test_4()
