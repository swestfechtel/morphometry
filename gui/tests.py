import nnunet_config
from examination import Examination
import SimpleITK as sitk


if __name__ == '__main__':
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
