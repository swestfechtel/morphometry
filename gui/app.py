import nnunet_config
import sys
import tempfile
import os
import logging
import webbrowser
import time
import torch
import shutil
import re
import pickle

import SimpleITK as sitk
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, make_response
from pathlib import Path
from collections import defaultdict
from skimage.transform import rotate
# from pynvml import *
from threading import Thread
from distutils.dir_util import copy_tree
from utils import *
from multiprocessing import freeze_support
from examination import Examination


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
workdir = os.path.dirname(os.path.realpath(__file__))


@app.route('/')
@app.route('/<error>')
def index(error=None):
    if error:
        return render_template('index.html', error=True)
    else:
        return render_template('index.html')


@app.route('/past/', methods=['GET', 'POST'])
def past():
    if 'session_id' in request.form:
        session_id = request.form.get('session_id')
        if session_id in [f.split('.')[0] for f in os.listdir(f'{workdir}/data/')]:
            with open(f'{workdir}/data/{session_id}.pkl', 'rb') as f:
                examination = pickle.load(f)
                image_layers, mask_layers = examination.to_base64()
                torsion_values = examination.get_torsion_values()

            response = make_response(render_template('view.html', image_layers=image_layers, mask_layers=mask_layers, torsion_values=torsion_values))
            response.set_cookie('session_id', session_id)
            return response

        return redirect('/error')
    else:
        return render_template('past.html')


@app.route('/upload/', methods=['GET', 'POST'])
def select_folder():
    logger = logging.getLogger('app')
    session_id = generate_session_id()

    files = request.files.getlist('dicom')

    with tempfile.TemporaryDirectory(dir=f'{workdir}/uploads/') as temp_dir:
        # temp_dir = tempfile.mkdtemp(dir=f'{workdir}/uploads/')
        dicom_file = re.compile(r'I\d+')
        digits = re.compile(r'[1-9]')
        double_digits = re.compile(re.compile(r'[1-9]{2}'))
        single_digits = re.compile(r'I[1-9]0+$')
        multiples_of_ten = re.compile(r'I[1-9]0+1')
        for i, file in enumerate(files):  # have to re-order layers because pacs export has dumb file naming scheme
            filename = file.filename.split('/')[-1]
            logger.debug(filename)
            if re.match(dicom_file, filename) is None:
                logger.debug(f'Could not match {filename}')
                continue  # filter out files we do not want, e.g. VERSION files

            if re.match(single_digits, filename):  # layers no.s 1-9
                number = re.search(digits, filename)[0]
                file.save(f'{temp_dir}/I00{number}')

            elif re.match(multiples_of_ten, filename):  # layers no.s 10, 20, 30, ...
                number = re.search(digits, filename)[0]
                file.save(f'{temp_dir}/I0{number}0')

            else:  # everything else, e.g. 11-19, 21-29, ...
                number = re.search(double_digits, filename)[0]
                file.save(f'{temp_dir}/I0{number}')

        examination = Examination()
        examination.read_dicom_series(temp_dir)
        examination.read_dicom_metadata(temp_dir)
        examination.split_series()
        # examination.compute_segmentations()
        examination.read_segmentation('hip', '/home/simon/Downloads/hip_seg.nii.gz')
        examination.read_segmentation('knee', '/home/simon/Downloads/knee_seg.nii.gz')
        examination.read_segmentation('ankle', '/home/simon/Downloads/ankle_seg.nii.gz')
        examination.compute_torsional_alignment()
        examination.combine_masks()
        image_layers, mask_layers = examination.to_base64()
        torsion_values = examination.get_torsion_values()
        examination.save_to_pickle(f'{workdir}/data/{session_id}.pkl')

    response = make_response(render_template('view.html', image_layers=image_layers, mask_layers=mask_layers, torsion_values=torsion_values))
    response.set_cookie('session_id', session_id)

    return response


@app.route('/publications/')
def publications():
    return render_template('publications.html')


@app.route('/view/')
def view():
    pass


if __name__ == '__main__':
    init_logger('app')
    logger = logging.getLogger('app')
    create_directories_and_files()
    logger.debug(f'nnUNet results directory set to {os.environ["nnUNet_results"]}')
    app.run()
    logger.info('Application started.')
