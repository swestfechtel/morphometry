import sys

sys.path += ['./rotation_prediction/rotation', './rotation_prediction/nnunet_rotation']

from flask import Flask, render_template, request, redirect, url_for, make_response
from pathlib import Path
from collections import defaultdict
from skimage.transform import rotate
from pynvml import *
from threading import Thread
from distutils.dir_util import copy_tree
from utils import *
from multiprocessing import freeze_support

import SimpleITK as sitk
import numpy as np

import os
import re
import logging
import webbrowser
import time
import torch
import shutil
import re


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'


def open_gui():
    webbrowser.open_new_tab('http://127.0.0.1:5000')


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
        if session_id in map(lambda x: x.name, list(os.scandir('output'))):
            response = make_response(redirect('/view/hip'))
            response.set_cookie('session_id', session_id)
            return response
        else:
            return redirect('/error')
    else:
        return render_template('past.html')


@app.route('/upload/', methods=['GET', 'POST'])
def select_folder():
    logger = logging.getLogger('app')
    session_id = generate_session_id()
    clear_temp(session_id)
    create_directories(session_id)
    files = request.files.getlist('dicom')

    for i, file in enumerate(files):
        try:
            # m = re.match('(.+\/)+((IM_?\d+)|(MR_?\d+))(?:\.dcm)?', file.filename).groups()
            # m = re.match('(?:.+\/)+(\w+(\.dcm)?)', file.filename).groups()
            m = re.match('(.+\/)+(.*)', file.filename).groups()
        except:
            continue

        if len(m) < 2:
            logger.debug(f'Could not match {file.filename}')
            continue

        path = m[0]
        filename = m[1]

        Path(f'temp/{session_id}/preview/{path}').mkdir(parents=True, exist_ok=True)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}', 'preview', path, filename))

    series = list()
    get_series(Path(f'temp/{session_id}/preview/'), series)

    response = make_response(render_template('preview.html', serieses=series, session_id=session_id))
    response.set_cookie('session_id', session_id)

    return response


@app.route('/preview/', methods=['GET', 'POST'])
def preview():
    logger = logging.getLogger('app')

    logger.debug(request.form.keys())
    session_id = get_session_id()

    if len(list(request.form.keys())) == 1:
        copy_tree(list(request.form.keys())[0], f'temp/{session_id}/dicom/hip_tmp')
        shutil.rmtree(f'temp/{session_id}/dicom/hip')
        Path(f'temp/{session_id}/dicom/hip').mkdir(parents=True, exist_ok=True)
        files = sorted([x.name for x in os.scandir(f'temp/{session_id}/dicom/hip_tmp')])
        logger.debug(f'num files in hip_tmp: {len(files)}')
        logger.debug(f'files in hip_tmp: {files}')
        for i, file in enumerate(files):
            if 0 <= i <= 28:
                logger.debug(f'move file {file} to hip')
                shutil.copy(f'temp/{session_id}/dicom/hip_tmp/{file}', f'temp/{session_id}/dicom/hip/{file}')
            elif 29 <= i <= 57:
                logger.debug(f'move file {file} to knee')
                shutil.copy(f'temp/{session_id}/dicom/hip_tmp/{file}', f'temp/{session_id}/dicom/knee/{file}')
            else:
                logger.debug(f'move file {file} to ankle')
                shutil.copy(f'temp/{session_id}/dicom/hip_tmp/{file}', f'temp/{session_id}/dicom/ankle/{file}')

        shutil.rmtree(f'temp/{session_id}/dicom/hip_tmp')
        Path(f'temp/{session_id}/dicom/hip_tmp').mkdir(parents=True, exist_ok=True)

    else:
        copy_tree(list(request.form.keys())[0], f'temp/{session_id}/dicom/hip')
        copy_tree(list(request.form.keys())[1], f'temp/{session_id}/dicom/knee')
        copy_tree(list(request.form.keys())[2], f'temp/{session_id}/dicom/ankle')

    for i, file in enumerate([x.name for x in os.scandir(f'temp/{session_id}/dicom/hip')]):
        if i == 1:
            shutil.copy(f'temp/{session_id}/dicom/hip/{file}', f'output/{session_id}/meta.dcm')
            break

    if len(list(request.form.keys())) == 1:
        current_stack_okay, voxel_list = is_current_stack_ok()
        hip_background_voxels, knee_background_voxels, ankle_background_voxels = voxel_list
        if ankle_background_voxels > knee_background_voxels > hip_background_voxels:
            logger.info('Dicom stack order ok.')
        else:
            logger.info('Reordering dicom stacks...')
            lookup = {0: 'hip', 1: 'knee', 2: 'ankle'}
            background_voxels = np.array([hip_background_voxels, knee_background_voxels, ankle_background_voxels])
            ankle_index = np.argmax(background_voxels)
            hip_index = np.argmin(background_voxels)
            knee_index = [x for x in [0, 1, 2] if x not in (ankle_index, hip_index)][0]
            copy_tree(f'temp/{session_id}/dicom/hip', f'temp/{session_id}/dicom/hip_tmp')
            copy_tree(f'temp/{session_id}/dicom/knee', f'temp/{session_id}/dicom/knee_tmp')
            copy_tree(f'temp/{session_id}/dicom/ankle', f'temp/{session_id}/dicom/ankle_tmp')
            shutil.rmtree(f'temp/{session_id}/dicom/hip')
            shutil.rmtree(f'temp/{session_id}/dicom/knee')
            shutil.rmtree(f'temp/{session_id}/dicom/ankle')
            copy_tree(f'temp/{session_id}/dicom/{lookup[hip_index]}_tmp', f'temp/{session_id}/dicom/hip')
            copy_tree(f'temp/{session_id}/dicom/{lookup[knee_index]}_tmp', f'temp/{session_id}/dicom/knee')
            copy_tree(f'temp/{session_id}/dicom/{lookup[ankle_index]}_tmp', f'temp/{session_id}/dicom/ankle')

        spacing = '3.6,0.625,0.625'

        use_gpu = False

        if torch.cuda.is_available():
            use_gpu = True

            logger.log(level=logging.INFO, msg='Using GPU for computation')

            torch.cuda.empty_cache()

        compute_segmentations(spacing=spacing, use_gpu=use_gpu)

        if use_gpu:
            torch.cuda.empty_cache()

        logger.debug(f'Allocated memory: {torch.cuda.memory_allocated()}')
        logger.debug(f'Cached memory: {torch.cuda.memory_cached()}')
        cache_images()
        # clear_temp()
        return redirect('/view/hip')

    # res, tmp = is_current_stack_ok()
    # hip_background_voxels, knee_background_voxels, ankle_background_voxels = tmp
    l = list()

    filename = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(f'temp/{session_id}/dicom/hip')[1]
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tmp = re.sub('\s+', ' ', reader.GetMetaData('0008|103e').replace('\udce4', 'ä').replace('\udcfc', 'ü').strip(' '))
    l.append('dummy_hip' if tmp == '' else tmp)

    filename = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(f'temp/{session_id}/dicom/knee')[1]
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tmp = re.sub('\s+', ' ', reader.GetMetaData('0008|103e').replace('\udce4', 'ä').replace('\udcfc', 'ü').strip(' '))
    l.append('dummy_knee' if tmp == '' else tmp)

    filename = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(f'temp/{session_id}/dicom/ankle')[1]
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tmp = re.sub('\s+', ' ', reader.GetMetaData('0008|103e').replace('\udce4', 'ä').replace('\udcfc', 'ü').strip(' '))
    l.append('dummy_ankle' if tmp == '' else tmp)

    return render_template('sort.html', descs=l, session_id=session_id)


@app.route('/sort/', methods=['GET', 'POST'])
def sort():
    logger = logging.getLogger('app')
    session_id = get_session_id()
    dirs = list(request.form.values())

    logger.debug(dirs)

    copy_tree(dirs[0], f'temp/{session_id}/dicom/hip_tmp')
    copy_tree(dirs[1], f'temp/{session_id}/dicom/knee_tmp')
    copy_tree(dirs[2], f'temp/{session_id}/dicom/ankle_tmp')

    logger.debug(f'moved {dirs[0]} to hip')
    logger.debug(f'moved {dirs[1]} to knee')
    logger.debug(f'moved {dirs[2]} to ankle')

    shutil.rmtree(f'temp/{session_id}/dicom/hip')
    shutil.rmtree(f'temp/{session_id}/dicom/knee')
    shutil.rmtree(f'temp/{session_id}/dicom/ankle')

    Path(f'temp/{session_id}/dicom/hip').mkdir(parents=True, exist_ok=True)
    Path(f'temp/{session_id}/dicom/knee').mkdir(parents=True, exist_ok=True)
    Path(f'temp/{session_id}/dicom/ankle').mkdir(parents=True, exist_ok=True)

    copy_tree(f'temp/{session_id}/dicom/hip_tmp', f'temp/{session_id}/dicom/hip')
    copy_tree(f'temp/{session_id}/dicom/knee_tmp', f'temp/{session_id}/dicom/knee')
    copy_tree(f'temp/{session_id}/dicom/ankle_tmp', f'temp/{session_id}/dicom/ankle')

    spacing = '3.6,0.625,0.625'

    # use_gpu = True if request.form.get('gpu') else False
    use_gpu = False

    if torch.cuda.is_available():
        if torch.version.hip:
            try:
                gpu = pyamdgpuinfo.get_gpu(0)
                vram = gpu.memory_info['vram_size']

                if vram / (1024 * 1024) >= 8000:
                    use_gpu = True
                    logger.log(level=logging.INFO, msg='Using GPU for computation')
            except Exception:
                logger.error(msg='Could not evaluate VRAM due to unknown error')
        else:
            try:
                nvmlInit()
                h = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(h)

                if info.total / (1024 * 1024) >= 8000:  # convert B into MB
                    use_gpu = True

                    logger.log(level=logging.INFO, msg='Using GPU for computation')
            except (FileNotFoundError, nvml.NVMLError):
                logger.error(msg='Could not evaluate VRAM due to missing files')

    if use_gpu:
        torch.cuda.empty_cache()

    compute_segmentations(spacing=spacing, use_gpu=use_gpu)

    if use_gpu:
        torch.cuda.empty_cache()

    logger.debug(f'Allocated memory: {torch.cuda.memory_allocated()}')
    logger.debug(f'Cached memory: {torch.cuda.memory_cached()}')
    cache_images()
    # clear_temp()
    return redirect('/view/hip')


@app.route('/publications/')
def publications():
    return render_template('publications.html')


@app.route('/view/')
@app.route('/view/<bodypart>')
@app.route('/view/<bodypart>/<reference>')
def view(bodypart=None, reference=None):
    session_id = get_session_id()
    dirs = [x.name for x in os.scandir(f'output/{session_id}') if os.path.isdir(x)]
    lookup_table = {
        'hip': 'Hüfte',
        'knee': 'Knie',
        'ankle': 'Knöchel'
    }
    bparts = [lookup_table[x] for x in dirs]
    session_id = get_session_id()
    if bodypart is None:
        return render_template('view.html', dirs=dirs, bparts=bparts)
    else:
        images = defaultdict()
        if reference is None:
            if bodypart == 'hip':
                reference = 'lee'
                # images['left'] = load_cache(bodypart, 'left', reference_method='lee')
                # images['right'] = load_cache(bodypart, 'right', reference_method='lee')
            elif bodypart == 'knee':
                reference = None
                # images['left'] = load_cache(bodypart, 'left', reference_method=None)
                # images['right'] = load_cache(bodypart, 'right', reference_method=None)
            elif bodypart == 'ankle':
                reference = 'ellipse'
                # images['left'] = load_cache(bodypart, 'left', reference_method='ellipse')
                # images['right'] = load_cache(bodypart, 'right', reference_method='ellipse')
        # else:
        images['left'] = load_cache(bodypart, 'left', reference_method=reference)
        images['right'] = load_cache(bodypart, 'right', reference_method=reference)

        try:
            data = defaultdict(list)
            read_images(directory=Path(f'output/{session_id}'), res_list=data)
            left_angles, right_angles = parse_angles(file=f'output/{session_id}/angles', reference=reference)
        except FileNotFoundError:
            return redirect(url_for('index', error=True))

        reader = sitk.ImageFileReader()
        reader.SetFileName(f'output/{session_id}/meta.dcm')
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        accesion_number = reader.GetMetaData('0008|0050')

        isr = sitk.ImageSeriesReader()

        isr.SetFileNames(isr.GetGDCMSeriesFileNames(f'temp/{session_id}/dicom/hip'))
        hip_reference = isr.Execute()

        isr.SetFileNames(isr.GetGDCMSeriesFileNames(f'temp/{session_id}/dicom/knee'))
        knee_reference = isr.Execute()

        isr.SetFileNames(isr.GetGDCMSeriesFileNames(f'temp/{session_id}/dicom/ankle'))
        ankle_reference = isr.Execute()

        lengths = get_length(hip_reference, knee_reference, ankle_reference)

        lookup_table = {
            'left_femur': 'Länge Femur links',
            'right_femur': 'Länge Femur rechts',
            'left_tibia': 'Länge Tibia links',
            'right_tibia': 'Länge Tibia rechts',
            'left': 'Länge untere Extremität (Femur + Tibia) links',
            'right': 'Länge untere Extremität (Femur + Tibia) rechts'
        }

        l, r = dict(), dict()
        for key, value in lengths.items():
            if 'left' in key:
                l[lookup_table[key]] = float(value)
            else:
                r[lookup_table[key]] = float(value)

        return render_template('view.html', dirs=dirs, bparts=bparts, images=images, left_angles=left_angles,
                               right_angles=right_angles, accession_number=accesion_number, left_lengths=l,
                               right_lengths=r, left_longer=lengths['left'] > lengths['right'],
                               left_length=lengths['left'], right_length=lengths['right'])
        # return render_template('view.html', dirs=dirs, images=base64_encodings, angles=angles)
        # return render_template('view.html', dirs=dirs, images=data[bodypart], angles=angles)


if __name__ == '__main__':
    freeze_support()
    print('\n\nWarnung: Nicht für diagnostische Zwecke geeignet.\n\n')
    logger = logging.getLogger('app')
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename='app_log.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(filename)s - %(funcName)s - %(message)s"))
    logger.addHandler(fh)
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    # logging.basicConfig(filename='app_log.log', level=logging.NOTSET, filemode='w')

    handle = 'app'
    # logging.root.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # clear_temp()
    # create_directories()

    """
    app_thread = Thread(target=app.run)
    app_thread.daemon = True
    app_thread.start()

    browser_thread = Thread(target=open_gui)
    browser_thread.daemon = True
    browser_thread.start()

    while True:
        time.sleep(10)
    open_gui()
    app.run()
    """
    open_gui()
    app.run()