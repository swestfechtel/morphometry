import json
import logging
import os
import datetime
import multiprocessing
import asyncio

import numpy as np

from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Dict

from fastapi import FastAPI, UploadFile, status, Response, Form, Request, Body, File
from fastapi.middleware.cors import CORSMiddleware

from api.file_controller import FileController, ReceivedSeriesBuffer
from api.model_controller import ModelJob, TorsionModelJob
from api.examination import Examination, TorsionExamination, encode_figure
from api.utils import create_directories_and_files, init_logger


init_logger('api')
logger = logging.getLogger('api')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
logger.info('FastAPI server started.')

create_directories_and_files()

file_controller = FileController()
logger.info('FileController started.')

open_jobs = dict()
open_series_buffers = dict()
executor = ThreadPoolExecutor()


def task_callback(task):
    """
    Callback for asynchronous tasks.
    :param task: A task to run.
    :return:
    """
    try:
        result = task.result()
        logger.info(f'Task {task} finished with result: {result}')
        result.running = False
    except:
        logger.error(f'Task {task} failed with exception: {task.exception()}')
        del open_jobs[task.get_name()]  # make sure to clean up the job

    return None


def buffer_callback(examination: Examination, examination_type: str, model_job: str):
    """
    Receive a created examination object from a concluded buffer and start a job for it.
    :param examination: An Examination object to process.
    :param examination_type: The type of examination.
    :param model_job: The model job to run.
    :return:
    """
    assert examination_type in globals(), f'{examination_type} is not a valid class.'
    examination = globals()[examination_type](examination)

    file_controller.update_examination(examination)

    assert model_job in globals(), f'{model_job} is not a valid model job.'
    job = globals()[model_job](file_controller)

    job.identifier = examination.identifier

    async def run_in_executor(func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)


    task = asyncio.create_task(run_in_executor(job.execute, examination))

    job.running = True
    open_jobs[job.identifier] = job
    task.set_name(job.identifier)

    task.add_done_callback(task_callback)
    task.set_name(job.identifier)

    del open_series_buffers[examination.identifier]  # remove the buffer from the list of open buffers


@app.get('/examinations/', status_code=status.HTTP_200_OK)
def get_examinations():
    """
    Retrieve a list of all examinations saved on disk.
    :return:
    """
    examinations = file_controller.get_examinations()
    l = list()
    for examination_id in examinations:
        examination = file_controller.get_examination(examination_id)
        d = dict()
        # d['patient_name'] = examination.patient_name.family_comma_given()
        d['patient_name'] = 'Anonymised'
        d['study_date'] = examination.study_date
        d['study_time'] = examination.study_time
        d['study_description'] = examination.study_description
        d['accession_number'] = examination.accession_number
        d['status'] = examination.status

        assert d['accession_number'] == examination.accession_number == examination_id  # sanity check

        l.append(d)

    return {'examinations': l}


@app.get('/examinations/{examination_id}', status_code=status.HTTP_200_OK)
def get_examination_by_id(examination_id: str):
    """
    Retrieve an examination by its identifier.
    :param examination_id: The identifier of the examination to retrieve.
    :return:
    """

    examination = file_controller.get_examination(examination_id)

    if examination is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    d = dict()
    # d['patient_name'] = examination.patient_name.family_comma_given()
    d['patient_name'] = 'Anonymised'
    d['study_date'] = examination.study_date
    d['study_time'] = examination.study_time
    d['study_description'] = examination.study_description
    d['accession_number'] = examination.accession_number
    d['status'] = examination.status

    assert d['accession_number'] == examination.accession_number == examination_id  # sanity check

    if examination.image_b64 is None:
        examination.encode_images()
        file_controller.update_examination(examination)

    d['image'] = examination.image_b64
    d['shape'] = examination.transformed_image.shape

    if isinstance(examination, TorsionExamination):
        logger.debug(dict(examination.landmarks))
        d['landmarks'] = dict(examination.landmarks)
        d['knee_offset'] = examination.hip.shape[2]
        d['ankle_offset'] = examination.hip.shape[2] + examination.knee.shape[2]
        d['torsion'] = examination.get_torsion_values()

        if examination.image_segmentation_b64 is None:
            examination.encode_images()
            file_controller.update_examination(examination)

        d['segmentation'] = examination.image_segmentation_b64

    return d


@app.patch('/examinations/{examination_id}', status_code=status.HTTP_200_OK)
def update_examination(examination_id: str, examination_json: str = Body(...)):
    """
    Update an examination by its identifier.
    :param examination_json: Examination data to update in json form.
    :param examination_id: The identifier of the examination to update.
    :return:
    """
    examination = file_controller.get_examination(examination_id)
    examination_json = json.loads(examination_json)

    if examination is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    for k, v in examination_json.items():
        if hasattr(examination, k):
            setattr(examination, k, v)

    file_controller.update_examination(examination)
    return {'status': 'updated'}


@app.delete('/examinations/{examination_id}', status_code=status.HTTP_205_RESET_CONTENT)
def delete_file_by_id(examination_id: str):
    """
    Delete an examination by its identifier.
    :param examination_id: The identifier of the examination to delete.
    :return:
    """
    success = file_controller.delete_examination(examination_id)

    if not success:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    return Response(status_code=status.HTTP_205_RESET_CONTENT)  # tell client to refresh the UI


@app.post('/upload/', status_code=status.HTTP_201_CREATED)
async def files_from_html_form(request: Request):
    """
    Upload files to the server via HTML form.
    :param request: A list of files to upload.
    :return:
    """
    data = await request.form()

    for k in data:
        files = data.getlist(k)
        if isinstance(files[0], UploadFile):
            break

    examination = file_controller.save_files(files, 'ui')

    if examination is False:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    return {'examination_id': examination.identifier}


@app.post('/upload/orthanc', status_code=status.HTTP_201_CREATED)
async def files_from_orthanc(file: Annotated[bytes, File(...)], metadata: str = Body(...)):
    """
    Upload files to the server from Orthanc.
    :param file: The file to upload.
    :param metadata: Metadata associated with the file.
    :return:
    """

    if not isinstance(file, bytes):
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    workdir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{workdir}/filter_rules.json', 'r') as f:
        rules = json.load(f)

    model_job = None
    examination_type = None
    for rule in rules['Rules']:
        try:
            for tag, value in rule['DICOM Tags']:
                assert tag in metadata.keys(), f'Tag {tag} not found in metadata.'
                assert value == metadata[tag], f'Tag {tag} has value {metadata[tag]} instead of {value}.'

            model_job = rule['ModelJob']
            examination_type = rule['ExaminationType']
        except AssertionError as e:
            logger.debug(f'Rule {rule} does not apply to file {metadata["AccessionNumber"]}: {e}')

    if (model_job is None) or (examination_type is None):
        logger.error(f'No model/examination found for file {metadata["AccessionNumber"]}')
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    identifier = metadata['AccessionNumber']

    if identifier in open_series_buffers.keys():
        q: ReceivedSeriesBuffer = open_series_buffers[identifier]
        q.add(file)
    else:
        callback = partial(buffer_callback, model_job=model_job, examination_type=examination_type)
        q = ReceivedSeriesBuffer(identifier=identifier, file_controller=file_controller, callback=callback)
        q.add(file)
        open_series_buffers[identifier] = q


@app.post('/model/torsion/{examination_id}', status_code=status.HTTP_202_ACCEPTED)
async def compute_torsion(examination_id: str):
    """
    Compute torsional alignment for an examination.
    :param examination_id: The identifier of the examination to process.
    :return:
    """
    examination = file_controller.get_examination(examination_id)

    if examination is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    if not isinstance(examination, TorsionExamination):
        examination = TorsionExamination(examination)
        file_controller.update_examination(examination)

    job = TorsionModelJob(file_controller)
    job.identifier = examination.identifier

    async def run_in_executor(func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)

    if examination.status == 'unprocessed':
        task = asyncio.create_task(run_in_executor(job.execute, examination))
    else:
        task = asyncio.create_task(run_in_executor(job.compute_torsional_alignment, examination))

    job.running = True
    open_jobs[job.identifier] = job
    task.set_name(job.identifier)

    task.add_done_callback(task_callback)
    task.set_name(job.identifier)

    return {'job_id': str(job.identifier)}


@app.post('/model/segmentation/{examination_id}', status_code=status.HTTP_202_ACCEPTED)
async def compute_segmentation(examination_id: str):
    """
    Compute segmentation for an examination.
    :param examination_id: The identifier of the examination to process.
    :return:
    """
    examination = file_controller.get_examination(examination_id)

    if examination is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    if not isinstance(examination, TorsionExamination):
        examination = TorsionExamination(examination)
        file_controller.update_examination(examination)

    job = TorsionModelJob(file_controller)
    job.identifier = examination.identifier

    async def run_in_executor(func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)

    task = asyncio.create_task(run_in_executor(job.compute_segmentation, examination))
    job.running = True

    open_jobs[job.identifier] = job
    task.set_name(job.identifier)

    task.add_done_callback(task_callback)
    task.set_name(job.identifier)

    return {'job_id': str(job.identifier)}


@app.get('/jobs/{job_id}', status_code=status.HTTP_200_OK)
def get_job_by_id(job_id: str):
    """
    Retrieve the status of a running job.
    :param job_id: The identifier of the job to retrieve.
    :return:
    """
    if job_id not in open_jobs.keys():
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    job = open_jobs[job_id]

    if job.running is False:  # job has finished
        del open_jobs[job_id]
        return {'status': 'finished'}

    return {'status': 'running'}
