import logging
import os
import datetime
import multiprocessing
import asyncio

from functools import partial
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, status, Response, Form, Request
from fastapi.middleware.cors import CORSMiddleware

from api.file_controller import FileController
from api.model_controller import ModelJob
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
        return result
    except:
        logger.error(f'Task {task} failed with exception: {task.exception()}')
        del open_jobs[task.name]  # make sure to clean up the job

    return None


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
        d['patient_name'] = examination.patient_name.family_comma_given()

        tmp = examination.study_date
        tmp = datetime.datetime.strptime(tmp, '%Y%m%d')
        d['study_date'] = tmp.strftime('%Y-%m-%d')

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
    d['patient_name'] = examination.patient_name.family_comma_given()

    tmp = examination.study_date
    tmp = datetime.datetime.strptime(tmp, '%Y%m%d')
    d['study_date'] = tmp.strftime('%Y-%m-%d')

    d['study_description'] = examination.study_description
    d['accession_number'] = examination.accession_number
    d['status'] = examination.status

    assert d['accession_number'] == examination.accession_number == examination_id  # sanity check

    layers = [examination.transformed_image.array[:, :, i] for i in range(examination.transformed_image.shape[-1])]

    with multiprocessing.Pool() as pool:
        layers = pool.map(encode_figure, layers)

    d['image'] = layers
    d['shape'] = examination.transformed_image.shape

    if isinstance(examination, TorsionExamination):
        logger.debug(dict(examination.landmarks))
        d['landmarks'] = dict(examination.landmarks)
        d['knee_offset'] = examination.hip.shape[2]
        d['ankle_offset'] = examination.hip.shape[2] + examination.knee.shape[2]
        d['torsion'] = examination.get_torsion_values()

    return d


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


@app.post('/upload/{origin}', status_code=status.HTTP_201_CREATED)
async def upload_files(request: Request, origin: str = 'ui'):
    """
    Upload files to the server via HTML form.
    :param request: A list of files to upload.
    :param origin: The origin of the files to upload.
    :return:
    """
    data = await request.form()

    for k in data:
        files = data.getlist(k)
        if isinstance(files[0], UploadFile):
            break

    examination = file_controller.save_files(files, origin)

    if examination is False:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    return {'examination_id': examination.identifier}



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

    job = ModelJob(file_controller)
    job.identifier = examination.identifier

    async def run_in_executor(func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)

    if examination.status == 'unprocessed':
        task = asyncio.create_task(run_in_executor(job.segment_and_process, examination))
    else:
        task = asyncio.create_task(run_in_executor(job.compute_torsional_alignment, examination))

    job.running = True
    open_jobs[job.identifier] = job

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

    job = ModelJob(file_controller)
    job.identifier = examination.identifier

    async def run_in_executor(func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)

    task = asyncio.create_task(run_in_executor(job.compute_segmentation, examination))
    job.running = True

    open_jobs[job.identifier] = job

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
