import orthanc
import requests
import json
import logging
from io import BytesIO


def on_receive(data, origin):
    """
    Callback function to handle received data from Orthanc.
    :param data: The received data.
    :param origin: The origin of the data.
    :return:
    """
    print("Received data:", data)
    print("Origin:", origin)
    orthanc.LogInfo(f"Received data {data} from {origin}")
    # Process the received data as needed

    return orthanc.ReceivedInstanceAction.DISCARD, None


def on_stored_instance(dicom, instance_id):
    file = dicom.SerializeDicomInstance()
    metadata = json.loads(dicom.GetInstanceSimplifiedJson())
    files =  {'file': (instance_id, file, 'application/dicom')}
    data = {'metadata': json.dumps(metadata)}
    response = requests.post(url='http://localhost:8000/upload/orthanc', files=files, data=data)
    logger.info(f'Got status code {response.status_code} and message {response.text}')


logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename='/home/orthanc/orthanc_python_log.log', mode='w')
fh.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s - %(filename)s - %(funcName)s - %(message)s'))
logger.addHandler(fh)
logger.info('Logger initialised')

# orthanc.RegisterReceivedInstanceCallback(on_receive)
orthanc.RegisterOnStoredInstanceCallback(on_stored_instance)
logger.info('Callbacks registered')
