import orthanc
import pprint
import json


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


def on_change(change_type, level, resource):
    """
    Callback function to handle changes in Orthanc.
    :param change_type: The type of change (e.g., "Create", "Update", "Delete").
    :param level: The level of the change (e.g., "Instance", "Series", "Study").
    :param resource: The resource that changed.
    :return:
    """
    print("Change type:", change_type)
    print("Level:", level)
    print("Resource:", resource)
    orthanc.LogInfo(f"Change detected: {change_type} at {level} for {resource}")
    # Process the change as needed


def on_stored_instance(dicom, instance_id):
    print('Received instance %s of size %d (transfer syntax %s, SOP class UID %s)' % (
        instance_id, dicom.GetInstanceSize(),
        dicom.GetInstanceMetadata('TransferSyntax'),
        dicom.GetInstanceMetadata('SopClassUid')))

    # Print the origin information
    if dicom.GetInstanceOrigin() == orthanc.InstanceOrigin.DICOM_PROTOCOL:
        print('This instance was received through the DICOM protocol')
    elif dicom.GetInstanceOrigin() == orthanc.InstanceOrigin.REST_API:
        print('This instance was received through the REST API')

    # Print the DICOM tags
    pprint.pprint(json.loads(dicom.GetInstanceSimplifiedJson()))


if __name__ == '__main__':
    orthanc.RegisterReceivedInstanceCallback(on_receive)
    orthanc.RegisterOnChangeCallback(on_change)
    orthanc.RegisterOnStoredInstanceCallback(on_stored_instance)
