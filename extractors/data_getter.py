import os
import pickle
import concurrent.futures
from tensorflow.python.lib.io import file_io
from tf_agents.trajectories.trajectory import Trajectory
from google.cloud import storage

class ListWrapper(object):
    def __init__(self, list_to_wrap):
        self._list = list_to_wrap
        
    def as_list(self):
        return self._list

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Trajectory":
            return Trajectory
        if name == "ListWrapper":
            return ListWrapper

        return super(RenameUnpickler, self).find_class(module, name)

def rename_load(s):
    """Helper function analogous to pickle.loads()."""
    return RenameUnpickler(s, encoding='latin1').load()


def get_data_from_file(data_path):
    try:
        with file_io.FileIO(data_path, mode='rb') as fIn:
            data = rename_load(fIn)
        return data
    except:
        return None

def list_folder(bucket_name, folder):
    gcs = storage.Client()
    bucket_to_list = gcs.lookup_bucket(bucket_name)
    bucket_iterator = bucket_to_list.list_blobs(prefix=folder)
    return [resource.name for resource in bucket_iterator]

def get_data_from_folder(base_path):
        
    base_path = base_path.split('gs://')[-1]
    bucket_name = base_path.split('/')[0]
    folder = '/'.join(base_path.split('/')[1:])
    raw_data = []
    for file in list_folder(bucket_name, folder):
        full_path = os.path.join('gs://', bucket_name, file)
        data = get_data_from_file(full_path)
        if data is not None:
            raw_data.append(data)
    
    return raw_data

def write_file(file_path, data_object):
    with file_io.FileIO(file_path, mode='wb') as fOut:
        pickle.dump(file_path, fOut, protocol=4)
        
            