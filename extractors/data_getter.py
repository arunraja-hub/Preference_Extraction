import os
import pickle
import concurrent.futures
from tensorflow.python.lib.io import file_io
from tf_agents.trajectories.trajectory import Trajectory


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
    
def get_data_from_folder(base_path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    
    futures = []
    for i in range(5000):
        full_path = os.path.join(base_path, "ts"+str(i)+".pickle")
        future = executor.submit(get_data_from_file, full_path)
        futures.append(future)
    
    raw_data = []
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            raw_data.append(result)
    
    return raw_data
