import os
import pickle
import random
import sys
import time
import io
import collections
import concurrent.futures
import itertools

import urllib.request
import scipy
import PIL
import numpy as np

from urllib.error import HTTPError
from PIL import Image
from scipy import ndimage
from sklearn import metrics
from sklearn.utils import shuffle
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


def load_file(full_path):
    try:
        with open(full_path, 'rb') as f:
            data = rename_load(f)
            return data
    except:
        return None
    
def get_data_from_folder(base_path):
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    
    futures = []
    for i in range(len(os.listdir(base_path))):
        full_path = os.path.join(base_path, "ts"+str(i)+".pickle")
        future = executor.submit(load_file, full_path)
        futures.append(future)
    
    raw_data = []
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            raw_data.append(result)
    
    return raw_data

def get_data_from_file(data_path):
    with file_io.FileIO(data_path, mode='rb') as fIn:
        data = pickle.load(fIn)
    return data

