import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import copy
import os
import datetime
import json


class Dataset():

    def __init__(self):
        # load collection data from file
        collection_data = sio.loadmat("../data/backs.mat")
        self.collection_data = collection_data

        # declaration
        self.signal_samples = None
        self.image_samples = None
        self.labels = None
        self.indexes = None
        self.cpas = None

        self.n_samples = 0
        self.n_features = 0
        self.target_type = None
        self.cpa_range = []
        self.data_class = None
        self.snr_avg = 0.0


    def load(self, filename):
        path = "../datasets/"

        with open(path + "log.json") as log_file:
            log_info = None
            for line in log_file:
                line_info = json.loads(line)
                if line_info["filename"] == filename:
                    log_info = line_info
                    break
            if log_info is None:
                raise Exception("Cannot find this file!")
            self.n_samples = log_info["n_samples"]
            self.n_features = log_info["n_features"]
            self.target_type = log_info["target_type"]
            self.cpa_range = log_info["cpa_range"]
            self.data_class = log_info["data_class"]
            self.snr_avg = log_info["snr_avg"]

        load_dict = np.load(path + filename + ".npz")
        self.signal_samples = load_dict["signals"].astype(np.float32)
        self.image_samples = load_dict["images"].astype(np.float32)
        self.labels = load_dict["labels"].astype(np.int16)
        self.indexes = load_dict["indexes"].astype(np.int16)
        self.cpas = load_dict["cpas"].astype(np.int16)

        # load collection data from file
        collection_data = sio.loadmat("../data/backs.mat")
        self.collection_data = collection_data

        return self

