"""
Copyright (c) 2020-present, Royal Bank of Canada.
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.

"""

import os
import json
import pickle
import platform
import pandas as pd
import numpy as np
import torch
import wget
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from utils_neuron import parse_multiple_sim_experiment_files

PATH = "./"

def data_preparation_newdataset(params, data_directory="/home/"):
    """
    DATASET X: Template
    """
    
    file_list = [data_directory + file_str for file_str in os.listdir(data_directory)]
    datasets = parse_multiple_sim_experiment_files(file_list)

    data_split = [random_split(datasets[i].T, [params['train_split'], params['validation_split'],
                                               params['test_split']]) for i in range(len(datasets))]
    dataset_train = [dat[0] for dat in data_split]
    dataset_validation = [dat[1] for dat in data_split]
    dataset_test = [dat[2] for dat in data_split]

    train_loader = DataLoader(dataset_train, shuffle=params["shuffle"], batch_size=params["batch_size"])
    validation_loader = DataLoader(dataset_validation, shuffle=params["shuffle"], batch_size=params["batch_size"])
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=params["batch_size"])

    num_features = dataset_train.shape[1]
    num_tasks = params["num_tasks"]
    output_info = None

    return (train_loader, validation_loader, test_loader, num_features, num_tasks, output_info)

#%%
data_directory = '/Users/constb/Data/archive/Data_test_subthreshold/'
file_list = [data_directory + file_str for file_str in os.listdir(data_directory)]
datasets = parse_multiple_sim_experiment_files(file_list)

X, y_spike, y_soma, y_DVT = datasets
y_total = np.vstack([y_DVT,np.stack([np.float16(y_spike),np.float16(y_soma)])])


np.save(data_directory+'input_spikes.npy',X)
np.save(data_directory+'output.npy',y_total)
