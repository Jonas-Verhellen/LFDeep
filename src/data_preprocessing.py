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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

PATH = "./"

def data_preparation_newdataset(params, data_directory="/home/"):
    """
    DATASET X: Template
    """
    dataset_train = pd.read_csv(data_directory + "train_features.csv")
    dataset_validation = pd.read_csv(data_directory + "val_features.csv")
    dataset_test = pd.read_csv(data_directory + "test_features.csv")

    train_loader = DataLoader(dataset_train, shuffle=params["shuffle"], batch_size=params["batch_size"])
    validation_loader = DataLoader(dataset_validation, shuffle=params["shuffle"], batch_size=params["batch_size"])
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=params["batch_size"])

    num_features = dataset_train.shape[1]
    num_tasks = params["num_tasks"]
    output_info = None

    return (train_loader, validation_loader, test_loader, num_features, num_tasks, output_info)
