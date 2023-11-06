"""
Copyright (c) 2020-present, Royal Bank of Canada.
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.

Supporting code from MTL MMoE code
Written by Gabriel Oliveira and Raquel Aoki in pytorch

"""
import time
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    cohen_kappa_score,
    confusion_matrix,
    average_precision_score,
)
from collections import OrderedDict
from scipy import spatial


class TensorboardWriter:
    """
    A class for managing Tensorboard logging.

    This class simplifies the setup and usage of Tensorboard for logging training and evaluation metrics.

    Args:
        path_logger (str): The directory path for saving Tensorboard logs.
        name_config (str): The name of the configuration or experiment.

    Attributes:
        writer: The SummaryWriter instance for writing Tensorboard logs.

    Methods:
        get_date(): Get the current date and time formatted as a string.
        get_home_dir(): Get the home directory path.
        add_scalar(name_metric, value_metric, epoch): Add a scalar value to the Tensorboard log.
        end_writer(): Flush the Tensorboard writer to ensure all pending events are written to disk.

    """
    def __init__(self, path_logger, name_config):
        date = self.get_date()
        full_path = (
            self.get_home_dir() + "/" + path_logger + date + "/" + name_config + "/"
        )
        print("Tensorboard folder path - {}".format(full_path))
        self.writer = SummaryWriter(log_dir=full_path)

    def get_date(self):
        """
        Get the current date and time formatted as a string.

        Returns:
            str: The formatted date and time string.
        """
        now = datetime.now()  # Current date and time (Hour, minute)
        date = now.strftime("%Y_%m_%d_%H_%M")
        return date

    def get_home_dir(self):
        """
        Get the home directory path.

        Returns:
            str: The path to the home directory.
        """
        return os.getenv("HOME")

    def add_scalar(self, name_metric, value_metric, epoch):
        """
        Add a scalar value to the Tensorboard log.

        Args:
            name_metric (str): The name of the metric to log.
            value_metric (float): The value of the metric.
            epoch (int): The current epoch or step number.
        """
        self.writer.add_scalar(name_metric, value_metric, epoch)

    def end_writer(self):
        """
        Flush the Tensorboard writer to ensure all pending events are written to disk.
        """
        self.writer.flush()


def metrics_mimic(
    epoch,
    data_loader,
    model,
    device,
    tasksname,
    criterion,
    training=False,
    validation=False,
    testing=False,
    B=1000,
):
    """
    Calculate metrics for MIMIC tasks.

    This function calculates metrics for MIMIC tasks, including AUC and loss if a criterion is provided.
    
    Args:
        epoch (int): The current epoch number.
        data_loader: The data loader providing the dataset for evaluation.
        model: The model for evaluation.
        device (str): The device (e.g., 'cpu' or 'cuda') to perform calculations.
        tasksname (list of str): A list of task names.
        criterion: The criterion for calculating loss (if None, loss won't be calculated).
        training (bool): Whether the model is being evaluated during training (default: False).
        validation (bool): Whether validation metrics should be calculated (default: False).
        testing (bool): Whether testing metrics should be calculated (default: False).
        B (int): The batch size for calculations (default: 1000).

    Returns:
        tuple: A tuple containing the following data:
        - auc_per_task (list of float): AUC values for each task.
        - loss_per_task (list of float): Loss values for each task (if validation is True).
        - conf_interval_95 (list of tuple): Confidence intervals for AUC (if testing is True).
    """
    # Initialization of arrays
    y_pred_t0, y_pred_t1, y_pred_t2, y_pred_t3 = [], [], [], []
    y_obse_t0, y_obse_t1, y_obse_t2, y_obse_t3 = [], [], [], []
    y_pred_t0_max = []

    """
    1) Predict and save values in an array
    Note: Combine several batches is important to avoid problems to
    calculate the AUC. When we calculate the metrics using only one
    batch we frequently have errors because only one class is present.
    """
    for i, batch in enumerate(data_loader):
        y_pred_cross = model(batch[0].to(device))

        for task in range(model.num_tasks):
            # Recover task column number
            col = model.task_number[tasksname[task]]
            y_obs = batch[col + 1].long().detach().numpy()

            if tasksname[task] == "los":
                # Task 0
                ls0, ls1, ls2 = y_pred_cross[task].shape
                y_pred = y_pred_cross[task].reshape(ls0 * ls1, ls2)
                y_pred = torch.log_softmax(y_pred, dim=1)
                y_pred = (
                    y_pred.max(dim=1)[1].cpu().detach().numpy()
                )  # Getting the max value
                y_pred_t0_max.append(y_pred.reshape(ls0 * ls1))
                y_pred_t0.append(
                    y_pred_cross[task].cpu().detach().numpy().reshape(ls0 * ls1, ls2)
                )
                y_obse_t0.append(y_obs.reshape(ls0 * ls1))

            elif tasksname[task] == "pheno":
                # Task 1
                y_pred = y_pred_cross[task]
                y_pred = torch.sigmoid(y_pred)
                y_pred = y_pred.cpu().detach().numpy()
                s0, s1 = y_pred.shape
                # Matrix n x 25
                y_pred_t1.append(y_pred)
                y_obse_t1.append(y_obs)
            elif tasksname[task] == "decomp":
                y_pred = y_pred_cross[task]
                s0, s1, s2 = y_pred.shape
                y_pred = y_pred.reshape(s0 * s1, 1)
                y_pred = y_pred.cpu().detach().numpy()
                y_pred_t2.append(y_pred)
                y_obse_t2.append(y_obs.reshape(s0 * s1))
            else:
                y_pred = y_pred_cross[task]
                s0, s1 = y_pred.shape
                y_pred = y_pred.reshape(s0 * s1, 1)
                y_pred = y_pred.cpu().detach().numpy()
                y_pred_t3.append(y_pred)
                y_obse_t3.append(y_obs.reshape(s0 * s1))

        """To avoid memmory errors, the metrics on training set only use 250 batches"""
        if training and i > 250:
            break

    auc_per_task, loss_per_task = [], []
    if testing:
        conf_interval_95 = []

    """
    2) Calculate the metric
    Note: MIMIC tasks are very different of each other, so each one
    has their own subroutine to organize and reshape arrays
    """
    for task in range(model.num_tasks):
        if tasksname[task] == "los":
            """Organize and reshape arrays"""
            y_pred_t0, y_obse_t0, y_pred_t0_max = flat_list(
                y_pred_t0, y_obse_t0, y_pred_t0_max
            )
            s0 = int(len(y_pred_t0) / ls2)
            y_pred_t0 = np.array(y_pred_t0).reshape(s0, ls2)
            """Metrics"""
            auc_per_task.append(
                cohen_kappa_score(
                    y_obse_t0, y_pred_t0_max, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                )
            )
            if validation:
                loss_per_task.append(
                    mimic_validation_loss(
                        y_pred_t0, y_obse_t0, task, device, criterion, True
                    )
                )
            if testing:
                print("=================", tasksname[task], "=================")
                print(confusion_matrix(y_obse_t0, y_pred_t0_max))
                conf_interval_95.append(
                    boostrap(tasksname[task], y_pred_t0_max, y_obse_t0)
                )

        elif tasksname[task] == "pheno":
            """Organize and reshape arrays"""
            y_pred_t1, y_obse_t1 = flat_list(y_pred_t1, y_obse_t1)
            s1 = int(len(y_pred_t1) / 25)
            y_pred_t1 = np.array(y_pred_t1).reshape(s1, 25)
            y_obse_t1 = np.array(y_obse_t1).reshape(s1, 25)
            auc_per_pheno, loss_per_pheno = [], []
            """Metrics"""
            for pheno in range(y_pred_t1.shape[1]):
                if validation:
                    y_pred = torch.tensor(y_pred_t1[:, pheno]).to(device)
                    y_obs = torch.tensor(y_obse_t1[:, pheno]).to(device)
                    loss_per_pheno.append(
                        criterion[task](y_pred.float(), y_obs.float())
                        .cpu()
                        .detach()
                        .numpy()
                    )
            auc = roc_auc_score(y_obse_t1, y_pred_t1)
            auc_per_task.append(auc)
            if validation:
                loss_per_pheno = [float(item) for item in loss_per_pheno]
                loss_per_task.append(np.mean(loss_per_pheno))
            if testing:
                print("=================", tasksname[task], "=================")
                print(
                    "Average: ",
                    auc,
                    "(",
                    roc_auc_score(y_obse_t1, y_pred_t1, average=None),
                    ")",
                )
                conf_interval_95.append(boostrap(tasksname[task], y_pred_t1, y_obse_t1))
        elif tasksname[task] == "decomp":
            """Organize and reshape arrays"""
            y_pred_t2, y_obse_t2 = flat_list(y_pred_t2, y_obse_t2)
            """Metrics"""
            auc_per_task.append(roc_auc_score(y_obse_t2, y_pred_t2))
            if validation:
                loss_per_task.append(
                    mimic_validation_loss(y_pred_t2, y_obse_t2, task, device, criterion)
                )
            if testing:
                y01 = [1 if i > 0.5 else 0 for i in y_pred_t2]
                print("=================", tasksname[task], "=================")
                print(confusion_matrix(y_obse_t2, y01))
                conf_interval_95.append(boostrap(tasksname[task], y_pred_t2, y_obse_t2))
        else:
            """Organize and reshape arrays"""
            y_pred_t3, y_obse_t3 = flat_list(y_pred_t3, y_obse_t3)
            """Metrics"""
            auc_per_task.append(roc_auc_score(y_obse_t3, y_pred_t3))
            if validation:
                loss_per_task.append(
                    mimic_validation_loss(y_pred_t3, y_obse_t3, task, device, criterion)
                )
            if testing:
                y01 = [1 if i > 0.5 else 0 for i in y_pred_t3]
                print("=================", tasksname[task], "=================")
                print(confusion_matrix(y_obse_t3, y01))
                conf_interval_95.append(boostrap(tasksname[task], y_pred_t3, y_obse_t3))

    auc_per_task = [float(item) for item in auc_per_task]
    if validation:
        loss_per_task = [float(item) for item in loss_per_task]
    if testing:
        print("Confidence Interval:")
        for item in conf_interval_95:
            print(item[0], "\n", item[1])
        return auc_per_task, loss_per_task, conf_interval_95
    else:
        return auc_per_task, loss_per_task, ""


def flat_list(array1, array2, array3=None):
    """
    Flatten and concatenate lists or arrays.

    This auxiliary function is used by `metrics_mimic` to flatten and concatenate one or more lists or arrays.

    Args:
        array1: The first list or array to flatten and concatenate.
        array2: The second list or array to flatten and concatenate.
        array3: (Optional) The third list or array to flatten and concatenate (default: None).

    Returns:
        tuple: A tuple containing the following data:
        - array1_flat: The first array after flattening and converting to a numpy array.
        - array2_flat: The second array after flattening and converting to a numpy array.
        - array3_flat: The third array after flattening and converting to a numpy array (if provided, otherwise None).
    """
    array1 = np.concatenate(array1).ravel().tolist()
    array2 = np.concatenate(array2).ravel().tolist()
    if array3 is None:
        return np.array(array1), np.array(array2)
    else:
        array3 = np.concatenate(array3).ravel().tolist()
        return np.array(array1), np.array(array2), np.array(array3)


def mimic_validation_loss(y_pred, y_obs, task, device, criterion, los=False):
    """
    Calculate the validation loss for MIMIC tasks.

    This auxiliary function is used by `metrics_mimic` to calculate the validation loss for MIMIC tasks.

    Args:
        y_pred: The predicted values.
        y_obs: The observed values.
        task (int): The task number.
        device (str): The device (e.g., 'cpu' or 'cuda') for tensor calculations.
        criterion: The criterion for calculating the loss.
        los (bool): Whether the task is "los" (default: False).

    Returns:
        float: The validation loss.
    """
    y_pred = torch.tensor(y_pred).to(device)
    y_obs = torch.tensor(y_obs).to(device)
    if los:
        return criterion[task](y_pred.float(), y_obs.long()).cpu().detach().numpy()
    else:
        return criterion[task](y_pred, y_obs.float()).cpu().detach().numpy()


def metrics_census(
    epoch,
    data_loader,
    model,
    device,
    criterion=None,
    confidence_interval=False,
    train=False,
):
    """
    Calculate metrics for the Census classification task.

    This function calculates metrics for the Census classification task, including AUC and loss (if a criterion is provided).

    Args:
        epoch (int): The current epoch number.
        data_loader: The data loader providing the dataset for evaluation.
        model: The classification model for evaluation.
        device (str): The device (e.g., 'cpu' or 'cuda') to perform calculations.
        criterion: The criterion for calculating loss (if None, loss won't be calculated).
        confidence_interval (bool): Whether to calculate confidence intervals (default: False).
        train (bool): Whether the model is being evaluated during training (default: False).

    Returns:
        tuple: A tuple containing the following data:
        - auc_aux (list of float): AUC values for each task.
        - conf_interval_95 (list of tuple): Confidence intervals for AUC (if confidence_interval is True).
        - loss_val (list of float): Loss values for each task (if criterion is provided).
    """

    auc_aux = []
    loss_val = []
    conf_interval_95 = []
    y_obs, y_pred, y_pred_ = [], [], []
    n = 0
    """
    1) Predict and save values in an array
    Note: Combine several batches is important to avoid problems to
    calculate the AUC. When we calculate the metrics using only one
    batch we frequently have errors because only one class is present.
    """
    for i, batch in enumerate(data_loader):
        y_pred_cross = model(batch[0].to(device))
        y_obs0, y_pred0, y_pred_0 = [], [], []
        n0 = batch[1].shape[0]
        n += n0
        for task in range(model.num_tasks):
            y_obs0.append(np.array(batch[1][:, task].long().detach().numpy()))
            y_pred_0.append(np.array(y_pred_cross[task].cpu().detach().numpy()))
            y_pred0.append(
                np.array(torch.sigmoid(y_pred_cross[task]).cpu().detach().numpy())
            )
        if i == 0:
            y_obs = np.array(y_obs0).reshape(model.num_tasks, n0)
            y_pred_ = np.array(y_pred_0).reshape(model.num_tasks, n0)
            y_pred = np.array(y_pred0).reshape(model.num_tasks, n0)
        else:
            y_obs = np.concatenate(
                (y_obs, np.array(y_obs0).reshape(model.num_tasks, n0)), axis=1
            )
            y_pred_ = np.concatenate(
                (y_pred_, np.array(y_pred_0).reshape(model.num_tasks, n0)), axis=1
            )
            y_pred = np.concatenate(
                (y_pred, np.array(y_pred0).reshape(model.num_tasks, n0)), axis=1
            )

    """
    2) Calculating metrics
    """
    for task in range(model.num_tasks):
        try:
            auc_aux.append(roc_auc_score(y_obs[task], y_pred[task]))
        except ValueError:
            print("NAN VALUE")
            auc_aux.append(np.nan)

        if criterion is not None:
            loss = criterion[task](
                y_pred_cross[task], batch[1][:, task].float().to(device).reshape(-1, 1)
            )
            loss_val.append(loss.mean().cpu().detach().numpy())

        if confidence_interval:
            conf_interval_95.append(boostrap(task, y_pred[task], y_obs[task]))

    if confidence_interval:
        print("Confidence Interval", conf_interval_95)

    if criterion is None:
        return auc_aux, conf_interval_95
    else:
        loss_val = [float(val) for val in loss_val]
        return auc_aux, conf_interval_95, loss_val


def metrics_pcba(
    epoch,
    data_loader,
    model,
    device,
    criterion=None,
    confidence_interval=False,
    train=False,
):
    """
    Calculate metrics for the PCBA classification task.

    This function calculates metrics for the PCBA classification task, including AUC and loss (if a criterion is provided).

    Args:
        epoch (int): The current epoch number.
        data_loader: The data loader providing the dataset for evaluation.
        model: The classification model for evaluation.
        device (str): The device (e.g., 'cpu' or 'cuda') to perform calculations.
        criterion: The criterion for calculating loss (if None, loss won't be calculated).
        confidence_interval (bool): Whether to calculate confidence intervals (default: False).
        train (bool): Whether the model is being evaluated during training (default: False).

    Returns:
        tuple: A tuple containing the following data:
        - auc_aux (list of float): AUC values for each task.
        - conf_interval_95 (list of tuple): Confidence intervals for AUC (if confidence_interval is True).
        - loss_val (list of float): Loss values for each task (if criterion is provided).
    """
    auc_aux = []
    loss_val = []
    conf_interval_95 = []

    n = 0
    y_obs, y_pred, y_pred_, w = [], [], [], []
    """
    1) Predict and save values in an array
    Note: Combine several batches is important to avoid problems to
    calculate the AUC. When we calculate the metrics using only one
    batch we frequently have errors because only one class is present.
    """
    for i, batch in enumerate(data_loader):
        """To avoid memmory errors, the metrics on training set only use 200 batches"""
        if train and i > 200:
            break
        y_pred_cross = model(batch[0].to(device))
        n0 = batch[1].shape[0]
        n += n0
        y_obs0, y_pred0, y_pred_0, w0 = [], [], [], []
        for task in range(model.num_tasks):
            w0.append(np.array(batch[2][:, task].long().detach().numpy()))
            y_obs0.append(np.array(batch[1][:, task].long().detach().numpy()))
            y_pred_0.append(np.array(y_pred_cross[task].cpu().detach().numpy()))
            y_pred0.append(
                np.array(torch.sigmoid(y_pred_cross[task]).cpu().detach().numpy())
            )
        if i == 0:
            w = np.array(w0).reshape(model.num_tasks, n0)
            y_obs = np.array(y_obs0).reshape(model.num_tasks, n0)
            y_pred_ = np.array(y_pred_0).reshape(model.num_tasks, n0)
            y_pred = np.array(y_pred0).reshape(model.num_tasks, n0)
        else:
            w = np.concatenate((w, np.array(w0).reshape(model.num_tasks, n0)), axis=1)
            y_obs = np.concatenate(
                (y_obs, np.array(y_obs0).reshape(model.num_tasks, n0)), axis=1
            )
            y_pred_ = np.concatenate(
                (y_pred_, np.array(y_pred_0).reshape(model.num_tasks, n0)), axis=1
            )
            y_pred = np.concatenate(
                (y_pred, np.array(y_pred0).reshape(model.num_tasks, n0)), axis=1
            )

    """
    2) Calculating metrics
    """
    for task in range(model.num_tasks):
        try:
            auc_aux.append(
                roc_auc_score(y_obs[task][w[task] > 0], y_pred[task][w[task] > 0])
            )
        except ValueError:
            print("Error on task ", task)
            auc_aux.append(np.nan)
        assert len(y_obs[task]) == n, "Shapes are different"

        if criterion is not None:
            obs = (
                torch.tensor(y_obs[task][w[task] > 0]).float().to(device).reshape(-1, 1)
            )
            pred = (
                torch.tensor(y_pred_[task][w[task] > 0])
                .float()
                .to(device)
                .reshape(-1, 1)
            )
            loss = criterion[task](obs, pred)  # [task]
            loss_val.append(loss.mean().cpu().detach().numpy())

        if confidence_interval:
            conf_interval_95.append(
                boostrap(task, y_pred[task][w[task] > 0], y_obs[task][w[task] > 0])
            )

    if confidence_interval:
        print("Confidence Interval", conf_interval_95)

    if criterion is None:
        return auc_aux, conf_interval_95
    else:
        loss_val = [float(val) for val in loss_val]
        return auc_aux, conf_interval_95, loss_val


def metrics_newdata(
    epoch,
    data_loader,
    model,
    device,
    criterion=None,
    confidence_interval=False,
    train=False,
):
    """
    Placeholder function for calculating metrics on new data.

    Args:
        epoch (int): The current epoch number.
        data_loader: The data loader providing the dataset for evaluation.
        model: The model for evaluation.
        device (str): The device (e.g., 'cpu' or 'cuda') to perform calculations.
        criterion: The criterion for calculating loss (if None, loss won't be calculated).
        confidence_interval (bool): Whether to calculate confidence intervals (default: False).
        train (bool): Whether the model is being evaluated during training (default: False).

    Returns:
        tuple: A tuple containing placeholder values, typically empty strings. The structure of the tuple depends on the presence of criterion and confidence_interval flags.
    """
    print("NOT IMPLEMENTED")
    if criterion is None:
        return "", ""
    else:
        return "", "", ""


def maml_split(
    batch, model, device, prop=0.66, time=False, seqlen=None, data_pcba=False
):
    """
    Splits a batch of data for the MAML-MTL approach, especially when working with MMoEEx.

    Args:
        batch: The input batch of data to split.
        model: The model used for prediction.
        device (str): The device (e.g., 'cpu' or 'cuda') to perform calculations.
        prop (float): The size proportion for the inner data (default: 0.66).
        time (bool): Whether the dataset is a time-based dataset (default: False).
        seqlen (int): The sequence length for time-based datasets (required if time is True).
        data_pcba (bool): Indicates whether the dataset is the PCBA dataset (default: False).

    Returns:
        tuple: A tuple containing the split data and labels, which depends on the dataset and time flag:
            - For non-time-based datasets:
                - data_inner_pred: The inner data predicted by the model.
                - label_inner: The inner data labels.
                - data_outer: The outer data.
                - label_outer: The outer data labels.
            - For time-based datasets:
                - data_inner_pred: The inner data predicted by the model.
                - label_inner: A list containing inner data labels.
                - data_outer: The outer data.
                - label_outer: A list containing outer data labels.
    """
    inner_size = int(batch[0].shape[0] * prop)
    data_inner = batch[0][0:inner_size, :].to(device)
    label_inner = batch[1][0:inner_size, :].to(device)

    if data_pcba:
        print(
            "Warning! MAML-MTL is not suitable for the PCBA dataset due to the large number of tasks"
        )

    elif time:
        # MIMIC datset
        data_inner_pred = model(data_inner.float())

        data_outer = batch[0][inner_size:, :].to(device)
        label_outer = batch[1][inner_size:, :].to(device)

        label_inner2 = batch[2][0:inner_size, 0:seqlen].to(device)
        label_inner3 = batch[3][0:inner_size, 0:seqlen].to(device)
        label_inner4 = batch[4][0:inner_size].to(device)

        label_outer2 = batch[2][inner_size:, 0:seqlen].to(device)
        label_outer3 = batch[3][inner_size:, 0:seqlen].to(device)
        label_outer4 = batch[4][inner_size:].to(device)

        return (
            data_inner_pred,
            [label_inner, label_inner2, label_inner3, label_inner4],
            data_outer,
            [label_outer, label_outer2, label_outer3, label_outer4],
        )

    else:
        data_outer = batch[0][inner_size:, :].to(device)
        label_outer = batch[1][inner_size:, :].to(device)
        data_inner_pred = model(data_inner)
        return data_inner_pred, label_inner, data_outer, label_outer


def keep_exclusivity(model):
    """
    Sets the gradients to zero for closed connections between gates and experts.
    This is required when using MMoEEx or MD with Exclusivity.

    Args:
        model: The PyTorch model with MMoEEx or MD layers.

    Returns:
        tuple: A tuple containing the gradients for gate kernels and gate bias after setting them to zero.
    """
    for index, e in enumerate(model.MMoEEx.exclusivity):
        if e < model.num_tasks + 1:  # if not shared
            for task in range(model.num_tasks):
                if e != task:  # if not exclusive of that task
                    model.MMoEEx.gate_kernels.grad.data[task][:, index].zero_()
                    model.MMoEEx.gate_bias.grad.data[task][:, index].zero_()
    return (
        model.MMoEEx.gate_kernels.grad.data.clone(),
        model.MMoEEx.gate_bias.grad.data.clone(),
    )


def keep_exclusion(model):
    """
    Set the gradients to zero for closed connections between gates and experts.
    This is required when using MMoEEx or MD with Exclusion.

    Args:
        model: The PyTorch model with MMoEEx or MD layers.

    Returns:
        tuple: A tuple containing the gradients for gate kernels and gate bias after setting them to zero.
    """
    for index, e in enumerate(model.MMoEEx.exclusivity):
        if e < model.num_tasks + 1:  # if not shared
            model.MMoEEx.gate_kernels.grad.data[e][:, index].zero_()
            model.MMoEEx.gate_bias.grad.data[e][:, index].zero_()
    return (
        model.MMoEEx.gate_kernels.grad.data.clone(),
        model.MMoEEx.gate_bias.grad.data.clone(),
    )


def gradient_update_parameters(
    model, loss, params=None, step_size=0.05, first_order=False
):
    """
    Update model parameters with one step of gradient descent on the loss function.

    Reference:
        https://github.com/tristandeleu/pytorch-meta/blob/6db28dc9e7e22c8f6239169c2ce0761e87d5a1b3/torchmeta/utils/gradient_based.py#L7

    Parameters:
        model (torch.nn.Module): The model for which parameters need to be updated.
        loss (torch.Tensor): The value of the inner loss. This is the result of the training dataset through the loss function.
        params (OrderedDict, optional): Dictionary containing the meta-parameters of the model. If None, the values stored in `model.named_parameters()` are used.
        step_size (float or OrderedDict, optional): The step size in the gradient update. If an OrderedDict, the keys must match the keys in `params`.
        first_order (bool, optional): If True, use the first-order approximation of MAML.

    Returns:
        updated_params (OrderedDict): Dictionary containing the updated meta-parameters of the model with one gradient update wrt. the inner loss.
    """

    if params is None:
        params = OrderedDict(model.named_parameters())

    torch.autograd.set_detect_anomaly(True)

    grads = torch.autograd.grad(
        loss, params.values(), create_graph=not first_order, allow_unused=True
    )

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            if grad is not None:
                updated_params[name] = param - step_size * grad
            else:
                updated_params[name] = param

    return updated_params


def organizing_predictions(
    model, params, train_y_pred, obs, task, name=None, weight=None
):
    """
    Organize model predictions and observed values based on the specified parameters and data type.

    Parameters:
        model (torch.nn.Module): The model used for prediction.
        params (dict): A dictionary containing parameters related to the dataset and tasks.
        train_y_pred (torch.Tensor): The model's predictions.
        obs (torch.Tensor): The observed values.
        task (int): The index of the task.
        name (str, optional): Name of the task (for use in mimic dataset).
        weight (torch.Tensor, optional): Weight values for data filtering (for use in pcba dataset).

    Returns:
        Tuple: A tuple containing the organized model predictions and observed values.
    """
    if params["data"] == "census":
        return train_y_pred, obs[:, task].float().reshape(-1, 1)
    elif params["data"] == "pcba":
        return (
            train_y_pred[weight[:, task] > 0],
            obs[:, task].float().reshape(-1, 1)[weight[:, task] > 0],
        )
    elif params["data"] == "mimic":
        col = model.task_number[params["tasks"][task]] - 1
        if params["tasks"][task] == "los":
            s0, s1, s2 = train_y_pred.float().shape
            obs = obs[col + 1].long().reshape(s0 * s1)
            return train_y_pred.reshape(s0 * s1, s2), obs
        elif params["tasks"][task] == "ihm":  # if ihm
            s0, s1 = train_y_pred.float().shape
            obs = obs[col + 1].reshape(s0, s1)
            return train_y_pred.float(), obs.float()
        elif params["tasks"][task] == "pheno":  # if ihm
            obs = obs[col + 1]
            pred = train_y_pred
            s1, s2 = pred.shape
            return pred.reshape(s1 * s2), obs.reshape(s1 * s2).float()
        else:
            s0, s1, s2 = train_y_pred.float().shape
            obs = obs[col + 1]
            return (
                train_y_pred.reshape(s0 * s1, s2).float(),
                obs.reshape(s0 * s1, s2).float(),
            )


def model_CI(ci_test, model):
    """
    Calculate confidence interval for multiple runs of the model.

    Parameters:
        ci_test (numpy.ndarray): Testing AUC values with shape (runs, tasks).
        model (torch.nn.Module): The current model (used for parameters).

    Returns:
        None
    """
    ic_final_ = model_CI_boostrap(ci_test, boostrap=False)
    ic_final = []

    print("Number of tasks", model.num_tasks)
    for task in range(model.num_tasks):
        ic_final.append([ic_final_[0][task], ic_final_[1][task]])

    print(
        "\n Final AUC-test:",
        np.mean(ci_test, axis=0),
        "\nCI(95%):",
        ic_final,
        "\nMax",
        np.max(ci_test, axis=0),
    )


def boostrap(task, pred, obs, B=100):
    """
    Calculate AUC or other metrics for a given task using bootstrap resampling.

    Parameters:
        task (str): The name of the task ('pheno', 'los', or other).
        pred (numpy.ndarray): Predicted values.
        obs (numpy.ndarray): Observed (true) values.
        B (int): Number of bootstrap resamples (default: 100).

    Returns:
        tuple: Lower and upper bounds of the confidence interval for the given metric.
    """
    repetitions = []
    prob = 0.75
    if task == "pheno":
        for i in range(B):
            mask = np.random.binomial(1, prob, pred.shape[0])
            pheno_ = []
            for pheno in range(pred.shape[1]):
                pheno_.append(
                    roc_auc_score(obs[:, pheno][mask == 1], pred[:, pheno][mask == 1])
                )
            repetitions.append(np.mean(pheno_))
    elif task == "los":
        for i in range(B):
            mask = np.random.binomial(1, prob, len(obs))
            obs = np.array(obs)
            pred = np.array(pred)
            metric = cohen_kappa_score(
                obs[mask == 1], pred[mask == 1], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            )
            repetitions.append(metric)
    else:
        for i in range(B):
            obs = np.array(obs)
            pred = np.array(pred)
            mask = np.random.binomial(1, 0.9, len(obs))
            try:
                repetitions.append(roc_auc_score(obs[mask == 1], pred[mask == 1]))
            except ValueError:
                repetitions.append(np.nan)
    return model_CI_boostrap(repetitions)


def model_CI_boostrap(repetitions, bootstrap=True):
    """
    Calculate the 95% confidence interval for an array of repetitions.

    Parameters:
        repetitions (numpy.ndarray): An array containing repeated measurements.
        bootstrap (bool): If True, use bootstrapping for confidence interval calculation.
                          If False, use the formula-based method (default: True).

    Returns:
        list: Lower and upper bounds of the 95% confidence interval.
    """
    if bootstrap:
        confidence_interval = np.quantile(repetitions, [0.025, 0.975], axis=0)
    else:
        confidence_interval = [
            np.mean(repetitions, axis=0)
            - 1.96 * np.sqrt(np.var(repetitions, axis=0)) / len(repetitions),
            np.mean(repetitions, axis=0)
            + 1.96 * np.sqrt(np.var(repetitions, axis=0)) / len(repetitions),
        ]
    return confidence_interval


def output_file_creation(
    rep,
    num_tasks,
    auc_test,
    auc_val,
    auc_train,
    conf_interval,
    rep_start,
    params,
    precision_auc_test,
):
    """
    Create a DataFrame containing metrics and parameters for model evaluation and experimentation.

    Parameters:
    -----------
    rep : int
        The repetition number.
    num_tasks : int
        The number of tasks (e.g., classification targets).
    auc_test : list
        A list of test AUC (Area Under the ROC Curve) values, one per task.
    auc_val : list
        A list of validation AUC values, one per task.
    auc_train : list
        A list of training AUC values, one per task.
    conf_interval : list
        A list of confidence intervals for test AUC, one per task.
    rep_start : float
        The start time of the repetition.
    params : dict
        A dictionary containing various experiment parameters and settings.
    precision_auc_test : list
        A list of precision values for test AUC, one per task.

    Returns:
    --------
    data_output : pandas.DataFrame
        A DataFrame containing the specified metrics and parameters.

    Note:
    -----
    This function creates a DataFrame with columns for repetition, AUC values, confidence intervals,
    time taken, and various experiment parameters. It's useful for organizing and recording
    results during experimentation and model evaluation.
    """
    print("...output file creation")
    names = {"repetition"}
    _output = {"repetition": rep}
    for i in range(num_tasks):
        colname = "Task_" + str(i)
        names.add(colname + "_test")
        _output[colname + "_test"] = auc_test[i]
        names.add(colname + "_test_bs_l")  # bootstrap
        _output[colname + "_test_bs_l"] = conf_interval[i][0]
        names.add(colname + "_test_bs_u")  # bootstrap
        _output[colname + "_test_bs_u"] = conf_interval[i][1]
        names.add(colname + "_val")
        _output[colname + "_val"] = auc_val[i]
        names.add(colname + "_train")
        _output[colname + "_train"] = auc_train[i]

    names.add("time")
    _output["time"] = time.time() - rep_start
    names.add("params")
    _output["params"] = params

    names.add("data")
    names.add("tasks")
    names.add("model")
    names.add("batch_size")
    names.add("max_epochs")
    names.add("num_experts")
    names.add("num_units")
    names.add("expert")
    names.add("expert_blocks")
    names.add("use_early_stop")
    names.add("runits")
    names.add("seqlen")
    names.add("prop")
    names.add("lambda")
    names.add("cw_pheno")
    names.add("cw_decomp")
    names.add("cw_ihm")
    names.add("cw_los")
    names.add("lstm_nlayers")
    names.add("task_balance_method")
    names.add("type_exc")
    names.add("prob_exclusivity")

    _output["data"] = params["data"]
    _output["tasks"] = params["tasks"]
    _output["model"] = params["model"]
    _output["batch_size"] = params["batch_size"]
    _output["max_epochs"] = params["max_epochs"]
    _output["num_experts"] = params["num_experts"]
    _output["num_units"] = params["num_units"]
    _output["runits"] = params["runits"]
    _output["expert"] = try_keyerror("expert", params)
    _output["expert_blocks"] = try_keyerror("expert_blocks", params)
    _output["seqlen"] = try_keyerror("seqlen", params)
    _output["prop"] = params["prop"]
    _output["lambda"] = params["lambda"]
    _output["cw_pheno"] = try_keyerror("cw_pheno", params)
    _output["cw_decomp"] = try_keyerror("cw_decomp", params)
    _output["cw_ihm"] = try_keyerror("cw_ihm", params)
    _output["cw_los"] = try_keyerror("cw_los", params)
    _output["cw_pcba"] = try_keyerror("cw_pcba", params)
    _output["lstm_nlayers"] = try_keyerror("lstm_nlayers", params)
    _output["task_balance_method"] = params["task_balance_method"]
    _output["type_exc"] = params["type_exc"]
    _output["prob_exclusivity"] = params["prob_exclusivity"]

    data_output = pd.DataFrame(columns=names)
    data_output = data_output.append(_output, ignore_index=True)
    return data_output


def try_keyerror(name, params):
    """
    Check if a key exists in the parameters dictionary and return its value if present.

    Parameters:
    -----------
    name : str
        The name of the key to check in the dictionary.
    params : dict
        The dictionary containing parameters and their values.

    Returns:
    --------
    str or any
        The value associated with the specified key if it exists in the dictionary; otherwise, an empty string.
    
    Note:
    -----
    This function is useful for safely accessing key-value pairs in a dictionary without raising a KeyError
    when the key doesn't exist. If the key is not found in the dictionary, it returns an empty string.
    """
    try:
        return params[name]
    except KeyError:
        return ""


def measuring_diversity(data_loader, model, device, output_name, data):
    """
    Calculate the diversity among the experts in the testing set and save the diversity matrix to a CSV file.

    Parameters:
    -----------
    data_loader : DataLoader
        The DataLoader containing the testing data.
    model : torch.nn.Module
        The current model used for inference.
    device : torch.device
        The device (CPU or GPU) on which the model is running.
    output_name : str
        The name of the output CSV file where the diversity matrix will be saved.
    data : str
        The type of dataset, which can be "census," "pcba," or "mimic."

    Returns:
    --------
    None

    Note:
    -----
    This function calculates the diversity among experts' predictions for the given testing dataset.
    The diversity matrix is a square matrix where each element (i, j) represents the diversity between experts i and j.

    The method used to calculate diversity may vary depending on the dataset. For "census" and "pcba," it's calculated
    based on the Euclidean distance between expert predictions. For "mimic," only the first 400 samples are considered
    to avoid memory errors.

    The resulting diversity matrix is saved to a CSV file with the specified output name.

    Example:
    --------
    measuring_diversity(data_loader, my_model, device, "diversity_matrix.csv", "census")
    """

    if data == "census":
        for i, batch in enumerate(data_loader):
            experts_output = model(batch[0].to(device), diversity=True)
    elif data == "pcba":
        for i, batch in enumerate(data_loader):
            exp_out = model(batch[0].to(device), diversity=True)
            if i == 0:
                experts_output = exp_out
            else:
                experts_output = torch.cat((experts_output, exp_out), dim=1)
        experts_output = experts_output.reshape(experts_output.shape[0], -1)
        experts_output = experts_output.transpose(0, 1)
    elif data == "mimic":
        for i, batch in enumerate(data_loader):
            exp_out = model(batch[0].to(device), diversity=True)
            if i == 0:
                experts_output = exp_out
            else:
                experts_output = torch.cat((experts_output, div), dim=1)
            if experts_output.shape[1] > 400:
                # Only 400 samples for MIMIC due to memory errors
                break

        experts_output = experts_output.reshape(experts_output.shape[0], -1)
        experts_output = experts_output.transpose(0, 1)
    else:
        print("New dataset, new routine missing")

    # E x E matrix
    diversity = np.zeros((experts_output.shape[1], experts_output.shape[1]))

    for ex1 in range(experts_output.shape[1]):
        for ex2 in range(experts_output.shape[1]):
            array1 = experts_output[:, ex1].cpu().detach().numpy()
            array2 = experts_output[:, ex2].cpu().detach().numpy()
            diversity[ex1, ex2] = np.sqrt(pow(array1 - array2, 2).sum())

    diversity = pd.DataFrame(diversity)
    diversity.to_csv(
        "output//" + output_name + "diversity.csv", header=False, index=False
    )
