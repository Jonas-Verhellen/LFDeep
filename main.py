import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import copy
import yaml
import time
import warnings
from tqdm import tqdm

sys.path.append("src/")
from mmoeex import MMoE, MMoETowers, MMoEEx
from standard_optimization import standarOptimization
from task_balancing import TaskBalanceMTL
from utils import *
from data_preprocessing import *

warnings.filterwarnings("ignore")

def main(config_path):
    """Start: Parameters Loading"""
    with open(config_path) as f:
        config = yaml.load_all(f, Loader=yaml.FullLoader)
        for p in config:
            params = p["parameters"]

    try:
        SEED = params["SEED"]
    except KeyError:
        SEED = 2

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    try:
        print(params["rep_ci"])
    except KeyError:
        params["rep_ci"] = 1

    if params["rep_ci"] > 1:
        ci_test = []
    """End: Parameters Loading"""

    """Start: Tensorboard creation"""
    if params["save_tensor"]:
        path_logger = "tensorboard/"
    else:
        path_logger = "notsave/"
    config_name = ("model_" + params["model"] + "_Ntasks_" + str(len(params["tasks"])) + "_batch_" + str(params["batch_size"]) + "_N_experts_" + str(params["num_experts"]))
    if params["save_tensor"]:
        writer_tensorboard = TensorboardWriter(path_logger, config_name)

    # Starting date to folder creation
    date_start = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    """End: Tensorboard Creation"""

    for rep in range(params["rep_ci"]):
        rep_start = time.time()

        print("Start: Data Loading")
        (train_loader, validation_loader, test_loader, num_features, num_tasks, output_info,) = data_preparation_neuron(params)
        X, y = next(iter(train_loader))
        print("...batch shapes: ", X.shape, y.shape)
        print("End: Data Loading")

        print("Start: Model Initialization and Losses")
        if params["model"] == "Standard":
            # Shared bottom
            model = standarOptimization(data=params["data"], num_units=params["num_units"], num_experts=params["num_experts"], num_tasks=len(params["tasks"]), num_features=num_features, hidden_size=params["hidden_size"], tasks_name=params["tasks"])
        else:
            # MMoEEx or MMoE
            model = MMoETowers(data=params["data"], tasks_name=params["tasks"], num_tasks=num_tasks, num_experts=params["num_experts"], num_units=params["num_units"], num_features=num_features, modelname=params["model"], prob_exclusivity=params["prob_exclusivity"], type=params["type_exc"])

        lr = 1e-4  # Learning Rate
        wd = 0.01  # Adam weight_decay
        criterion = [nn.BCEWithLogitsLoss().to(device) for i in range(num_tasks)]

        if torch.cuda.is_available():
            model.to(device)
        print("End: Model Initialization")

        print("Start: Variables Initialization")
        alpha = 0.5
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params["gamma"])
        balance_tasks = TaskBalanceMTL(model.num_tasks, params["task_balance_method"], alpha_balance=alpha)

        # Loss variables initialization
        loss_ = []
        task_losses = np.zeros([model.num_tasks], dtype=np.float32)
        best_val_AUC = 0
        best_epoch = 0
        print("End: Variables Initialization")

        print("Start: Training Loop")
        for e in range(params["max_epochs"]):
            torch.cuda.empty_cache()
            for i, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                loss = 0

                if params["model"] == "Standard":
                    train_y_pred = model(batch[0].to(device))
                    for task in range(model.num_tasks):
                        label = batch[1][:, task].long().to(device).reshape(-1, 1)
                        loss_temp = (criterion[task](train_y_pred[task][batch[2][:, task] > 0], label.float()[batch[2][:, task] > 0],) * params["lambda"][task])
                        loss += loss_temp
                    loss.backward()
                    optimizer.step()
                else:
                    if params["model"] == "MMoE" or params["model"] == "Md":
                        train_y_pred = model(batch[0].to(device))
                        for task in range(model.num_tasks):
                                label = (batch[1][:, task].long().to(device).reshape(-1, 1))
                                loss += criterion[task](train_y_pred[task], label.float())
                                task_losses[task] = (criterion[task](train_y_pred[task], label.float()) * params["lambda"][task])
                        loss.backward()
                        if params["model"] == "Md":
                            if params["type_exc"] == "exclusivity":
                                (model.MMoEEx.gate_kernels.grad.data, model.MMoEEx.gate_bias.grad.data) = keep_exclusivity(model)
                            else:
                                (model.MMoEEx.gate_kernels.grad.data, model.MMoEEx.gate_bias.grad.data) = keep_exclusion(model)
                        optimizer.step()
                    else:
                        """
                        1) For MAML-MTL, we split the training set in inner and outer loss calculation
                        """

                        (train_y_pred, label_inner, train_outer, label_outer,) = maml_split(batch, model, device, params["maml_split_prop"])
                        loss_task_train = []

                        """
                        2) Deepcopy to save the model before temporary updates
                        """
                        model_copy = copy.deepcopy(model)
                        for task in range(model.num_tasks):
                            """
                            3) Inner loss / loss in the current model
                            """
                            pred, obs = organizing_predictions(model, params, train_y_pred[task], label_inner, task)
                            inner_loss = criterion[task](pred, obs).float()
                            """
                            4) Temporary update per task
                            """
                            params_ = gradient_update_parameters(model, inner_loss, step_size=optimizer.param_groups[0]["lr"],)
                            """
                            5) Calculate outer loss / loss in the temporary model
                            """
                            current_y_pred = model(train_outer, params=params_)
                            pred, obs = organizing_predictions(model, params, current_y_pred[task], label_outer, task,)
                            loss_out = (criterion[task](pred, obs).float() * params["lambda"][task])
                            task_losses[task] = loss_out
                            loss += loss_out
                            loss_task_train.append(loss_out.cpu().detach().numpy())
                            """
                            6) Reset temporary model
                            """
                            for (_0, p_), (_1, p_b) in zip(model.named_parameters(), model_copy.named_parameters()):
                                p_.data = p_b.data

                        loss.backward()
                        # Keeping gates 'closed'
                        if params["type_exc"] == "exclusivity":
                            (model.MMoEEx.gate_kernels.grad.data, model.MMoEEx.gate_bias.grad.data,) = keep_exclusivity(model)
                        else: 
                            (model.MMoEEx.gate_kernels.grad.data, model.MMoEEx.gate_bias.grad.data,) = keep_exclusion(model)
                        optimizer.step()

                """ Optional task balancing step"""
                if params["task_balance_method"] == "LBTW":
                    for task in range(model.num_tasks):
                        if i == 0:  # First batch
                            balance_tasks.get_initial_loss(task_losses[task], task,)
                        balance_tasks.LBTW(task_losses[task], task)
                        weights = balance_tasks.get_weights()
                        params["lambda"] = weights

            if params["task_balance_method"] == "LBTW":
                print("... Current weights LBTW: ", params["lambda"])

            """ Saving losses per epoch"""
            loss_.append(loss.cpu().detach().numpy())

            print("... calculating metrics")
            print("Validation")
            auc_val, _, loss_val = metrics_census(e, validation_loader, model, device, criterion)
            print("Train")
            auc_train, _ = metrics_census(e, train_loader, model, device, train=True)

            """Updating tensorboard"""
            if params["save_tensor"]:
                writer_tensorboard.add_scalar("Loss/train_Total", loss.cpu().detach().numpy(), e)
                for task in range(model.num_tasks):
                    writer_tensorboard.add_scalar("Auc/train_T" + str(task + 1), auc_train[task], e)
                    writer_tensorboard.add_scalar("Auc/Val_T" + str(task + 1), auc_val[task], e)
                writer_tensorboard.end_writer()

            """Printing Outputs """
            if e % 1 == 0:
                if params["gamma"] < 1 and e % 10 == 0 and e > 1:
                    opt_scheduler.step()

            """Saving the model with best validation AUC"""
            if params["best_validation_test"]:
                current_val_AUC = np.nansum(auc_val)
                if current_val_AUC > best_val_AUC:
                    best_epoch = e
                    best_val_AUC = current_val_AUC
                    print("better AUC ... saving model")
                    # Path to save model
                    path = (".//output//" + date_start + "/" + params["model"] + "-" + params["data"] + "-" + str(params["num_experts"]) + "-" + params["output"] + "/")
                    Path(path).mkdir(parents=True, exist_ok=True)
                    path = path + "net_best.pth"
                    torch.save(model.state_dict(), path)
                print("...best epoch", best_epoch)

            """Optional: DWA task balancing"""
            if params["task_balance_method"] == "DWA":
                # Add losses to history structure
                balance_tasks.add_loss_history(task_losses)
                balance_tasks.last_elements_history()
                balance_tasks.DWA(task_losses, e)
                weights = balance_tasks.get_weights()
                params["lambda"] = weights
                print("... Current weights DWA: ", params["lambda"])

            """Reset array with loss per task"""
            task_losses[:] = 0.0

        loss_ = np.array(loss_).flatten().tolist()
        torch.cuda.empty_cache()

        if params["best_validation_test"]:
            print("...Loading best validation epoch")
            path = (".//output//" + date_start + "/" + params["model"] + "-" + params["data"] + "-" + str(params["num_experts"]) + "-" + params["output"] + "/")
            path = path + "net_best.pth"
            model.load_state_dict(torch.load(path))

        print("... calculating metrics on testing set")
        auc_test, _, conf_interval = metrics_newdata(epoch=e, data_loader=test_loader, model=model, device=device, confidence_interval=True,)

        print("... calculating diversity on testing set experts")
        if params["model"] != "Standard":
            measuring_diversity(test_loader, model, device, params["output"], params["data"])

        """Creating and saving output files"""
        if params["rep_ci"] <= 1:
            print("\nFinal AUC-Test: {}".format(auc_test))
            print("...Creating the output file")
            if params["create_outputfile"]:
                data_output = output_file_creation(rep, model.num_tasks, auc_test, auc_val, auc_train, conf_interval, rep_start, params, precision_auc_test,)
                path = (".//output//" + date_start + "/"+ params["model"] + "-" + params["data"] + "-" + str(params["num_experts"]) + "-")
                data_output.to_csv(path + params["output"] + ".csv", header=True, index=False)

        else:
            ci_test.append(auc_test)
            if params["create_outputfile"]:
                if rep == 0:
                    data_output = output_file_creation(rep, model.num_tasks, auc_test, auc_val, auc_train, conf_interval, rep_start, params,)
                    path = (".//output//" + date_start + "/"  + params["model"] + "-" + params["data"] + "-" + str(params["num_experts"]) + "-" + params["task_balance_method"] + "/")
                    data_output.to_csv(path + params["output"] + ".csv", header=True, index=False)
                else:
                    _output = {"repetition": rep}
                    for i in range(model.num_tasks):
                        colname = "Task_" + str(i)
                        _output[colname + "_test"] = auc_test[i]
                        _output[colname + "_test_bs_l"] = conf_interval[i][0]
                        _output[colname + "_test_bs_u"] = conf_interval[i][1]
                        _output[colname + "_val"] = auc_val[i]
                        _output[colname + "_train"] = auc_train[i]
                    _output["time"] = time.time() - rep_start
                    _output["params"] = params
                    _output["data"] = params["data"]
                    _output["tasks"] = params["tasks"]
                    _output["model"] = params["model"]
                    _output["batch_size"] = params["batch_size"]
                    _output["max_epochs"] = params["max_epochs"]
                    _output["num_experts"] = params["num_experts"]
                    _output["num_units"] = params["num_units"]
                    _output["expert"] = try_keyerror("expert", params)
                    _output["expert_blocks"] = try_keyerror("expert_blocks", params)
                    _output["seqlen"] = try_keyerror("seqlen", params)
                    _output["runits"] = params["runits"]
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
                    data_output = data_output.append(_output, ignore_index=True)
                    data_output.to_csv("output//" + params["output"] + ".csv", header=True, index=False)

    """Calculating the Confidence Interval using Bootstrap"""
    if params["rep_ci"] > 1:
        model_CI(ci_test, model)
    print("...Best Epoch: ", best_epoch)
    print(params)


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
    main(config_path=sys.argv[1])
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
