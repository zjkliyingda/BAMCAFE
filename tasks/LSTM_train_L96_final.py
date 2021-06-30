# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Only for LSTM training
# This notebook is only for LSTM training; everytime using different model

# %%
import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")

sys.path.append(".")


from pkg.helper import patt_corr, rmse
from pkg.utils.logging import get_logger, init_logging
from pkg.utils.misc import (
    AverageMeter,
    compare_metric_value,
    export_json,
    makedirs,
    savefig,
    set_rand_seed,
)

# %matplotlib inline


def get_parser():
    parser = argparse.ArgumentParser(
        description="EnKS sampling and ML prediction"
    )
    parser.add_argument(
        "--rand_seed", type=int, default=50, help="Default: 200"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Two_layer_L96_final",
        help="Default: Two_layer_L96_final",
    )
    parser.add_argument(
        "--foldername",
        type=str,
        default="/n_steps=200001,dim_I=10,sparse_obs=True,"
        + "f=4.0,h=2.0,b=2.0,c=2.0,sigma_obs=1.0,sigma_u=1.0,sigma_v=1.0",
        help="/sigma_obs=1.0,sigma_u=1.0,sigma_v=1.0,n_steps=50001",
    )
    # this num_layers need to change to the dimension of the model
    parser.add_argument("--num_layers", type=int, default=1, help="Default: 1")
    parser.add_argument(
        "--hidden_size", type=int, default=32, help="Default: 32"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Default: 1e-3")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Default: 50"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Default: 0")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Default: 0"
    )
    parser.add_argument(
        "--training_type",
        type=str,
        default="imperfect_large",
        help="Default: imperfect",
    )
    parser.add_argument("--lead_time", type=int, default=1, help="Default: 1")
    parser.add_argument(
        "--n_segments", type=int, default=20, help="Default: 10"
    )
    parser.add_argument(
        "--num_ensembles_train", type=int, default=20, help="Default: 10"
    )
    return parser


def get_hparam_str(args):
    hparams = [
        "hidden_size",
        "num_layers",
        "lr",
        "batch_size",
        "num_epochs",
        "weight_decay",
        "n_segments",
    ]

    return ",".join("{}={}".format(p, getattr(args, p)) for p in hparams)


args = get_parser().parse_args()


# %%
root_folder = "/grad/yli678/workspace/smoother_sampling/output/"
model = args.model
foldername = args.foldername
output_path = root_folder + model + foldername
lstm_output_path = osp.join(
    output_path, "_lstm_training_type_all_val=" + args.training_type
)
lstm_output_path = osp.join(
    lstm_output_path, "lead_time=" + str(args.lead_time)
)
makedirs([lstm_output_path])

# %%
set_rand_seed(args.rand_seed)
init_logging(
    lstm_output_path, "logging_" + osp.join(get_hparam_str(args)) + ".log"
)
logger = get_logger(__file__)
logger.info(args)
logger.info(output_path)

# %%
# load data
data_sampling = np.load(output_path + "/sampling.npz")
model_prediction_data = np.load(output_path + "/dynamicalmodels.npz")


true_state = data_sampling["true_state"]
pred_dt = model_prediction_data["pred_dt"]
obs_dt = model_prediction_data["obs_dt"]
pred_start_time = model_prediction_data["pred_start_time"]
pred_total_time = model_prediction_data["pred_total_time"]
last_lead_t = model_prediction_data["last_lead_t"]
pred_last_lead_t = model_prediction_data["pred_last_lead_t"]
pred_num_ensembles = model_prediction_data["pred_num_ensembles"]
predict_mean = model_prediction_data["predict_mean"]
rmse_perfect = model_prediction_data["rmse_perfect"]
corr_perfect = model_prediction_data["corr_perfect"]
predict_mean_DA = model_prediction_data["predict_mean_DA"]
rmse_perfect_DA = model_prediction_data["rmse_perfect_DA"]
corr_perfect_DA = model_prediction_data["corr_perfect_DA"]


# %% [markdown]
# # Machine learning training

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# %%
json_filename = output_path + "/config.json"
with open(json_filename, "r") as f:
    configs = json.loads(f.read())

n_steps = configs["n_steps"]
dt = configs["dt"]
dim_I = configs["dim_I"]

pred_n_steps = int(dt * (n_steps - 1) / pred_dt) + 1
pred_K = int(pred_dt / dt)


t_pred = np.linspace(0, (pred_n_steps - 1) * pred_dt, pred_n_steps)
pred_idx = np.linspace(0, int((pred_n_steps - 1)) * pred_K, pred_n_steps)
pred_idx = pred_idx.astype(int)

obs_n_steps = int(dt * (n_steps - 1) / obs_dt) + 1
pred_obs_n_steps = int(obs_dt * (obs_n_steps - 1) / pred_dt) + 1
pred_obs_K = int(pred_dt / obs_dt)

pred_obs_idx = np.linspace(
    0, int((pred_obs_n_steps - 1)) * pred_obs_K, pred_obs_n_steps
)
pred_obs_idx = pred_obs_idx.astype(int)

model = configs["model"]
model_dim = true_state.shape[1]

lead_step = pred_K
pred_start_pred_step = int(pred_start_time / pred_dt)
pred_total_pred_steps = int(pred_total_time / pred_dt) + 1
lead_time_steps = np.linspace(
    0, last_lead_t, int(last_lead_t / lead_step) + 1
).astype(int)


# %%
class SampledTrajectoriesData(Dataset):
    def __init__(self, data):
        """
        data is a tuple, given the input of lstm and the true output
        """
        self.data = data

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        X, y = self.data[0], self.data[1]
        return X[idx], y[idx]  # use iterable as return, no need to implement


def data_transform(x, traj_len, lead_time):
    """
    change input of training dataset with length total_length into small pieces
    use 0-traj_len of data as input, use traj_len + 1 as the true label
    input:
        x: orginal training dataset
           shape: batch_size * total_length * model_dim
        traj_len: change
    """
    batch_size, total_length, model_dim = x.shape
    X = []
    y = []
    for i in range(batch_size):
        for j in range(total_length - traj_len - lead_time):
            X.append(x[i, j : j + traj_len + lead_time, :])
            y.append(x[i, j + traj_len + lead_time, :])
    return (
        torch.from_numpy(np.asarray(X)).float(),
        torch.from_numpy(np.asarray(y)).float(),
    )


# %%
# using all data in pred_start_pred_step as the training
num_ensembles_train = args.num_ensembles_train
# number of sampled trajectories used in LSTM
logger.info(f"{num_ensembles_train} of ensembles used in training")
if args.training_type == "truth":
    Y_samples_short = true_state[pred_idx][:pred_start_pred_step]
    logger.info("Training data: sampled trajectories from true signal")
elif args.training_type == "imperfect_large":
    Y_samples_short = data_sampling["gamma_ensembles"][
        :pred_start_pred_step, :, :num_ensembles_train
    ]
    predict_wrong = model_prediction_data["predict_wrong"]
    rmse_wrong = model_prediction_data["rmse_wrong"]
    corr_wrong = model_prediction_data["corr_wrong"]
    predict_wrong_DA = model_prediction_data["predict_wrong_DA"]
    rmse_wrong_DA = model_prediction_data["rmse_wrong_DA"]
    corr_wrong_DA = model_prediction_data["corr_wrong_DA"]
    logger.info("Training data: sampled trajectories from imperfect model")
elif args.training_type == "imperfect_onelayer":
    data_sampling_onelayer = np.load(output_path + "/sampling_onelayer.npz")
    Y_samples_short = data_sampling_onelayer["gamma_ensembles_onelayer"][
        :pred_start_pred_step, :, :num_ensembles_train
    ]
    predict_wrong = model_prediction_data["predict_wrong_onelayer"]
    rmse_wrong = model_prediction_data["rmse_wrong_onelayer"]
    corr_wrong = model_prediction_data["corr_wrong_onelayer"]
    predict_wrong_DA = model_prediction_data["predict_wrong_DA_onelayer"]
    rmse_wrong_DA = model_prediction_data["rmse_wrong_DA_onelayer"]
    corr_wrong_DA = model_prediction_data["corr_wrong_DA_onelayer"]
    logger.info(
        "Training data: sampled trajectories from onelayer L96"
        + " imperfect model"
    )

Y_samples_short = np.einsum("ijk->kij", Y_samples_short)
logger.info(f"Shape of trajectory: {Y_samples_short.shape}")
logger.info(
    f"{Y_samples_short.shape[0]} of total ensembles, "
    + f"{Y_samples_short.shape[1]} of total length, "
    + f"{Y_samples_short.shape[2]} of dimension"
)


# using large scale only
train_dim_idx = np.arange(0, dim_I)
Y_samples_short = Y_samples_short[:, :, train_dim_idx]
train_dim = len(train_dim_idx)
obs_dims = np.arange(1, dim_I, 2)
hidden_dims = np.arange(0, dim_I, 2)

n_segments = args.n_segments

train_length_total = Y_samples_short.shape[1]
train_length_each = int(train_length_total / n_segments)
logger.info(f"length of each training samples: {train_length_each}")

# in order to make sure the trajectories can but
Y_samples_short_train = Y_samples_short[:, : train_length_each * n_segments, :]


n_segments_val = 5
logger.info(
    f"Train segments: {n_segments - n_segments_val}; "
    + f" Validation segments: {n_segments_val}"
)
logger.info(f"Train : val = {(n_segments - n_segments_val) / n_segments_val}")
Y_samples_short_train = Y_samples_short[
    :, : train_length_each * (n_segments - n_segments_val), :
]
Y_samples_short_val = Y_samples_short[
    :, train_length_each * (n_segments - n_segments_val) :, :
]

logger.info(
    f"training shape: {Y_samples_short_train.shape}; "
    + f" validation shape: {Y_samples_short_val.shape}"
)

Y_samples_short_train = Y_samples_short_train.reshape(
    (n_segments - n_segments_val) * num_ensembles_train, -1, train_dim
)
logger.info(Y_samples_short.shape)

if train_length_each // 2 < args.lead_time:
    logger.info("The length of segment is too short")

X_train, y_train = (
    torch.from_numpy(
        np.asarray(Y_samples_short_train[:, : -args.lead_time, :])
    ).float(),
    torch.from_numpy(
        np.asarray(Y_samples_short_train[:, args.lead_time :, :])
    ).float(),
)

sampled_traj_data = SampledTrajectoriesData((X_train, y_train))
logger.info(f"total number of training data: {sampled_traj_data.__len__()}")
logger.info(f"X_train.shape: {X_train.shape}")
logger.info(f"Equivalent {X_train.shape[1] * pred_dt} time units")

# %%
X_val, y_val = (
    torch.from_numpy(
        np.asarray(Y_samples_short_val[:, : -args.lead_time, :])
    ).float(),
    torch.from_numpy(
        np.asarray(Y_samples_short_val[:, args.lead_time :, :])
    ).float(),
)
batch_val = X_val.shape[0]
logger.info(f"Shape of validation to dataLoader: {X_val.shape}")


# %%
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.rnn(
            x
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out)
        return out

    def train_epoch(
        self,
        train_dataloader,
        optimizer,
        valid_dataloader=None,
        **kwargs,
    ):
        self.train()
        train_metrics = defaultdict(AverageMeter)
        train_metrics_ind = []
        for dim in range(train_dim):
            train_metrics_ind.append(defaultdict(AverageMeter))
        step = 0

        for X, y in train_dataloader:
            output = self(X)
            loss_ind = [0] * train_dim
            for dim in range(train_dim):
                loss_ind[dim] = F.mse_loss(output[:, :, dim], y[:, :, dim])
            loss = F.mse_loss(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            for dim in range(train_dim):
                train_metrics_ind[dim]["loss"].update(
                    loss_ind[dim].item(), X.shape[0]
                )
            train_metrics["loss"].update(loss.item(), X.shape[0])

        if valid_dataloader:
            (
                eval_metrics,
                eval_metrics_ind,
                eval_metrics_ind_details_full,
            ) = self.evaluate(valid_dataloader)
        else:
            eval_metrics = None

        return (
            train_metrics,
            train_metrics_ind,
            eval_metrics,
            eval_metrics_ind,
            eval_metrics_ind_details_full,
        )

    def evaluate(self, dataloader):
        metrics = defaultdict(AverageMeter)
        metrics_ind = []
        for dim in range(train_dim):
            metrics_ind.append(defaultdict(AverageMeter))
        self.eval()
        step = 0
        with torch.no_grad():
            for X, y in dataloader:
                output = self(X)
                loss_ind = [0] * train_dim
                loss_ind_full = [[] for _ in range(train_dim)]
                for dim in range(train_dim):
                    loss_ind[dim] = F.mse_loss(output[:, :, dim], y[:, :, dim])
                    metrics_ind[dim]["loss"].update(
                        loss_ind[dim].item(), X.shape[0]
                    )
                loss = F.mse_loss(output, y)
                step += 1
                metrics["loss"].update(loss.item(), X.shape[0])
                # only for compute the loss_ind_full
                for dim in range(train_dim):
                    for sampled_dim in range(X.shape[0]):
                        loss_ind_full[dim].append(
                            (
                                y[sampled_dim, :, dim]
                                - output[sampled_dim, :, dim]
                            ).tolist()
                        )
        return metrics, metrics_ind, loss_ind_full


# %%
def train_nn_model(model, optimizer, num_epochs, batch_size):
    sampled_traj_data = SampledTrajectoriesData((X_train, y_train))
    val_traj_data = SampledTrajectoriesData((X_val, y_val))
    train_dataloader = DataLoader(sampled_traj_data, batch_size=batch_size)
    valid_dataloader = DataLoader(val_traj_data, batch_size=int(batch_val))
    train_arr_all = []
    train_arr_ind_all = []
    val_arr_all = []
    val_arr_ind_all = []
    best_metric = float("nan")
    suffix = osp.join(get_hparam_str(args))
    for epoch in tqdm(range(num_epochs)):
        (
            train_metrics,
            train_metrics_ind,
            valid_metrics,
            valid_metrics_ind,
            valid_metrics_ind_details_full,
        ) = model.train_epoch(
            train_dataloader,
            optimizer,
            valid_dataloader,
        )

        msg = f"[Training] Epoch={epoch}"
        for k, v in train_metrics.items():
            msg += f", {k}={v.avg:.4g}"
            train_arr_all.append(v.avg)
        logger.info(msg)
        msg = f"[Validation] Epoch={epoch}"
        for k, v in valid_metrics.items():
            msg += f", {k}={v.avg:.4g}"
            val_arr_all.append(v.avg)
        logger.info(msg)
        train_arr_ind = []
        val_arr_ind = []
        for dim in range(train_dim):
            for k, v in train_metrics_ind[dim].items():
                train_arr_ind.append(v.avg)
            for k, v in valid_metrics_ind[dim].items():
                val_arr_ind.append(v.avg)
        train_arr_ind_all.append(train_arr_ind)
        val_arr_ind_all.append(val_arr_ind)
        if compare_metric_value(valid_metrics["loss"].avg, best_metric):
            if epoch > args.num_epochs // 2:
                logger.info(f"Found a better model at epoch {epoch}.")
            best_metric = valid_metrics["loss"].avg
            train_arr_ind_dict = {"train_arr_ind": train_arr_ind}
            val_arr_ind_dict = {"val_arr_ind": val_arr_ind}
            val_arr_ind_dict_details_full = {
                "val_arr_ind_details": valid_metrics_ind_details_full
            }
            export_json(
                train_arr_ind_dict,
                osp.join(lstm_output_path, "train_err_" + suffix + ".json"),
            )
            export_json(
                val_arr_ind_dict,
                osp.join(lstm_output_path, "val_err_" + suffix + ".json"),
            )
            export_json(
                val_arr_ind_dict_details_full,
                osp.join(
                    lstm_output_path,
                    "val_err_details_full_" + suffix + ".json",
                ),
            )
            torch.save(
                model.state_dict(),
                osp.join(lstm_output_path, "model_" + suffix + ".pt"),
            )

    model.load_state_dict(
        torch.load(osp.join(lstm_output_path, "model_" + suffix + ".pt"))
    )

    return (
        model,
        train_arr_all,
        val_arr_all,
    )


num_epochs = args.num_epochs

lstm_model = LSTM(
    input_size=train_dim,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    output_size=train_dim,
)
learning_rate = args.lr
optimizer = torch.optim.Adam(
    lstm_model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
)

(lstm_model, train_arr_all, val_arr_all,) = train_nn_model(
    lstm_model, optimizer, num_epochs, batch_size=args.batch_size
)

# %%
fig = plt
plt.plot(np.asarray(train_arr_all), label="train")
plt.plot(val_arr_all, label="validation")
plt.legend()
plt.yscale("log")
plt.title("loss_" + osp.join(get_hparam_str(args)) + ".pdf")
savefig(
    fig,
    osp.join(
        lstm_output_path,
        "loss_" + osp.join(get_hparam_str(args)) + ".pdf",
    ),
)

# %% [markdown]
# ## LSTM prediction

# %%
# using perfect initial condition
pred_metrics = defaultdict(AverageMeter)
lstm_model.eval()

# some length for initialize the model prediction
# len_init = X_train.shape[
#     1
# ]  # use the entire length of X_train to initialize the LSTM model
# int(pred_n_steps / args.n_segments): total length for each segment
len_init = 1


initial_values = true_state[pred_idx][
    pred_start_pred_step
    - args.lead_time
    - len_init : pred_start_pred_step
    + pred_total_pred_steps,
    train_dim_idx,
]

initial_values_LSTM = np.empty(
    (
        1,
        pred_total_pred_steps + args.lead_time + len_init,
        train_dim,
    )
)
for i in range(1):
    initial_values_LSTM[i] = initial_values

target = true_state[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps,
    train_dim_idx,
]

target = torch.from_numpy(target).float()

X_pred, y_pred = (
    torch.from_numpy(
        np.asarray(initial_values_LSTM[:, : -args.lead_time, :])
    ).float(),
    torch.from_numpy(
        np.asarray(initial_values_LSTM[:, args.lead_time :, :])
    ).float(),
)
pred_traj_data = SampledTrajectoriesData((X_pred, y_pred))
pred_dataloader = DataLoader(pred_traj_data, batch_size=int(1))
logger.info(f"X_pred shape for perfect initial condition: {X_pred.shape}")
with torch.no_grad():
    for X, y in pred_dataloader:

        output = lstm_model(X)
        predict_lstm_out = output[:, len_init:, :]
        logger.info(
            "predict_lstm_out shape for perfect initial condition: "
            + f"{predict_lstm_out.shape}"
        )
        predict_lstm = torch.mean(predict_lstm_out, axis=0)
        logger.info(
            "predict_lstm shape for perfect initial condition: "
            + f" {predict_lstm.shape}"
        )
        #         predict_lstm = predict_lstm_out[0]
        loss = F.mse_loss(predict_lstm, target)
        pred_metrics["loss"].update(loss.item(), X.shape[0])
        logger.info(pred_metrics["loss"].avg)
        predict_lstm = predict_lstm.numpy()

# %%
lead_time = args.lead_time

# %%
if args.training_type == "imperfect_large":
    gamma_mean_smooth = data_sampling["gamma_mean_smooth"]
    data_sampling_for_IC = np.load(output_path + "/sampling_for_IC.npz")
    gamma_ensembles_for_IC_short = data_sampling_for_IC[
        "gamma_ensembles_for_IC_short"
    ]
elif args.training_type == "imperfect_onelayer":
    gamma_mean_smooth = data_sampling_onelayer["gamma_mean_smooth_onelayer"]
    data_sampling_for_IC = np.load(
        output_path + "/sampling_for_IC_onelayer.npz"
    )
    gamma_ensembles_for_IC_short = data_sampling_for_IC[
        "gamma_ensembles_for_IC_short"
    ]
# %%
initial_values_mean_smooth = gamma_mean_smooth[
    pred_start_pred_step
    - args.lead_time
    - len_init : pred_start_pred_step
    + pred_total_pred_steps,
    train_dim_idx,
]
samples_initial_mean = np.mean(
    gamma_ensembles_for_IC_short[
        (pred_last_lead_t - lead_time) : (
            pred_last_lead_t - lead_time + pred_total_pred_steps
        ),
        :,
        :,
        :,
    ],
    axis=3,
)

# %%
# using DA IC
num_plt_dims = 6
fig, axs = plt.subplots(
    nrows=num_plt_dims,
    ncols=2,
    figsize=(18, 2 * num_plt_dims),
    constrained_layout=True,
)
target = true_state[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
]

pred_metrics = defaultdict(AverageMeter)
lstm_model.eval()

predict_lstm_DA_out = torch.empty(
    (pred_total_pred_steps, pred_num_ensembles, train_dim)
)
warmup_len = 1
for i in range(pred_num_ensembles):
    initial_samples = gamma_ensembles_for_IC_short[
        (pred_last_lead_t - lead_time) : (
            pred_last_lead_t - lead_time + pred_total_pred_steps
        ),
        :,
        train_dim_idx,
        i,
    ]
    for plt_dim in range(num_plt_dims):
        axs[plt_dim, 0].plot(
            initial_values[len_init : -args.lead_time][:, plt_dim],
            "black",
        )
        axs[plt_dim, 0].plot(initial_samples[:, -1, plt_dim], "lime")
        axs[plt_dim, 0].plot(
            initial_values_mean_smooth[len_init : -args.lead_time][:, plt_dim],
            "orange",
        )
        axs[plt_dim, 0].plot(samples_initial_mean[:, -1, plt_dim], "r")
        axs[plt_dim, 0].set_xlim([0, 400])
    #     logger.info(initial_samples.shape)
    initial_values_LSTM = initial_samples[:, warmup_len:, :]
    # y_pred here is useless!!!
    X_pred, y_pred = (
        torch.from_numpy(np.asarray(initial_values_LSTM[:, :, :])).float(),
        torch.from_numpy(np.asarray(initial_values_LSTM[:, :, :])).float(),
    )
    pred_traj_data = SampledTrajectoriesData((X_pred, y_pred))
    pred_dataloader = DataLoader(
        pred_traj_data, batch_size=int(pred_total_pred_steps)
    )
    with torch.no_grad():
        for X, y in pred_dataloader:
            output = lstm_model(X)
            predict_lstm_DA_out[:, i, :] = output[:, -1, :]
    for plt_dim in range(num_plt_dims):
        axs[plt_dim, 1].plot(
            predict_lstm_DA_out[:, i, plt_dim].numpy(), "lime"
        )
        axs[plt_dim, 1].plot(target[:, plt_dim], "black")
        axs[plt_dim, 1].set_xlim([0, 400])
predict_lstm_DA = torch.mean(predict_lstm_DA_out, axis=1)
pred_metrics["loss"].update(loss.item(), X.shape[0])
logger.info(pred_metrics["loss"].avg)
predict_lstm_DA = predict_lstm_DA.numpy()
for plt_dim in range(num_plt_dims):
    axs[plt_dim, 1].plot(predict_lstm_DA[:, plt_dim], "r")
    axs[plt_dim, 1].plot(predict_mean_DA[lead_time, :, plt_dim], "b")

# %%
target = true_state[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps,
    train_dim_idx,
]
rmse_lstm_DA = rmse(
    predict_lstm_DA,
    target,
)
logger.info(rmse_lstm_DA)

# %%
target = true_state[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps,
    train_dim_idx,
]
rmse_lstm = rmse(
    predict_lstm,
    target,
)
rmse_lstm_DA = rmse(
    predict_lstm_DA,
    target,
)
corr_lstm = 0
corr_lstm_DA = 0

corr_lstm = patt_corr(
    predict_lstm,
    target,
)
corr_lstm_DA = patt_corr(
    predict_lstm_DA,
    target,
)

np.savez(
    osp.join(
        lstm_output_path,
        "results_DA_" + osp.join(get_hparam_str(args)) + ".npz",
    ),
    predict_lstm_DA=predict_lstm_DA,
    rmse_lstm_DA=rmse_lstm_DA,
    corr_lstm_DA=corr_lstm_DA,
)
np.savez(
    osp.join(
        lstm_output_path, "results_" + osp.join(get_hparam_str(args)) + ".npz"
    ),
    predict_lstm=predict_lstm,
    rmse_lstm=rmse_lstm,
    corr_lstm=corr_lstm,
)

# %%
logger.info(
    "Avg RMSE for perfect model using perfect IC:"
    + f" {rmse_perfect[args.lead_time][:dim_I].mean()}"
)
logger.info(
    "Avg RMSE for imperfect model using perfect IC: "
    + f"{rmse_wrong[args.lead_time][:dim_I].mean()}"
)
logger.info(
    "Avg RMSE for lstm model using perfect IC: "
    + f"{rmse_lstm[:dim_I].mean()}"
)
logger.info("\n")
logger.info(
    "Avg Corr for perfect model using perfect IC: "
    + f"{corr_perfect[args.lead_time][:dim_I].mean()}"
)
logger.info(
    "Avg Corr for imperfect model using perfect IC: "
    + f"{corr_wrong[args.lead_time][:dim_I].mean()}"
)
logger.info(
    "Avg Corr for lstm model using perfect IC: "
    + f"{corr_lstm[:dim_I].mean()}"
)
logger.info("\n")


# %%
logger.info(
    "Avg RMSE for perfect model using DA IC: "
    + f"{rmse_perfect_DA[args.lead_time][:dim_I].mean()}"
)
logger.info(
    "Avg RMSE for imperfect model using DA IC: "
    + f"{rmse_wrong_DA[args.lead_time][:dim_I].mean()}"
)
logger.info(
    "Avg RMSE for lstm model using DA IC: " + f"{rmse_lstm_DA[:dim_I].mean()}"
)
logger.info("\n")
logger.info(
    "Avg Corr for perfect model using DA IC: "
    + f"{corr_perfect_DA[args.lead_time][:dim_I].mean()}"
)
logger.info(
    "Avg Corr for imperfect model using DA IC: "
    + f"{corr_wrong_DA[args.lead_time][:dim_I].mean()}"
)

logger.info(
    "Avg Corr for lstm model using DA IC: " + f"{corr_lstm_DA[:dim_I].mean()}"
)
# %%
t = np.linspace(0, (n_steps - 1) * dt, n_steps)
t_pred = np.linspace(0, (pred_n_steps - 1) * pred_dt, pred_n_steps)
predict_idx = args.lead_time

plt.figure(figsize=(18, 2))

num_plt_dims = 6
fig, axs = plt.subplots(
    nrows=num_plt_dims,
    ncols=3,
    figsize=(18, 2 * num_plt_dims),
    gridspec_kw={"width_ratios": [9, 1, 1]},
    constrained_layout=True,
)

for i in range(num_plt_dims):
    pred_dim = i
    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        target[:, pred_dim],
        "black",
        label="Truth",
    )
    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        predict_mean[predict_idx, :, pred_dim],
        "b",
        label="Perfect model",
    )
    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        predict_wrong[predict_idx, :, pred_dim],
        "r",
        label="Imperfect model",
    )

    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        predict_lstm[:, pred_dim],
        "lime",
        label="LSTM",
    )
    axs[i, 0].set_title(
        f"Lead time = {lead_time_steps[predict_idx] * dt}, "
        + f"pred_dim = {pred_dim + 1}"
    )

    axs[i, 1].plot([0.5], [rmse_perfect[args.lead_time, pred_dim]], "bo")
    axs[i, 1].plot([0.5], [rmse_wrong[args.lead_time, pred_dim]], "ro")
    axs[i, 1].plot([0.5], [rmse_lstm[pred_dim]], "o", color="lime")
    axs[i, 1].set_title("RMSE")

    axs[i, 2].plot([0.5], [corr_perfect[args.lead_time, pred_dim]], "bo")
    axs[i, 2].plot([0.5], [corr_wrong[args.lead_time, pred_dim]], "ro")
    axs[i, 2].plot([0.5], [0.5], "o", color="black")
    axs[i, 2].plot([0.5], [corr_lstm[pred_dim]], "o", color="lime")
    axs[i, 2].set_title("Corr")

plt.suptitle(
    "Ave RMSE perfect model: "
    + f"{rmse_perfect[args.lead_time][:dim_I].mean().round(3)}, "
    + "Ave RMSE imperfect model: "
    + f"{rmse_wrong[args.lead_time][:dim_I].mean().round(3)}, "
    + f"Ave RMSE LSTM: {rmse_lstm[:dim_I].mean().round(3)},"
    + "Ave hid_dim RMSE LSTM: "
    + f"{rmse_lstm[:dim_I][hidden_dims].mean().round(3)}"
    + f"Ave obs_dim RMSE LSTM: {rmse_lstm[:dim_I][obs_dims].mean().round(3)}",
    fontsize="x-large",
)
savefig(
    fig,
    osp.join(
        lstm_output_path,
        "TRAJ_u_" + osp.join(get_hparam_str(args)) + ".pdf",
    ),
)


# %%
t = np.linspace(0, (n_steps - 1) * dt, n_steps)
t_pred = np.linspace(0, (pred_n_steps - 1) * pred_dt, pred_n_steps)
predict_idx = args.lead_time

plt.figure(figsize=(18, 2))

fig, axs = plt.subplots(
    nrows=num_plt_dims,
    ncols=3,
    figsize=(18, num_plt_dims * 2),
    gridspec_kw={"width_ratios": [9, 1, 1]},
    constrained_layout=True,
)

for i in range(num_plt_dims):
    pred_dim = i
    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        target[:, pred_dim],
        "black",
        label="Truth",
    )
    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        predict_mean_DA[predict_idx, :, pred_dim],
        "b",
        label="Perfect model",
    )
    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        predict_wrong_DA[predict_idx, :, pred_dim],
        "r",
        label="Imperfect model",
    )

    axs[i, 0].plot(
        t_pred[
            pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
        ],
        predict_lstm_DA[:, pred_dim],
        "lime",
        label="LSTM",
    )
    axs[i, 0].set_title(
        f"Lead time = {lead_time_steps[predict_idx] * dt},"
        + f" pred_dim = {pred_dim + 1}"
    )

    axs[i, 1].plot([0.5], [rmse_perfect_DA[args.lead_time, pred_dim]], "bo")
    axs[i, 1].plot([0.5], [rmse_wrong_DA[args.lead_time, pred_dim]], "ro")
    axs[i, 1].plot([0.5], [rmse_lstm_DA[pred_dim]], "o", color="lime")
    axs[i, 1].set_title("RMSE")

    axs[i, 2].plot([0.5], [corr_perfect_DA[args.lead_time, pred_dim]], "bo")
    axs[i, 2].plot([0.5], [corr_wrong_DA[args.lead_time, pred_dim]], "ro")
    axs[i, 2].plot([0.5], [0.5], "o", color="black")
    axs[i, 2].plot([0.5], [corr_lstm_DA[pred_dim]], "o", color="lime")
    axs[i, 2].set_title("Corr")

plt.suptitle(
    "Ave RMSE perfect model: "
    + f"{rmse_perfect_DA[args.lead_time][:dim_I].mean().round(3)}, "
    + "Ave RMSE imperfect model: "
    + f"{rmse_wrong_DA[args.lead_time][:dim_I].mean().round(3)}, "
    + f"Ave RMSE LSTM: {rmse_lstm_DA[:dim_I].mean().round(3)},"
    + "Ave hid_dim RMSE LSTM: "
    + f"{rmse_lstm_DA[:dim_I][hidden_dims].mean().round(3)}"
    + "Ave obs_dim RMSE LSTM: "
    + f"{rmse_lstm_DA[:dim_I][obs_dims].mean().round(3)}",
    fontsize="x-large",
)
savefig(
    fig,
    osp.join(
        lstm_output_path,
        "TRAJ_DA_u_" + osp.join(get_hparam_str(args)) + ".pdf",
    ),
)
