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
# # Basic setup and helper functions

# %%
import argparse
import os
import os.path as osp
import sys
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm
from tqdm import tqdm

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.helper import (
    compute_kde_axis,
    estimated_autocorrelation,
    patt_corr,
    rmse,
)
from pkg.utils.logging import get_logger, init_logging
from pkg.models.model_two_layer_L96 import (
    EnsembleKalmanFilterSmootherL96,
    EnsembleKalmanFilterSmootherL96OneLayer,
    EnsembleKalmanFilterSmootherL96SPEKF,
)
from pkg.models.model_two_layer_L96_perfect_numba_class import (
    EnsembleKalmanFilterSmootherL96Numba,
)
from pkg.models.model_two_layer_L96_spekf_numba_class import (
    EnsembleKalmanFilterSmootherL96SPEKFNumba,
)
from pkg.utils.misc import export_json, makedirs, savefig, set_rand_seed

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


def get_parser():
    parser = argparse.ArgumentParser(
        description="EnKS sampling and ML prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Two_layer_L96_final",
        help="Default: Two_layer_L96",
    )
    parser.add_argument(
        "--n_steps", type=int, default=50001, help="Default: 100001"
    )
    parser.add_argument(
        "--pred_start_time", type=int, default=20, help="Default: 20"
    )
    parser.add_argument(
        "--pred_total_time", type=int, default=20, help="Default: 20"
    )
    parser.add_argument(
        "--last_lead_time", type=int, default=3, help="Default: 3"
    )
    parser.add_argument("--dt", type=float, default=1e-3, help="Default: 5e-3")
    parser.add_argument(
        "--obs_dt", type=float, default=5e-2, help="Default: 5e-2"
    )
    parser.add_argument("--h", type=float, default=2.0, help="Default: 2.")
    parser.add_argument("--b", type=float, default=2.0, help="Default: 2.")
    parser.add_argument("--c", type=float, default=2.0, help="Default: 2.")
    parser.add_argument("--f", type=float, default=4.0, help="Default: 4.")
    parser.add_argument("--dim_I", type=int, default=40, help="Default: 5")
    parser.add_argument("--dim_J", type=int, default=4, help="Default: 4")
    parser.add_argument(
        "--sigma_obs", type=float, default=1.0, help="Default: 0.4"
    )
    parser.add_argument(
        "--sigma_u", type=float, default=1.0, help="Default: 0.5"
    )
    parser.add_argument(
        "--sigma_v", type=float, default=1.0, help="Default: 1"
    )
    parser.add_argument(
        "--stop_at_sampling", type=bool, default=False, help="Default: False"
    )
    parser.add_argument(
        "--stop_at_modeling", type=bool, default=False, help="Default: False"
    )
    parser.add_argument(
        "--finalized_mode", type=bool, default=False, help="Default: False"
    )
    parser.add_argument(
        "--sparse_obs", type=bool, default=False, help="Default: False"
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="Default: output"
    )
    parser.add_argument(
        "--rand_seed", type=int, default=200, help="Default: 200"
    )
    return parser


def get_hparam_str(args):
    hparams = [
        "n_steps",
        "dim_I",
        "sparse_obs",
        "f",
        "h",
        "b",
        "c",
        "sigma_obs",
        "sigma_u",
        "sigma_v",
    ]

    return ",".join("{}={}".format(p, getattr(args, p)) for p in hparams)


Params = namedtuple(
    "Params",
    [
        "sigma_obs",
        "sigma_u",
        "sigma_v",
        "dim_I",
        "dim_J",
        "h",
        "b",
        "c",
        "f",
        "d_v",
        "v_hat",
        "sigma_v_hat",
    ],
)


args = get_parser().parse_args()

output_path = osp.join(args.output_path, args.model, get_hparam_str(args))
makedirs([output_path])


def plot_v(state):
    return np.hstack((state[:, dim_I:], state[:, dim_I][:, None]))


def plot_u(state):
    return np.hstack((state[:, :dim_I], state[:, 0][:, None]))


dim_I = args.dim_I
dim_J = args.dim_J


n_steps, dt, obs_dt, model_dim = (
    args.n_steps,
    args.dt,
    args.obs_dt,
    dim_I + dim_I * dim_J,
)
obs_n_steps = int(dt * (n_steps - 1) / obs_dt) + 1
K = int(obs_dt / dt)
t = np.linspace(0, (n_steps - 1) * dt, n_steps)
t_obs = np.linspace(0, (obs_n_steps - 1) * obs_dt, obs_n_steps)
obs_idx = np.linspace(0, int((obs_n_steps - 1)) * K, obs_n_steps)
obs_idx = obs_idx.astype(int)


params_truth = Params(
    args.sigma_obs,
    args.sigma_u,  # sigma_u
    args.sigma_v,  # sigma_v,
    dim_I,  # dim_I
    dim_J,  # dim_J
    args.h,  # h
    args.b,  # b
    args.c,  # c
    args.f,  # f
    None,
    None,
    None,
)

init_logging(output_path, "logging_data_generating.log")
logger = get_logger(__file__)
logger.info(args)
logger.info(output_path)

logger.info(params_truth)
params_truth_arr = []
for k, v in zip(params_truth._fields, params_truth):
    params_truth_arr.append(v)
params_truth_arr = np.asarray(params_truth_arr, dtype="float64")

if args.sparse_obs:
    obs_dims = list(range(1, dim_I, 2))
else:
    obs_dims = list(range(0, dim_I))
num_obs_dims = len(obs_dims)
trans_mat = np.zeros((num_obs_dims, model_dim))
obs_noise = np.ones(num_obs_dims) * args.sigma_obs
for i in range(num_obs_dims):
    trans_mat[i, obs_dims[i]] = 1

logger.info(f"model_dim={model_dim}")
logger.info(f"obs_dims = {obs_dims}")
logger.info(f"trans_mat.shape = {trans_mat.shape}")
# logger.info(obs_noise)

model_dim_onelayer = dim_I
trans_mat_onelayer = np.zeros((num_obs_dims, dim_I))
trans_mat_onelayer = trans_mat[:, :dim_I]
obs_noise_onelayer = obs_noise
export_json(
    params_truth._asdict(), osp.join(output_path, "true_params_used.json")
)
config_dict = vars(args)
config_dict.update({"obs_dims": obs_dims})
export_json(
    config_dict,
    osp.join(output_path, "config.json"),
)

# %%
L96_EnKF = EnsembleKalmanFilterSmootherL96(
    n_steps, model_dim, params_truth, obs_dt, trans_mat, obs_noise, dt
)

L96_EnKF_numba = EnsembleKalmanFilterSmootherL96Numba(
    n_steps, model_dim, obs_dt, trans_mat, obs_noise, dt
)

# %%
start_time = time.time()
obs, true_state = L96_EnKF_numba.simulate(
    params_truth_arr, np.zeros(model_dim), args.rand_seed
)
logger.info("--- %s seconds ---" % (time.time() - start_time))


# %%
# start_time = time.time()
# true_state_long_traj = L96_EnKF_numba.simulate_model_only(
#     int(500 / dt), params_truth_arr, np.zeros(model_dim)
# )
# logger.info("--- %s seconds ---" % (time.time() - start_time))

# %%
from scipy.optimize import curve_fit


def func(x, a, b):
    return a * np.exp(-b * x) + 1 - a


# %%
if n_steps > 200000:
    burnin = 600  # in dt space
else:
    burnin = 150
obs_burnin = burnin // K  # in obs_dt space

# %%
auto_obs_steps = obs_n_steps - obs_burnin
t_auto = np.linspace(0, (auto_obs_steps - 1) * obs_dt, auto_obs_steps)
acf_u_truth = np.zeros((auto_obs_steps, model_dim))
for i in tqdm(range(model_dim)):
    acf_u_truth[:, i] = estimated_autocorrelation(
        true_state[obs_idx, i][obs_burnin:]
    )

# %%
i = dim_I
estimated_autocorrelation_v = estimated_autocorrelation(
    true_state[obs_idx, i][obs_burnin:]
)
auto_trunk = int(5 / obs_dt)
popt, pcov = curve_fit(
    func, t_auto[:auto_trunk], estimated_autocorrelation_v[:auto_trunk]
)

# %%
plt.plot(t_auto, acf_u_truth[:, dim_I], "b")
plt.plot(
    t_auto[:auto_trunk],
    func(t_auto[:auto_trunk], *popt),
    "r-",
)
plt.xlim([0, 5])

# %%
v_mean = np.mean(true_state[obs_idx, dim_I][obs_burnin:])
v_var = np.var(true_state[obs_idx, dim_I][obs_burnin:])

# %%
d_v = 1 / np.sum(func(t_auto[:auto_trunk], *popt)) / args.obs_dt
sigma_v_hat = np.sqrt(v_var * 2 * d_v)
logger.info(v_mean)
logger.info(d_v)
logger.info(sigma_v_hat)

# %%
params_wrong = Params(
    params_truth.sigma_obs,
    params_truth.sigma_u,  # sigma_u
    params_truth.sigma_v,  # sigma_v,
    params_truth.dim_I,  # dim_I
    params_truth.dim_J,  # dim_J
    params_truth.h,  # h
    params_truth.b,  # b
    params_truth.c,  # c
    params_truth.f,  # f
    d_v,  # d_v,  # d_v
    v_mean,  # v_mean,  # v_hat
    sigma_v_hat,  # sigma_v_hat,  # sigm
)

params_onelayer = Params(
    params_truth.sigma_obs,
    params_truth.sigma_u,  # sigma_u
    params_truth.sigma_v,  # sigma_v,
    params_truth.dim_I,  # dim_I
    0,  # dim_J
    params_truth.h,  # h
    params_truth.b,  # b
    params_truth.c,  # c
    params_truth.f,  # f
    params_truth.d_v,  # d_v,  # d_v
    params_truth.v_hat,  # v_mean,  # v_hat
    params_truth.sigma_v_hat,  # sigma_v_hat,  # sigm
)

params_wrong_arr = []
for k, v in zip(params_wrong._fields, params_wrong):
    params_wrong_arr.append(v)
params_wrong_arr = np.asarray(params_wrong_arr, dtype="float64")

export_json(
    params_wrong._asdict(),
    osp.join(output_path, "SPEKF_model_params.json"),
)

export_json(
    params_onelayer._asdict(),
    osp.join(output_path, "onelayer_params.json"),
)

# %%
L96_EnKF_SPEKF = EnsembleKalmanFilterSmootherL96SPEKF(
    n_steps, model_dim, params_wrong, obs_dt, trans_mat, obs_noise, dt
)

L96_EnKF_spekf_numba = EnsembleKalmanFilterSmootherL96SPEKFNumba(
    n_steps, model_dim, obs_dt, trans_mat, obs_noise, dt
)
L96_EnKF_onelayer = EnsembleKalmanFilterSmootherL96OneLayer(
    n_steps,
    model_dim_onelayer,
    params_onelayer,
    obs_dt,
    trans_mat_onelayer,
    obs_noise_onelayer,
    dt,
)
# %%
start_time = time.time()
_, wrong_state = L96_EnKF_spekf_numba.simulate(
    params_wrong_arr, np.zeros(model_dim), args.rand_seed
)
logger.info("--- %s seconds ---" % (time.time() - start_time))

set_rand_seed(args.rand_seed)
_, wrong_state_onelayer = L96_EnKF_onelayer.simulate()
# %%
u_truth, v_truth = true_state[:, 0], true_state[:, dim_I]

kde_u = gaussian_kde(u_truth.flatten())
u_axis = np.linspace(u_truth.min(), u_truth.max(), num=100)
kde_v = gaussian_kde(v_truth.flatten())
v_axis = np.linspace(v_truth.min(), v_truth.max(), num=100)

mean_u, std_u = norm.fit(u_truth)
mean_v, std_v = norm.fit(v_truth)

u_wrong, v_wrong = wrong_state[:, 0], wrong_state[:, dim_I]

kde_u_wrong = gaussian_kde(u_wrong.flatten())
u_axis_wrong = np.linspace(u_wrong.min(), u_wrong.max(), num=100)
kde_v_wrong = gaussian_kde(v_wrong.flatten())
v_axis_wrong = np.linspace(v_wrong.min(), v_wrong.max(), num=100)

mean_u_wrong, std_u_wrong = norm.fit(u_wrong)
mean_v_wrong, std_v_wrong = norm.fit(v_wrong)

u_wrong_onelayer = wrong_state_onelayer[:, 0]

kde_u_wrong_onelayer = gaussian_kde(u_wrong_onelayer.flatten())
u_axis_wrong_onelayer = np.linspace(
    u_wrong_onelayer.min(), u_wrong_onelayer.max(), num=100
)

mean_u_wrong_onelayer, std_u_wrong_onelayer = norm.fit(u_wrong_onelayer)

# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("legend", fontsize=14)
fig, axs = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(15, 5),
    gridspec_kw={"width_ratios": [4, 1]},
    constrained_layout=True,
)
# fig.suptitle(output_path, fontsize=16)
if n_steps > 50001:
    x_left = 10
    x_right = 50
else:
    x_left = 0
    x_right = n_steps * dt

axs[0, 0].plot(t, true_state[:, 0], color="b", label="Truth")
axs[0, 0].scatter(
    t_obs, obs[:, 0], s=40, facecolors="none", label="obs", edgecolors="black"
)
axs[0, 0].plot(t, wrong_state[:, 0], color="r", label="Imperfect model")
axs[0, 0].set_title("(a) Trajectory of $u_1$")
axs[0, 0].set_xlim([x_left, x_right])

axs[0, 1].plot(u_axis, kde_u(u_axis), color="b")
axs[0, 1].plot(u_axis_wrong, kde_u_wrong(u_axis_wrong), color="r")
axs[0, 1].set_title("(c) PDF of $u_1$")

for i in range(1):

    axs[1 + i, 0].plot(t, true_state[:, dim_I + i], color="b")
    axs[1 + i, 0].plot(t, wrong_state[:, dim_I + i], color="r")
    axs[1 + i, 0].set_title("(b) Trajectory of $v_{11}$")
    axs[1, 0].set_xlim([x_left, x_right])

axs[1, 1].plot(v_axis, kde_v(v_axis), color="b")
axs[1, 1].plot(v_axis_wrong, kde_v_wrong(v_axis_wrong), color="r")
axs[1, 1].set_title("(d) PDF of $v_{11}$")


handles, labels = axs[0, 0].get_legend_handles_labels()
axs[1, 0].legend(
    handles, labels, ncol=3, loc="upper right", bbox_to_anchor=(0.85, -0.15)
)
savefig(
    fig, osp.join(output_path, "trajectory_wrong_SPEKF_" + args.model + ".pdf")
)

plt.rcdefaults()


# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("legend", fontsize=14)
fig, axs = plt.subplots(
    ncols=2,
    figsize=(15, 3),
    gridspec_kw={"width_ratios": [4, 1]},
    constrained_layout=True,
)
# fig.suptitle(output_path, fontsize=16)

if n_steps > 50001:
    x_left = 10
    x_right = 50
else:
    x_left = 0
    x_right = n_steps * dt

axs[0].plot(t, true_state[:, 0], color="b", label="Truth")
axs[0].scatter(
    t_obs, obs[:, 0], s=40, facecolors="none", label="obs", edgecolors="black"
)
axs[0].plot(t, wrong_state_onelayer[:, 0], color="r", label="Imperfect model")
axs[0].set_title("(a) Trajectory of $u_1$")
axs[0].set_xlim([x_left, x_right])

axs[1].plot(u_axis, kde_u(u_axis), color="b")
axs[1].plot(
    u_axis_wrong_onelayer,
    kde_u_wrong_onelayer(u_axis_wrong_onelayer),
    color="r",
)
axs[1].set_title("(c) PDF of $u_1$")

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(
    handles, labels, ncol=3, loc="upper right", bbox_to_anchor=(0.85, -0.15)
)
savefig(
    fig,
    osp.join(output_path, "trajectory_wrong_one_layer_" + args.model + ".pdf"),
)

plt.rcdefaults()


# %%
fig, axs = plt.subplots(
    ncols=4,
    figsize=(10, 4),
    #     constrained_layout=True,
)
if n_steps > 40000:
    t_start = int(20 / obs_dt)
    t_end = int(40 / obs_dt) + 1
else:
    t_start = int(0 / obs_dt)
    t_end = int((n_steps * dt) / obs_dt) + 1


X, Y = np.meshgrid(np.arange(0, (dim_I + 1)), t_obs[t_start:t_end])

# Y *= args.obs_dt


Z = plot_u(true_state)[obs_idx][t_start:t_end]
cf = axs[0].contourf(X, Y, Z, 10, cmap="jet")
axs[0].set_title("(a) True signal u")

Z = plot_u(wrong_state)[obs_idx][t_start:t_end]
cf = axs[1].contourf(X, Y, Z, 10, cmap="jet")
axs[1].set_title("(b) Imperfect model u")

X, Y = np.meshgrid(np.arange(0, (dim_J * dim_I + 1)), t_obs[t_start:t_end])

Z = plot_v(true_state)[obs_idx][t_start:t_end]
cf = axs[2].contourf(X, Y, Z, 10, cmap="jet")
axs[2].set_title("(c) True signal u")

Z = plot_v(wrong_state)[obs_idx][t_start:t_end]
cf = axs[3].contourf(X, Y, Z, 10, cmap="jet")
axs[3].set_title("(d) Imperfect model v")

plt.tight_layout()
plt.colorbar(cf, ax=axs)
plt.show()

savefig(
    fig,
    osp.join(
        output_path, "Hovmoeller_perf_vs_imperf_SPEKF_" + args.model + ".pdf"
    ),
)

# %%
fig, axs = plt.subplots(
    ncols=2,
    figsize=(5, 4),
    #     constrained_layout=True,
)
if n_steps > 40000:
    t_start = int(20 / obs_dt)
    t_end = int(40 / obs_dt) + 1
else:
    t_start = int(0 / obs_dt)
    t_end = int((n_steps * dt) / obs_dt) + 1


X, Y = np.meshgrid(np.arange(0, (dim_I + 1)), t_obs[t_start:t_end])


Z = plot_u(true_state)[obs_idx][t_start:t_end]
cf = axs[0].contourf(X, Y, Z, 10, cmap="jet")
axs[0].set_title("(a) True signal u")

Z = plot_u(wrong_state_onelayer)[obs_idx][t_start:t_end]
cf = axs[1].contourf(X, Y, Z, 10, cmap="jet")
axs[1].set_title("(b) Imperfect model u")


plt.tight_layout()
plt.colorbar(cf, ax=axs)
plt.show()

savefig(
    fig,
    osp.join(
        output_path,
        "Hovmoeller_perf_vs_imperf_onelayer_" + args.model + ".pdf",
    ),
)
# %%
if args.stop_at_sampling:
    sys.exit()
# %%
num_ensembles = 300
init_mu = np.zeros(model_dim)
init_mu = np.random.randn(model_dim) * 0.01
init_R = np.eye(model_dim) * args.sigma_obs

Y_init = np.random.multivariate_normal(init_mu, init_R, num_ensembles).T
inflation = 0
if num_obs_dims > 5:
    is_localization = True
    local_Ls = 3
else:
    is_localization = False
    local_Ls = 2
Lag = 16
set_rand_seed(args.rand_seed)
start_time = time.time()
(
    gamma_mean_trace,
    gamma_cov_trace,
    gamma_mean_smooth,
    gamma_cov_smooth,
    gamma_ensembles,
) = L96_EnKF_SPEKF.online_ETKS_save(
    obs,
    Y_init,
    Lag=Lag,
    obs_dim_idx=obs_dims,
    inflation=inflation,
    localization=is_localization,
    radius_L=local_Ls,
)
logger.info(
    "SPEKF_model_ETKS " + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
set_rand_seed(args.rand_seed)
start_time = time.time()
(
    gamma_mean_trace_perfect,
    gamma_cov_trace_perfect,
    gamma_mean_smooth_perfect,
    gamma_cov_smooth_perfect,
    gamma_ensembles_perfect,
) = L96_EnKF.online_ETKS_save(
    obs,
    Y_init,
    Lag=Lag,
    obs_dim_idx=obs_dims,
    inflation=inflation,
    localization=is_localization,
    radius_L=local_Ls,
)
logger.info(
    "Perfect_model_ETKS" + "--- %s seconds ---" % (time.time() - start_time)
)


# %%
num_ensembles = 300
init_mu = np.zeros(dim_I)
init_mu = np.random.randn(dim_I) * 0.01
init_R = np.eye(dim_I) * args.sigma_obs

Y_init = np.random.multivariate_normal(init_mu, init_R, num_ensembles).T
start_time = time.time()
set_rand_seed(args.rand_seed)
(
    gamma_mean_trace_onelayer,
    gamma_cov_trace_onelayer,
    gamma_mean_smooth_onelayer,
    gamma_cov_smooth_onelayer,
    gamma_ensembles_onelayer,
) = L96_EnKF_onelayer.online_ETKS_save(
    obs,
    Y_init,
    Lag=Lag,
    obs_dim_idx=obs_dims,
    inflation=inflation,
    localization=is_localization,
    radius_L=local_Ls,  # localization radius
)
logger.info(
    "Onelayer_model_ETKS: "
    + " --- %s seconds ---" % (time.time() - start_time)
)

# %%
smoother_dict = {
    "inflation": inflation,
    "is_localization": is_localization,
    "local_Ls": local_Ls,
    "Lag": Lag,
}

export_json(
    smoother_dict,
    osp.join(output_path, "smoother_dict.json"),
)

# %%
if args.finalized_mode:
    np.savez_compressed(
        osp.join(output_path, "sampling.npz"),
        true_state=true_state,
        wrong_state=wrong_state,
        obs=obs,
        gamma_mean_trace=gamma_mean_trace,
        gamma_cov_trace=gamma_cov_trace,
        gamma_mean_smooth=gamma_mean_smooth,
        gamma_cov_smooth=gamma_cov_smooth,
        gamma_ensembles=gamma_ensembles[:, :, :50],
    )
    np.savez_compressed(
        osp.join(output_path, "sampling_onelayer.npz"),
        wrong_state_onelayer=wrong_state_onelayer,
        gamma_mean_trace_onelayer=gamma_mean_trace_onelayer,
        gamma_cov_trace_onelayer=gamma_cov_trace_onelayer,
        gamma_mean_smooth_onelayer=gamma_mean_smooth_onelayer,
        gamma_cov_smooth_onelayer=gamma_cov_smooth_onelayer,
        gamma_ensembles_onelayer=gamma_ensembles_onelayer[:, :, :50],
    )
    np.savez_compressed(
        osp.join(output_path, "sampling_perfect.npz"),
        gamma_mean_trace_perfect=gamma_mean_trace_perfect,
        gamma_cov_trace_perfect=gamma_cov_trace_perfect,
        gamma_mean_smooth_perfect=gamma_mean_smooth_perfect,
        gamma_cov_smooth_perfect=gamma_cov_smooth_perfect,
        gamma_ensembles_perfect=gamma_ensembles_perfect[:, :, :50],
    )

# %%
fig, axs = plt.subplots(
    nrows=4,
    figsize=(14, 8),
    constrained_layout=True,
)
i = 0

if n_steps > 50001:
    x_left = 10
    x_right = 50
else:
    x_left = 0
    x_right = n_steps * dt

axs[i].plot(
    t,
    true_state[:, i],
    "blue",
    label="truth",
)
axs[i].plot(
    t_obs,
    gamma_mean_smooth[:, i],
    "orange",
    label="Smoother cov from imperfect model",
)
axs[i].plot(
    t_obs,
    gamma_ensembles[:, i, 0],
    "lime",
    label="Sampled trajectories from imperfect model",
)
axs[i].legend()
axs[i].set_title("$u_1$ (hidden)")
axs[i].set_xlim([x_left, x_right])


j = 0
axs[j + 1].plot(
    t,
    true_state[:, dim_I + i * dim_J + j],
    "blue",
)
axs[j + 1].plot(
    t_obs,
    gamma_mean_smooth[:, dim_I + i * dim_J + j],
    "orange",
)
axs[j + 1].plot(
    t_obs,
    gamma_ensembles[:, dim_I + i * dim_J + j, 0],
    "lime",
)
axs[j + 1].set_title("$v_{11}$")
axs[j + 1].set_xlim([x_left, x_right])


i = 1
axs[i + 1].plot(
    t,
    true_state[:, i],
    "blue",
    label="truth",
)
axs[i + 1].plot(
    t_obs,
    gamma_mean_smooth[:, i],
    "orange",
    label="Smoother cov from imperfect model",
)
axs[i + 1].plot(
    t_obs,
    gamma_ensembles[:, i, 0],
    "lime",
    label="Sampled trajectories from imperfect model",
)
axs[i + 1].set_title("$u_2$ (obs)")
axs[i + 1].set_xlim([x_left, x_right])


j = 0
axs[j + 3].plot(
    t,
    true_state[:, dim_I + i * dim_J + j],
    "blue",
)
axs[j + 3].plot(
    t_obs,
    gamma_mean_smooth[:, dim_I + i * dim_J + j],
    "orange",
)
axs[j + 3].plot(
    t_obs,
    gamma_ensembles[:, dim_I + i * dim_J + j, 0],
    "lime",
)
axs[j + 3].set_title("$v_{21}$")
axs[j + 3].set_xlim([x_left, x_right])


savefig(fig, osp.join(output_path, "mean_u_v_" + args.model + ".pdf"))


# %%
fig, axs = plt.subplots(
    nrows=4,
    figsize=(14, 8),
    constrained_layout=True,
)
i = 0
axs[i].plot(
    t_obs,
    gamma_cov_smooth_perfect[:, i, i],
    "black",
    label="Smoother cov from perfect model",
)
axs[i].plot(
    t_obs,
    gamma_cov_smooth[:, i, i],
    "r",
    label="Smoother cov from imperfect model",
)
axs[i].legend()
axs[i].set_title("$u_1$ (hidden)")
axs[i].set_xlim([x_left, x_right])


j = 0
axs[j + 1].plot(
    t_obs,
    gamma_cov_smooth_perfect[:, dim_I + i * dim_J + j, dim_I + i * dim_J + j],
    "black",
)
axs[j + 1].plot(
    t_obs,
    gamma_cov_smooth[:, dim_I + i * dim_J + j, dim_I + i * dim_J + j],
    "r",
)
axs[j + 1].set_title("$v_{11}$")
axs[j + 1].set_xlim([x_left, x_right])


i = 1
axs[i + 1].plot(
    t_obs,
    gamma_cov_smooth_perfect[:, i, i],
    "black",
    label="Smoother cov from perfect model",
)
axs[i + 1].plot(
    t_obs,
    gamma_cov_smooth[:, i, i],
    "r",
    label="Smoother cov from imperfect model",
)
axs[i + 1].set_title("$u_2$ (obs)")
axs[i + 1].set_xlim([x_left, x_right])


j = 0
axs[j + 3].plot(
    t_obs,
    gamma_cov_smooth_perfect[:, dim_I + i * dim_J + j, dim_I + i * dim_J + j],
    "black",
)
axs[j + 3].plot(
    t_obs,
    gamma_cov_smooth[:, dim_I + i * dim_J + j, dim_I + i * dim_J + j],
    "r",
)
axs[j + 3].set_title("$v_{21}$")
axs[j + 3].set_xlim([x_left, x_right])


savefig(fig, osp.join(output_path, "cov_u_v_" + args.model + ".pdf"))

# %%
fig, axs = plt.subplots(
    nrows=5,
    figsize=(14, 8),
    constrained_layout=True,
)
if dim_I > 5:
    plt_range = 5
else:
    plt_range = dim_I
for i in range(plt_range):
    axs[i].plot(
        t_obs,
        gamma_cov_smooth[:, i, i],
        "r",
        label="Smoother cov from imperfect model",
    )
    axs[i].plot(
        t_obs,
        gamma_cov_smooth_perfect[:, i, i],
        "black",
        label="Smoother cov from perfect model",
    )
    axs[i].set_xlim([x_left, x_right])
axs[0].legend()
savefig(fig, osp.join(output_path, "cov_u_" + args.model + ".pdf"))

# %%
fig, axs = plt.subplots(
    ncols=4,
    figsize=(10, 4),
    #     constrained_layout=True,
)

if n_steps > 40000:
    t_start = int(20 / obs_dt)
    t_end = int(40 / obs_dt) + 1
else:
    t_start = int(0 / obs_dt)
    t_end = int((n_steps * dt) / obs_dt) + 1

X, Y = np.meshgrid(np.arange(0, (dim_I + 1)), t_obs[t_start:t_end])

# Y *= args.obs_dt

vmax = 5.5
vmin = -2
Z = plot_u(true_state)[obs_idx][t_start:t_end]
cf = axs[0].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[0].set_title("(a) True signal")

Z = plot_u(wrong_state)[obs_idx][t_start:t_end]
cf = axs[1].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[1].set_title("(b) Imperfect model")

Z = plot_u(gamma_mean_smooth)[t_start:t_end]
cf = axs[2].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[2].set_title("(c) Smoother mean")

Z = plot_u(gamma_ensembles[:, :, 0])[t_start:t_end]
cf = axs[3].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[3].set_title("(d) Sampled trajectories")

plt.tight_layout()
plt.colorbar(cf, ax=axs)
plt.show()

savefig(
    fig, osp.join(output_path, "Hovmoeller_u_SPEKF_" + args.model + ".pdf")
)

# %%
fig, axs = plt.subplots(
    ncols=4,
    figsize=(10, 4),
    #     constrained_layout=True,
)

if n_steps > 40000:
    t_start = int(20 / obs_dt)
    t_end = int(40 / obs_dt) + 1
else:
    t_start = int(0 / obs_dt)
    t_end = int((n_steps * dt) / obs_dt) + 1

X, Y = np.meshgrid(np.arange(0, (dim_I + 1)), t_obs[t_start:t_end])

# Y *= args.obs_dt

vmax = 5.5
vmin = -2
Z = plot_u(true_state)[obs_idx][t_start:t_end]
cf = axs[0].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[0].set_title("(a) True signal")

Z = plot_u(wrong_state_onelayer)[obs_idx][t_start:t_end]
cf = axs[1].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[1].set_title("(b) Imperfect model")

Z = plot_u(gamma_mean_smooth_onelayer)[t_start:t_end]
cf = axs[2].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[2].set_title("(c) Smoother mean")

Z = plot_u(gamma_ensembles_onelayer[:, :, 0])[t_start:t_end]
cf = axs[3].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[3].set_title("(d) Sampled trajectories")

plt.tight_layout()
plt.colorbar(cf, ax=axs)
plt.show()

savefig(
    fig, osp.join(output_path, "Hovmoeller_u_onelayer_" + args.model + ".pdf")
)

# %%
fig, axs = plt.subplots(
    ncols=4,
    figsize=(10, 4),
    #     constrained_layout=True,
)

if n_steps > 40000:
    t_start = int(20 / obs_dt)
    t_end = int(40 / obs_dt) + 1
else:
    t_start = int(0 / obs_dt)
    t_end = int((n_steps * dt) / obs_dt) + 1

X, Y = np.meshgrid(np.arange(0, (dim_J * dim_I + 1)), t_obs[t_start:t_end])

X = X / dim_J

vmin = -1.6
vmax = 2.8

Z = plot_v(true_state)[obs_idx][t_start:t_end]
cf = axs[0].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[0].set_title("(a) True signal")

Z = plot_v(wrong_state)[obs_idx][t_start:t_end]
cf = axs[1].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[1].set_title("(b) Imperfect model")

Z = plot_v(gamma_mean_smooth)[t_start:t_end]
cf = axs[2].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[2].set_title("(c) Smoother mean")

Z = plot_v(gamma_ensembles[:, :, 0])[t_start:t_end]
cf = axs[3].contourf(X, Y, Z, 10, cmap="jet", vmin=vmin, vmax=vmax)
axs[3].set_title("(d) Sampled trajectories")

plt.tight_layout()
plt.colorbar(cf, ax=axs)
plt.show()

savefig(
    fig, osp.join(output_path, "Hovmoeller_v_SPEKF_" + args.model + ".pdf")
)

# %%
auto_truth = np.zeros((model_dim, auto_obs_steps))

# %%
plt_dims = [0, dim_I, 1, dim_I + dim_J]
auto_obs_related_variables = np.zeros((len(plt_dims) * 3, auto_obs_steps))
obs_related_variables = np.hstack(
    (
        gamma_mean_trace[:, plt_dims],
        gamma_mean_smooth[:, plt_dims],
        gamma_ensembles[:, plt_dims, 0],
    )
)
for i in tqdm(plt_dims):
    auto_truth[i] = estimated_autocorrelation(
        true_state[obs_idx][obs_burnin:][:, i]
    )
for i in tqdm(range(auto_obs_related_variables.shape[0])):
    auto_obs_related_variables[i] = estimated_autocorrelation(
        obs_related_variables[obs_burnin:][:, i]
    )


# %%
kde_values = np.zeros((len(plt_dims) * 3, 100))
axis_values = np.zeros((len(plt_dims) * 3, 100))

for i in range(len(plt_dims) * 3):
    kde, axis = compute_kde_axis(obs_related_variables[:, i])
    kde_values[i], axis_values[i] = kde(axis), axis


# %%
kde_values_truth = np.zeros((model_dim, 100))
axis_values_truth = np.zeros((model_dim, 100))
for i in plt_dims:
    kde, axis = compute_kde_axis(true_state[:, i])
    kde_values_truth[i], axis_values_truth[i] = kde(axis), axis


# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("legend", fontsize=14)
fig, axs = plt.subplots(
    nrows=4,
    ncols=3,
    figsize=(18, 10),
    gridspec_kw={"width_ratios": [5, 1, 1]},
    constrained_layout=True,
)
x_left = 0
x_right = n_steps * dt

titles = ["$u_1$ (hidden)", "$v_{11}$", "$u_2$ (observed)", "$v_{21}$"]

for i in range(len(plt_dims)):
    plt_dim = plt_dims[i]
    axs[i, 0].plot(
        t_obs,
        true_state[obs_idx, plt_dim],
        color="b",
        label="truth at obs state",
    )
    axs[i, 0].plot(
        t_obs, gamma_mean_trace[:, plt_dim], color="r", label="filter mean"
    )
    axs[i, 0].plot(
        t_obs,
        gamma_mean_smooth[:, plt_dim],
        color="orange",
        label="smoother mean",
    )
    axs[i, 0].plot(
        t_obs,
        gamma_ensembles[:, plt_dim, 0],
        color="lime",
        label="sampled trajectories",
    )
    axs[i, 0].set_xlim([x_left, x_right])
    axs[i, 0].set_title(titles[i])

    axs[i, 1].plot(
        axis_values_truth[plt_dim],
        kde_values_truth[plt_dim],
        color="b",
        label="Truth",
    )
    axs[i, 1].plot(
        axis_values[len(plt_dims) * 0 + i],
        kde_values[len(plt_dims) * 0 + i],
        color="r",
        label="filter mean",
    )
    axs[i, 1].plot(
        axis_values[len(plt_dims) * 1 + i],
        kde_values[len(plt_dims) * 1 + i],
        color="orange",
        label="smoother mean",
    )
    axs[i, 1].plot(
        axis_values[len(plt_dims) * 2 + i],
        kde_values[len(plt_dims) * 2 + i],
        color="lime",
        label="sampled trajectories",
    )
    axs[i, 1].set_title("PDF of " + titles[i])

    axs[i, 2].plot(t_obs[:auto_obs_steps], auto_truth[plt_dim], "b")
    axs[i, 2].plot(
        t_obs[:auto_obs_steps],
        auto_obs_related_variables[len(plt_dims) * 0 + i],
        color="r",
    )
    axs[i, 2].plot(
        t_obs[:auto_obs_steps],
        auto_obs_related_variables[len(plt_dims) * 1 + i],
        color="orange",
    )
    axs[i, 2].plot(
        t_obs[:auto_obs_steps],
        auto_obs_related_variables[len(plt_dims) * 2 + i],
        color="lime",
    )
    axs[i, 2].set_xlim([0, 10])


handles, labels = axs[0, 1].get_legend_handles_labels()
axs[3, 0].legend(
    handles, labels, ncol=4, loc="upper right", bbox_to_anchor=(0.85, -0.15)
)
savefig(
    fig,
    osp.join(
        output_path,
        "Sampling_PDF_ACF_SPEKF_" + args.model + ".pdf",
    ),
)
plt.rcdefaults()


# %%
plt_dims_onelayer = [0, 1]
auto_obs_related_variables_onelayer = np.zeros(
    (len(plt_dims_onelayer) * 3, auto_obs_steps)
)
obs_related_variables_onelayer = np.hstack(
    (
        gamma_mean_trace_onelayer[:, plt_dims_onelayer],
        gamma_mean_smooth_onelayer[:, plt_dims_onelayer],
        gamma_ensembles_onelayer[:, plt_dims_onelayer, 0],
    )
)

for i in tqdm(range(auto_obs_related_variables_onelayer.shape[0])):
    auto_obs_related_variables_onelayer[i] = estimated_autocorrelation(
        obs_related_variables_onelayer[obs_burnin:][:, i]
    )

for i in tqdm(plt_dims_onelayer):
    auto_truth[i] = estimated_autocorrelation(
        true_state[obs_idx][obs_burnin:][:, i]
    )

kde_values_onelayer = np.zeros((len(plt_dims_onelayer) * 3, 100))
axis_values_onelayer = np.zeros((len(plt_dims_onelayer) * 3, 100))

for i in tqdm(range(len(plt_dims_onelayer) * 3)):
    kde, axis = compute_kde_axis(obs_related_variables_onelayer[:, i])
    kde_values_onelayer[i], axis_values_onelayer[i] = kde(axis), axis
# do this because the plotting dim for SPEKF and onelayer might not be the same
for i in plt_dims_onelayer:
    kde, axis = compute_kde_axis(true_state[:, i])
    kde_values_truth[i], axis_values_truth[i] = kde(axis), axis

# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("legend", fontsize=14)
fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(18, 5),
    gridspec_kw={"width_ratios": [5, 1, 1]},
    constrained_layout=True,
)
x_left = 0
x_right = n_steps * dt

titles = ["$u_1$ (hidden)", "$u_2$ (observed)"]

for i in range(len(plt_dims_onelayer)):
    plt_dim = plt_dims_onelayer[i]
    axs[i, 0].plot(
        t_obs,
        true_state[obs_idx, plt_dim],
        color="b",
        label="truth at obs state",
    )
    axs[i, 0].plot(
        t_obs,
        gamma_mean_trace_onelayer[:, plt_dim],
        color="r",
        label="filter mean",
    )
    axs[i, 0].plot(
        t_obs,
        gamma_mean_smooth_onelayer[:, plt_dim],
        color="orange",
        label="smoother mean",
    )
    axs[i, 0].plot(
        t_obs,
        gamma_ensembles_onelayer[:, plt_dim, 0],
        color="lime",
        label="sampled trajectories",
    )
    axs[i, 0].set_xlim([x_left, x_right])
    axs[i, 0].set_title(titles[i])

    axs[i, 1].plot(
        axis_values_truth[plt_dim],
        kde_values_truth[plt_dim],
        color="b",
        label="Truth",
    )
    axs[i, 1].plot(
        axis_values_onelayer[len(plt_dims_onelayer) * 0 + i],
        kde_values_onelayer[len(plt_dims_onelayer) * 0 + i],
        color="r",
        label="filter mean",
    )
    axs[i, 1].plot(
        axis_values_onelayer[len(plt_dims_onelayer) * 1 + i],
        kde_values_onelayer[len(plt_dims_onelayer) * 1 + i],
        color="orange",
        label="smoother mean",
    )
    axs[i, 1].plot(
        axis_values_onelayer[len(plt_dims_onelayer) * 2 + i],
        kde_values_onelayer[len(plt_dims_onelayer) * 2 + i],
        color="lime",
        label="sampled trajectories",
    )
    axs[i, 1].set_title("PDF of " + titles[i])

    axs[i, 2].plot(t_obs[:auto_obs_steps], auto_truth[plt_dim], "b")
    axs[i, 2].plot(
        t_obs[:auto_obs_steps],
        auto_obs_related_variables_onelayer[len(plt_dims_onelayer) * 0 + i],
        color="r",
    )
    axs[i, 2].plot(
        t_obs[:auto_obs_steps],
        auto_obs_related_variables_onelayer[len(plt_dims_onelayer) * 1 + i],
        color="orange",
    )
    axs[i, 2].plot(
        t_obs[:auto_obs_steps],
        auto_obs_related_variables_onelayer[len(plt_dims_onelayer) * 2 + i],
        color="lime",
    )
    axs[i, 2].set_title("ACF of " + titles[i])
    axs[i, 2].set_xlim([0, 10])

handles, labels = axs[0, 1].get_legend_handles_labels()
axs[1, 0].legend(
    handles, labels, ncol=4, loc="upper right", bbox_to_anchor=(0.85, -0.15)
)
savefig(
    fig,
    osp.join(
        output_path,
        "Sampling_PDF_ACF_onelayer_" + args.model + ".pdf",
    ),
)
plt.rcdefaults()
# %%
if args.stop_at_sampling:
    sys.exit()

# %% [markdown]
# # Model based prediction

# %%
from pkg.helper import model_prediction_one_traj_new_new

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

pred_dt = obs_dt
pred_n_steps = int(dt * (n_steps - 1) / pred_dt) + 1
pred_K = int(pred_dt / dt)


t_pred = np.linspace(0, (pred_n_steps - 1) * pred_dt, pred_n_steps)
pred_idx = np.linspace(0, int((pred_n_steps - 1)) * pred_K, pred_n_steps)
pred_idx = pred_idx.astype(int)

# %%
pred_obs_n_steps = int(obs_dt * (obs_n_steps - 1) / pred_dt) + 1
pred_obs_K = int(pred_dt / obs_dt)


pred_obs_idx = np.linspace(
    0, int((pred_obs_n_steps - 1)) * pred_obs_K, pred_obs_n_steps
)
pred_obs_idx = pred_obs_idx.astype(int)

# %%
pred_start_time = args.pred_start_time
pred_total_time = args.pred_total_time

pred_start_pred_step = int(pred_start_time / pred_dt)
pred_total_pred_steps = int(pred_total_time / pred_dt) + 1

logger.info(f"pred_start_pred_step: {pred_start_pred_step}")
logger.info(f"pred_total_pred_steps: {pred_total_pred_steps}")

logger.info(f"start time unit: {pred_start_pred_step * pred_dt}")
logger.info(
    "end time unit: "
    + f"{(pred_start_pred_step + pred_total_pred_steps) * pred_dt}"
)
logger.info(f"length of entire trajectories: {(n_steps - 1) * dt}")

last_lead_t = int(args.last_lead_time / dt)
lead_step = pred_K

pred_last_lead_t = last_lead_t // pred_K
pred_lead_step = lead_step // pred_K

lead_time_steps = np.linspace(
    0, last_lead_t, int(last_lead_t / lead_step) + 1
).astype(
    int
)  # only use for plotting
logger.info(
    "lead_time_steps (in terms of in true trajectories space, dt):"
    + f" {lead_time_steps}"
)
logger.info(f"final lead time: {last_lead_t * dt}")
logger.info(f"total_lead_time_steps: {len(lead_time_steps)}")

# %%
pred_num_ensembles = 50

# %%
t_target = t[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
]
target = true_state[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
]

# %%
initial_values = true_state[pred_idx][
    pred_start_pred_step
    - pred_last_lead_t : pred_start_pred_step
    + pred_total_pred_steps
]
initial_mean = gamma_mean_trace[pred_obs_idx][
    pred_start_pred_step
    - pred_last_lead_t : pred_start_pred_step
    + pred_total_pred_steps
]
initial_cov = gamma_cov_trace[pred_obs_idx][
    pred_start_pred_step
    - pred_last_lead_t : pred_start_pred_step
    + pred_total_pred_steps
]
initial_mean_onelayer = gamma_mean_trace_onelayer[pred_obs_idx][
    pred_start_pred_step
    - pred_last_lead_t : pred_start_pred_step
    + pred_total_pred_steps
]
initial_cov_onelayer = gamma_cov_trace_onelayer[pred_obs_idx][
    pred_start_pred_step
    - pred_last_lead_t : pred_start_pred_step
    + pred_total_pred_steps
]
initial_mean_perfect = gamma_mean_trace_perfect[pred_obs_idx][
    pred_start_pred_step
    - pred_last_lead_t : pred_start_pred_step
    + pred_total_pred_steps
]
initial_cov_perfect = gamma_cov_trace_perfect[pred_obs_idx][
    pred_start_pred_step
    - pred_last_lead_t : pred_start_pred_step
    + pred_total_pred_steps
]
initial_DA = np.zeros((pred_num_ensembles, initial_mean.shape[0], model_dim))
initial_DA_onelayer = np.zeros(
    (pred_num_ensembles, initial_mean.shape[0], dim_I)
)
initial_DA_perfect = np.zeros(
    (pred_num_ensembles, initial_mean.shape[0], model_dim)
)
initial_values_repeat = np.zeros(
    (pred_num_ensembles, initial_mean.shape[0], model_dim)
)
initial_values_repeat_onelayer = np.zeros(
    (pred_num_ensembles, initial_mean.shape[0], dim_I)
)
for i in range(pred_num_ensembles):
    initial_values_repeat[i] = initial_values
    initial_values_repeat_onelayer[i] = initial_values[:, :dim_I]

for i in range(initial_mean.shape[0]):
    initial_DA[:, i, :] = np.random.multivariate_normal(
        initial_mean[i], initial_cov[i], pred_num_ensembles
    )
    initial_DA_onelayer[:, i, :] = np.random.multivariate_normal(
        initial_mean_onelayer[i], initial_cov_onelayer[i], pred_num_ensembles
    )
    initial_DA_perfect[:, i, :] = np.random.multivariate_normal(
        initial_mean_perfect[i], initial_cov_perfect[i], pred_num_ensembles
    )

# %%
# check initial conditions
for i in [0]:
    plt.figure(figsize=(20, 2))
    plt.plot(initial_values[:, i], "b")
    plt.plot(np.mean(initial_DA_perfect, axis=0)[:, i], "lime")
    plt.plot(np.mean(initial_DA, axis=0)[:, i], "r")
    plt.plot(np.mean(initial_DA_onelayer, axis=0)[:, i])
    for j in range(3):
        plt.figure(figsize=(20, 2))
        plt.plot(initial_values[:, dim_I + i * dim_J + j], "b")
        plt.plot(
            np.mean(initial_DA_perfect, axis=0)[:, dim_I + i * dim_J + j],
            "lime",
        )
        plt.plot(np.mean(initial_DA, axis=0)[:, dim_I + i * dim_J + j], "r")

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_mean = model_prediction_one_traj_new_new(
    model=L96_EnKF,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_values_repeat,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
logger.info(
    "Perfect prediction" + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_mean_DA = model_prediction_one_traj_new_new(
    model=L96_EnKF,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_DA_perfect,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
logger.info(
    "Perfect prediction (DA)"
    + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_wrong = model_prediction_one_traj_new_new(
    model=L96_EnKF_SPEKF,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_values_repeat,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
logger.info(
    "SPEKF prediction" + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_wrong_DA = model_prediction_one_traj_new_new(
    model=L96_EnKF_SPEKF,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_DA,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
logger.info(
    "SPEKF prediction (DA)" + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_wrong_onelayer = model_prediction_one_traj_new_new(
    model=L96_EnKF_onelayer,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_values_repeat_onelayer,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
logger.info(
    "Onelayer prediction" + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_wrong_DA_onelayer = model_prediction_one_traj_new_new(
    model=L96_EnKF_onelayer,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_DA_onelayer,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
logger.info(
    "Onelayer prediction (DA)"
    + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
rmse_perfect = np.zeros((len(predict_mean), model_dim))
rmse_perfect_DA = np.zeros((len(predict_mean), model_dim))
rmse_wrong = np.zeros((len(predict_mean), model_dim))
rmse_wrong_DA = np.zeros((len(predict_mean), model_dim))

rmse_wrong_onelayer = np.zeros((len(predict_mean), dim_I))
rmse_wrong_DA_onelayer = np.zeros((len(predict_mean), dim_I))
for i in range(len(predict_mean)):
    rmse_perfect[i] = rmse(predict_mean[i], target)
    rmse_perfect_DA[i] = rmse(predict_mean_DA[i], target)
    rmse_wrong[i] = rmse(predict_wrong[i], target)
    rmse_wrong_DA[i] = rmse(predict_wrong_DA[i], target)
    rmse_wrong_onelayer[i] = rmse(predict_wrong_onelayer[i], target[:, :dim_I])
    rmse_wrong_DA_onelayer[i] = rmse(
        predict_wrong_DA_onelayer[i], target[:, :dim_I]
    )

corr_perfect = np.zeros((len(predict_mean), model_dim))
corr_perfect_DA = np.zeros((len(predict_mean), model_dim))
corr_wrong = np.zeros((len(predict_mean), model_dim))
corr_wrong_DA = np.zeros((len(predict_mean), model_dim))

corr_wrong_onelayer = np.zeros((len(predict_mean), dim_I))
corr_wrong_DA_onelayer = np.zeros((len(predict_mean), dim_I))
for i in range(len(predict_mean)):
    corr_perfect[i] = patt_corr(predict_mean[i], target)
    corr_perfect_DA[i] = patt_corr(predict_mean_DA[i], target)
    corr_wrong[i] = patt_corr(predict_wrong[i], target)
    corr_wrong_DA[i] = patt_corr(predict_wrong_DA[i], target)
    corr_wrong_onelayer[i] = patt_corr(
        predict_wrong_onelayer[i], target[:, :dim_I]
    )
    corr_wrong_DA_onelayer[i] = patt_corr(
        predict_wrong_DA_onelayer[i], target[:, :dim_I]
    )


# %%
pred_time = 10
for i in [0, 1, 2, 3]:
    plt.figure(figsize=(20, 2))
    plt.plot(t_target, target[:, i], "black")
    plt.plot(t_target, predict_mean[pred_time, :, i], "b")
    #     plt.plot(t_target, predict_mean_DA[pred_time, :, i], "lime")
    plt.plot(t_target, predict_wrong[pred_time, :, i], "r")
    plt.plot(t_target, predict_wrong_onelayer[pred_time, :, i], "m")
#     plt.plot(t_target, predict_wrong_DA[pred_time, :, i], "r")
#         plt.plot(t_target, predict_mean_numba[pred_time, :, i], "r")

#     plt.figure(figsize=(20, 2))
#     for j in range(3):
#         plt.figure(figsize=(20, 2))
#         plt.plot(t_target, target[:, dim_I + i * dim_J + j], "black")
#         plt.plot(
#             t_target,
#             predict_mean[pred_time, :, dim_I + i * dim_J + j],
#             "b",
#         )
#         plt.plot(
#             t_target,
#             predict_mean_DA[pred_time, :, dim_I + i * dim_J + j],
#             "lime",
#         )
#         plt.plot(
#             t_target,
#             predict_wrong[pred_time, :, dim_I + i * dim_J + j],
#             "m",
#         )
#         plt.plot(
#             t_target,
#             predict_wrong_DA[pred_time, :, dim_I + i * dim_J + j],
#             "r",
#         )


# %%
np.savez_compressed(
    osp.join(output_path, "dynamicalmodels.npz"),
    pred_dt=pred_dt,
    obs_dt=obs_dt,
    pred_start_time=pred_start_time,
    pred_total_time=pred_total_time,
    pred_last_lead_t=pred_last_lead_t,
    pred_num_ensembles=pred_num_ensembles,
    last_lead_t=last_lead_t,
    predict_mean=predict_mean,
    predict_mean_DA=predict_mean_DA,
    predict_wrong=predict_wrong,
    predict_wrong_DA=predict_wrong_DA,
    predict_wrong_onelayer=predict_wrong_onelayer,
    predict_wrong_DA_onelayer=predict_wrong_DA_onelayer,
    rmse_perfect=rmse_perfect,
    rmse_perfect_DA=rmse_perfect_DA,
    rmse_wrong=rmse_wrong,
    rmse_wrong_DA=rmse_wrong_DA,
    rmse_wrong_onelayer=rmse_wrong_onelayer,
    rmse_wrong_DA_onelayer=rmse_wrong_DA_onelayer,
    corr_perfect=corr_perfect,
    corr_perfect_DA=corr_perfect_DA,
    corr_wrong=corr_wrong,
    corr_wrong_DA=corr_wrong_DA,
    corr_wrong_onelayer=corr_wrong_onelayer,
    corr_wrong_DA_onelayer=corr_wrong_DA_onelayer,
)

# # %%
export_json(
    {
        "lead_time_steps": list(lead_time_steps.astype(float)),
        "total_lead_time_steps:": len(lead_time_steps),
    },
    osp.join(output_path, "lead_time_steps.json"),
)


# %%
fig, axs = plt.subplots(
    nrows=2,
    ncols=5,
    figsize=(14, 6),
    #     gridspec_kw={"width_ratios": [4, 1]},
    #     constrained_layout=True,
)
title_RMSE = [
    "RMSE u1",
    "RMSE u2",
    "RMSE u3",
    "RMSE u4",
    "RMSE u5",
]
title_corr = [
    "Corr u1",
    "Corr u2",
    "Corr u3",
    "Corr u4",
    "Corr u5",
]
for i in range(5):
    axs[0, i].plot(
        lead_time_steps * dt,
        rmse_perfect[:, i],
        "b",
        label="perfect model",
    )
    axs[0, i].plot(
        lead_time_steps * dt,
        rmse_perfect_DA[:, i],
        "--",
        color="b",
        label="perfect model with assimilated IC",
    )
    axs[0, i].plot(
        lead_time_steps * dt,
        rmse_wrong[:, i],
        "r",
        label="imperfect model, (SPEKF)",
    )
    axs[0, i].plot(
        lead_time_steps * dt,
        rmse_wrong_DA[:, i],
        "--",
        color="r",
        label="imperfect model with assimilated IC, (SPEKF)",
    )
    axs[0, i].plot(
        lead_time_steps * dt,
        rmse_wrong_onelayer[:, i],
        "m",
        label="imperfect model, (one layer)",
    )
    axs[0, i].plot(
        lead_time_steps * dt,
        rmse_wrong_DA_onelayer[:, i],
        "--",
        color="m",
        label="imperfect model with assimilated IC, (one layer)",
    )

    axs[0, i].plot(
        [0, lead_time_steps[-1] * dt],
        [
            np.sqrt(np.var(true_state[:, i])),
            np.sqrt(np.var(true_state[:, i])),
        ],
        "--",
    )
    axs[0, i].set_title(title_RMSE[i % 5])

    axs[1, i].plot(
        lead_time_steps * dt,
        corr_perfect[:, i],
        "b",
    )
    axs[1, i].plot(
        lead_time_steps * dt,
        corr_perfect_DA[:, i],
        "--",
        color="b",
    )
    axs[1, i].plot(
        lead_time_steps * dt,
        corr_wrong_DA[:, i],
        "--",
        color="r",
    )
    axs[1, i].plot(
        lead_time_steps * dt,
        corr_wrong[:, i],
        color="r",
    )
    axs[1, i].plot(
        lead_time_steps * dt,
        corr_wrong_DA_onelayer[:, i],
        "--",
        color="m",
    )
    axs[1, i].plot(
        lead_time_steps * dt,
        corr_wrong_onelayer[:, i],
        color="m",
    )

    axs[1, i].plot(
        [0, lead_time_steps[-1] * dt],
        [0.5, 0.5],
        "--",
    )
    axs[1, i].set_title(title_corr[i % 5])

handles, labels = axs[0, 0].get_legend_handles_labels()
axs[1, 0].legend(handles, labels, ncol=3, bbox_to_anchor=(6, -0.2))
savefig(
    fig,
    osp.join(
        output_path,
        "physical_model_based_pred_corr_with" + args.model + ".pdf",
    ),
)

# %%
fig, axs = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(12, 6),
    #     gridspec_kw={"width_ratios": [4, 1]},
    #     constrained_layout=True,
)
title_RMSE = [
    "RMSE v11",
    "RMSE v12",
    "RMSE v13",
    "RMSE v14",
]
title_corr = [
    "Corr v11",
    "Corr v12",
    "Corr v13",
    "Corr v14",
]
for j in range(4):
    i = dim_I + j
    axs[0, j].plot(
        lead_time_steps * dt,
        rmse_perfect[:, i],
        "b",
        label="perfect model",
    )
    axs[0, j].plot(
        lead_time_steps * dt,
        rmse_perfect_DA[:, i],
        "--",
        color="b",
        label="perfect model with assimilated IC",
    )
    axs[0, j].plot(
        lead_time_steps * dt,
        rmse_wrong[:, i],
        "r",
        label="imperfect model",
    )
    axs[0, j].plot(
        lead_time_steps * dt,
        rmse_wrong_DA[:, i],
        "--",
        color="r",
        label="imperfect model with assimilated IC",
    )

    axs[0, j].plot(
        [0, lead_time_steps[-1] * dt],
        [
            np.sqrt(np.var(true_state[:, i])),
            np.sqrt(np.var(true_state[:, i])),
        ],
        "--",
        color="black",
    )
    axs[0, j].set_title(title_RMSE[j % 4])
    axs[0, j].set_xlim([0, 0.5])

    axs[1, j].plot(
        lead_time_steps * dt,
        corr_perfect[:, i],
        "b",
    )
    axs[1, j].plot(
        lead_time_steps * dt, corr_perfect_DA[:, i], "--", color="b"
    )
    axs[1, j].plot(
        lead_time_steps * dt,
        corr_wrong_DA[:, i],
        "--",
        color="r",
    )
    axs[1, j].plot(
        lead_time_steps * dt,
        corr_wrong[:, i],
        color="r",
    )

    axs[1, j].plot(
        [0, lead_time_steps[-1] * dt], [0.5, 0.5], "--", color="black"
    )
    axs[1, j].set_title(title_corr[j % 4])
    axs[1, j].set_xlim([0, 0.5])

handles, labels = axs[0, 0].get_legend_handles_labels()
axs[1, 0].legend(handles, labels, ncol=4, bbox_to_anchor=(5, -0.2))
savefig(
    fig,
    osp.join(
        output_path,
        "physical_model_based_pred_v_" + args.model + ".pdf",
    ),
)

# %%
plt.rc("axes", titlesize=10)  # using a size in points
plt.rc("xtick", labelsize=10)
fig, axs = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(8, 12),
)

if pred_total_time > 10:
    t_start = 0
    t_end = t_start + 10
else:
    t_start = 0
    t_end = t_start + 2

t_start_step = int(t_start / pred_dt)
t_end_step = int(t_end / pred_dt) + 1

X, Y = np.meshgrid(
    np.arange(0, (dim_I + 1)), t_target[t_start_step:t_end_step]
)

# vmax = 3.5
# vmin = -2.5
# control the time step that we want to predict
lead_time_dim_options = [3, 8, 10]
for i in range(3):

    lead_time = lead_time_dim_options[i]
    Z = plot_u(target[t_start_step:t_end_step])
    cf = axs[0, i].contourf(X, Y, Z, 10, cmap="jet")
    axs[0, i].set_title("Truth")
    axs[0, i].set_ylabel("t")

    Z = plot_u(predict_mean[lead_time][t_start_step:t_end_step])

    cf = axs[1, i].contourf(X, Y, Z, 10, cmap="jet")
    axs[1, i].set_title("Lead time %8.2f" % (lead_time * dt * pred_K))
    axs[1, i].set_ylabel("t")

    Z = plot_u(predict_wrong[lead_time][t_start_step:t_end_step])
    cf = axs[2, i].contourf(X, Y, Z, 10, cmap="jet")
    axs[2, i].set_title("Lead time %8.2f" % (lead_time * dt * pred_K))
    axs[2, i].set_ylabel("t")


plt.colorbar(cf, ax=axs)

axs[1, 0].text(-7, t_start + pred_start_time, "Perfect model", fontsize=12)
axs[2, 0].text(-7, t_start + pred_start_time, "Imperfect model", fontsize=12)


savefig(
    fig,
    osp.join(
        output_path, "physical_model_pred_hovmoller_u_" + args.model + ".pdf"
    ),
)
plt.rcdefaults()

# %%
plt.rc("axes", titlesize=10)  # using a size in points
plt.rc("xtick", labelsize=10)
fig, axs = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(8, 12),
)


if pred_total_time > 10:
    t_start = 0
    t_end = t_start + 10
else:
    t_start = 0
    t_end = t_start + 2

t_start_step = int(t_start / pred_dt)
t_end_step = int(t_end / pred_dt) + 1

X, Y = np.meshgrid(
    np.arange(0, (dim_J * dim_I + 1)), t_target[t_start_step:t_end_step]
)

X = X / dim_J

# control the time step that we want to predict
lead_time_dim_options = [3, 8, 10]
for i in range(3):

    lead_time = lead_time_dim_options[i]
    Z = plot_v(target[t_start_step:t_end_step])
    cf = axs[0, i].contourf(X, Y, Z, 10, cmap="jet")
    axs[0, i].set_title("Truth")
    axs[0, i].set_ylabel("t")

    Z = plot_v(predict_mean[lead_time][t_start_step:t_end_step])

    cf = axs[1, i].contourf(X, Y, Z, 10, cmap="jet")
    axs[1, i].set_title("Lead time %8.2f" % (lead_time * dt * pred_K))
    axs[1, i].set_ylabel("t")

    Z = plot_v(predict_wrong[lead_time][t_start_step:t_end_step])
    cf = axs[2, i].contourf(X, Y, Z, 10, cmap="jet")
    axs[2, i].set_title("Lead time %8.2f" % (lead_time * dt * pred_K))
    axs[2, i].set_ylabel("t")


plt.colorbar(cf, ax=axs)

axs[1, 0].text(-7, t_start + pred_start_time, "Perfect model", fontsize=12)
axs[2, 0].text(-7, t_start + pred_start_time, "Imperfect model", fontsize=12)


savefig(
    fig,
    osp.join(
        output_path, "physical_model_pred_hovmoller_v_" + args.model + ".pdf"
    ),
)
plt.rcdefaults()

# %%
if args.finalized_mode:
    num_ensembles = 300
    init_mu = np.random.randn(model_dim) * 0.01
    init_R = np.eye(model_dim) * args.sigma_obs

    Y_init = np.random.multivariate_normal(init_mu, init_R, num_ensembles).T
    L_init = Lag  # this is in the pred_idx space
    set_rand_seed(args.rand_seed)
    (
        _,
        _,
        _,
        _,
        _,
        gamma_ensembles_for_IC_short,
    ) = L96_EnKF_SPEKF.online_ETKS_for_IC_save(
        obs,
        Y_init,
        Lag=Lag,
        obs_dim_idx=obs_dims,
        pred_start_pred_step=pred_start_pred_step,
        pred_total_pred_steps=pred_total_pred_steps,
        pred_last_lead_t=pred_last_lead_t,
        L_init=L_init,
        pred_num_ensembles=pred_num_ensembles,
        inflation=inflation,
        localization=is_localization,
        radius_L=local_Ls,  # localization radius
    )
    np.savez_compressed(
        osp.join(output_path, "sampling_for_IC.npz"),
        gamma_ensembles_for_IC_short=gamma_ensembles_for_IC_short,
    )

    set_rand_seed(args.rand_seed)
    (
        _,
        _,
        _,
        _,
        _,
        gamma_ensembles_for_IC_short_perfect,
    ) = L96_EnKF.online_ETKS_for_IC_save(
        obs,
        Y_init,
        Lag=Lag,
        obs_dim_idx=obs_dims,
        pred_start_pred_step=pred_start_pred_step,
        pred_total_pred_steps=pred_total_pred_steps,
        pred_last_lead_t=pred_last_lead_t,
        L_init=L_init,
        pred_num_ensembles=pred_num_ensembles,
        inflation=inflation,
        localization=is_localization,
        radius_L=local_Ls,  # localization radius
    )
    np.savez_compressed(
        osp.join(output_path, "sampling_for_IC_perfect.npz"),
        gamma_ensembles_for_IC_short=gamma_ensembles_for_IC_short_perfect,
    )

    init_mu = np.zeros(dim_I)
    init_mu = np.random.randn(dim_I) * 0.01
    init_R = np.eye(dim_I) * args.sigma_obs
    Y_init = np.random.multivariate_normal(init_mu, init_R, num_ensembles).T
    set_rand_seed(args.rand_seed)
    (
        _,
        _,
        _,
        _,
        _,
        gamma_ensembles_for_IC_short_onelayer,
    ) = L96_EnKF_onelayer.online_ETKS_for_IC_save(
        obs,
        Y_init,
        Lag=Lag,
        obs_dim_idx=obs_dims,
        pred_start_pred_step=pred_start_pred_step,
        pred_total_pred_steps=pred_total_pred_steps,
        pred_last_lead_t=pred_last_lead_t,
        L_init=L_init,
        pred_num_ensembles=pred_num_ensembles,
        inflation=inflation,
        localization=is_localization,
        radius_L=local_Ls,  # localization radius
    )
    np.savez_compressed(
        osp.join(output_path, "sampling_for_IC_onelayer.npz"),
        gamma_ensembles_for_IC_short=gamma_ensembles_for_IC_short_onelayer,
    )
