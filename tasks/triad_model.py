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
from scipy.stats import gaussian_kde, norm, skew, kurtosis
from tqdm import tqdm

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.ensembleKalmanfilter import (
    EnsembleKalmanFilterSmootherTriad,
    EnsembleKalmanFilterSmootherTriadApprox,
)
from pkg.helper import (
    compute_kde_axis,
    estimated_autocorrelation,
    patt_corr,
    rmse,
)
from pkg.utils.misc import export_json, makedirs, savefig, set_rand_seed

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


def get_parser():
    parser = argparse.ArgumentParser(
        description="Paramter estimation for triad model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="triad_model_final",
        help="Default: triad_model_final",
    )
    parser.add_argument(
        "--n_steps", type=int, default=200001, help="Default: 500001"
    )
    parser.add_argument(
        "--pred_start_time", type=int, default=800, help="Default: 20"
    )
    parser.add_argument(
        "--pred_total_time", type=int, default=80, help="Default: 20"
    )
    parser.add_argument(
        "--last_lead_time", type=float, default=10.0, help="Default: 3"
    )
    parser.add_argument("--dt", type=float, default=5e-3, help="Default: 5e-3")

    parser.add_argument("--regime", type=int, default=1, help="Default: 5e-3")
    parser.add_argument(
        "--obs_dt", type=float, default=5e-2, help="Default: 5e-2"
    )
    parser.add_argument(
        "--sigma_obs", type=float, default=0.2, help="Default: 0.5"
    )
    parser.add_argument(
        "--stop_at_modeling", type=bool, default=False, help="Default: False"
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="Default: output"
    )
    parser.add_argument(
        "--rand_seed", type=int, default=201, help="Default: 200"
    )
    return parser


def get_hparam_str(args):
    hparams = ["n_steps", "regime", "sigma_obs"]

    return ",".join("{}={}".format(p, getattr(args, p)) for p in hparams)


args = get_parser().parse_args([])
print(args)

output_path = osp.join(args.output_path, args.model, get_hparam_str(args))
makedirs([output_path])

export_json(vars(args), osp.join(output_path, "config.json"))

# %%
n_steps, dt, obs_dt, model_dim = args.n_steps, args.dt, args.obs_dt, 3
obs_n_steps = int(dt * (n_steps - 1) / obs_dt) + 1
K = int(obs_dt / dt)
t = np.linspace(0, (n_steps - 1) * dt, n_steps)
t_obs = np.linspace(0, (obs_n_steps - 1) * obs_dt, obs_n_steps)
obs_idx = np.linspace(0, int((obs_n_steps - 1)) * K, obs_n_steps)
obs_idx = obs_idx.astype(int)
Params = namedtuple(
    "Params", ["gamma_s", "L_s", "sigma_s", "I", "epsilon", "F"]
)
# ["gamma_s", "L_s", "sigma_s", "I", "epsilon", "F"]
# # regime 2
# params_truth = Params(
#     [2, 0.6, 0.4],
#     [1, 0.5, 0],
#     [0.5, 0.1, 0.1],
#     5,  # I
#     0.1,  # epsilon
#     2,  # F
# )
# # regime 1
if args.regime == 1:
    params_truth = Params(
        [2, 0.2, 0.4],
        [0.2, 0.1, 0],
        [0.5, 1.2, 0.8],
        5,  # I
        1,  # epsilon
        2,  # F
    )
elif args.regime == 4:
    params_truth = Params(
        [2 * 2, 0.2, 0.4 / 2],
        [0.2, 0.1, 0],
        [0.5, 1.2, 0.8],
        5,  # I
        1,  # epsilon
        2,  # F
    )
elif args.regime == 5:
    params_truth = Params(
        [2 / 2, 0.2, 0.4 * 2],
        [0.2, 0.1, 0],
        [0.5, 1.2, 0.8],
        5,  # I
        1,  # epsilon
        2,  # F
    )
elif args.regime == 3:
    # regime 3
    params_truth = Params(
        [2, 0.6, 0.4],  # gamma_s
        [1, 1, 10],  # L_s
        [0.5 / np.sqrt(0.1), 0.1, 0.1],  # sigma_s
        5,  # I
        0.1,  # epsilon
        2,  # F
    )
else:

    print("regime not set")
    sys.exit()
print(params_truth)
export_json(
    params_truth._asdict(), osp.join(output_path, "true_params_used.json")
)
# %%
trans_mat = np.asarray([[1, 0, 0]])
obs_noise = np.asarray([1]) * args.sigma_obs
obs_noise

# %%
triad_EnKF = EnsembleKalmanFilterSmootherTriad(
    n_steps, model_dim, params_truth, obs_dt, trans_mat, obs_noise
)

# %%
set_rand_seed(args.rand_seed)
obs, true_state = triad_EnKF.simulate()


# %%
for i in range(model_dim):
    print(f"Skewness {skew(true_state[obs_idx, i])}")
    print(f"Kurtosis {kurtosis(true_state[obs_idx, i], fisher=False)}")
    print("\n")

# %% [markdown]
# # The perfect model


# %%
if n_steps > 20000:
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
fig, axs = plt.subplots(
    nrows=model_dim,
    ncols=3,
    figsize=(14, 6),
    gridspec_kw={"width_ratios": [4, 1, 1]},
    constrained_layout=True,
)
x_left = 50
x_right = 100
t = np.linspace(0, (n_steps - 1) * dt, n_steps)
t_obs = np.linspace(0, (obs_n_steps - 1) * obs_dt, obs_n_steps)

for i in range(model_dim):
    u = true_state[:, i]
    mean_u, std_u = norm.fit(u)
    axs[i, 0].plot(t, u, color="b")
    kde_u, u_axis = compute_kde_axis(u)
    axs[i, 1].plot(u_axis, kde_u(u_axis), color="b")

    axs[i, 1].plot(
        u_axis, norm.pdf(u_axis, mean_u, std_u), dashes=[6, 2], color="r"
    )
    axs[0, 1].set_title("PDF")
    axs[0, 0].set_title("(a) Trajectory of $u$")
    axs[i, 0].set_xlim([x_left, x_right])

    axs[i, 2].plot(t_auto, acf_u_truth[:, i], color="b")
    axs[i, 2].set_xlim([0, 15])
    axs[0, 2].set_title("ACF")

savefig(fig, osp.join(output_path, args.model + "_trajectory.pdf"))

# %%
for i in range(model_dim):
    print(f"Skewness truth: {skew(true_state[obs_idx, i])}")
    print(f"Kurtosis truth: {kurtosis(true_state[obs_idx, i], fisher=False)}")


# %% [markdown]
# # Imperfect model

# %%
from scipy.optimize import curve_fit


def func(x, a, b):
    return a * np.exp(-b * x) + 1 - a


# %%
mean_s_arr = []
var_s = []
start_dim = 1
for i in range(start_dim, model_dim):
    mean_s_arr.append(np.mean(true_state[:, i]))
    var_s.append(np.var(true_state[:, i]))

d_s_arr = []
sigma_s_arr = []
for i in range(start_dim, model_dim):
    autocorr_v = estimated_autocorrelation(true_state[obs_idx, i][obs_burnin:])

    auto_trunk = int(15 / obs_dt)
    popt, pcov = curve_fit(
        func, t_auto[:auto_trunk], np.real(autocorr_v)[:auto_trunk]
    )
    plt.figure()
    plt.plot(t_auto, acf_u_truth[:, i], "b")
    plt.plot(
        t_auto[:auto_trunk],
        func(t_auto[:auto_trunk], *popt),
        "r-",
    )
    plt.xlim([0, 15])

    d_s = 1 / np.sum(func(t_auto[:auto_trunk], *popt)) / args.obs_dt
    d_s_arr.append(d_s)
    sigma_s_arr.append(np.sqrt(var_s[i - start_dim] * 2 * d_s))

# %%
Params_M = namedtuple(
    "Params",
    [
        "gamma_s",
        "L_s",
        "sigma_s",
        "I",
        "epsilon",
        "F",
        "d_M",
        "mean_M",
        "sigma_M",
    ],
)

# %%
params_wrong = Params_M(
    params_truth.gamma_s,
    params_truth.L_s,
    params_truth.sigma_s,
    params_truth.I,
    params_truth.epsilon,
    params_truth.F,
    d_s_arr,
    mean_s_arr,
    sigma_s_arr,
)
export_json(
    params_wrong._asdict(), osp.join(output_path, "wrong_params_used.json")
)

# %%
triad_EnKF_wrong = EnsembleKalmanFilterSmootherTriadApprox(
    n_steps, model_dim, params_wrong, obs_dt, trans_mat, obs_noise
)

# %%
params_wrong

# %%
_, wrong_state = triad_EnKF_wrong.simulate()

# %%
acf_u_wrong = np.zeros((auto_obs_steps, model_dim))
for i in tqdm(range(model_dim)):
    acf_u_wrong[:, i] = estimated_autocorrelation(
        wrong_state[obs_idx, i][obs_burnin:]
    )

# %%
fig, axs = plt.subplots(
    nrows=model_dim,
    ncols=3,
    figsize=(20, 6),
    gridspec_kw={"width_ratios": [4, 1, 1]},
    constrained_layout=True,
)
x_left = 50
x_right = 100
t = np.linspace(0, (n_steps - 1) * dt, n_steps)
t_obs = np.linspace(0, (obs_n_steps - 1) * obs_dt, obs_n_steps)

for i in range(model_dim):
    u = true_state[:, i]
    mean_u, std_u = norm.fit(u)
    axs[i, 0].plot(t, u, color="b")
    kde_u, u_axis = compute_kde_axis(u)
    axs[i, 1].plot(u_axis, kde_u(u_axis), color="b", label="truth")
    axs[0, 0].set_title("(a) Trajectory of $u$")
    axs[i, 0].set_xlim([x_left, x_right])

    if i >= 1:
        # Gaussian fit from the perfect model
        mean_u, std_u = norm.fit(u)
        axs[i, 1].plot(
            u_axis, norm.pdf(u_axis, mean_u, std_u), dashes=[6, 2], color="r"
        )

        u = wrong_state[obs_idx, i][obs_burnin:]
        kde_u, u_axis = compute_kde_axis(u)
        axs[i, 1].plot(
            u_axis, kde_u(u_axis), color="r", label="imperfect model"
        )

    axs[i, 2].plot(t_auto, acf_u_truth[:, i], color="b")
    axs[i, 2].plot(t_auto, acf_u_wrong[:, i], color="r")
    axs[i, 2].set_xlim([0, 10])

handles, labels = axs[1, 1].get_legend_handles_labels()
axs[-1, 0].legend(
    handles, labels, ncol=4, loc="upper right", bbox_to_anchor=(1, -0.20)
)
savefig(fig, osp.join(output_path, args.model + "_trajectory_vs_wrong.pdf"))


# %%
if args.stop_at_modeling:
    sys.exit()
# %% [markdown]
# # Filter, smoother, and sampled trajectories

# %%
num_ensembles = 300
init_mu = np.zeros(model_dim)
init_mu = np.random.randn(model_dim) * 0.01
init_R = np.eye(model_dim) * args.sigma_obs

Y_init = np.random.multivariate_normal(init_mu, init_R, num_ensembles).T
inflation = 0
Lag = 16
set_rand_seed(args.rand_seed)
start_time = time.time()
(
    gamma_mean_trace,
    gamma_cov_trace,
    gamma_mean_smooth,
    gamma_cov_smooth,
    gamma_ensembles,
) = triad_EnKF_wrong.online_EnKS(
    obs,
    Y_init,
    Lag=Lag,
    inflation=inflation,
)
print("SPEKF_model_ETKS " + "--- %s seconds ---" % (time.time() - start_time))


# %%
acf_u_smooth = np.zeros((auto_obs_steps, model_dim))
acf_u_sampled = np.zeros((auto_obs_steps, model_dim))
for i in tqdm(range(model_dim)):
    acf_u_smooth[:, i] = estimated_autocorrelation(
        gamma_mean_smooth[:, i][obs_burnin:]
    )
    acf_u_sampled[:, i] = estimated_autocorrelation(
        gamma_ensembles[:, i, 0][obs_burnin:]
    )

# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("legend", fontsize=14)

fig, axs = plt.subplots(
    nrows=model_dim,
    ncols=3,
    figsize=(15, 2 * (model_dim)),
    constrained_layout=True,
    gridspec_kw={"width_ratios": [5, 1, 1]},
)

title = [
    "$u_1$",
    "$u_2$",
    "$u_3$",
]
if n_steps > 10000:
    x_left = 100
    x_right = 150
else:
    x_left = obs_burnin * obs_dt
    x_right = n_steps * dt

for i in range(model_dim):
    axs[i, 0].plot(
        t_obs[obs_burnin:],
        np.real(true_state)[obs_idx, i][obs_burnin:],
        "b",
        label="Truth",
    )
    axs[i, 0].set_xlim([x_left, x_right])
    u = np.real(true_state[obs_idx, i])[obs_burnin:]
    kde_u, u_axis = compute_kde_axis(u, dist=3)
    axs[i, 1].plot(u_axis, kde_u(u_axis), color="b")

    axs[i, 2].plot(u_axis, kde_u(u_axis), color="b")
    if i >= 1:
        u = wrong_state[obs_idx, i][obs_burnin:]
        axs[i, 0].plot(
            t_obs[obs_burnin:],
            u,
            "r",
            label="imperfect model",
        )
        kde_u, _ = compute_kde_axis(u, dist=3)
        axs[i, 1].plot(u_axis, kde_u(u_axis), color="r")

        axs[i, 2].plot(u_axis, kde_u(u_axis), color="r")
    j = 1
    # j is number of the sampled trajectories
    u = np.real(gamma_ensembles)[:, i, j][obs_burnin:]
    kde_u, _ = compute_kde_axis(u, dist=3)
    axs[i, 1].plot(u_axis, kde_u(u_axis), color="lime")

    axs[i, 2].plot(u_axis, kde_u(u_axis), color="lime")

    smoothing_upper_u = np.real(gamma_mean_smooth)[:, i][
        obs_burnin:
    ] + np.sqrt(np.real(gamma_cov_smooth)[:, i, i][obs_burnin:])
    smoothing_lower_u = np.real(gamma_mean_smooth)[:, i][
        obs_burnin:
    ] - np.sqrt(np.real(gamma_cov_smooth)[:, i, i][obs_burnin:])
    axs[i, 0].plot(
        t_obs[obs_burnin:],
        np.real(gamma_ensembles)[:, i, j][obs_burnin:],
        "lime",
        label="Sampled trajectories",
    )

    axs[i, 0].set_title(f"True trajectory of $u_{i + 1}$")

    axs[i, 1].set_title("PDF of " + title[i])

    axs[i, 2].set_title("PDF of " + title[i] + " log scale")

handles, labels = axs[1, 0].get_legend_handles_labels()
axs[-1, 0].legend(
    handles, labels, ncol=4, loc="upper right", bbox_to_anchor=(1, -0.20)
)
savefig(
    fig,
    osp.join(output_path, "PDFs_SPEKF_" + args.model + ".pdf"),
)


# %%
skew(true_state[:, 1])

# %%
# for i in range(1,):
i = 1
j = 1
skewness_ave = 0
kurtosis_ave = 0
print(f"Skewness truth: {skew(true_state[obs_idx, i])}")
print(f"Skewness imperfect: {skew(wrong_state[obs_idx, i])}")
for j in range(10):
    skewness_ave += skew(gamma_ensembles[:, i, j])
print(f"Skewness sampled: {skew(gamma_ensembles[:, i, j])}")
print(f"Skewness sampled ave: {skewness_ave / 10}")
print("\n")
print(f"Kurtosis truth: {kurtosis(true_state[obs_idx, i], fisher=False)}")
print(f"Kurtosis imperfect: {kurtosis(wrong_state[obs_idx, i], fisher=False)}")
for j in range(10):
    kurtosis_ave += kurtosis(gamma_ensembles[:, i, j], fisher=False)
print(f"Kurtosis sampled: {kurtosis(gamma_ensembles[:, i, j], fisher=False)}")
print(f"Kurtosis sampled ave: {kurtosis_ave / 10}")
print("\n")

# %%
num_ensembles = 300
init_mu = np.zeros(model_dim)
init_mu = np.random.randn(model_dim) * 0.01
init_R = np.eye(model_dim) * args.sigma_obs

Y_init = np.random.multivariate_normal(init_mu, init_R, num_ensembles).T
set_rand_seed(args.rand_seed)
start_time = time.time()
(
    gamma_mean_trace_perfect,
    gamma_cov_trace_perfect,
    gamma_mean_smooth_perfect,
    gamma_cov_smooth_perfect,
    gamma_ensembles_perfect,
) = triad_EnKF.online_EnKS(
    obs,
    Y_init,
    Lag=Lag,
    inflation=inflation,
)
print(
    "Perfect_model_ETKS " + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
smoother_dict = {
    "inflation": inflation,
    "Lag": Lag,
}

export_json(
    smoother_dict,
    osp.join(output_path, "smoother_dict.json"),
)

# %%
pred_num_ensembles = 50
np.savez_compressed(
    osp.join(output_path, "sampling.npz"),
    true_state=true_state,
    wrong_state=wrong_state,
    obs=obs,
    gamma_mean_trace=gamma_mean_trace,
    gamma_cov_trace=gamma_cov_trace,
    gamma_mean_smooth=gamma_mean_smooth,
    gamma_cov_smooth=gamma_cov_smooth,
    gamma_ensembles=gamma_ensembles[:, :, :pred_num_ensembles],
)
np.savez_compressed(
    osp.join(output_path, "sampling_perfect.npz"),
    gamma_mean_trace_perfect=gamma_mean_trace_perfect,
    gamma_cov_trace_perfect=gamma_cov_trace_perfect,
    gamma_mean_smooth_perfect=gamma_mean_smooth_perfect,
    gamma_cov_smooth_perfect=gamma_cov_smooth_perfect,
    gamma_ensembles_perfect=gamma_ensembles_perfect[:, :, :pred_num_ensembles],
)


# %%
acf_u_smooth = np.zeros((auto_obs_steps, model_dim))
acf_u_sampled = np.zeros((auto_obs_steps, model_dim))
for i in tqdm(range(model_dim)):
    acf_u_smooth[:, i] = estimated_autocorrelation(
        np.real(gamma_mean_smooth[:, i])[obs_burnin:]
    )
    acf_u_sampled[:, i] = estimated_autocorrelation(
        np.real(gamma_ensembles[:, i, 0])[obs_burnin:]
    )

# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("legend", fontsize=14)

fig, axs = plt.subplots(
    nrows=model_dim,
    ncols=3,
    figsize=(15, 2 * (model_dim)),
    constrained_layout=True,
    gridspec_kw={"width_ratios": [5, 1, 1]},
)

title = [
    "$u_1$",
    "$u_2$",
    "$u_3$",
]
if n_steps > 10000:
    x_left = 100
    x_right = 150
else:
    x_left = obs_burnin * obs_dt
    x_right = n_steps * dt

for i in range(model_dim):
    axs[i, 0].plot(
        t_obs[obs_burnin:],
        np.real(true_state)[obs_idx, i][obs_burnin:],
        "b",
        label="Truth",
    )
    axs[i, 0].set_xlim([x_left, x_right])
    u = np.real(true_state[obs_idx, i])[obs_burnin:]
    kde_u, u_axis = compute_kde_axis(u)
    axs[i, 1].plot(u_axis, kde_u(u_axis), color="b")
    axs[i, 2].plot(t_auto, acf_u_truth[:, i], color="b")
    u = np.real(gamma_mean_smooth)[:, i][obs_burnin:]
    kde_u, u_axis = compute_kde_axis(u)
    axs[i, 1].plot(u_axis, kde_u(u_axis), color="orange")
    axs[i, 2].plot(t_auto, acf_u_smooth[:, i], color="orange")
    for j in range(1):
        # j is number of the sampled trajectories
        u = np.real(gamma_ensembles)[:, i, j][obs_burnin:]
        kde_u, u_axis = compute_kde_axis(u)
        axs[i, 1].plot(u_axis, kde_u(u_axis), color="lime")
        axs[i, 2].plot(t_auto, acf_u_sampled[:, i], color="lime")
        smoothing_upper_u = np.real(gamma_mean_smooth)[:, i][
            obs_burnin:
        ] + np.sqrt(np.real(gamma_cov_smooth)[:, i, i][obs_burnin:])
        smoothing_lower_u = np.real(gamma_mean_smooth)[:, i][
            obs_burnin:
        ] - np.sqrt(np.real(gamma_cov_smooth)[:, i, i][obs_burnin:])
        axs[i, 0].plot(
            t_obs[obs_burnin:],
            np.real(gamma_ensembles)[:, i, j][obs_burnin:],
            "lime",
            label="Sampled trajectories",
        )
        axs[i, 0].fill_between(
            t_obs[obs_burnin:],
            smoothing_upper_u,
            smoothing_lower_u,
            facecolor="lime",
            alpha=0.2,
            label="1 std from smoother mean",
        )

    axs[i, 0].plot(
        t_obs[obs_burnin:],
        np.real(gamma_mean_smooth)[:, i][obs_burnin:],
        "orange",
        label="Smoother mean",
    )
    axs[i, 0].set_title(f"True trajectory of " + title[i])

    axs[i, 1].set_title("PDF of " + title[i])

    axs[i, 2].set_title("ACF of " + title[i])
    axs[i, 2].set_xlim([0, 15])
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[-1, 0].legend(
    handles, labels, ncol=4, loc="upper right", bbox_to_anchor=(1, -0.20)
)
savefig(
    fig,
    osp.join(output_path, "PDFs_ACFs_SPEKF_" + args.model + ".pdf"),
)


# %% [markdown]
# # Model based prediction

# %%
from pkg.helper import model_prediction_one_traj_new_new

# %%
pred_dt = obs_dt
pred_n_steps = int(dt * (n_steps - 1) / pred_dt) + 1
pred_K = int(pred_dt / dt)


t_pred = np.linspace(0, (pred_n_steps - 1) * pred_dt, pred_n_steps)
pred_idx = np.linspace(0, int((pred_n_steps - 1)) * pred_K, pred_n_steps)
pred_idx = pred_idx.astype(int)

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

print(f"pred_start_pred_step: {pred_start_pred_step}")
print(f"pred_total_pred_steps: {pred_total_pred_steps}")

print(f"start time unit: {pred_start_pred_step * pred_dt}")
print(
    "end time unit: "
    + f"{(pred_start_pred_step + pred_total_pred_steps) * pred_dt}"
)
print(f"length of entire trajectories: {(n_steps - 1) * dt}")

last_lead_t = int(args.last_lead_time / dt)
lead_step = pred_K

pred_last_lead_t = last_lead_t // pred_K
pred_lead_step = lead_step // pred_K

lead_time_steps = np.linspace(
    0, last_lead_t, int(last_lead_t / lead_step) + 1
).astype(
    int
)  # only use for plotting
print(
    "lead_time_steps (in terms of in true trajectories space, dt):"
    + f" {lead_time_steps}"
)
print(f"final lead time: {last_lead_t * dt}")
print(f"total_lead_time_steps: {len(lead_time_steps)}")


# %%
t_target = t[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
]
target = true_state[pred_idx][
    pred_start_pred_step : pred_start_pred_step + pred_total_pred_steps
]
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
initial_DA_perfect = np.zeros(
    (pred_num_ensembles, initial_mean.shape[0], model_dim)
)
initial_values_repeat = np.zeros(
    (pred_num_ensembles, initial_mean.shape[0], model_dim)
)
for i in range(pred_num_ensembles):
    initial_values_repeat[i] = initial_values

for i in range(initial_mean.shape[0]):
    initial_DA[:, i, :] = np.random.multivariate_normal(
        initial_mean[i], initial_cov[i], pred_num_ensembles
    )
    initial_DA_perfect[:, i, :] = np.random.multivariate_normal(
        initial_mean_perfect[i], initial_cov_perfect[i], pred_num_ensembles
    )


# %%
for i in range(model_dim):
    plt.figure(figsize=(20, 2))
    plt.plot(initial_values[:, i], "b")
    plt.plot(np.mean(initial_DA_perfect, axis=0)[:, i], "lime")
    plt.plot(np.mean(initial_DA, axis=0)[:, i], "r")


# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_mean = model_prediction_one_traj_new_new(
    model=triad_EnKF,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_values_repeat,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
print("Perfect prediction" + "--- %s seconds ---" % (time.time() - start_time))

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_mean_DA = model_prediction_one_traj_new_new(
    model=triad_EnKF,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_DA_perfect,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
print(
    "Perfect prediction (DA)"
    + "--- %s seconds ---" % (time.time() - start_time)
)

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_wrong = model_prediction_one_traj_new_new(
    model=triad_EnKF_wrong,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_values_repeat,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
print("SPEKF prediction" + "--- %s seconds ---" % (time.time() - start_time))

# %%
start_time = time.time()
set_rand_seed(args.rand_seed)
predict_wrong_DA = model_prediction_one_traj_new_new(
    model=triad_EnKF_wrong,
    num_ensembles=pred_num_ensembles,
    initial_values=initial_DA,
    last_lead_t=pred_last_lead_t,
    lead_step=pred_lead_step,
    K=pred_K,
)
print(
    "SPEKF prediction (DA)" + "--- %s seconds ---" % (time.time() - start_time)
)


# %%
rmse_perfect = np.zeros((len(predict_mean), model_dim))
rmse_perfect_DA = np.zeros((len(predict_mean), model_dim))
rmse_wrong = np.zeros((len(predict_mean), model_dim))
rmse_wrong_DA = np.zeros((len(predict_mean), model_dim))

for i in range(len(predict_mean)):
    rmse_perfect[i] = rmse(predict_mean[i], target)
    rmse_perfect_DA[i] = rmse(predict_mean_DA[i], target)
    rmse_wrong[i] = rmse(predict_wrong[i], target)
    rmse_wrong_DA[i] = rmse(predict_wrong_DA[i], target)

corr_perfect = np.zeros((len(predict_mean), model_dim))
corr_perfect_DA = np.zeros((len(predict_mean), model_dim))
corr_wrong = np.zeros((len(predict_mean), model_dim))
corr_wrong_DA = np.zeros((len(predict_mean), model_dim))

for i in range(len(predict_mean)):
    corr_perfect[i] = patt_corr(predict_mean[i], target)
    corr_perfect_DA[i] = patt_corr(predict_mean_DA[i], target)
    corr_wrong[i] = patt_corr(predict_wrong[i], target)
    corr_wrong_DA[i] = patt_corr(predict_wrong_DA[i], target)


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
    rmse_perfect=rmse_perfect,
    rmse_perfect_DA=rmse_perfect_DA,
    rmse_wrong=rmse_wrong,
    rmse_wrong_DA=rmse_wrong_DA,
    corr_perfect=corr_perfect,
    corr_perfect_DA=corr_perfect_DA,
    corr_wrong=corr_wrong,
    corr_wrong_DA=corr_wrong_DA,
)

export_json(
    {
        "lead_time_steps": list(lead_time_steps.astype(float)),
        "total_lead_time_steps:": len(lead_time_steps),
    },
    osp.join(output_path, "lead_time_steps.json"),
)


# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("legend", fontsize=16)
fig, axs = plt.subplots(
    nrows=2,
    ncols=model_dim,
    figsize=(model_dim * 3, 6),
    #     gridspec_kw={"width_ratios": [4, 1]},
    #     constrained_layout=True,
)
title_RMSE = [
    "RMSE u_1",
    "RMSE u_2",
    "RMSE u_3",
]
title_corr = [
    "Corr u_1",
    "Corr u_2",
    "Corr u_3",
]
for i in range(model_dim):
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
        label="imperfect model",
    )
    axs[0, i].plot(
        lead_time_steps * dt,
        rmse_wrong_DA[:, i],
        "--",
        color="r",
        label="imperfect model with assimilated IC",
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
        corr_wrong[:, i],
        "r",
    )
    axs[1, i].plot(
        lead_time_steps * dt,
        corr_wrong_DA[:, i],
        "--",
        color="r",
    )

    axs[1, i].plot(
        [0, lead_time_steps[-1] * dt],
        [0.5, 0.5],
        "--",
    )
    axs[1, i].set_title(title_corr[i % 5])

handles, labels = axs[0, 0].get_legend_handles_labels()
axs[1, 0].legend(handles, labels, ncol=2, bbox_to_anchor=(4, -0.2))
savefig(
    fig,
    osp.join(
        output_path,
        "physical_model_based_pred_corr_" + args.model + ".pdf",
    ),
)
plt.rcdefaults()

# %%
plt.rc("axes", titlesize=16)  # using a size in points
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("legend", fontsize=16)
fig, axs = plt.subplots(
    nrows=model_dim,
    ncols=2,
    figsize=(30, 2 * model_dim),
)
lead_time_dim_options = [5, 10]
x_left = args.pred_start_time
x_right = x_left + 80

for i in range(model_dim):
    pred_dim = i
    for j in [0, 1]:
        pred_time = lead_time_dim_options[j]
        axs[i, j].plot(
            t_target,
            predict_mean[pred_time, :, pred_dim],
            "b",
            label="perfect model",
        )
        axs[i, j].plot(
            t_target,
            predict_wrong[pred_time, :, pred_dim],
            "r",
            label="imperfect model",
        )
        axs[i, j].plot(t_target, target[:, pred_dim], "black")
        axs[i, j].set_xlim([x_left, x_right])

for j in [0, 1]:
    pred_time = lead_time_dim_options[j]
    axs[0, j].set_title("Lead time =  %4.2f" % (pred_time * dt * pred_K))

savefig(
    fig,
    osp.join(
        output_path,
        "physical_model_based_pred_traj_" + args.model + ".pdf",
    ),
)
plt.rcdefaults()

# %%
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
) = triad_EnKF_wrong.online_EnKS_for_IC(
    obs,
    Y_init,
    Lag=Lag,
    pred_start_pred_step=pred_start_pred_step,
    pred_total_pred_steps=pred_total_pred_steps,
    pred_last_lead_t=pred_last_lead_t,
    L_init=L_init,
    pred_num_ensembles=pred_num_ensembles,
    inflation=inflation,
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
) = triad_EnKF.online_EnKS_for_IC(
    obs,
    Y_init,
    Lag=Lag,
    pred_start_pred_step=pred_start_pred_step,
    pred_total_pred_steps=pred_total_pred_steps,
    pred_last_lead_t=pred_last_lead_t,
    L_init=L_init,
    pred_num_ensembles=pred_num_ensembles,
    inflation=inflation,
)
np.savez_compressed(
    osp.join(output_path, "sampling_for_IC_perfect.npz"),
    gamma_ensembles_for_IC_short=gamma_ensembles_for_IC_short_perfect,
)

# %%

# %%

# %%
