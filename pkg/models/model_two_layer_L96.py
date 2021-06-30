import numpy as np
from ..ensembleKalmanfilter import EnsembleKalmanFilterSmootherBase
from tqdm import tqdm


def SVD_J(J_i, num_ensembles):
    e_Tao_sq, e_P = np.linalg.eigh(J_i)

    e_Tao_sq[e_Tao_sq < 1e-5] = 0
    e_Tao = np.sqrt(e_Tao_sq)
    e_Tao_inv = np.zeros(e_Tao.shape)
    for dim in range(num_ensembles):
        if e_Tao[dim] != 0:
            e_Tao_inv[dim] = 1 / e_Tao[dim]
    e_Tao_sq = np.diag(e_Tao_sq)
    e_Tao_sq_inv = e_Tao_inv ** 2
    e_Tao = np.diag(e_Tao)
    e_Tao_inv = np.diag(e_Tao_inv)
    e_Tao_sq_inv = np.diag(e_Tao_sq_inv)
    return e_Tao_sq_inv, e_Tao_inv, e_P


class EnsembleKalmanFilterSmootherL96(EnsembleKalmanFilterSmootherBase):
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        if params:
            params = params
        else:
            params = self.params
        model_dim = Y_prev.shape[0]
        model_noise = np.zeros((model_dim, model_dim))
        for i in range(params.dim_I):
            model_noise[i, i] = params.sigma_u
        for i in range(params.dim_I, model_dim):
            model_noise[i, i] = params.sigma_v

        Y = (
            np.zeros((model_dim))
            if num_ensembles == 1
            else np.zeros((model_dim, num_ensembles))
        )
        u_prev = Y_prev[: params.dim_I]
        if params.dim_J != 0:
            v_prev = Y_prev[params.dim_I :]
            v_prev_sum = (
                np.sum(
                    v_prev.reshape(params.dim_I, params.dim_J, num_ensembles),
                    axis=1,
                ).flatten()
                if num_ensembles == 1
                else np.sum(
                    v_prev.reshape(params.dim_I, params.dim_J, num_ensembles),
                    axis=1,
                )
            )  # shape is dim_I * num_ensembles
            u_prev_reap = np.repeat(u_prev, params.dim_J, axis=0)
            u = (
                u_prev
                + (
                    -np.roll(u_prev, 1, axis=0)
                    * (
                        np.roll(u_prev, 2, axis=0)
                        - np.roll(u_prev, -1, axis=0)
                    )
                    - u_prev
                    + params.f
                    - params.h * params.c / params.dim_J * v_prev_sum
                )
                * self.dt
            )
            v = (
                v_prev
                + (
                    -params.b
                    * params.c
                    * np.roll(v_prev, -1, axis=0)
                    * (
                        np.roll(v_prev, -2, axis=0)
                        - np.roll(v_prev, 1, axis=0)
                    )
                    - params.c * v_prev
                    + params.h * params.c / params.dim_J * u_prev_reap
                )
                * self.dt
            )
            Y[params.dim_I :] = v
        else:
            u = (
                u_prev
                + (
                    -np.roll(u_prev, 1, axis=0)
                    * (
                        np.roll(u_prev, 2, axis=0)
                        - np.roll(u_prev, -1, axis=0)
                    )
                    - u_prev
                    + params.f
                )
                * self.dt
            )

        Y[: params.dim_I] = u

        if num_ensembles == 1:
            Y += np.sqrt(self.dt) * model_noise @ np.random.randn(model_dim)
        else:
            Y += (
                np.sqrt(self.dt)
                * model_noise
                @ np.random.randn(model_dim, num_ensembles)
            )
        return Y

    def online_ETKS(
        self,
        obs,
        Y_init,
        Lag,
        inflation=0,
        localization=None,
        radius_L=1,  # localization radius
        obs_n_steps=None,
    ):

        """
        The first version
        """
        obs_dt = self.obs_dt
        dt = self.dt
        model_dim = self.model_dim
        if obs_n_steps:
            obs_n_steps = obs_n_steps
        else:
            obs_n_steps = self.obs_n_steps
        trans_mat = self.trans_mat
        obs_noise = self.obs_noise
        num_ensembles = Y_init.shape[1]
        obs_dim = self.obs_dim

        K = int(obs_dt / dt)  # number of timestampes between obs
        dim_I = self.params.dim_I
        dim_J = self.params.dim_J
        gamma_mean_trace = np.zeros((obs_n_steps, model_dim))
        gamma_cov_trace = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_smooth = np.zeros((obs_n_steps, model_dim))
        gamma_cov_smooth = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_trace[0] = np.mean(Y_init, axis=1)
        gamma_cov_trace[0] = np.cov(Y_init)
        Y_en = Y_init
        Y_ensembles = np.zeros((obs_n_steps * model_dim, num_ensembles))
        gamma_ensembles = np.zeros((obs_n_steps, model_dim, num_ensembles))
        Y_ensembles[:model_dim, :] = Y_en
        for i in tqdm(range(1, obs_n_steps)):
            for k in range(K):
                Y_init = self.simulate_state(Y_init, num_ensembles)
            Y_prior = Y_init
            # compute the last step, this will used in the observations part
            prior_mean = np.mean(Y_prior, axis=1)
            U_i = Y_prior - prior_mean.reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            H_prior_mean = (
                trans_mat @ prior_mean
            )  # apply observation operator to proir mean
            Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            if localization:
                for l in range(obs_dim):
                    # deal with observations
                    # for radius = 1 of the localization
                    # loc_dim is for observations with in radius L
                    loc_dim = np.arange(l - radius_L, l + radius_L + 1) % dim_I
                    V_i_loc = V_i[loc_dim]
                    H_prior_mean_loc = H_prior_mean[loc_dim]
                    obs_noise_loc = obs_noise[loc_dim]
                    R_0_inv_loc = np.linalg.inv(np.diag(obs_noise_loc ** 2))
                    J_i_loc = (num_ensembles - 1) / (1 + inflation) * np.eye(
                        num_ensembles
                    ) + V_i_loc.T @ R_0_inv_loc @ V_i_loc
                    e_Tao_sq_inv_loc, e_Tao_inv_loc, e_P_loc = SVD_J(
                        J_i_loc, num_ensembles
                    )

                    # start to select local_domain
                    #  = [u_l, v_{l,1}, ...v_{l,dim_J}]
                    J_loc = dim_I + l * dim_J
                    idx_dim = np.hstack(
                        (np.asarray([l]), np.arange(J_loc, J_loc + dim_J))
                    )
                    # smoother part
                    # select prior mean to update
                    if i > Lag:
                        update_dim = idx_dim + model_dim * (i - Lag)
                        for _ in range(i - Lag, i):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_ensembles[update_dim, :]
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((Lag + 1) * (1 + dim_J), 1)
                    else:
                        update_dim = idx_dim
                        for _ in range(1, i + 1):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_ensembles[update_dim, :]
                        # writing in this way to double check the dimension
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((i + 1) * (1 + dim_J), 1)

                    # combine smoother Lag as well as local domain for updating
                    # update local_domain^{i - Lag}, ... local_domain^i

                    # update mean and perturbations

                    prior_mean_all_loc = np.mean(Y_prior_all_loc, axis=1)
                    kal_gain_loc = (
                        U_prior_all_loc
                        @ e_P_loc
                        @ e_Tao_sq_inv_loc
                        @ e_P_loc.T
                        @ V_i_loc.T
                        @ R_0_inv_loc
                    )
                    # posterior mean (smoother) at local domain
                    gamma_mean_all_loc = prior_mean_all_loc + kal_gain_loc @ (
                        obs[i][loc_dim] - H_prior_mean_loc
                    )
                    # transform matrix at local domain
                    T_loc = (
                        np.sqrt(num_ensembles - 1)
                        * e_P_loc
                        @ e_Tao_inv_loc
                        @ e_P_loc.T
                    )
                    # posterior perturbation (smoother) at local domain
                    U_post_all_loc = U_prior_all_loc @ T_loc

                    # select ensemble members to update
                    # update Y_ensemles for previous Lag/i at local domain
                    for idx_i in range(len(update_dim)):
                        idx_dim_i = update_dim[idx_i]
                        Y_ensembles[idx_dim_i] = (
                            gamma_mean_all_loc[idx_i] + U_post_all_loc[idx_i]
                        )
                    # update initial values for next step
                    Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :][
                        idx_dim
                    ]
                    Y_init[idx_dim] = Y_en
                    gamma_mean_trace[i][idx_dim] = np.mean(Y_en, axis=1)
                    gamma_cov_trace[i, idx_dim[:, None], idx_dim] = np.cov(
                        Y_en
                    )
            else:
                J_i = (num_ensembles - 1) / (1 + inflation) * np.eye(
                    num_ensembles
                ) + V_i.T @ R_0_inv @ V_i

                e_Tao_sq_inv, e_Tao_inv, e_P = SVD_J(J_i, num_ensembles)

                # select prior mean to update
                if i > Lag:
                    # here Y_prior_all is different
                    Y_prior_all = Y_ensembles[
                        (i - Lag) * model_dim : (i + 1) * model_dim, :
                    ]
                    U_prior_all = Y_prior_all - np.mean(
                        Y_prior_all, axis=1
                    ).reshape((Lag + 1) * model_dim, 1)
                else:
                    Y_prior_all = Y_ensembles[: (i + 1) * model_dim, :]
                    U_prior_all = Y_prior_all - np.mean(
                        Y_prior_all, axis=1
                    ).reshape((i + 1) * model_dim, 1)

                # update mean and perturbations
                prior_mean_all = np.mean(Y_prior_all, axis=1)
                kal_gain = (
                    U_prior_all @ e_P @ e_Tao_sq_inv @ e_P.T @ V_i.T @ R_0_inv
                )

                gamma_mean_all = prior_mean_all + kal_gain @ (
                    obs[i] - trans_mat @ prior_mean
                )
                # Check this!! if will add to all previous states
                T = np.sqrt(num_ensembles - 1) * e_P @ e_Tao_inv @ e_P.T
                U_post_all = U_prior_all @ T

                # select ensemble members to update
                if i > Lag:
                    Y_ensembles[
                        (i - Lag) * model_dim : (i + 1) * model_dim, :
                    ] = (gamma_mean_all[:, None] + U_post_all)
                else:
                    Y_ensembles[: (i + 1) * model_dim, :] = (
                        gamma_mean_all[:, None] + U_post_all
                    )
                # update the last ensemble member, which is the filter mean and
                # also the initial values for the prediction in next step
                # think about this if need to change!!
                Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :]
                gamma_mean_trace[i] = np.mean(Y_en, axis=1)
                gamma_cov_trace[i] = np.cov(Y_en)
                Y_init = Y_en

            if i == obs_n_steps - 1:
                for j in tqdm(range(obs_n_steps)):
                    Y_en = Y_ensembles[j * model_dim : (j + 1) * model_dim, :]
                    gamma_mean_smooth[j] = np.mean(Y_en, axis=1)
                    gamma_cov_smooth[j] = np.cov(Y_en)
                    gamma_ensembles[j, :] = Y_en
        return (
            gamma_mean_trace,
            gamma_cov_trace,
            gamma_mean_smooth,
            gamma_cov_smooth,
            gamma_ensembles,
        )

    def online_ETKS_save(
        self,
        obs,
        Y_init,
        Lag,
        obs_dim_idx,
        inflation=0,
        localization=None,
        radius_L=1,  # localization radius
        obs_n_steps=None,
    ):

        """
        The first version
        """
        obs_dt = self.obs_dt
        dt = self.dt
        model_dim = self.model_dim
        if obs_n_steps:
            obs_n_steps = obs_n_steps
        else:
            obs_n_steps = self.obs_n_steps
        trans_mat = self.trans_mat
        obs_noise = self.obs_noise
        num_ensembles = Y_init.shape[1]

        K = int(obs_dt / dt)  # number of timestampes between obs
        dim_I = self.params.dim_I
        dim_J = self.params.dim_J
        gamma_mean_trace = np.zeros((obs_n_steps, model_dim))
        gamma_cov_trace = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_smooth = np.zeros((obs_n_steps, model_dim))
        gamma_cov_smooth = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_trace[0] = np.mean(Y_init, axis=1)
        gamma_cov_trace[0] = np.cov(Y_init)
        Y_en = Y_init
        Y_ensembles = np.zeros(((Lag + 1) * model_dim, num_ensembles))
        # decide later if only save a few ensemble members
        gamma_ensembles = np.zeros((obs_n_steps, model_dim, num_ensembles))
        Y_ensembles[:model_dim, :] = Y_en
        for i in tqdm(range(1, obs_n_steps)):
            for k in range(K):
                Y_init = self.simulate_state(Y_init, num_ensembles)
            Y_prior = Y_init
            # compute the last step, this will used in the observations part
            prior_mean = np.mean(Y_prior, axis=1)
            U_i = Y_prior - prior_mean.reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            H_prior_mean = (
                trans_mat @ prior_mean
            )  # apply observation operator to proir mean
            # select prior mean/perturbation to update later
            if i > Lag:
                Y_prior_all_prev = Y_ensembles[model_dim:, :]
                Y_prior_all = np.concatenate(
                    (Y_prior_all_prev, Y_prior), axis=0
                )
                U_prior_all = Y_prior_all - np.mean(
                    Y_prior_all, axis=1
                ).reshape((Lag + 1) * model_dim, 1)
            else:
                Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
                Y_prior_all = Y_ensembles[: (i + 1) * model_dim, :]
                U_prior_all = Y_prior_all - np.mean(
                    Y_prior_all, axis=1
                ).reshape((i + 1) * model_dim, 1)
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            if localization:
                for l in range(dim_I):
                    # deal with observations
                    # for radius = 1 of the localization
                    # loc_dim is for observations with in radius L
                    # loc_dim_in_space is the affecting radius in all large scales
                    # but this is not those in observation dims
                    loc_dim_in_space = (
                        np.arange(l - radius_L, l + radius_L + 1) % dim_I
                    )
                    inter = np.intersect1d(loc_dim_in_space, obs_dim_idx)
                    loc_dim = np.asarray([])
                    for idx in inter:
                        loc_dim = np.concatenate(
                            (loc_dim, np.where(obs_dim_idx == idx)[0])
                        )
                    loc_dim = loc_dim.astype(int)
                    # dim_inter = []
                    # for dim1 in obs_dim_idx:
                    #     for dim2 in loc_dim_in_space:
                    #         if dim2 == dim1:
                    #             dim_inter.append(dim2)
                    # loc_dim = []
                    # for ii in range(len(obs_dim_idx)):
                    #     for dim2 in dim_inter:
                    #         if dim2 == obs_dim_idx[ii]:
                    #             loc_dim.append(ii)
                    # loc_dim = np.asarray(loc_dim)
                    V_i_loc = V_i[loc_dim]
                    H_prior_mean_loc = H_prior_mean[loc_dim]
                    obs_noise_loc = obs_noise[loc_dim]
                    R_0_inv_loc = np.linalg.inv(np.diag(obs_noise_loc ** 2))
                    J_i_loc = (num_ensembles - 1) / (1 + inflation) * np.eye(
                        num_ensembles
                    ) + V_i_loc.T @ R_0_inv_loc @ V_i_loc
                    e_Tao_sq_inv_loc, e_Tao_inv_loc, e_P_loc = SVD_J(
                        J_i_loc, num_ensembles
                    )

                    # start to select local_domain
                    #  = [u_l, v_{l,1}, ...v_{l,dim_J}]
                    J_loc = dim_I + l * dim_J
                    idx_dim = np.hstack(
                        (np.asarray([l]), np.arange(J_loc, J_loc + dim_J))
                    )
                    # smoother part
                    # select prior mean to update
                    if i > Lag:
                        update_dim = idx_dim
                        for _ in range(Lag):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_prior_all[update_dim, :]
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((Lag + 1) * (1 + dim_J), 1)
                    else:
                        update_dim = idx_dim
                        for _ in range(1, i + 1):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_ensembles[update_dim, :]
                        # writing in this way to double check the dimension
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((i + 1) * (1 + dim_J), 1)

                    # combine smoother Lag as well as local domain for updating
                    # update local_domain^{i - Lag}, ... local_domain^i

                    # update mean and perturbations

                    prior_mean_all_loc = np.mean(Y_prior_all_loc, axis=1)
                    kal_gain_loc = (
                        U_prior_all_loc
                        @ e_P_loc
                        @ e_Tao_sq_inv_loc
                        @ e_P_loc.T
                        @ V_i_loc.T
                        @ R_0_inv_loc
                    )
                    # posterior mean (smoother) at local domain
                    gamma_mean_all_loc = prior_mean_all_loc + kal_gain_loc @ (
                        obs[i][loc_dim] - H_prior_mean_loc
                    )
                    # transform matrix at local domain
                    T_loc = (
                        np.sqrt(num_ensembles - 1)
                        * e_P_loc
                        @ e_Tao_inv_loc
                        @ e_P_loc.T
                    )
                    # posterior perturbation (smoother) at local domain
                    U_post_all_loc = U_prior_all_loc @ T_loc

                    # select ensemble members to update
                    # update Y_ensemles for previous Lag/i at local domain
                    # update initial values for next step
                    for idx_i in range(len(update_dim)):
                        idx_dim_i = update_dim[idx_i]
                        Y_ensembles[idx_dim_i] = (
                            gamma_mean_all_loc[idx_i] + U_post_all_loc[idx_i]
                        )
                    if i > Lag:
                        Y_en = Y_ensembles[-model_dim:, :][idx_dim]
                        for j in range(Lag + 1):
                            Y_temp = Y_ensembles[
                                j * model_dim : (j + 1) * model_dim, :
                            ][idx_dim]
                            gamma_mean_smooth[i - Lag + j][idx_dim] = np.mean(
                                Y_temp, axis=1
                            )
                            gamma_cov_smooth[
                                i - Lag + j, idx_dim[:, None], idx_dim
                            ] = np.cov(Y_temp)
                            gamma_ensembles[i - Lag + j, :][idx_dim] = Y_temp
                    else:
                        Y_en = Y_ensembles[
                            i * model_dim : (i + 1) * model_dim, :
                        ][idx_dim]
                        for j in range(i + 1):
                            Y_temp = Y_ensembles[
                                j * model_dim : (j + 1) * model_dim, :
                            ][idx_dim]
                            gamma_mean_smooth[j][idx_dim] = np.mean(
                                Y_temp, axis=1
                            )
                            gamma_cov_smooth[
                                j, idx_dim[:, None], idx_dim
                            ] = np.cov(Y_temp)
                            gamma_ensembles[j, :][idx_dim] = Y_temp
                    Y_init[idx_dim] = Y_en
                    gamma_mean_trace[i][idx_dim] = np.mean(Y_en, axis=1)
                    gamma_cov_trace[i, idx_dim[:, None], idx_dim] = np.cov(
                        Y_en
                    )
            else:
                J_i = (num_ensembles - 1) / (1 + inflation) * np.eye(
                    num_ensembles
                ) + V_i.T @ R_0_inv @ V_i

                e_Tao_sq_inv, e_Tao_inv, e_P = SVD_J(J_i, num_ensembles)

                # update mean and perturbations
                prior_mean_all = np.mean(Y_prior_all, axis=1)
                kal_gain = (
                    U_prior_all @ e_P @ e_Tao_sq_inv @ e_P.T @ V_i.T @ R_0_inv
                )

                gamma_mean_all = prior_mean_all + kal_gain @ (
                    obs[i] - trans_mat @ prior_mean
                )
                # Check this!! if will add to all previous states
                T = np.sqrt(num_ensembles - 1) * e_P @ e_Tao_inv @ e_P.T
                U_post_all = U_prior_all @ T

                # select ensemble members to update
                # update the last ensemble member, which is the filter mean
                # also the initial values for the prediction in next step
                if i > Lag:
                    Y_ensembles = gamma_mean_all[:, None] + U_post_all
                    Y_en = Y_ensembles[-model_dim:, :]
                    for j in range(Lag + 1):
                        Y_temp = Y_ensembles[
                            j * model_dim : (j + 1) * model_dim, :
                        ]
                        gamma_mean_smooth[i - Lag + j] = np.mean(
                            Y_temp, axis=1
                        )
                        gamma_cov_smooth[i - Lag + j] = np.cov(Y_temp)
                        gamma_ensembles[i - Lag + j, :] = Y_temp
                else:
                    Y_ensembles[: (i + 1) * model_dim, :] = (
                        gamma_mean_all[:, None] + U_post_all
                    )
                    Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :]
                    for j in range(i + 1):
                        Y_temp = Y_ensembles[
                            j * model_dim : (j + 1) * model_dim, :
                        ]
                        gamma_mean_smooth[j] = np.mean(Y_temp, axis=1)
                        gamma_cov_smooth[j] = np.cov(Y_temp)
                        gamma_ensembles[j, :] = Y_temp

                gamma_mean_trace[i] = np.mean(Y_en, axis=1)
                gamma_cov_trace[i] = np.cov(Y_en)
                Y_init = Y_en
        return (
            gamma_mean_trace,
            gamma_cov_trace,
            gamma_mean_smooth,
            gamma_cov_smooth,
            gamma_ensembles,
        )

    def online_ETKS_for_IC(
        self,
        obs,
        Y_init,
        Lag,
        pred_start_pred_step,
        pred_total_pred_steps,
        pred_last_lead_t,
        L_init,
        pred_num_ensembles,
        inflation=0,
        localization=None,
        radius_L=1,  # localization radius
        obs_n_steps=None,
    ):

        """
        The first version
        """
        obs_dt = self.obs_dt
        dt = self.dt
        model_dim = self.model_dim
        if obs_n_steps:
            obs_n_steps = obs_n_steps
        else:
            obs_n_steps = self.obs_n_steps
        trans_mat = self.trans_mat
        obs_noise = self.obs_noise
        num_ensembles = Y_init.shape[1]
        obs_dim = self.obs_dim

        K = int(obs_dt / dt)  # number of timestampes between obs
        dim_I = self.params.dim_I
        dim_J = self.params.dim_J
        gamma_mean_trace = np.zeros((obs_n_steps, model_dim))
        gamma_cov_trace = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_smooth = np.zeros((obs_n_steps, model_dim))
        gamma_cov_smooth = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_trace[0] = np.mean(Y_init, axis=1)
        gamma_cov_trace[0] = np.cov(Y_init)
        Y_en = Y_init
        Y_ensembles = np.zeros((obs_n_steps * model_dim, num_ensembles))
        gamma_ensembles = np.zeros((obs_n_steps, model_dim, num_ensembles))
        Y_ensembles[:model_dim, :] = Y_en

        gamma_ensembles_for_IC = np.empty(
            (
                (pred_total_pred_steps + pred_last_lead_t),
                L_init,
                model_dim,
                pred_num_ensembles,
            )
        )
        for i in tqdm(range(1, obs_n_steps)):
            for k in range(K):
                Y_init = self.simulate_state(Y_init, num_ensembles)
            Y_prior = Y_init
            # compute the last step, this will used in the observations part
            prior_mean = np.mean(Y_prior, axis=1)
            U_i = Y_prior - prior_mean.reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            H_prior_mean = (
                trans_mat @ prior_mean
            )  # apply observation operator to proir mean
            Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            if localization:
                for l in range(obs_dim):
                    # deal with observations
                    # for radius = 1 of the localization
                    # loc_dim is for observations with in radius L
                    loc_dim = np.arange(l - radius_L, l + radius_L + 1) % dim_I
                    V_i_loc = V_i[loc_dim]
                    H_prior_mean_loc = H_prior_mean[loc_dim]
                    obs_noise_loc = obs_noise[loc_dim]
                    R_0_inv_loc = np.linalg.inv(np.diag(obs_noise_loc ** 2))
                    J_i_loc = (num_ensembles - 1) / (1 + inflation) * np.eye(
                        num_ensembles
                    ) + V_i_loc.T @ R_0_inv_loc @ V_i_loc
                    e_Tao_sq_inv_loc, e_Tao_inv_loc, e_P_loc = SVD_J(
                        J_i_loc, num_ensembles
                    )

                    # start to select local_domain = [u_l, v_{l,1}, ...v_{l,dim_J}]
                    J_loc = dim_I + l * dim_J
                    idx_dim = np.hstack(
                        (np.asarray([l]), np.arange(J_loc, J_loc + dim_J))
                    )
                    # smoother part
                    # select prior mean to update
                    if i + 1 > Lag:
                        update_dim = idx_dim + model_dim * (i - Lag)
                        for _ in range(i - Lag, i):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_ensembles[update_dim, :]
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((Lag + 1) * (1 + dim_J), 1)
                    else:
                        update_dim = idx_dim
                        for _ in range(1, i + 1):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_ensembles[update_dim, :]
                        # writing in this way to double check the dimension
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((i + 1) * (1 + dim_J), 1)

                    # combine smoother Lag as well as local domain for updating
                    # update local_domain^{i - Lag}, ... local_domain^i

                    # update mean and perturbations

                    prior_mean_all_loc = np.mean(Y_prior_all_loc, axis=1)
                    kal_gain_loc = (
                        U_prior_all_loc
                        @ e_P_loc
                        @ e_Tao_sq_inv_loc
                        @ e_P_loc.T
                        @ V_i_loc.T
                        @ R_0_inv_loc
                    )
                    # posterior mean (smoother) at local domain
                    gamma_mean_all_loc = prior_mean_all_loc + kal_gain_loc @ (
                        obs[i][loc_dim] - H_prior_mean_loc
                    )
                    # transform matrix at local domain
                    T_loc = (
                        np.sqrt(num_ensembles - 1)
                        * e_P_loc
                        @ e_Tao_inv_loc
                        @ e_P_loc.T
                    )
                    # posterior perturbation (smoother) at local domain
                    U_post_all_loc = U_prior_all_loc @ T_loc

                    # select ensemble members to update
                    # update Y_ensemles for previous Lag/i at local domain
                    for idx_i in range(len(update_dim)):
                        idx_dim_i = update_dim[idx_i]
                        Y_ensembles[idx_dim_i] = (
                            gamma_mean_all_loc[idx_i] + U_post_all_loc[idx_i]
                        )
                    # update initial values for next step

                    Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :][
                        idx_dim
                    ]
                    Y_init[idx_dim] = Y_en
                    gamma_mean_trace[i][idx_dim] = np.mean(Y_en, axis=1)
                    gamma_cov_trace[i, idx_dim[:, None], idx_dim] = np.cov(
                        Y_en
                    )
            else:
                J_i = (num_ensembles - 1) / (1 + inflation) * np.eye(
                    num_ensembles
                ) + V_i.T @ R_0_inv @ V_i

                e_Tao_sq_inv, e_Tao_inv, e_P = SVD_J(J_i, num_ensembles)

                # select prior mean to update
                if i + 1 > Lag:
                    Y_prior_all = Y_ensembles[
                        (i - Lag) * model_dim : (i + 1) * model_dim, :
                    ]
                    U_prior_all = Y_prior_all - np.mean(
                        Y_prior_all, axis=1
                    ).reshape((Lag + 1) * model_dim, 1)
                else:
                    Y_prior_all = Y_ensembles[: (i + 1) * model_dim, :]
                    U_prior_all = Y_prior_all - np.mean(
                        Y_prior_all, axis=1
                    ).reshape((i + 1) * model_dim, 1)

                # update mean and perturbations
                prior_mean_all = np.mean(Y_prior_all, axis=1)
                kal_gain = (
                    U_prior_all @ e_P @ e_Tao_sq_inv @ e_P.T @ V_i.T @ R_0_inv
                )

                gamma_mean_all = prior_mean_all + kal_gain @ (
                    obs[i] - trans_mat @ prior_mean
                )
                # Check this!! if will add to all previous states
                T = np.sqrt(num_ensembles - 1) * e_P @ e_Tao_inv @ e_P.T
                U_post_all = U_prior_all @ T

                # select ensemble members to update
                if i + 1 > Lag:
                    Y_ensembles[
                        (i - Lag) * model_dim : (i + 1) * model_dim, :
                    ] = (gamma_mean_all[:, None] + U_post_all)
                else:
                    Y_ensembles[: (i + 1) * model_dim, :] = (
                        gamma_mean_all[:, None] + U_post_all
                    )

                # update the last ensemble member, which is the filter mean and
                # also the initial values for the prediction in next step
                # think about this if need to change!!
                Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :]

                gamma_mean_trace[i] = np.mean(Y_en, axis=1)
                gamma_cov_trace[i] = np.cov(Y_en)
                Y_init = Y_en

            if i >= pred_start_pred_step - pred_last_lead_t and i < (
                pred_start_pred_step + pred_total_pred_steps
            ):
                for j in range(L_init):
                    j_start = i - L_init + j + 1
                    Y_en = Y_ensembles[
                        j_start * model_dim : (j_start + 1) * model_dim, :
                    ]
                    gamma_ensembles_for_IC[
                        i - pred_start_pred_step + pred_last_lead_t, j
                    ] = Y_en[:, :pred_num_ensembles]

            if i == obs_n_steps - 1:
                for j in tqdm(range(obs_n_steps)):
                    Y_en = Y_ensembles[j * model_dim : (j + 1) * model_dim, :]
                    gamma_mean_smooth[j] = np.mean(Y_en, axis=1)
                    gamma_cov_smooth[j] = np.cov(Y_en)
                    gamma_ensembles[j, :] = Y_en
        return (
            gamma_mean_trace,
            gamma_cov_trace,
            gamma_mean_smooth,
            gamma_cov_smooth,
            gamma_ensembles,
            gamma_ensembles_for_IC,
        )

    def online_ETKS_for_IC_save(
        self,
        obs,
        Y_init,
        Lag,
        pred_start_pred_step,
        pred_total_pred_steps,
        pred_last_lead_t,
        L_init,
        pred_num_ensembles,
        obs_dim_idx,
        inflation=0,
        localization=None,
        radius_L=1,  # localization radius
        obs_n_steps=None,
    ):

        """
        The first version
        """
        obs_dt = self.obs_dt
        dt = self.dt
        model_dim = self.model_dim
        if obs_n_steps:
            obs_n_steps = obs_n_steps
        else:
            obs_n_steps = self.obs_n_steps
        trans_mat = self.trans_mat
        obs_noise = self.obs_noise
        num_ensembles = Y_init.shape[1]

        K = int(obs_dt / dt)  # number of timestampes between obs
        dim_I = self.params.dim_I
        dim_J = self.params.dim_J
        gamma_mean_trace = np.zeros((obs_n_steps, model_dim))
        gamma_cov_trace = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_smooth = np.zeros((obs_n_steps, model_dim))
        gamma_cov_smooth = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_trace[0] = np.mean(Y_init, axis=1)
        gamma_cov_trace[0] = np.cov(Y_init)
        Y_en = Y_init
        Y_ensembles = np.zeros(((Lag + 1) * model_dim, num_ensembles))
        # decide later if only save a few ensemble members
        gamma_ensembles = np.zeros((obs_n_steps, model_dim, num_ensembles))
        Y_ensembles[:model_dim, :] = Y_en
        gamma_ensembles_for_IC = np.empty(
            (
                (pred_total_pred_steps + pred_last_lead_t),
                L_init,
                model_dim,
                pred_num_ensembles,
            )
        )
        for i in tqdm(range(1, obs_n_steps)):
            for k in range(K):
                Y_init = self.simulate_state(Y_init, num_ensembles)
            Y_prior = Y_init
            # compute the last step, this will used in the observations part
            prior_mean = np.mean(Y_prior, axis=1)
            U_i = Y_prior - prior_mean.reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            H_prior_mean = (
                trans_mat @ prior_mean
            )  # apply observation operator to proir mean
            # select prior mean/perturbation to update later
            if i > Lag:
                Y_prior_all_prev = Y_ensembles[model_dim:, :]
                Y_prior_all = np.concatenate(
                    (Y_prior_all_prev, Y_prior), axis=0
                )
                U_prior_all = Y_prior_all - np.mean(
                    Y_prior_all, axis=1
                ).reshape((Lag + 1) * model_dim, 1)
            else:
                Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
                Y_prior_all = Y_ensembles[: (i + 1) * model_dim, :]
                U_prior_all = Y_prior_all - np.mean(
                    Y_prior_all, axis=1
                ).reshape((i + 1) * model_dim, 1)
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            if localization:
                for l in range(dim_I):
                    # deal with observations
                    # for radius = 1 of the localization
                    # loc_dim is for observations with in radius L
                    loc_dim_in_space = (
                        np.arange(l - radius_L, l + radius_L + 1) % dim_I
                    )
                    inter = np.intersect1d(loc_dim_in_space, obs_dim_idx)
                    loc_dim = np.asarray([])
                    for idx in inter:
                        loc_dim = np.concatenate(
                            (loc_dim, np.where(obs_dim_idx == idx)[0])
                        )
                    loc_dim = loc_dim.astype(int)
                    V_i_loc = V_i[loc_dim]
                    H_prior_mean_loc = H_prior_mean[loc_dim]
                    obs_noise_loc = obs_noise[loc_dim]
                    R_0_inv_loc = np.linalg.inv(np.diag(obs_noise_loc ** 2))
                    J_i_loc = (num_ensembles - 1) / (1 + inflation) * np.eye(
                        num_ensembles
                    ) + V_i_loc.T @ R_0_inv_loc @ V_i_loc
                    e_Tao_sq_inv_loc, e_Tao_inv_loc, e_P_loc = SVD_J(
                        J_i_loc, num_ensembles
                    )

                    # start to select local_domain
                    #  = [u_l, v_{l,1}, ...v_{l,dim_J}]
                    J_loc = dim_I + l * dim_J
                    idx_dim = np.hstack(
                        (np.asarray([l]), np.arange(J_loc, J_loc + dim_J))
                    )
                    # smoother part
                    # select prior mean to update
                    if i > Lag:
                        update_dim = idx_dim
                        for _ in range(Lag):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_prior_all[update_dim, :]
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((Lag + 1) * (1 + dim_J), 1)
                    else:
                        update_dim = idx_dim
                        for _ in range(1, i + 1):
                            update_dim = np.concatenate(
                                (
                                    update_dim,
                                    update_dim[-(1 + dim_J) :] + model_dim,
                                )
                            )
                        Y_prior_all_loc = Y_ensembles[update_dim, :]
                        # writing in this way to double check the dimension
                        U_prior_all_loc = Y_prior_all_loc - np.mean(
                            Y_prior_all_loc, axis=1
                        ).reshape((i + 1) * (1 + dim_J), 1)

                    # combine smoother Lag as well as local domain for updating
                    # update local_domain^{i - Lag}, ... local_domain^i

                    # update mean and perturbations

                    prior_mean_all_loc = np.mean(Y_prior_all_loc, axis=1)
                    kal_gain_loc = (
                        U_prior_all_loc
                        @ e_P_loc
                        @ e_Tao_sq_inv_loc
                        @ e_P_loc.T
                        @ V_i_loc.T
                        @ R_0_inv_loc
                    )
                    # posterior mean (smoother) at local domain
                    gamma_mean_all_loc = prior_mean_all_loc + kal_gain_loc @ (
                        obs[i][loc_dim] - H_prior_mean_loc
                    )
                    # transform matrix at local domain
                    T_loc = (
                        np.sqrt(num_ensembles - 1)
                        * e_P_loc
                        @ e_Tao_inv_loc
                        @ e_P_loc.T
                    )
                    # posterior perturbation (smoother) at local domain
                    U_post_all_loc = U_prior_all_loc @ T_loc

                    # select ensemble members to update
                    # update Y_ensemles for previous Lag/i at local domain
                    # update initial values for next step
                    for idx_i in range(len(update_dim)):
                        idx_dim_i = update_dim[idx_i]
                        Y_ensembles[idx_dim_i] = (
                            gamma_mean_all_loc[idx_i] + U_post_all_loc[idx_i]
                        )
                    if i > Lag:
                        Y_en = Y_ensembles[-model_dim:, :][idx_dim]
                        for j in range(Lag + 1):
                            Y_temp = Y_ensembles[
                                j * model_dim : (j + 1) * model_dim, :
                            ][idx_dim]
                            gamma_mean_smooth[i - Lag + j][idx_dim] = np.mean(
                                Y_temp, axis=1
                            )
                            gamma_cov_smooth[
                                i - Lag + j, idx_dim[:, None], idx_dim
                            ] = np.cov(Y_temp)
                            gamma_ensembles[i - Lag + j, :][idx_dim] = Y_temp
                    else:
                        Y_en = Y_ensembles[
                            i * model_dim : (i + 1) * model_dim, :
                        ][idx_dim]
                        for j in range(i + 1):
                            Y_temp = Y_ensembles[
                                j * model_dim : (j + 1) * model_dim, :
                            ][idx_dim]
                            gamma_mean_smooth[j][idx_dim] = np.mean(
                                Y_temp, axis=1
                            )
                            gamma_cov_smooth[
                                j, idx_dim[:, None], idx_dim
                            ] = np.cov(Y_temp)
                            gamma_ensembles[j, :][idx_dim] = Y_temp
                    Y_init[idx_dim] = Y_en
                    gamma_mean_trace[i][idx_dim] = np.mean(Y_en, axis=1)
                    gamma_cov_trace[i, idx_dim[:, None], idx_dim] = np.cov(
                        Y_en
                    )
            else:
                J_i = (num_ensembles - 1) / (1 + inflation) * np.eye(
                    num_ensembles
                ) + V_i.T @ R_0_inv @ V_i

                e_Tao_sq_inv, e_Tao_inv, e_P = SVD_J(J_i, num_ensembles)

                # update mean and perturbations
                prior_mean_all = np.mean(Y_prior_all, axis=1)
                kal_gain = (
                    U_prior_all @ e_P @ e_Tao_sq_inv @ e_P.T @ V_i.T @ R_0_inv
                )

                gamma_mean_all = prior_mean_all + kal_gain @ (
                    obs[i] - trans_mat @ prior_mean
                )
                # Check this!! if will add to all previous states
                T = np.sqrt(num_ensembles - 1) * e_P @ e_Tao_inv @ e_P.T
                U_post_all = U_prior_all @ T

                # select ensemble members to update
                # update the last ensemble member, which is the filter mean
                # also the initial values for the prediction in next step
                if i > Lag:
                    Y_ensembles = gamma_mean_all[:, None] + U_post_all
                    Y_en = Y_ensembles[-model_dim:, :]
                    for j in range(Lag + 1):
                        Y_temp = Y_ensembles[
                            j * model_dim : (j + 1) * model_dim, :
                        ]
                        gamma_mean_smooth[i - Lag + j] = np.mean(
                            Y_temp, axis=1
                        )
                        gamma_cov_smooth[i - Lag + j] = np.cov(Y_temp)
                        gamma_ensembles[i - Lag + j, :] = Y_temp
                else:
                    Y_ensembles[: (i + 1) * model_dim, :] = (
                        gamma_mean_all[:, None] + U_post_all
                    )
                    Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :]
                    for j in range(i + 1):
                        Y_temp = Y_ensembles[
                            j * model_dim : (j + 1) * model_dim, :
                        ]
                        gamma_mean_smooth[j] = np.mean(Y_temp, axis=1)
                        gamma_cov_smooth[j] = np.cov(Y_temp)
                        gamma_ensembles[j, :] = Y_temp

                gamma_mean_trace[i] = np.mean(Y_en, axis=1)
                gamma_cov_trace[i] = np.cov(Y_en)
                Y_init = Y_en

            if i >= pred_start_pred_step - pred_last_lead_t and i < (
                pred_start_pred_step + pred_total_pred_steps
            ):
                for j in range(L_init):
                    j_start = Lag - L_init + j + 1
                    Y_en = Y_ensembles[
                        j_start * model_dim : (j_start + 1) * model_dim, :
                    ]
                    gamma_ensembles_for_IC[
                        i - pred_start_pred_step + pred_last_lead_t, j
                    ] = Y_en[:, :pred_num_ensembles]

        return (
            gamma_mean_trace,
            gamma_cov_trace,
            gamma_mean_smooth,
            gamma_cov_smooth,
            gamma_ensembles,
            gamma_ensembles_for_IC,
        )

    def basic_ETKF(
        self, obs, Y_init, inflation=0, localization=None, radius_L=1
    ):

        """
        for filter only
        """
        obs_dt = self.obs_dt
        dt = self.dt
        model_dim = self.model_dim
        obs_n_steps = self.obs_n_steps
        trans_mat = self.trans_mat
        obs_noise = self.obs_noise
        num_ensembles = Y_init.shape[1]
        obs_dim = self.obs_dim

        # true init
        dim_I = self.params.dim_I
        dim_J = self.params.dim_J
        K = int(obs_dt / dt)  # number of timestampes between obs
        gamma_mean_trace = np.zeros((obs_n_steps, model_dim))
        gamma_cov_trace = np.zeros((obs_n_steps, model_dim, model_dim))

        gamma_mean_trace[0] = np.mean(Y_init, axis=1)
        gamma_cov_trace[0] = np.cov(Y_init)
        Y_prior = np.zeros((model_dim, num_ensembles))

        for i in tqdm(range(1, obs_n_steps)):
            for k in range(K):
                Y_prior_one = self.simulate_state(Y_init, num_ensembles)
                Y_init = Y_prior_one
            Y_prior = Y_prior_one

            prior_mean = np.mean(Y_prior, axis=1)
            U_i = Y_prior - prior_mean.reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            H_prior_mean = (
                trans_mat @ prior_mean
            )  # apply observation operator to proir mean
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            if localization:
                for l in range(obs_dim):
                    J_loc = dim_I + l * dim_J
                    idx_dim = np.hstack(
                        (np.asarray([l]), np.arange(J_loc, J_loc + dim_J))
                    )
                    prior_mean_loc = prior_mean[idx_dim]
                    U_i_loc = U_i[idx_dim]
                    # for radius = 1 of the localization
                    loc_dim = np.arange(l - radius_L, l + radius_L + 1) % dim_I
                    V_i_loc = V_i[loc_dim]

                    H_prior_mean_loc = H_prior_mean[loc_dim]
                    obs_noise_loc = obs_noise[loc_dim]
                    R_0_inv_loc = np.linalg.inv(np.diag(obs_noise_loc ** 2))

                    J_i_loc = (num_ensembles - 1) / (1 + inflation) * np.eye(
                        num_ensembles
                    ) + V_i_loc.T @ R_0_inv_loc @ V_i_loc
                    e_Tao_sq_inv_loc, e_Tao_inv_loc, e_P_loc = SVD_J(
                        J_i_loc, num_ensembles
                    )

                    kal_gain_loc = (
                        U_i_loc
                        @ e_P_loc
                        @ e_Tao_sq_inv_loc
                        @ e_P_loc.T
                        @ V_i_loc.T
                        @ R_0_inv_loc
                    )

                    gamma_mean_loc = prior_mean_loc + kal_gain_loc @ (
                        obs[i][loc_dim] - H_prior_mean_loc
                    )

                    T_loc = (
                        np.sqrt(num_ensembles - 1)
                        * e_P_loc
                        @ e_Tao_inv_loc
                        @ e_P_loc.T
                    )
                    U_post_loc = U_i_loc @ T_loc

                    Y_init[idx_dim] = (
                        gamma_mean_loc.reshape(dim_J + 1, 1) + U_post_loc
                    )

                    gamma_mean_trace[i][idx_dim] = gamma_mean_loc
                    gamma_cov_trace[i, idx_dim[:, None], idx_dim] = (
                        U_post_loc @ U_post_loc.T / (num_ensembles - 1)
                    )
            else:

                J_i = (num_ensembles - 1) / (1 + inflation) * np.eye(
                    num_ensembles
                ) + V_i.T @ R_0_inv @ V_i

                e_Tao_sq_inv, e_Tao_inv, e_P = SVD_J(J_i, num_ensembles)

                kal_gain = U_i @ e_P @ e_Tao_sq_inv @ e_P.T @ V_i.T @ R_0_inv
                gamma_mean = prior_mean + kal_gain @ (obs[i] - H_prior_mean)
                T = np.sqrt(num_ensembles - 1) * e_P @ e_Tao_inv @ e_P.T
                U_post = U_i @ T

                Y_init = gamma_mean.reshape(model_dim, 1) + U_post

                gamma_mean_trace[i] = gamma_mean
                gamma_cov_trace[i] = U_post @ U_post.T / (num_ensembles - 1)

        return (
            gamma_mean_trace,
            gamma_cov_trace,
        )


class EnsembleKalmanFilterSmootherL96SPEKF(EnsembleKalmanFilterSmootherL96):
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        if params:
            params = params
        else:
            params = self.params
        model_dim = Y_prev.shape[0]
        model_noise = np.zeros((model_dim, model_dim))
        for i in range(params.dim_I):
            model_noise[i, i] = params.sigma_u
        for i in range(params.dim_I, model_dim):
            model_noise[i, i] = params.sigma_v_hat

        Y = (
            np.zeros((model_dim))
            if num_ensembles == 1
            else np.zeros((model_dim, num_ensembles))
        )
        u_prev = Y_prev[: params.dim_I]
        v_prev = Y_prev[params.dim_I :]
        v_prev_sum = (
            np.sum(
                v_prev.reshape(params.dim_I, params.dim_J, num_ensembles),
                axis=1,
            ).flatten()
            if num_ensembles == 1
            else np.sum(
                v_prev.reshape(params.dim_I, params.dim_J, num_ensembles),
                axis=1,
            )
        )  # shape is dim_I * num_ensembles

        u = (
            u_prev
            + (
                -np.roll(u_prev, 1, axis=0)
                * (np.roll(u_prev, 2, axis=0) - np.roll(u_prev, -1, axis=0))
                - u_prev
                + params.f
                - params.h * params.c / params.dim_J * v_prev_sum
            )
            * self.dt
        )
        v = v_prev + (-params.d_v * (v_prev - params.v_hat)) * self.dt

        Y[: params.dim_I] = u
        Y[params.dim_I :] = v
        if num_ensembles == 1:
            Y += np.sqrt(self.dt) * model_noise @ np.random.randn(model_dim)
        else:
            Y += (
                np.sqrt(self.dt)
                * model_noise
                @ np.random.randn(model_dim, num_ensembles)
            )
        return Y


class EnsembleKalmanFilterSmootherL96OneLayer(EnsembleKalmanFilterSmootherL96):
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        if params:
            params = params
        else:
            params = self.params
        model_dim = Y_prev.shape[0]
        model_noise = np.zeros((model_dim, model_dim))
        for i in range(params.dim_I):
            model_noise[i, i] = params.sigma_u
        for i in range(params.dim_I, model_dim):
            model_noise[i, i] = params.sigma_v_hat

        Y = (
            np.zeros((model_dim))
            if num_ensembles == 1
            else np.zeros((model_dim, num_ensembles))
        )
        u_prev = Y_prev[: params.dim_I]
        u = (
            u_prev
            + (
                -np.roll(u_prev, 1, axis=0)
                * (np.roll(u_prev, 2, axis=0) - np.roll(u_prev, -1, axis=0))
                - u_prev
                + params.f
            )
            * self.dt
        )

        Y[: params.dim_I] = u
        if num_ensembles == 1:
            Y += np.sqrt(self.dt) * model_noise @ np.random.randn(model_dim)
        else:
            Y += (
                np.sqrt(self.dt)
                * model_noise
                @ np.random.randn(model_dim, num_ensembles)
            )
        return Y
