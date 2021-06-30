import numpy as np
from abc import abstractmethod
from tqdm import tqdm

"""
This is the class for Ensemble Kalman filter and smoother and
related sampling algorithms
"""


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


class EnsembleKalmanFilterSmootherBase:
    """
    Parameters
    ----------
    model:
        np.array((model_dim))
        the perfect or imperfect model used in filter/smoother
        algorithm (state transition function)

    obs:
        np.array((obs_dim))
        the observation trajectories

    params:
        Params
        parameters used in the model

    n_steps:
        int
        length of the trajectories
    dt:
        float
        integrate time step in the model

    obs_dt:
        float
        observation time step

    trans_mat:
        np.array((obs_dim, model_dim))
        observation matrix (it could be square or non-square matrix),
        dimension should consistant with obs

    obs_noise:
        np.array((obs_dim))
        noise in observations; the dimension should be consistant with,
        dimensions

    num_ensembles:
        int
        number of ensembles used in algorithm

    """

    def __init__(
        self,
        n_steps: int,
        model_dim: int,
        params,
        obs_dt: float,
        trans_mat: np.array,
        obs_noise: np.array,
        dt: float = 0.005,
    ):
        self.n_steps = n_steps
        self.model_dim = model_dim
        self.params = params
        self.trans_mat = trans_mat  # symbol A in the paper
        self.dt = dt
        self.obs_dt = obs_dt
        self.obs_dim = trans_mat.shape[0]
        self.obs_n_steps = int(dt * (n_steps - 1) / obs_dt) + 1
        self.obs_noise = obs_noise

    def simulate(self, init=None):
        obs_n_steps = self.obs_n_steps
        obs_dim = self.obs_dim
        n_steps = self.n_steps
        model_dim = self.model_dim
        self.obs = np.zeros((obs_n_steps, obs_dim))
        self.state_truth = np.zeros((n_steps, model_dim))
        if init is not None:
            self.state_truth[0] = init
        else:
            self.state_truth[0] = np.zeros(model_dim)
        self.obs[0] = self.trans_mat @ self.state_truth[
            0
        ] + self.obs_noise * np.random.randn(obs_dim)
        delta_obs = int(self.obs_dt / self.dt)
        for i in tqdm(range(1, n_steps)):
            self.state_truth[i] = self.simulate_state(
                self.state_truth[i - 1], num_ensembles=1
            )
            if i % delta_obs == 0:
                self.obs[i // delta_obs] = self.trans_mat @ self.state_truth[
                    i
                ] + self.obs_noise * np.random.randn(obs_dim)
        return self.obs, self.state_truth

    @abstractmethod
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        # Y_prev dim: (model_dim, )
        pass

    def filtering_observation_state_only(
        self, obs, Y_init, inflation=0, obs_n_steps=None
    ):

        """
        for obs states only
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

        # true init
        K = int(obs_dt / dt)  # number of timestampes between obs
        obs_dim = self.obs_dim
        aux_C = np.zeros((obs_n_steps - 1, model_dim, model_dim))
        gamma_mean_prior = np.zeros((obs_n_steps, model_dim))
        gamma_cov_prior = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_trace = np.zeros((obs_n_steps, model_dim))
        gamma_cov_trace = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_trace[0] = np.mean(Y_init, axis=1)
        gamma_cov_trace[0] = np.cov(Y_init)
        Y_prior = np.zeros((model_dim, num_ensembles))
        Y_en = Y_init
        for i in tqdm(range(1, obs_n_steps)):
            for k in range(K):
                Y_prior_one = self.simulate_state(Y_init, num_ensembles)
                Y_init = Y_prior_one
            Y_prior = Y_prior_one

            # Y_en is the ensembles from previous obs point
            Y_prev = Y_en
            Y_join = np.vstack((Y_prev, Y_prior))
            cov_join = np.cov(Y_join)
            gamma_mean_prior[i] = np.mean(Y_prior, axis=1)
            prior_cov = cov_join[model_dim:, model_dim:]
            prior_cov = prior_cov * (1 + inflation)
            gamma_cov_prior[i] = prior_cov
            R_12 = cov_join[:model_dim, model_dim:]

            aux_C[i - 1] = R_12 @ np.linalg.inv(prior_cov)

            kal_gain = (
                prior_cov
                @ trans_mat.T
                @ np.linalg.inv(
                    trans_mat @ prior_cov @ trans_mat.T
                    + np.diag(obs_noise ** 2)
                )
            )
            Y_en = Y_prior + kal_gain @ (
                obs[i].reshape(obs_dim, 1)
                - trans_mat @ Y_prior_one
                - obs_noise.reshape(obs_dim, 1)
                * np.random.randn(obs_dim, num_ensembles)
            )

            gamma_mean_trace[i] = np.mean(Y_en, axis=1)
            gamma_cov_trace[i] = np.cov(Y_en)
            Y_init = Y_en
        return (
            gamma_mean_trace,
            gamma_cov_trace,
            aux_C,
            gamma_mean_prior,
            gamma_cov_prior,
        )

    def smoothing_observation_state_only(
        self,
        gamma_mean_trace,
        gamma_cov_trace,
        aux_C,
        gamma_mean_prior,
        gamma_cov_prior,
    ):
        """
        for obs state only
        """

        N_obs, dim_N = gamma_mean_trace.shape
        gamma_mean_smooth = np.zeros((N_obs, dim_N))
        gamma_cov_smooth = np.zeros((N_obs, dim_N, dim_N))
        gamma_mean_smooth[-1] = gamma_mean_trace[-1]
        gamma_cov_smooth[-1] = gamma_cov_trace[-1]
        Y_sampled = np.zeros((N_obs, dim_N))

        Y_sampled[-1] = np.random.multivariate_normal(
            gamma_mean_trace[-1], gamma_cov_trace[-1]
        ).T

        for i in tqdm(range(N_obs - 1, 0, -1)):
            gamma_mean_trace_cur = gamma_mean_trace[i - 1]
            gamma_cov_trace_cur = gamma_cov_trace[i - 1]
            gamma_mean_smooth_next = gamma_mean_smooth[i]
            gamma_cov_smooth_next = gamma_cov_smooth[i]

            gamma_mean_prior_next = gamma_mean_prior[i]
            gamma_cov_prior_next = gamma_cov_prior[i]

            gamma_mean_smooth[i - 1] = gamma_mean_trace_cur + aux_C[i - 1] @ (
                gamma_mean_smooth_next - gamma_mean_prior_next
            )

            gamma_cov_smooth[i - 1] = (
                gamma_cov_trace_cur
                + aux_C[i - 1]
                @ (gamma_cov_smooth_next - gamma_cov_prior_next)
                @ aux_C[i - 1].T
            )
            sample_mean = gamma_mean_trace_cur + aux_C[i - 1] @ (
                Y_sampled[i] - gamma_mean_prior_next
            )
            sample_cov = (
                gamma_cov_trace_cur
                - aux_C[i - 1] @ gamma_cov_prior_next @ aux_C[i - 1].T
            )
            Y_sampled[i - 1] = np.random.multivariate_normal(
                sample_mean, sample_cov
            ).T

        return gamma_mean_smooth, gamma_cov_smooth, Y_sampled

    def sampling_observation_state_only(
        self,
        gamma_mean_trace,
        gamma_cov_trace,
        aux_C,
        gamma_mean_prior,
        gamma_cov_prior,
    ):
        """
        for obs state only
        """

        N_obs, dim_N = gamma_mean_trace.shape
        Y_sampled = np.zeros((N_obs, dim_N))

        Y_sampled[-1] = np.random.multivariate_normal(
            gamma_mean_trace[-1], gamma_cov_trace[-1]
        ).T

        for i in range(N_obs - 1, 0, -1):
            gamma_mean_trace_cur = gamma_mean_trace[i - 1]
            gamma_cov_trace_cur = gamma_cov_trace[i - 1]

            gamma_mean_prior_next = gamma_mean_prior[i]
            gamma_cov_prior_next = gamma_cov_prior[i]

            sample_mean = gamma_mean_trace_cur + aux_C[i - 1] @ (
                Y_sampled[i] - gamma_mean_prior_next
            )
            sample_cov = (
                gamma_cov_trace_cur
                - aux_C[i - 1] @ gamma_cov_prior_next @ aux_C[i - 1].T
            )
            Y_sampled[i - 1] = np.random.multivariate_normal(
                sample_mean, sample_cov
            ).T

        return Y_sampled

    def EAKF(self, obs, Y_init, inflation=0):

        """
        for obs states only
        """
        obs_dt = self.obs_dt
        dt = self.dt
        model_dim = self.model_dim
        obs_n_steps = self.obs_n_steps
        trans_mat = self.trans_mat
        obs_noise = self.obs_noise
        num_ensembles = Y_init.shape[1]

        # true init
        K = int(obs_dt / dt)  # number of timestampes between obs
        aux_C = np.zeros((obs_n_steps - 1, model_dim, model_dim))
        gamma_mean_prior = np.zeros((obs_n_steps, model_dim))
        gamma_cov_prior = np.zeros((obs_n_steps, model_dim, model_dim))
        gamma_mean_trace = np.zeros((obs_n_steps, model_dim))
        gamma_cov_trace = np.zeros((obs_n_steps, model_dim, model_dim))

        gamma_mean_trace[0] = np.mean(Y_init, axis=1)
        gamma_cov_trace[0] = np.cov(Y_init)
        Y_prior = np.zeros((model_dim, num_ensembles))
        Y_en = Y_init
        for i in tqdm(range(1, obs_n_steps)):
            for k in range(K):
                Y_prior_one = self.simulate_state(Y_init, num_ensembles)
                Y_init = Y_prior_one
            Y_prior = Y_prior_one

            # Y_en is the ensembles from previous obs point
            Y_prev = Y_en
            Y_join = np.vstack((Y_prev, Y_prior))
            cov_join = np.cov(Y_join)

            prior_mean = np.mean(Y_prior, axis=1)
            gamma_mean_prior[i] = prior_mean
            prior_cov = cov_join[model_dim:, model_dim:]
            prior_cov *= 1 + inflation
            U_prior = Y_prior - prior_mean.reshape(model_dim, 1)
            V_prior = Y_prev - np.mean(Y_prev, axis=1).reshape(model_dim, 1)

            e_Sigma_sq, e_P = np.linalg.eigh(prior_cov)
            e_Sigma_sq[e_Sigma_sq < 1e-5] = 0
            e_Sigma = np.sqrt(e_Sigma_sq)
            e_Sigma_inv = np.zeros(e_Sigma.shape)
            for dim in range(model_dim):
                if e_Sigma[dim] != 0:
                    e_Sigma_inv[dim] = 1 / e_Sigma[dim]
            e_Sigma_sq = np.diag(e_Sigma_sq)
            e_Sigma_sq_inv = e_Sigma_inv ** 2
            e_Sigma = np.diag(e_Sigma)
            e_Sigma_inv = np.diag(e_Sigma_inv)
            e_Sigma_sq_inv = np.diag(e_Sigma_sq_inv)

            gamma_cov_prior[i] = e_P @ e_Sigma_sq @ e_P.T
            aux_C[i - 1] = (
                V_prior
                @ U_prior.T
                / (num_ensembles - 1)
                @ e_P
                @ e_Sigma_sq_inv
                @ e_P.T
            )
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            M_temp = (
                e_Sigma
                @ e_P.T
                @ trans_mat.T
                @ R_0_inv
                @ trans_mat
                @ e_P
                @ e_Sigma
            )
            e_D, e_X = np.linalg.eigh(M_temp)
            I_D = (e_D + np.ones(model_dim)) ** (1 / 2)
            I_D_inv = 1 / I_D
            ad_A = (
                e_P
                @ e_Sigma
                @ e_X
                @ np.diag(I_D_inv)
                @ np.sqrt(e_Sigma_sq_inv)
                @ e_P.T
            )
            U_post = ad_A @ U_prior
            ad_L = (
                e_P @ e_Sigma_sq_inv @ e_P.T
                + trans_mat.T @ R_0_inv @ trans_mat
            )
            ad_y = (
                e_P @ e_Sigma_sq_inv @ e_P.T @ prior_mean
                + trans_mat.T @ R_0_inv @ obs[i]
            )
            ad_x = U_post @ U_post.T @ ad_y / (num_ensembles)
            ad_x = np.linalg.inv(ad_L) @ ad_y
            Y_en = ad_x.reshape(model_dim, 1) + U_post

            gamma_mean_trace[i] = ad_x
            gamma_cov_trace[i] = np.linalg.inv(ad_L)
            Y_init = Y_en

        return (
            gamma_mean_trace,
            gamma_cov_trace,
            aux_C,
            gamma_mean_prior,
            gamma_cov_prior,
        )

    def EAKS(
        self,
        gamma_mean_trace,
        gamma_cov_trace,
        aux_C,
        gamma_mean_prior,
        gamma_cov_prior,
    ):
        """
        for obs state only
        """

        N_obs, dim_N = gamma_mean_trace.shape
        gamma_mean_smooth = np.zeros((N_obs, dim_N))
        gamma_cov_smooth = np.zeros((N_obs, dim_N, dim_N))
        gamma_mean_smooth[-1] = gamma_mean_trace[-1]
        gamma_cov_smooth[-1] = gamma_cov_trace[-1]
        Y_sampled = np.zeros((N_obs, dim_N))

        Y_sampled[-1] = np.random.multivariate_normal(
            gamma_mean_trace[-1], gamma_cov_trace[-1]
        ).T

        for i in tqdm(range(N_obs - 1, 0, -1)):
            gamma_mean_trace_cur = gamma_mean_trace[i - 1]
            gamma_cov_trace_cur = gamma_cov_trace[i - 1]
            gamma_mean_smooth_next = gamma_mean_smooth[i]
            gamma_cov_smooth_next = gamma_cov_smooth[i]

            gamma_mean_prior_next = gamma_mean_prior[i]
            gamma_cov_prior_next = gamma_cov_prior[i]

            gamma_mean_smooth[i - 1] = gamma_mean_trace_cur + aux_C[i - 1] @ (
                gamma_mean_smooth_next - gamma_mean_prior_next
            )

            gamma_cov_smooth[i - 1] = (
                gamma_cov_trace_cur
                + aux_C[i - 1] @ gamma_cov_smooth_next @ aux_C[i - 1].T
                - aux_C[i - 1] @ gamma_cov_prior_next @ aux_C[i - 1].T
            )

            sample_mean = gamma_mean_trace_cur + aux_C[i - 1] @ (
                Y_sampled[i] - gamma_mean_prior_next
            )
            sample_cov = (
                gamma_cov_trace_cur
                - aux_C[i - 1] @ gamma_cov_prior_next @ aux_C[i - 1].T
            )
            Y_sampled[i - 1] = np.random.multivariate_normal(
                sample_mean, sample_cov
            ).T

        return gamma_mean_smooth, gamma_cov_smooth, Y_sampled

    def online_EnKS(self, obs, Y_init, Lag, inflation=0, obs_n_steps=None):

        """
        for obs states only
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
            U_i = Y_prior - np.mean(Y_prior, axis=1).reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
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

            kal_gain = (
                U_prior_all
                @ V_i.T
                / (num_ensembles - 1)
                @ np.linalg.inv(
                    V_i @ V_i.T / (num_ensembles - 1) + np.diag(obs_noise ** 2)
                )
            )
            if i + 1 > Lag:
                Y_ensembles[
                    (i - Lag) * model_dim : (i + 1) * model_dim, :
                ] = Y_ensembles[
                    (i - Lag) * model_dim : (i + 1) * model_dim, :
                ] + kal_gain @ (
                    obs[i].reshape(obs_dim, 1)
                    - trans_mat @ Y_prior
                    - obs_noise.reshape(obs_dim, 1)
                    * np.random.randn(obs_dim, num_ensembles)
                )
            else:
                Y_ensembles[: (i + 1) * model_dim, :] = Y_ensembles[
                    : (i + 1) * model_dim, :
                ] + kal_gain @ (
                    obs[i].reshape(obs_dim, 1)
                    - trans_mat @ Y_prior
                    - obs_noise.reshape(obs_dim, 1)
                    * np.random.randn(obs_dim, num_ensembles)
                )
            Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :]
            gamma_mean_trace[i] = np.mean(Y_en, axis=1)
            gamma_cov_trace[i] = np.cov(Y_en)
            Y_init = Y_en
            if i == obs_n_steps - 1:
                for j in range(obs_n_steps):
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

    def basic_ETKF(self, obs, Y_init, inflation=0):

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

        # true init
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
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            J_i = (num_ensembles - 1) / (1 + inflation) * np.eye(
                num_ensembles
            ) + V_i.T @ R_0_inv @ V_i
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

            kal_gain = U_i @ e_P @ e_Tao_sq_inv @ e_P.T @ V_i.T @ R_0_inv
            gamma_mean = prior_mean + kal_gain @ (
                obs[i] - trans_mat @ prior_mean
            )
            T = np.sqrt(num_ensembles - 1) * e_P @ e_Tao_inv @ e_P.T
            U_post = U_i @ T

            Y_init = gamma_mean.reshape(model_dim, 1) + U_post

            gamma_mean_trace[i] = gamma_mean
            gamma_cov_trace[i] = U_post @ U_post.T / (num_ensembles - 1)

        return (
            gamma_mean_trace,
            gamma_cov_trace,
        )

    def online_ETKS(self, obs, Y_init, Lag, inflation=0, obs_n_steps=None):

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
            #         for i in [1]:
            for k in range(K):
                Y_init = self.simulate_state(Y_init, num_ensembles)
            Y_prior = Y_init

            prior_mean = np.mean(Y_prior, axis=1)
            U_i = Y_prior - np.mean(Y_prior, axis=1).reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))
            J_i = (num_ensembles - 1) / (1 + inflation) * np.eye(
                num_ensembles
            ) + V_i.T @ R_0_inv @ V_i

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
            prior_mean_all = np.mean(Y_prior_all, axis=1)
            kal_gain = (
                U_prior_all @ e_P @ e_Tao_sq_inv @ e_P.T @ V_i.T @ R_0_inv
            )

            gamma_mean_all = prior_mean_all + kal_gain @ (
                obs[i] - trans_mat @ prior_mean
            )
            T = np.sqrt(num_ensembles - 1) * e_P @ e_Tao_inv @ e_P.T
            U_post_all = U_prior_all @ T
            if i + 1 > Lag:
                Y_ensembles[(i - Lag) * model_dim : (i + 1) * model_dim, :] = (
                    gamma_mean_all[:, None] + U_post_all
                )
            else:
                Y_ensembles[: (i + 1) * model_dim, :] = (
                    gamma_mean_all[:, None] + U_post_all
                )

            Y_en = Y_ensembles[i * model_dim : (i + 1) * model_dim, :]

            gamma_mean_trace[i] = np.mean(Y_en, axis=1)
            gamma_cov_trace[i] = np.cov(Y_en)
            Y_init = Y_en
            if i == obs_n_steps - 1:
                for j in range(obs_n_steps):
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

    def online_EnKS_for_IC(
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
            prior_mean = np.mean(Y_prior, axis=1)
            U_i = Y_prior - prior_mean.reshape(model_dim, 1)
            V_i = trans_mat @ U_i
            Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
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
            kal_gain = (
                U_prior_all
                @ V_i.T
                / (num_ensembles - 1)
                @ np.linalg.inv(
                    V_i @ V_i.T / (num_ensembles - 1) + np.diag(obs_noise ** 2)
                )
            )
            # select ensemble members to update
            if i + 1 > Lag:
                Y_ensembles[
                    (i - Lag) * model_dim : (i + 1) * model_dim, :
                ] = Y_ensembles[
                    (i - Lag) * model_dim : (i + 1) * model_dim, :
                ] + kal_gain @ (
                    obs[i].reshape(obs_dim, 1)
                    - trans_mat @ Y_prior
                    - obs_noise.reshape(obs_dim, 1)
                    * np.random.randn(obs_dim, num_ensembles)
                )
            else:
                Y_ensembles[: (i + 1) * model_dim, :] = Y_ensembles[
                    : (i + 1) * model_dim, :
                ] + kal_gain @ (
                    obs[i].reshape(obs_dim, 1)
                    - trans_mat @ Y_prior
                    - obs_noise.reshape(obs_dim, 1)
                    * np.random.randn(obs_dim, num_ensembles)
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
            Y_ensembles[i * model_dim : (i + 1) * model_dim, :] = Y_prior
            R_0_inv = np.linalg.inv(np.diag(obs_noise ** 2))

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
                Y_ensembles[(i - Lag) * model_dim : (i + 1) * model_dim, :] = (
                    gamma_mean_all[:, None] + U_post_all
                )
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


class EnsembleKalmanFilterSmootherDyad(EnsembleKalmanFilterSmootherBase):
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        if params:
            params = params
        else:
            params = self.params
        model_noise = np.asarray([[params.sigma_u, 0], [0, params.sigma_v]])
        model_dim = Y_prev.shape[0]
        Y = (
            np.zeros((model_dim))
            if num_ensembles == 1
            else np.zeros((model_dim, num_ensembles))
        )
        u_prev, v_prev = Y_prev[0], Y_prev[1]
        u = (
            u_prev
            + (
                -params.d_u * u_prev
                + params.c_1 * v_prev * u_prev
                + params.F_u
            )
            * self.dt
        )
        v = (
            v_prev
            + (-params.d_v * v_prev + params.c_2 * u_prev ** 2 + params.F_v)
            * self.dt
        )
        Y[0] = u
        Y[1] = v
        if num_ensembles == 1:
            Y += np.sqrt(self.dt) * model_noise @ np.random.randn(model_dim)
        else:
            Y += (
                np.sqrt(self.dt)
                * model_noise
                @ np.random.randn(model_dim, num_ensembles)
            )
        return Y


class EnsembleKalmanFilterSmootherDyadApprox(EnsembleKalmanFilterSmootherBase):
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        if params:
            params = params
        else:
            params = self.params
        model_noise = np.asarray([[params.sigma_u, 0], [0, params.sigma_v_M]])
        model_dim = Y_prev.shape[0]
        Y = (
            np.zeros((model_dim))
            if num_ensembles == 1
            else np.zeros((model_dim, num_ensembles))
        )
        u_prev, v_prev = Y_prev[0], Y_prev[1]
        u = (
            u_prev
            + (
                -params.d_u * u_prev
                + params.c_1 * v_prev * u_prev
                + params.F_u
            )
            * self.dt
        )
        v = v_prev - params.d_v_M * (v_prev - params.v_M) * self.dt
        Y[0] = u
        Y[1] = v
        if num_ensembles == 1:
            Y += np.sqrt(self.dt) * model_noise @ np.random.randn(model_dim)
        else:
            Y += (
                np.sqrt(self.dt)
                * model_noise
                @ np.random.randn(model_dim, num_ensembles)
            )
        return Y


class EnsembleKalmanFilterSmootherTriad(EnsembleKalmanFilterSmootherBase):    
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        if params:
            params = params
        else:
            params = self.params
        u1_prev, u2_prev, u3_prev = Y_prev[0], Y_prev[1], Y_prev[2]
        sigma_1, sigma_2, sigma_3 = params.sigma_s
        L12, L13, L23 = params.L_s
        gamma_1, gamma_2, gamma_3 = params.gamma_s
        model_noise = np.asarray(
            [
                [sigma_1, 0, 0],
                [0, sigma_2 / np.sqrt(params.epsilon), 0],
                [0, 0, sigma_3 / np.sqrt(params.epsilon)],
            ]
        )
        model_dim = Y_prev.shape[0]
        Y = (
            np.zeros((model_dim))
            if num_ensembles == 1
            else np.zeros((model_dim, num_ensembles))
        )
        u1 = (
            u1_prev
            + (
                -gamma_1 * u1_prev
                + L12 * u2_prev
                + L13 * u3_prev
                + params.I * u1_prev * u2_prev
                + params.F 
            )
            * self.dt
        )
        u2 = (
            u2_prev
            + (
                -L12 * u1_prev
                - gamma_2 / params.epsilon * u2_prev
                + L23 * u3_prev
                - params.I * u1_prev ** 2
            )
            * self.dt
        )
        u3 = (
            u3_prev
            + (
                -L13 * u1_prev
                - L23 * u2_prev
                - gamma_3 / params.epsilon * u3_prev
            )
            * self.dt
        )
        Y[0] = u1
        Y[1] = u2
        Y[2] = u3
        if num_ensembles == 1:
            Y += np.sqrt(self.dt) * model_noise @ np.random.randn(model_dim)
        else:
            Y += (
                np.sqrt(self.dt)
                * model_noise
                @ np.random.randn(model_dim, num_ensembles)
            )
        return Y
    
class EnsembleKalmanFilterSmootherTriadApprox(EnsembleKalmanFilterSmootherBase):    
    def simulate_state(self, Y_prev, num_ensembles, params=None):
        if params:
            params = params
        else:
            params = self.params
        u1_prev, u2_prev, u3_prev = Y_prev[0], Y_prev[1], Y_prev[2]
        sigma_1, sigma_2, sigma_3 = params.sigma_s
        L12, L13, L23 = params.L_s
        gamma_1, gamma_2, gamma_3 = params.gamma_s
        
        d_M2, d_M3 = params.d_M
        mean_M2, mean_M3 = params.mean_M
        sigma_M2, sigma_M3 = params.sigma_M
        model_noise = np.asarray(
            [
                [sigma_1, 0, 0],
                [0, sigma_M2, 0],
                [0, 0, sigma_M3],
            ]
        )
        model_dim = Y_prev.shape[0]
        Y = (
            np.zeros((model_dim))
            if num_ensembles == 1
            else np.zeros((model_dim, num_ensembles))
        )
        u1 = (
            u1_prev
            + (
                -gamma_1 * u1_prev
                + L12 * u2_prev
                + L13 * u3_prev
                + params.I * u1_prev * u2_prev
                + params.F 
            )
            * self.dt
        )
        u2 = u2_prev - d_M2 * (u2_prev - mean_M2) * self.dt
        u3 = u3_prev - d_M3 * (u3_prev - mean_M3) * self.dt
        Y[0] = u1
        Y[1] = u2
        Y[2] = u3
        if num_ensembles == 1:
            Y += np.sqrt(self.dt) * model_noise @ np.random.randn(model_dim)
        else:
            Y += (
                np.sqrt(self.dt)
                * model_noise
                @ np.random.randn(model_dim, num_ensembles)
            )
        return Y