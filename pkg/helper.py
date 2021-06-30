import numpy as np
from tqdm import tqdm


def autocorr(x):
    from tqdm import tqdm

    half_num = x.size // 2 + 1
    autocorr_x = np.zeros((half_num))
    var_x = np.var(x[:half_num])
    samples = x[:half_num]
    for i in tqdm(range(half_num)):
        samples_temp = x[i : half_num + i]
        autocorr_x[i] = np.cov(samples, samples_temp)[0, 1] / var_x
    return autocorr_x


def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")
    result = r / (variance * n)
    return result[-n:]


def compute_kde_axis(x, num=100, dist=0):
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(x.flatten())
    axis = np.linspace(x.min() - dist, x.max() + dist, num)
    return kde, axis


def rmse(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2), axis=0))


def patt_corr(predictions, targets):
    p_m = np.mean(predictions, axis=0)
    t_m = np.mean(targets, axis=0)
    return np.sum((predictions - p_m) * (targets - t_m), axis=0) / np.sqrt(
        np.sum((predictions - p_m) ** 2, axis=0)
        * np.sum((targets - t_m) ** 2, axis=0)
    )


def normalize(x, mu=None, std=None):
    if mu:
        mu = mu
        std = std
    else:
        mu = np.mean(x)
        std = np.std(x)
    return (x - mu) / std, mu, std


def denormalize(x, mu, std):
    return x * std + mu


def model_prediction_one_traj(
    model, num_ensembles, initial_values, model_dim, lead_time_step
):
    """
    initial_value: this should be the portion of trajectories that we want to make the prediction
                   shape: model_dim * total_pred_steps
    """
    total_pred_steps = initial_values.shape[0]
    predicts = np.zeros((num_ensembles, total_pred_steps, model_dim))
    predicts = np.repeat(initial_values[np.newaxis, :], num_ensembles, axis=0)
    if lead_time_step == 0:
        return predicts
    else:
        for i in range(num_ensembles):
            for j in range(lead_time_step):
                predicts[i] = model.simulate_state(
                    predicts[i].reshape(model_dim, total_pred_steps),
                    total_pred_steps,
                ).reshape(
                    total_pred_steps, model_dim
                )  # output shape: total_pred_steps * model dim
        return predicts


def model_prediction_one_traj_new(
    model, num_ensembles, initial_values, last_lead_t, lead_step, K=1
):
    """
    initial_value: this should be the portion of trajectories that we want to make the prediction
                   shape: model_dim * total_pred_steps
    """
    model_dim = model.model_dim
    total_pred_steps = initial_values.shape[0] - last_lead_t
    total_pred_step_temps = initial_values.shape[0]
    predict_temps = np.zeros((num_ensembles, total_pred_step_temps, model_dim))
    total_lead_steps = int(last_lead_t / lead_step) + 1
    predicts_en = np.zeros(
        (num_ensembles, total_lead_steps, total_pred_steps, model_dim)
    )
    predicts = np.zeros((total_lead_steps, total_pred_steps, model_dim))
    predicts[0] = initial_values[last_lead_t:]

    for i in range(num_ensembles):
        predict_temps[i, :, :] = initial_values
    for i in tqdm(range(num_ensembles)):
        for j in range(1, (last_lead_t + 1) * K):
            predict_temps[i] = model.simulate_state(
                predict_temps[i].T, total_pred_step_temps
            ).T  # output shape: model_dim * total_pred_step_temps
            if j % (lead_step * K) == 0:
                predicts_en[i, j // (lead_step * K)] = predict_temps[
                    i,
                    last_lead_t
                    - j // K : last_lead_t
                    - j // K
                    + total_pred_steps,
                    :,
                ]
    predicts[1:] = np.mean(predicts_en[:, 1:], axis=0)

    return predicts


def model_prediction_one_traj_new_new(
    model, num_ensembles, initial_values, last_lead_t, lead_step, K=1
):
    """
    initial_value: this should be the portion of trajectories that we want to make the prediction
                   shape: (num_ensembles, total_pred_step_temps, model_dim)
    """
    model_dim = model.model_dim
    total_pred_steps = initial_values.shape[1] - last_lead_t
    total_pred_step_temps = initial_values.shape[1]
    total_lead_steps = int(last_lead_t / lead_step) + 1
    predicts_en = np.zeros(
        (num_ensembles, total_lead_steps, total_pred_steps, model_dim)
    )
    predicts = np.zeros((total_lead_steps, total_pred_steps, model_dim))
    predict_temps = initial_values.copy()
    predicts_en[:, 0, :, :] = initial_values[:, last_lead_t:, :]
    for i in tqdm(range(num_ensembles)):
        for j in range(1, (last_lead_t + 1) * K):
            predict_temps[i] = model.simulate_state(
                predict_temps[i].T, total_pred_step_temps
            ).T  # output shape: model_dim * total_pred_step_temps
            if j % (lead_step * K) == 0:
                predicts_en[i, j // (lead_step * K)] = predict_temps[
                    i,
                    last_lead_t
                    - j // K : last_lead_t
                    - j // K
                    + total_pred_steps,
                    :,
                ]
    predicts = np.mean(predicts_en, axis=0)

    return predicts
