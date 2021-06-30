import json
import os
import time
import shutil


def set_rand_seed(seed):
    import random
    import numpy as np

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_printoptions(precision=4, linewidth=160):
    import numpy as np
    import torch

    np.set_printoptions(precision=precision, linewidth=linewidth)
    torch.set_printoptions(precision=precision, linewidth=linewidth)


class RelativeChangeMonitor(object):
    def __init__(self, tol):

        self.tol = tol
        # self._best_loss = float('inf')
        # self._curr_loss = float('inf')
        self._losses = []
        self._best_loss = float("inf")

    @property
    def save(self):
        return len(self._losses) > 0 and self._losses[-1] == self._best_loss

    @property
    def stop(self):
        return (
            len(self._losses) > 1
            and abs((self._losses[-1] - self._losses[-2]) / self._best_loss)
            < self.tol
        )

    def register(self, loss):
        self._losses.append(loss)
        self._best_loss = min(self._best_loss, loss)


class EarlyStoppingMonitor(object):
    def __init__(self, patience):

        self._patience = patience
        self._best_loss = float("inf")
        self._curr_loss = float("inf")
        self._n_fails = 0

    @property
    def save(self):
        return self._curr_loss == self._best_loss

    @property
    def stop(self):
        return self._n_fails > self._patience

    def register(self, loss):

        self._curr_loss = loss
        if loss < self._best_loss:
            self._best_loss = loss
            self._n_fails = 0

        else:
            self._n_fails += 1


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[{}] ".format(self.name), end="")
        dt = time.time() - self.tstart
        if dt < 60:
            print("Elapsed: {:.4f} sec.".format(dt))
        elif dt < 3600:
            print("Elapsed: {:.4f} min.".format(dt / 60))
        elif dt < 86400:
            print("Elapsed: {:.4f} hour.".format(dt / 3600))
        else:
            print("Elapsed: {:.4f} day.".format(dt / 86400))


def makedirs(dirs):
    assert isinstance(dirs, list), "Argument dirs needs to be a list"

    for dir in dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir)


def export_json(obj, path):

    with open(path, "w") as fout:
        json.dump(obj, fout, indent=4)


def export_csv(df, path, append=False, index=False):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    mode = "a" if append else "w"
    with open(path, mode) as f:
        df.to_csv(f, header=f.tell() == 0, index=index)


def counting_proc_to_event_seq(count_proc):
    """Convert a counting process sample to event sequence

    Args:
      count_proc (list of ndarray): each array in the list contains the
        timestamps of events occurred on that dimension.

    Returns:
      (list of 2-tuples): each tuple is of (t, c), where c denotes the event
        type
    """
    seq = []
    for i, ts in enumerate(count_proc):
        seq += [(t, i) for t in ts]

    seq = sorted(seq, key=lambda x: x[0])
    return seq


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def split_dataloader(dataloader, ratio: float):
    import torch
    from torch.utils.data import DataLoader
    dataset = dataloader.dataset
    n = len(dataset)
    lengths = [int(n * ratio), n - int(n * ratio)]
    datasets = torch.utils.data.random_split(dataset, lengths)

    copied_fields = ["batch_size", "num_workers", "collate_fn", "drop_last"]
    dataloaders = []
    for d in datasets:
        dataloaders.append(
            DataLoader(
                dataset=d, **{k: getattr(dataloader, k) for k in copied_fields}
            )
        )

    return tuple(dataloaders)


def compare_metric_value(val1: float, val2: float) -> bool:
    """Compare whether val1 is "better" than val2.
    Args:
        val1 (float):
        val2 (float): can be NaN.
        metric (str): metric name
    Returns:
        (bool): True only if val1 is better than val2.
    """
    from math import isnan

    if isnan(val2):
        return True
    elif isnan(val1):
        return False
    return val1 < val2

        

def save_checkpoint(state, output_folder, is_best, filename="checkpoint.tar"):
    import torch

    torch.save(state, os.path.join(output_folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(output_folder, filename),
            os.path.join(output_folder, "model_best.tar"),
        )


def get_freer_gpu(by="n_proc"):
    """Return the GPU index which has the largest available memory

    Returns:
        int: the index of selected GPU.
    """
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetComputeRunningProcesses,
        nvmlDeviceGetMemoryInfo,
    )

    nvmlInit()
    n_devices = nvmlDeviceGetCount()
    gpu_id, gpu_state = None, None
    for i in range(0, n_devices):
        handle = nvmlDeviceGetHandleByIndex(i)
        if by == "n_proc":
            temp = -len(nvmlDeviceGetComputeRunningProcesses(handle))
        elif by == "free_mem":
            temp = nvmlDeviceGetMemoryInfo(handle).free
        else:
            raise ValueError("`by` can only be 'n_proc', 'free_mem'.")
        if gpu_id is None or gpu_state < temp:
            gpu_id, gpu_state = i, temp

    return gpu_id


def savefig(fig, path, save_pickle=False):
    """save matplotlib figure

    Args:
        fig (matplotlib.figure.Figure): figure object
        path (str): [description]
        save_pickle (bool, optional): Defaults to True. Whether to pickle the
          figure object as well.
    """

    fig.savefig(path, bbox_inches="tight")
    if save_pickle:
        import matplotlib
        import pickle

        # the `inline` of IPython will fail the pickle/unpickle; if so, switch
        # the backend temporarily
        if "inline" in matplotlib.get_backend():
            raise (
                "warning: the `inline` of IPython will fail the pickle/"
                "unpickle. Please use `matplotlib.use` to switch to other "
                "backend."
            )
        else:
            with open(path + ".pkl", "wb") as f:
                pickle.dump(fig, f)
