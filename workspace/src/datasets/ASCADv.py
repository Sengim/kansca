import torch
import h5py
from pathlib import Path
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        trace_transforms=[],
        label_transforms=[],
        profiling=True,
        scale=True
    ):

        self.dataset_path = dataset_path
        self.trace_transforms = trace_transforms
        self.label_transforms = label_transforms

        # Load data
        if profiling:
            root_key = 'Profiling_traces'
        else:
            root_key = 'Attack_traces'
        with h5py.File(Path(dataset_path, 'ascad-variable.h5'), 'r') as f:
            trace = np.array(f['Profiling_traces/traces'], dtype=np.float32)
            u, v = self.calc_scale(trace)
            if not profiling:
                trace = np.array(f[root_key+'/traces'], dtype=np.float32)

            if scale:
                self.traces = self.trace_scaler(trace, u, v)
            else:
                self.traces = trace
            self.plaintext = np.array(
                f[root_key+'/metadata']['plaintext'], dtype=np.uint8)
            self.key = np.array(f[root_key+'/metadata']['key'], dtype=np.uint8)
            self.masks = np.array(
                f[root_key+'/metadata']['masks'], dtype=np.uint8)
        self._len = self.traces.shape[0]
        self.trace_len = 1400  # Constant

    # Calculate mean and sandard diviation of the trace
    def calc_scale(self, trace):
        return np.mean(trace, axis=0), np.std(trace, axis=0)

    # Z-score normalization
    def trace_scaler(self, trace, u, v):
        return (trace-u)/v

    def __len__(self):
        return self._len

    def set_key_hyposesis(self, key):
        self._external_key = True
        self.key = key

    def __getitem__(self, i):
        x = self.traces[i]
        for f in self.trace_transforms:
            x = f(x)
        t = (self.plaintext[i], self.key[i], self.masks[i])
        for g in self.label_transforms:
            if isinstance(t, tuple):
                t = g(*t)
            else:
                t = g(t)
        return x, t
