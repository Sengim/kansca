from . import base
import h5py
from pathlib import Path
import numpy as np


class Dataset(base.BaseDataset):
    def __init__(
        self,
        dataset_path,
        profiling=True,
        scale=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dataset_path = dataset_path
        self._external_key = False

        # Load data
        if profiling:
            root_key = 'Profiling_traces'
        else:
            root_key = 'Attack_traces'
        with h5py.File(Path(dataset_path, 'ASCAD.h5'), 'r') as f:
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
        self.trace_len = 700  # Constant

    # Calculate mean and sandard diviation of the trace
    def calc_scale(self, trace):
        return np.mean(trace, axis=0), np.std(trace, axis=0)

    # Z-score normalization
    def trace_scaler(self, trace, u, v):
        return (trace-u)/v

    def __len__(self):
        return self._len
