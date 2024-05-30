import torch
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trace_transforms=[],
        label_transforms=[],
    ):

        self.trace_transforms = trace_transforms
        self.label_transforms = label_transforms
        self._external_key = False

    def set_key_hyposesis(self, key):
        self._external_key = True
        self.ext_key = np.ones(
            (self.key.shape[1], ), dtype=np.int32) * key

    def __getitem__(self, i):
        x = self.get_trace(i)
        t = self.get_label(i)
        return x, t

    def get_trace(self, i):
        x = self.traces[i]
        for f in self.trace_transforms:
            x = f(x)
        return x

    def get_label(self, i):
        if self._external_key:
            t = (self.plaintext[i], self.ext_key, self.masks[i])
        else:
            t = (self.plaintext[i], self.key[i], self.masks[i])
        for g in self.label_transforms:
            if isinstance(t, tuple):
                t = g(*t)
            else:
                t = g(t)
        return t
