from . import base
import numpy as np


class Dataset(base.BaseDataset):
    def __init__(
        self,
        n_shares,
        bus_width=8,
        n_traces=-1,
        sigma=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_shares = n_shares
        self.bus_width = bus_width
        if n_traces <= 0:
            n_traces = (2**bus_width)**n_shares
        self.shares = np.random.randint(
            0, 2**self.bus_width, (n_traces, n_shares), dtype=np.uint8)
        self.sigma = sigma
        self._len = n_traces
        HW_table = np.array([bin(i).count('1')
                             for i in range(2**self.bus_width)])
        self.HW = np.vectorize(
            lambda x: HW_table[x])
        self.traces = self.make_traces()

    def calc_ivs(self, i):
        """ Calculate intermediate variables (IVs) of 1st round
        Please refar the paper (Table 1)
        https://eprint.iacr.org/2018/053.pdf
        Args:
            i: int
                Sample index

        Returns: shares
        """
        t = 0
        for v in self.shares[i]:
            t = t ^ v
        return t

    def make_traces(self):
        tr = self.HW(self.shares)
        tr = tr.astype(np.float32) / (self.bus_width)
        tr += np.random.randn(*tr.shape) * self.sigma
        return tr

    def __len__(self):
        return self._len

    def get_label(self, i):
        t = self.calc_ivs(i)
        for g in self.label_transforms:
            t = g(t)
        return t
