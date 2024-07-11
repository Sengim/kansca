from . import base
import numpy as np
from src import aes_utils, sca_utils


def RandomDataset(n_traces, **kwargs):
    pt = np.random.randint(0, 255, (n_traces, 16), dtype=np.uint8)
    key = np.random.randint(0, 255, (1, 16), dtype=np.uint8)
    key = np.tile(key, (n_traces, 1))
    masks = np.random.randint(0, 255, (n_traces, 18), dtype=np.uint8)
    ds = Dataset(pt, key, masks, **kwargs)
    return ds


class Dataset(base.BaseDataset):
    def __init__(
        self,
        pt, key, masks, target_byte, sigma=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.plaintext = pt
        self.key = key
        self.masks = masks
        self.sigma = sigma
        self._external_key = False
        self._len = pt.shape[0]
        self.target_byte = target_byte
        self.traces = self.make_traces()

    def calc_ivs(self, i):
        """ Calculate intermediate variables (IVs) of 1st round
        Please refar the paper (Table 1)
        https://eprint.iacr.org/2018/053.pdf
        Args:
            i: int
                Sample index

        Returns:
            unmasked_sbox_out
                sbox( pt[i] ^ k[i] )
            masked_sbox_out
                sbox( pt[i] ^ k[i] ) ^ r_out
            sbox_out_mask
                r_out
            masked_sbox_out_linear
                sbox( pt[i] ^ k[i] ) ^ r[i]
            sbox_out_mask_linear
                r[i]
        """
        pt = self.plaintext[i, self.target_byte]
        key = self.key[i, self.target_byte]
        mask = self.masks[i, self.target_byte]

        unmasked_sbox_out = aes_utils.aes_sbox[pt ^ key]
        masked_sbox_out_linear = aes_utils.aes_sbox[pt ^ key] ^ mask
        sbox_out_mask_linear = mask

        r_out = self.masks[i, 2]
        masked_sbox_out = aes_utils.aes_sbox[pt ^ key] ^ r_out
        sbox_out_mask = r_out

        return (unmasked_sbox_out, masked_sbox_out,
                sbox_out_mask, masked_sbox_out_linear,
                sbox_out_mask_linear)

    def gen_trace(self, iv):
        # HW leakage model
        trace = np.array(
            [sca_utils.calc_hw(v) for v in iv], dtype=np.float32)
        trace /= 8  # Scale to [0, 1]
        trace += np.random.randn(*trace.shape)*self.sigma
        return trace

    def make_traces(self):
        tr = [self.gen_trace(self.calc_ivs(i))[np.newaxis, ...]
              for i in range(len(self))]
        return np.concatenate(tr, axis=0)

    def __len__(self):
        return self._len
