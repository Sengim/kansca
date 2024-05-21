import numpy as np
from src import aes_utils, sca_utils


def RandomDataset(n_traces, **kwargs):
    pt = np.random.randint(0, 255, (n_traces, 16), dtype=np.uint8)
    key = np.random.randint(0, 255, (1, 16), dtype=np.uint8)
    key = np.tile(key, (n_traces, 1))
    masks = np.random.randint(0, 255, (n_traces, 18), dtype=np.uint8)
    ds = Dataset(pt, key, masks, **kwargs)
    return ds


class Dataset:
    def __init__(
        self,
        pt, key, masks, target_byte, sigma=0.5,
        trace_transforms=[],
        label_transforms=[]
    ):
        self.plaintext = pt
        self.key = key
        self.masks = masks
        self.sigma = sigma
        self.trace_transforms = trace_transforms
        self.label_transforms = label_transforms
        self._external_key = False
        self._len = pt.shape[0]
        self.target_byte = target_byte

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
        r_out = self.masks[i, 2]

        unmasked_sbox_out = aes_utils.aes_sbox[pt ^ key]
        masked_sbox_out = aes_utils.aes_sbox[pt ^ key] ^ r_out
        sbox_out_mask = r_out
        masked_sbox_out_linear = aes_utils.aes_sbox[pt ^ key] ^ mask
        sbox_out_mask_linear = mask

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

    def __len__(self):
        return self._len

    def set_key_hyposesis(self, key):
        self._external_key = True
        self.ext_key = np.ones(
            (self.key.shape[1], ), dtype=np.int32) * key

    def __getitem__(self, i):
        iv = self.calc_ivs(i)
        x = self.gen_trace(iv[1:])
        for f in self.trace_transforms:
            x = f(x)

        if self._external_key:
            t = (self.plaintext[i], self.ext_key, self.masks[i])
        else:
            t = (self.plaintext[i], self.key[i], self.masks[i])
        for g in self.label_transforms:
            if isinstance(t, tuple):
                t = g(*t)
            else:
                t = g(t)

        return x, t
