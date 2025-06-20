import numpy as np
from tqdm import tqdm


hw_table = [bin(i).count('1') for i in range(256)]


# Calculate whether the pos-th bit is 0 or 1
def calc_bit(pos):
    return lambda v: (v & (2**pos)) >> pos


def calc_LSB():  # alias to calc_bit(v, 0)
    f = calc_bit(0)
    return lambda v: f(v)


def calc_MSB():  # alias to calc_bit(v, 7)
    f = calc_bit(7)
    return lambda v: f(v)


def calc_hw(v):  # Calculate Hamming weight
    return hw_table[v]


def calc_v_eq_n(n):
    def f(v, n):
        return int(v == n)
    return lambda v: f(v, n)


def to_onehot(n_classes):
    eye = np.eye(n_classes)
    return lambda v: eye[v]


def calc_multilabel():  # Calculate multilabel
    return lambda v: np.array([calc_bit(i)(v) for i in range(8)])


def set_pois(pois):
    def f(trace, pois):
        t = []
        for p in pois:
            t.append(trace[p[0]:p[1]:p[2]])
        return np.concatenate(t)

    return lambda trace: f(trace, pois=pois)


# From https://github.com/Sengim/kansca/tree/main
# utils.py
def snr_fast(x, y):
    ns = x.shape[1]
    unique = np.unique(y)
    means = np.zeros((len(unique), ns))
    variances = np.zeros((len(unique), ns))

    for i, u in enumerate(unique):
        new_x = x[np.argwhere(y == int(u))]
        means[i] = np.mean(new_x, axis=0)
        variances[i] = np.var(new_x, axis=0)
    return np.var(means, axis=0) / np.mean(variances, axis=0)


def calc_guessing_entropy(
        preds, label_hyposesis, target_key,
        n_attack_traces, n_trial=40):
    """ Compute Guessing Entropy

    Args:
        preds: np.ndarray
            Predictions, shape: (# whole attack traces, # classes)
        label_hyposesis: np.ndarray
            Labels for each key hyposesis,
            shape: (# key hyposesis, # whole attack traces)
        target_key: int
            Target key
        n_attack_traces: int
            # attack traces for each trial,
            n_attack_traces < # whole attack traces
        n_trial: int
            # trials to calculate guessing entropy
        report_interval: int
            # trials per making report

    Returns: np.ndarray
        Guessing entropy, shape: (n_attack_traces, )
    """

    # Prevent to reach "NaN" by adding small constant
    preds = preds.astype(np.float64)
    preds = np.log(preds + 1e-36)

    # Set probabilities for each key hyposesis
    probs_all_key_hyposesis = np.concatenate([
        p[t][np.newaxis, ...]
        for p, t in zip(preds, label_hyposesis.T)
        ], axis=0).T

    # Calculate guessing entropy
    correct_key_ranks = []
    for _ in tqdm(range(n_trial), desc="Calculating GE [/trials]"):
        idx = np.random.choice(
            range(len(preds)), n_attack_traces, replace=False
        )
        atk_probs = probs_all_key_hyposesis[:, idx]

        prob = np.cumsum(atk_probs, dtype=np.float64, axis=1)
        rank = np.argsort(prob, axis=0)[::-1, :]
        correct_key_ranks.append(
            np.argmax(rank == target_key, axis=0)[np.newaxis, ...])
    correct_key_ranks = np.concatenate(correct_key_ranks, axis=0)
    guessing_entropy = np.mean(correct_key_ranks, axis=0)

    return guessing_entropy
