import numpy as np


def resample_indices(n_items, n_boot, seed):
    #one (n_boot, n_items) index matrix; reused for every condition so differences are paired
    rng= np.random.default_rng(seed)

    return rng.integers(0, n_items, size=(n_boot, n_items))


def bootstrap_mean_ci(values, idx, level=0.95):
    #values: one number per item (e.g. per pair); returns the point estimate and the percentile interval
    values= np.asarray(values, dtype=np.float64)
    means= values[idx].mean(axis=1)
    lo, hi= np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])

    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def paired_difference_ci(a, b, idx, level=0.95):
    #a, b: per-item values for two conditions on the SAME items in the same order; resampled with shared idx
    return bootstrap_mean_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), idx, level)
