"""
Cluster selection methods for ToMATo.

Each method turns a persistence diagram into a single persistence threshold:
classes with persistence below the threshold are merged into their parents,
classes above it become clusters.

All functions here are stateless. ToMAToCore keeps thin get_threshold_by_*
wrappers that call into this module and record diagnostics on the estimator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import stats

MIN_NOISE_SPACINGS = 3
N_NULL_SIMS = 1000
DEFAULT_ALPHA = 0.2


@dataclass(frozen=True)
class ThresholdResult:
    """
    A selected threshold plus diagnostics about how it was chosen.
    """

    threshold: float
    method: str
    pvalue: float = np.nan


def log_ratio_persistences(diagram) -> np.ndarray:
    """
    Persistence measured as |log(death) - log(birth)| = |log(death / birth)|.

    Works for both filtration directions (ascending: death > birth,
    descending: birth > death). Requires strictly positive f values.
    Non-finite entries (e.g. the essential class) come back as np.nan.
    """
    if diagram is None:
        raise ValueError("Must call fit() before computing log-ratio persistences")

    diagram = np.asarray(diagram, dtype=float)
    birth, death = diagram[:, 0], diagram[:, 1]

    finite = np.isfinite(birth) & np.isfinite(death)
    if np.any(birth[finite] <= 0) or np.any(death[finite] <= 0):
        raise ValueError(
            "Ratio persistence needs strictly positive f values. "
            "Shift/rescale f (e.g. f - f.min() + eps) or use a density instead."
        )

    out = np.full(diagram.shape[0], np.nan)
    out[finite] = np.abs(np.log(death[finite]) - np.log(birth[finite]))
    return out


def _finite_pairs(persistences, diagram):
    """
    Raw and log-ratio persistences, restricted to finite entries.
    """
    p = np.asarray(persistences, dtype=float)
    r = log_ratio_persistences(diagram)
    finite = np.isfinite(p) & np.isfinite(r)
    return p[finite], r[finite]


def profile_likelihood_split(x_sorted) -> Optional[int]:
    """
    Find the best two-regime split of a sorted 1-D array (Zhu & Ghodsi, 2006).

    Fits two Gaussians with a shared variance, one to x[:q] and one to x[q:],
    and returns the q that maximizes the likelihood. O(n) via prefix sums.

    Returns None if there are fewer than 2 points.
    """
    x = np.asarray(x_sorted, dtype=float)
    n = len(x)
    if n < 2:
        return None

    x = x - x.mean()
    cum_sum = np.cumsum(x)
    cum_sq = np.cumsum(x * x)

    split = np.arange(1, n)
    n_low, n_high = split, n - split
    sum_low, sq_low = cum_sum[:-1], cum_sq[:-1]
    sum_high = cum_sum[-1] - sum_low
    sq_high = cum_sq[-1] - sq_low

    within_ss = (sq_low - sum_low**2 / n_low) + (sq_high - sum_high**2 / n_high)
    pooled_var = np.maximum(within_ss / n, 1e-12)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * pooled_var) - 0.5 * n

    return int(split[np.argmax(log_likelihood)])


def _normalized_spacings(x_sorted) -> np.ndarray:
    """
    e_j = j * (spacing between the j-th and (j+1)-th largest values).
    """
    descending = np.asarray(x_sorted, dtype=float)[::-1]
    spacings = descending[:-1] - descending[1:]
    return np.arange(1, len(descending)) * spacings


def _gap_from_normalized(normalized: np.ndarray, n: int, q: int) -> float:
    k = n - q
    if k < 1 or q < 2:
        return np.nan

    gap_e = normalized[k - 1]
    window = max(MIN_NOISE_SPACINGS, q // 2)
    noise_e = normalized[k : k + window]
    if len(noise_e) < MIN_NOISE_SPACINGS:
        return np.nan

    scale = np.median(noise_e) / np.log(2)
    if scale <= 1e-12:
        return np.nan
    return float(gap_e / scale)


def gap_statistic(x_sorted, q: int) -> float:
    """
    Scale-free size of the gap between x_sorted[q-1] and x_sorted[q].

    Under an exponential upper tail, the normalized spacings
        e_j = j * (spacing between the j-th and (j+1)-th largest values)
    are i.i.d. exponential, with j counted from the top of the WHOLE sample.
    The gap has j = k = n - q (k points lie above it). The scale is estimated
    robustly from the next spacings down (j = k+1, k+2, ...), i.e. the upper
    part of the noise group.

    Returns gap_e / scale, or NaN if the scale cannot be estimated.
    """
    x = np.asarray(x_sorted, dtype=float)
    return _gap_from_normalized(_normalized_spacings(x), len(x), q)


def gap_statistics(x_sorted) -> np.ndarray:
    """
    Gap statistic for every candidate split. Entry q is the statistic for the
    gap between x_sorted[q-1] and x_sorted[q]; untestable splits are NaN.
    """
    x = np.asarray(x_sorted, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < 3:
        return out

    normalized = _normalized_spacings(x)
    for q in range(2, n):
        out[q] = _gap_from_normalized(normalized, n, q)
    return out


@lru_cache(maxsize=32)
def _null_gap_statistics_cached(n: int, n_sims: int, min_spacings: int) -> np.ndarray:
    if n < 3:
        out = np.empty(0)
        out.setflags(write=False)
        return out

    rng = np.random.default_rng(0)
    j = np.arange(1, n)

    e = rng.exponential(size=(n_sims, n - 1))
    x = np.zeros((n_sims, n))
    x[:, 1:] = np.cumsum((e / j)[:, ::-1], axis=1)

    xc = x - x.mean(axis=1, keepdims=True)
    cum_sum = np.cumsum(xc, axis=1)
    cum_sq = np.cumsum(xc * xc, axis=1)

    split = np.arange(1, n)
    sum_low, sq_low = cum_sum[:, :-1], cum_sq[:, :-1]
    sum_high = cum_sum[:, -1:] - sum_low
    sq_high = cum_sq[:, -1:] - sq_low
    within_ss = (sq_low - sum_low**2 / split) + (sq_high - sum_high**2 / (n - split))
    q = split[np.argmin(np.maximum(within_ss / n, 1e-12), axis=1)]

    k = n - q
    rows = np.arange(n_sims)
    gap_e = e[rows, k - 1]

    window = np.maximum(min_spacings, q // 2)
    cols = np.arange(n - 1)[None, :]
    in_window = (cols >= k[:, None]) & (cols < (k + window)[:, None])
    n_window = in_window.sum(axis=1)

    padded = np.where(in_window, e, np.inf)
    padded.sort(axis=1)
    lo = np.maximum(n_window - 1, 0) // 2
    hi = np.minimum(n_window // 2, n - 2)
    scale = 0.5 * (padded[rows, lo] + padded[rows, hi]) / np.log(2)

    valid = (
        (q >= 2)
        & (n_window >= min_spacings)
        & np.isfinite(scale)
        & (scale > 1e-12)
    )
    out = gap_e[valid] / scale[valid]
    out.setflags(write=False)
    return out


def null_gap_statistics(n: int) -> np.ndarray:
    """
    Null distribution of the gap statistic for a sample of size n, obtained by
    running the full pipeline (profile split, then gap statistic) on pure
    exponential samples. Cached per n. The statistic is location- and
    scale-free, so Exp(1) is sufficient.
    """
    return _null_gap_statistics_cached(int(n), N_NULL_SIMS, MIN_NOISE_SPACINGS)


def gap_pvalue(x_sorted, q: int) -> float:
    """
    Monte Carlo p-value for the gap at q: the fraction of null samples whose
    selected gap is at least this large. Because the null runs the same split
    selection, the p-value accounts for the split having been chosen.

    Returns NaN if the gap is not testable.
    """
    observed = gap_statistic(x_sorted, q)
    if not np.isfinite(observed):
        return np.nan
    null = null_gap_statistics(len(x_sorted))
    if len(null) == 0:
        return np.nan
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def largest_raw_jump(p) -> float:
    """
    Safe default: midpoint of the largest gap in sorted raw persistence.
    The most persistent class is excluded.
    """
    p = np.asarray(p, dtype=float)
    sorted_p = np.sort(p)[:-1]
    if len(sorted_p) < 2:
        return float(sorted_p[0]) if len(sorted_p) else float(p.min())

    i = int(np.argmax(np.diff(sorted_p)))
    return float(0.5 * (sorted_p[i] + sorted_p[i + 1]))


def threshold_by_profile_likelihood(
    persistences, diagram, alpha: float = DEFAULT_ALPHA
) -> ThresholdResult:
    """
    Two-regime split of the sorted log-ratio persistences chosen by profile
    likelihood, accepted only if the Monte Carlo gap p-value is <= alpha.
    Otherwise falls back to the largest raw persistence jump.
    """
    p, r = _finite_pairs(persistences, diagram)

    if len(r) < 3:
        return ThresholdResult(
            largest_raw_jump(p) if len(p) else 0.0, "raw_max_jump"
        )

    order = np.argsort(r)[:-1]
    r_sorted, p_sorted = r[order], p[order]

    pvalue = np.nan
    q = profile_likelihood_split(r_sorted)
    if q is not None:
        pvalue = gap_pvalue(r_sorted, q)
        if pvalue <= alpha:
            threshold = 0.5 * (p_sorted[q - 1] + p_sorted[q])
            return ThresholdResult(float(threshold), "profile_likelihood", pvalue)

    return ThresholdResult(largest_raw_jump(p), "raw_max_jump", pvalue)


def threshold_by_max_jump(persistences, diagram) -> ThresholdResult:
    """
    Split at the largest gap by gap statistic: evaluate the scale-free gap
    statistic at every candidate split of the sorted log-ratio persistences
    and cut at the maximum. Unlike the profile-likelihood method this always
    returns a split (no significance test). Falls back to the largest raw
    persistence jump only if no gap is testable.
    """
    p, r = _finite_pairs(persistences, diagram)

    if len(r) < 3:
        return ThresholdResult(
            largest_raw_jump(p) if len(p) else 0.0, "raw_max_jump"
        )

    order = np.argsort(r)[:-1]
    r_sorted, p_sorted = r[order], p[order]

    gaps = gap_statistics(r_sorted)
    if not np.any(np.isfinite(gaps)):
        return ThresholdResult(largest_raw_jump(p), "raw_max_jump")

    q = int(np.nanargmax(gaps))
    threshold = 0.5 * (p_sorted[q - 1] + p_sorted[q])
    return ThresholdResult(float(threshold), "max_jump")


def threshold_by_significance(persistences, sig: float) -> float:
    """
    Persistence threshold from an exponential fit at significance level sig.
    """
    if persistences is None:
        raise ValueError("Must call fit() before get_threshold_by_significance()")

    scale = np.asarray(persistences).mean()
    return float(stats.expon.ppf(1 - sig, scale=scale))


def threshold_by_n_clusters_after_merging(
    persistences, n_clusters: int, n_modes: int, count_clusters
) -> float:
    """
    Threshold that leaves at least n_clusters clusters once outlier modes have
    been merged away.

    Starts from the plain n_clusters threshold and lowers it (walking down the
    merge tree), retaining more modes until count_clusters(threshold) >=
    n_clusters. The number of retained modes is found by galloping then bisecting,
    which assumes the cluster count generally grows as more modes are retained.
    The returned threshold always satisfies the count; it is the smallest such
    one whenever the count is monotone. Returns 0.0, which keeps every mode, with
    a warning if the count is not reached even then.

    Parameters
    ----------
    persistences : array-like
        Persistence of every mode.
    n_clusters : int
        Number of clusters to retain.
    n_modes : int
        Total number of modes.
    count_clusters : callable
        Maps a persistence threshold to the number of clusters that remain
        after outlier modes are merged into their parents.
    """
    if persistences is None:
        raise ValueError("Must call fit() before get_threshold_by_n_clusters()")

    sorted_persistences = np.flip(np.sort(persistences))

    def threshold_for(m):
        if m >= n_modes:
            return 0.0
        return 0.5 * (sorted_persistences[m - 1] + sorted_persistences[m])

    def enough(m):
        return count_clusters(threshold_for(m)) >= n_clusters

    start = max(n_clusters, 1)
    if start <= n_modes and enough(start):
        return float(threshold_for(start))

    failed, step, found = start, 1, None
    while failed < n_modes:
        candidate = min(failed + step, n_modes)
        if enough(candidate):
            found = candidate
            break
        failed, step = candidate, step * 2

    if found is None:
        warnings.warn(
            f"Could not retain {n_clusters} clusters after merging outlier modes. "
            + "Returning the finest clustering."
        )
        return 0.0

    while found - failed > 1:
        mid = (failed + found) // 2
        if enough(mid):
            found = mid
        else:
            failed = mid

    return float(threshold_for(found))


def threshold_by_n_clusters(
    persistences, n_clusters: int, n_modes: int, verbose: bool = False
) -> float:
    """
    Threshold between two adjacent persistences that yields n_clusters clusters.
    Returns 0.0 (keep every mode) if n_clusters >= n_modes.
    """
    if persistences is None:
        raise ValueError("Must call fit() before get_threshold_by_n_clusters()")

    if n_clusters > n_modes:
        warnings.warn(
            "n_clusters exceeds number of detectable modes. "
            + f"Returning {n_modes} clusters."
        )
        return 0.0

    if n_clusters == n_modes:
        return 0.0

    sorted_persistences = np.flip(np.sort(persistences))
    threshold_upper = sorted_persistences[n_clusters - 1]
    threshold_lower = sorted_persistences[n_clusters]
    threshold = 0.5 * (threshold_upper + threshold_lower)

    if verbose:
        print(f"[ToMATo] Cluster threshold for {n_clusters} clusters: {threshold}")
        print(
            f"[ToMATo] Clusters above threshold: {(sorted_persistences > threshold).sum()}"
        )

    return threshold


CLUSTER_SELECTION_METHODS = (
    "profile_likelihood",
    "max_jump",
    "significance",
    "persistence_threshold",
    "n_clusters",
)


def validate_cluster_selection_method(method: str) -> str:
    """
    Return method if it is a known cluster selection method, otherwise raise.
    """
    if method not in CLUSTER_SELECTION_METHODS:
        raise ValueError(
            f"Unknown cluster_selection_method {method!r}. "
            f"Choose from {list(CLUSTER_SELECTION_METHODS)}."
        )
    return method


def resolve_cluster_selection_method(
    method: Optional[str],
    n_clusters: Optional[int] = None,
    persistence_threshold: Optional[float] = None,
    sig: Optional[float] = None,
) -> str:
    """
    Return the cluster selection method to use.

    An explicit method is returned as given, with no inference and no check of
    the other parameters. If method is None it is inferred from whichever of
    n_clusters, persistence_threshold and sig is set, in that order of
    precedence, and is "profile_likelihood" if none is. A warning naming the
    method used and the parameters ignored is raised if more than one is set.
    """
    if method is not None:
        return validate_cluster_selection_method(method)

    candidates = (
        ("n_clusters", "n_clusters", n_clusters),
        ("persistence_threshold", "persistence_threshold", persistence_threshold),
        ("sig", "significance", sig),
    )
    given = [(name, chosen) for name, chosen, value in candidates if value is not None]
    if not given:
        return "profile_likelihood"

    chosen = given[0][1]
    if len(given) > 1:
        names = ", ".join(name for name, _ in given)
        ignored = ", ".join(name for name, _ in given[1:])
        warnings.warn(
            f"Conflicting cluster selection parameters were given ({names}). "
            f"Using cluster_selection_method='{chosen}' and ignoring {ignored}. "
            "Set cluster_selection_method explicitly to choose a method."
        )
    return chosen