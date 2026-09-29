VALID_METRICS = (
    "euclidean",
    "cityblock",
    "cosine",
    "chebyshev",
    "minkowski",
    "hamming",
    "l1",
    "l2",
    "manhattan",
    "minkowski",
)


METRIC_ALIASES = {
    "euclidean": "euclidean",
    "cityblock": "manhattan",
    "cosine": "cosine",
    "chebyshev": "chebyshev",
    "minkowski": "minkowski",
    "hamming": "hamming",
    "l1": "manhattan",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "minkowski": "minkowski",
}


def check_metric(metric: str) -> str:
    """
    Validate and standardize the metric name.

    Parameters
    ----------
    metric : str
        The distance metric to validate.

    Returns
    -------
    str
        The standardized metric name.

    Raises
    ------
    ValueError
        If the metric is not recognized.
    """
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}'. Valid options are: {VALID_METRICS}"
        )
    return METRIC_ALIASES[metric]
