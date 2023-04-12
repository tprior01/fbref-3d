"""
Code adapted from https://github.com/ckjellson/textalloc
"""

import numpy as np


def non_overlapping_with_points(
    scatter_xy: np.ndarray, candidates: np.ndarray, xmargin: float, ymargin: float
) -> np.ndarray:
    """Finds candidates not overlapping with points.
    Args:
        scatter_xy (np.ndarray): Array of shape (N,2) containing coordinates for all scatter-points
        candidates (np.ndarray): Array of shape (K,4) with K candidate boxes
        xmargin (float): fraction of the x-dimension to use as margins for text boxes
        ymargin (float): fraction of the y-dimension to use as margins for text boxes
    Returns:
        np.ndarray: Boolean array of shape (K,) with True for non-overlapping candidates with points
    """
    return np.invert(
        np.bitwise_or.reduce(
            np.bitwise_and(
                candidates[:, 0][:, None] - xmargin < scatter_xy[:, 0],
                np.bitwise_and(
                    candidates[:, 2][:, None] + xmargin > scatter_xy[:, 0],
                    np.bitwise_and(
                        candidates[:, 1][:, None] - ymargin < scatter_xy[:, 1],
                        candidates[:, 3][:, None] + ymargin > scatter_xy[:, 1],
                    ),
                ),
            ),
            axis=1,
        )
    )


def ccw(x1y1: np.ndarray, x2y2: np.ndarray, x3y3: np.ndarray, cand: bool) -> np.ndarray:
    """CCW used in line intersect
    Args:
        x1y1 (np.ndarray):
        x2y2 (np.ndarray):
        x3y3 (np.ndarray):
        cand (bool): using candidate positions (different broadcasting)
    Returns:
        np.ndarray:
    """
    if cand:
        return (
            (-(x1y1[:, 1][:, None] - x3y3[:, 1]))
            * np.repeat(x2y2[:, 0:1] - x1y1[:, 0:1], x3y3.shape[0], axis=1)
        ) > (
            np.repeat(x2y2[:, 1:2] - x1y1[:, 1:2], x3y3.shape[0], axis=1)
            * (-(x1y1[:, 0][:, None] - x3y3[:, 0]))
        )
    return (
        (-(x1y1[:, 1][:, None] - x3y3[:, 1])) * (-(x1y1[:, 0][:, None] - x2y2[:, 0]))
    ) > ((-(x1y1[:, 1][:, None] - x2y2[:, 1])) * (-(x1y1[:, 0][:, None] - x3y3[:, 0])))


def non_overlapping_with_boxes(
    box_arr: np.ndarray, candidates: np.ndarray, xmargin: float, ymargin: float
) -> np.ndarray:
    """Finds candidates not overlapping with allocated boxes.
    Args:
        box_arr (np.ndarray): array with allocated boxes
        candidates (np.ndarray): candidate boxes
        xmargin (float): fraction of the x-dimension to use as margins for text boxes
        ymargin (float): fraction of the y-dimension to use as margins for text boxes
    Returns:
        np.ndarray: Boolean array of shape (K,) with True for non-overlapping candidates with boxes.
    """
    return np.invert(
        np.any(
            np.invert(
                np.bitwise_or(
                    candidates[:, 0][:, None] - xmargin > box_arr[:, 2],
                    np.bitwise_or(
                        candidates[:, 2][:, None] + xmargin < box_arr[:, 0],
                        np.bitwise_or(
                            candidates[:, 1][:, None] - ymargin > box_arr[:, 3],
                            candidates[:, 3][:, None] + ymargin < box_arr[:, 1],
                        ),
                    ),
                )
            ),
            axis=1,
        )
    )


def inside_plot(
    xmin_bound: float,
    ymin_bound: float,
    xmax_bound: float,
    ymax_bound: float,
    candidates: np.ndarray,
) -> np.ndarray:
    """Finds candidates that are inside the plot bounds
    Args:
        xmin_bound (float):
        ymin_bound (float):
        xmax_bound (float):
        ymax_bound (float):
        candidates (np.ndarray): candidate boxes
    Returns:
        np.ndarray: Boolean array of shape (K,) with True for non-overlapping candidates with boxes.
    """
    return np.invert(
        np.bitwise_or(
            candidates[:, 0] < xmin_bound,
            np.bitwise_or(
                candidates[:, 1] < ymin_bound,
                np.bitwise_or(
                    candidates[:, 2] > xmax_bound, candidates[:, 3] > ymax_bound
                ),
            ),
        )
    )