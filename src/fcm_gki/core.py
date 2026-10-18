"""
Modified Fuzzy C-Means clustering with spatial distance to cluster centre of
gravity (H term) and a fuzzy local neighbourhood factor (G_ki term).

Pure NumPy. Images follow the OpenCV convention: a NumPy array of shape
(H, W) or (H, W, C), indexed [y, x]. Channels are treated as generic
features, so BGR, RGB, Lab, HSV or greyscale all work; outputs keep the
input's channel order (BGR in, BGR out).

Pixels are flattened row-major, so flat index n is at (y, x) = divmod(n, W).
Every reshape here uses that convention.

References
  - C. Gauge, "Modified Fuzzy C-Means Clustering Algorithm with Spatial
    Distance to Cluster Center of Gravity", IEEE ISM 2010,
    doi:10.1109/ISM.2010.53
  - S. Krinidis and V. Chatzis, "A Robust Fuzzy Local Information C-Means
    Clustering Algorithm", IEEE TIP 2010 (source of the G_ki factor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

_EPS = 1e-10

# Same values as cv2.TERM_CRITERIA_* so criteria tuples are interchangeable.
TERM_CRITERIA_COUNT = 1
TERM_CRITERIA_MAX_ITER = 1
TERM_CRITERIA_EPS = 2

__all__ = [
    "FCMResult", "ModifiedFCM", "fuzzy_cmeans",
    "render_segmented", "render_clusters",
    "TERM_CRITERIA_COUNT", "TERM_CRITERIA_MAX_ITER", "TERM_CRITERIA_EPS",
]


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class FCMResult:
    """Output of a run.

    labels     (H, W) int32 hard labels (argmax of the memberships)
    U          (H*W, K) float memberships; each row sums to 1
    centers    (K, C) float32 cluster centres in the input's feature space
    positions  (K, 2) float centre-of-gravity positions as (x, y)
    iterations number of iterations performed
    history    objective J after each iteration
    converged  True if the tolerance was met before max_iter / cancel
    """

    labels: np.ndarray
    U: np.ndarray
    centers: np.ndarray
    positions: np.ndarray
    iterations: int
    history: list = field(default_factory=list)
    converged: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape

    @property
    def memberships(self) -> np.ndarray:
        """(H, W, K) float32 membership maps, usable as soft masks in OpenCV."""
        h, w = self.labels.shape
        return self.U.reshape(h, w, -1).astype(np.float32)

    def membership(self, k: int) -> np.ndarray:
        """(H, W) float32 membership map of cluster k."""
        return self.memberships[..., k]

    def mask(self, k: int) -> np.ndarray:
        """(H, W) uint8 mask (0/255) of pixels whose hard label is k."""
        return np.where(self.labels == k, 255, 0).astype(np.uint8)

    @property
    def compactness(self) -> float:
        return self.history[-1] if self.history else float("nan")

    # Backwards-compatible alias for the first Python port.
    @property
    def colours(self) -> np.ndarray:
        return self.centers


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _as_features(image: np.ndarray) -> tuple[np.ndarray, int, int, np.dtype]:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"expected an (H, W) or (H, W, C) image, got shape {image.shape}")
    h, w, c = image.shape
    if h * w == 0:
        raise ValueError("image is empty")
    return image.reshape(h * w, c).astype(np.float64), h, w, image.dtype


def _sqdist(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance (N, K) via |x|^2 - 2x.v + |v|^2 (O(N*K) memory)."""
    d = (X * X).sum(1)[:, None] - 2.0 * X @ V.T + (V * V).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _seed_top_colours(X: np.ndarray, k: int, min_sep: float) -> np.ndarray:
    """Most frequent distinct values, skipping near-duplicates (original method).

    The C# version took the k most frequent exact colours, which in a noisy
    image are often near-identical. A candidate within `min_sep` of an
    already chosen centre is skipped; the threshold is relaxed if too few
    remain.
    """
    values, counts = np.unique(X, axis=0, return_counts=True)
    values = values[np.argsort(-counts, kind="stable")]
    k = min(k, len(values))
    sep = min_sep
    while True:
        chosen: list[np.ndarray] = []
        for v in values:
            if not chosen or np.min(np.linalg.norm(np.array(chosen) - v, axis=1)) >= sep:
                chosen.append(v)
                if len(chosen) == k:
                    return np.array(chosen)
        if sep <= 1e-12:
            return np.array(chosen)
        sep /= 2.0


def _seed_kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (as cv2.KMEANS_PP_CENTERS); works for float features."""
    idx = [rng.integers(len(X))]
    d2 = _sqdist(X, X[idx]).ravel()
    for _ in range(1, k):
        total = d2.sum()
        if total <= 0:
            break
        idx.append(rng.choice(len(X), p=d2 / total))
        d2 = np.minimum(d2, _sqdist(X, X[idx[-1:]]).ravel())
    return X[idx].copy()


# --------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------

class ModifiedFCM:
    """Modified Fuzzy C-Means (spatial centre-of-gravity term + G_ki).

    Parameters
    ----------
    n_clusters : number of clusters K
    fuzziness : m > 1 (2 is standard)
    use_spatial : penalise membership by spatial distance to each cluster's
        centre of gravity (the ISM 2010 modification)
    use_gki : add the fuzzy local neighbourhood factor G_ki
    window : neighbourhood radius for G_ki (1 => 3x3, 2 => 5x5, ...)
    max_iter, tol : stop after max_iter or when max |delta U| < tol
    init : "top_colours" (original method; best for 8-bit images),
        "kmeans++", "random", or a (K, C) array of initial centres
    seed : int or numpy Generator for reproducible runs
    """

    def __init__(self, n_clusters: int = 5, fuzziness: float = 2.0,
                 use_spatial: bool = False, use_gki: bool = False,
                 window: int = 1, max_iter: int = 20, tol: float = 1e-5,
                 init: str | np.ndarray = "top_colours",
                 seed: int | np.random.Generator | None = None,
                 verbose: bool = False):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if fuzziness <= 1.0:
            raise ValueError("fuzziness (m) must be > 1")
        if window < 1:
            raise ValueError("window radius must be >= 1")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.n_clusters = n_clusters
        self.m = fuzziness
        self.use_spatial = use_spatial
        self.use_gki = use_gki
        self.window = window
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.result_: FCMResult | None = None

    # ---- update steps ----------------------------------------------------

    def _memberships(self, D: np.ndarray) -> np.ndarray:
        """u_ik = 1 / sum_j (D_ik / D_ij)^(1/(m-1)), D = squared dissimilarity."""
        D = np.maximum(D, _EPS)
        inv = (D / D.min(axis=1, keepdims=True)) ** (-1.0 / (self.m - 1.0))
        return inv / inv.sum(axis=1, keepdims=True)

    @staticmethod
    def _spatial_penalty(U: np.ndarray, xy: np.ndarray, P: np.ndarray) -> np.ndarray:
        """U - H with H_ik = s_ik / (1 + sum_j s_ij), clipped and renormalised.

        The C# version min-max rescaled each row afterwards, forcing one
        cluster to exactly 0 and producing NaN rows when entries were equal.
        """
        dx = xy[:, 0:1] - P[None, :, 0]
        dy = xy[:, 1:2] - P[None, :, 1]
        S = np.sqrt(dx * dx + dy * dy)
        H = S / (1.0 + S.sum(axis=1, keepdims=True))
        Up = np.clip(U - H, 0.0, None)
        sums = Up.sum(axis=1, keepdims=True)
        dead = sums[:, 0] <= _EPS
        Up[~dead] /= sums[~dead]
        Up[dead] = U[dead]
        return Up

    def _gki(self, U: np.ndarray, D2: np.ndarray, h: int, w: int) -> np.ndarray:
        """G_ki = sum_{j in N_i, j != i} 1/(d_ij+1) (1-u_kj)^m ||x_j - v_k||^2.

        Fixes from the C# version: d_ij is the true spatial distance
        (arguments were passed in the wrong order) and the last row/column
        are included.
        """
        k = U.shape[1]
        r = self.window
        term = ((1.0 - U) ** self.m * D2).reshape(h, w, k)
        padded = np.pad(term, ((r, r), (r, r), (0, 0)))   # zeros outside => no contribution
        G = np.zeros((h, w, k))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy or dx:
                    G += padded[r + dy:r + dy + h, r + dx:r + dx + w] / (np.hypot(dy, dx) + 1.0)
        return G.reshape(h * w, k)

    def _update_centres(self, X, xy, U, P_prev, lo, hi):
        Um = U ** self.m
        V = (Um.T @ X) / np.maximum(Um.sum(axis=0), _EPS)[:, None]
        labels = U.argmax(axis=1)
        P = P_prev.copy()
        for k in range(U.shape[1]):
            sel = labels == k
            if sel.any():               # empty cluster keeps its position
                P[k] = xy[sel].mean(axis=0)
        return np.clip(V, lo, hi), P

    def _initial_centres(self, X: np.ndarray, dtype: np.dtype) -> np.ndarray:
        k = self.n_clusters
        if isinstance(self.init, np.ndarray) or isinstance(self.init, (list, tuple)):
            V = np.asarray(self.init, dtype=np.float64)
            if V.ndim == 1:
                V = V[:, None]
            if V.shape[1] != X.shape[1]:
                raise ValueError(f"init centres have {V.shape[1]} channels, image has {X.shape[1]}")
            return V
        if self.init == "top_colours":
            # Separation of 1/16 of the value range (16 levels for 8-bit).
            span = float(np.ptp(X, axis=0).max()) if np.issubdtype(dtype, np.floating) else \
                float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 255.0
            return _seed_top_colours(X, k, span / 16.0)
        if self.init == "kmeans++":
            return _seed_kmeanspp(X, k, self.rng)
        if self.init == "random":
            return X[self.rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
        raise ValueError(f"unknown init {self.init!r}")

    # ---- driver ----------------------------------------------------------

    def fit(self, image: np.ndarray,
            callback: Callable[[FCMResult, float], bool | None] | None = None) -> FCMResult:
        """Cluster an (H, W) or (H, W, C) image.

        callback(result, delta) is called after each iteration; return True
        to stop early (used for live previews and cancelling).
        """
        X, h, w, dtype = _as_features(image)
        lo, hi = X.min(axis=0), X.max(axis=0)
        ys, xs = np.divmod(np.arange(h * w), w)
        xy = np.stack([xs, ys], axis=1).astype(np.float64)

        V = self._initial_centres(X, dtype)
        k = len(V)
        if k < self.n_clusters and self.verbose:
            print(f"only {k} distinct values in image; using {k} clusters")
        P = np.column_stack([self.rng.uniform(0, w, k), self.rng.uniform(0, h, k)])

        # Proper initial memberships (C# stored raw distances, inverting them).
        U = self._memberships(_sqdist(X, V))
        if self.use_spatial:
            U = self._spatial_penalty(U, xy, P)

        history: list[float] = []
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):
            V, P = self._update_centres(X, xy, U, P, lo, hi)
            D2 = _sqdist(X, V)
            G = self._gki(U, D2, h, w) if self.use_gki else 0.0
            # C# used the *spatial* distance in the ratio's denominator.
            U_new = self._memberships(D2 + G)
            if self.use_spatial:
                U_new = self._spatial_penalty(U_new, xy, P)

            delta = float(np.abs(U_new - U).max())
            U = U_new
            history.append(float(((U ** self.m) * (D2 + G)).sum()))
            if self.verbose:
                print(f"iter {it:3d}  J={history[-1]:.6e}  max|dU|={delta:.3e}")
            converged = delta < self.tol
            if callback is not None:
                snap = FCMResult(U.argmax(1).reshape(h, w).astype(np.int32), U,
                                 V.astype(np.float32), P, it, history, converged)
                if callback(snap, delta):
                    break
            if converged:
                break

        self.result_ = FCMResult(U.argmax(1).reshape(h, w).astype(np.int32), U,
                                 V.astype(np.float32), P, it, history, converged)
        return self.result_

    def fit_predict(self, image: np.ndarray) -> np.ndarray:
        return self.fit(image).labels


# --------------------------------------------------------------------------
# cv2.kmeans-style function
# --------------------------------------------------------------------------

def fuzzy_cmeans(data: np.ndarray, K: int,
                 criteria: Sequence | None = None, attempts: int = 1,
                 *, fuzziness: float = 2.0, use_spatial: bool = False,
                 use_gki: bool = False, window: int = 1,
                 init: str | np.ndarray = "top_colours",
                 seed: int | None = None):
    """Drop-in style counterpart to ``cv2.kmeans`` for images.

    >>> compactness, labels, centers, res = fuzzy_cmeans(
    ...     img, 4, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-5),
    ...     use_gki=True)
    >>> seg = centers[labels.ravel()].reshape(img.shape).astype(np.uint8)

    Unlike cv2.kmeans, ``data`` is the image itself (H, W[, C]) because the
    spatial and G_ki terms need pixel positions. ``labels`` is (H*W, 1)
    int32 and ``centers`` (K, C) float32, as cv2.kmeans returns; the full
    FCMResult (memberships, positions) is returned last. With attempts > 1
    the run with the lowest objective is kept.
    """
    max_iter, tol = 20, 1e-5
    if criteria is not None:
        ctype, cmax, ceps = criteria
        max_iter = int(cmax) if ctype & TERM_CRITERIA_MAX_ITER else 10_000
        tol = float(ceps) if ctype & TERM_CRITERIA_EPS else 0.0
    rng = np.random.default_rng(seed)
    best: FCMResult | None = None
    for _ in range(max(1, attempts)):
        res = ModifiedFCM(K, fuzziness, use_spatial, use_gki, window, max_iter,
                          tol, init, rng).fit(data)
        if best is None or res.compactness < best.compactness:
            best = res
    return best.compactness, best.labels.reshape(-1, 1), best.centers, best


# --------------------------------------------------------------------------
# Rendering (pure NumPy; outputs keep the input dtype and channel order)
# --------------------------------------------------------------------------

def _cast_like(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(dtype)
    return arr.astype(dtype)


def render_segmented(result: FCMResult, mark_centres: bool = False,
                     marker_colour: Sequence[float] | None = None,
                     dtype: np.dtype = np.uint8) -> np.ndarray:
    """Each pixel painted with its cluster centre.

    Returns (H, W, C), or (H, W) for single-channel input. If mark_centres,
    a 3x3 marker is drawn at each centre of gravity; marker_colour is in the
    input's channel order (default: red for 3-channel BGR, i.e. (0, 0, 255),
    or max value for greyscale). Pass (255, 0, 0) for RGB images.
    """
    out = result.centers[result.labels].astype(np.float64)
    c = out.shape[-1]
    if mark_centres:
        if marker_colour is None:
            marker_colour = (0, 0, 255) if c == 3 else (255,) * c
        h, w = result.labels.shape
        for x, y in result.positions:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                out[max(yi - 1, 0):yi + 2, max(xi - 1, 0):xi + 2] = marker_colour
    out = _cast_like(out, np.dtype(dtype))
    return out[..., 0] if c == 1 else out


def render_clusters(image: np.ndarray, result: FCMResult,
                    background: float = 0) -> list[np.ndarray]:
    """One image per cluster: original pixels where the label is k, else background."""
    image = np.asarray(image)
    sel = result.labels if image.ndim == 2 else result.labels[..., None]
    return [np.where(sel == k, image, np.asarray(background, image.dtype))
            for k in range(len(result.centers))]
