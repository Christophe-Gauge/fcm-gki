# API reference

```python
import fcm_gki
```

All core functions take an image as a NumPy array of shape **(H, W)** or **(H, W, C)**, indexed
`[y, x]` (the OpenCV and NumPy convention). Any numeric dtype works. Channels are treated as
generic features, and outputs keep the input's channel order.

---

## `ModifiedFCM`

```python
ModifiedFCM(n_clusters=5, fuzziness=2.0, use_spatial=False, use_gki=False,
            window=1, max_iter=20, tol=1e-5, init="top_colours",
            seed=None, verbose=False)
```

| Argument | Type | Description |
|---|---|---|
| `n_clusters` | int ≥ 1 | number of clusters K |
| `fuzziness` | float > 1 | fuzzifier m |
| `use_spatial` | bool | enable the centre-of-gravity spatial penalty |
| `use_gki` | bool | enable the G<sub>ki</sub> neighbourhood factor |
| `window` | int ≥ 1 | G<sub>ki</sub> neighbourhood radius (1 → 3×3) |
| `max_iter` | int ≥ 1 | maximum iterations |
| `tol` | float | stop when the largest membership change < `tol` |
| `init` | str or array | `"top_colours"`, `"kmeans++"`, `"random"`, or a (K, C) array of centres |
| `seed` | int / `np.random.Generator` / None | randomness for seeding and initial positions |
| `verbose` | bool | print J and ΔU every iteration |

Raises `ValueError` for invalid parameters.

### `fit(image, callback=None) -> FCMResult`

Clusters the image and returns an `FCMResult`, which is also stored as `self.result_`.

`callback(result, delta)` is called after every iteration with a snapshot `FCMResult` and the
largest membership change. If it returns `True`, the run stops. Use this for progress bars, live
previews or cancellation:

```python
stop = threading.Event()
res = fcm.fit(img, callback=lambda r, d: stop.is_set())
```

### `fit_predict(image) -> ndarray`

Shorthand for `fit(image).labels`.

---

## `FCMResult`

| Attribute | Shape / type | Description |
|---|---|---|
| `labels` | (H, W) `int32` | hard labels (argmax of memberships) |
| `U` | (H·W, K) `float64` | memberships, row-major pixel order; rows sum to 1 |
| `centers` | (K, C) `float32` | cluster centres in the input's feature space |
| `positions` | (K, 2) `float64` | centres of gravity as (x, y) |
| `iterations` | int | iterations performed |
| `history` | list of float | objective J after each iteration |
| `converged` | bool | `True` if `tol` was reached |

| Member | Returns |
|---|---|
| `memberships` | (H, W, K) `float32` membership maps |
| `membership(k)` | (H, W) `float32` membership map of cluster k, usable as a soft mask |
| `mask(k)` | (H, W) `uint8` mask, 255 where the label is k (for `cv2.bitwise_and`, `cv2.findContours`) |
| `compactness` | final J, the counterpart of `cv2.kmeans`'s compactness |
| `shape` | (H, W) |

K can be smaller than requested if the image has fewer distinct values than `n_clusters`;
check `len(res.centers)`.

---

## `fuzzy_cmeans`

```python
compactness, labels, centers, result = fuzzy_cmeans(
    data, K, criteria=None, attempts=1, *,
    fuzziness=2.0, use_spatial=False, use_gki=False, window=1,
    init="top_colours", seed=None)
```

Mirrors `cv2.kmeans(data, K, None, criteria, attempts, flags)`:

- `criteria` is a `(type, max_iter, epsilon)` tuple using `cv2.TERM_CRITERIA_*` (or the
  identical `fcm_gki.TERM_CRITERIA_*`). Without `MAX_ITER` the iteration cap is 10 000; without
  `EPS` the run continues to the cap. The default is `(EPS + MAX_ITER, 20, 1e-5)`.
- `attempts` runs the algorithm several times and keeps the lowest objective.
- Returns `labels` as (H·W, 1) `int32` and `centers` as (K, C) `float32`, like `cv2.kmeans`,
  plus the full `FCMResult`.

**Difference from `cv2.kmeans`:** `data` is the image itself, (H, W[, C]), not an (N, C) sample
matrix, because the spatial and G<sub>ki</sub> terms need pixel positions.

---

## Rendering

### `render_segmented(result, mark_centres=False, marker_colour=None, dtype=np.uint8)`

Paints every pixel with its cluster centre. Returns (H, W, C), or (H, W) for single-channel
input, cast and clipped to `dtype`. If `mark_centres` is set, a 3×3 marker is drawn at each
centre of gravity. The marker colour is in the input's channel order: the default is
`(0, 0, 255)`, red in BGR; pass `(255, 0, 0)` for RGB images. For float feature spaces, pass
`dtype=np.float32`.

### `render_clusters(image, result, background=0) -> list[ndarray]`

Returns one image per cluster: the original pixels where the label is k, and `background`
elsewhere. The equivalent of the original app's `ClusterN.png` files.

---

## `fcm_gki.io` (needs Pillow)

| Function | Description |
|---|---|
| `load_image(path, max_dim=None, order="RGB")` | uint8 array in `"RGB"`, `"BGR"` or `"L"` (grey) order, with EXIF orientation applied and optional downscale |
| `save_image(arr, path, order="RGB")` | save an array whose channels are in `order` |

OpenCV users can use `cv2.imread` / `cv2.imwrite` instead.

---

## Command line: `fcm-gki`

```
fcm-gki IMAGE [-c K] [-m M] [--spatial] [--gki] [-w R] [--max-iter N] [--tol T]
              [--init {top_colours,kmeans++,random}] [--seed S] [--grey]
              [--max-dim PX] [--no-markers] [-o DIR] [-q]
```

Writes `segmented.png` and `Cluster0.png … ClusterN.png` to `DIR`. Centre markers are drawn when
`--spatial` is on, unless `--no-markers` is given.

## GUI: `fcm-gki-gui [IMAGE]`

See the README for a description of the window.
