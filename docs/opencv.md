# Using fcm-gki with OpenCV

`fcm_gki` takes and returns plain NumPy arrays in OpenCV's layout, so it needs no conversion
code. The runnable version of these recipes is [examples/opencv_example.py](../examples/opencv_example.py).

## Conventions

| | OpenCV | fcm_gki |
|---|---|---|
| Colour image | (H, W, 3) BGR `uint8` | accepted as-is; outputs stay BGR |
| Greyscale | (H, W) `uint8` | accepted; outputs are (H, W) |
| Float images | (H, W, C) `float32` | accepted; use `init="kmeans++"` and `render_segmented(..., dtype=np.float32)` |
| Masks | (H, W) `uint8`, 0/255 | `res.mask(k)` |
| Termination criteria | `(type, max_iter, eps)` | same tuple in `fuzzy_cmeans` |
| Labels / centres | (N, 1) `int32` / (K, C) `float32` | same from `fuzzy_cmeans` |

`cv2.imread` applies EXIF orientation by default, so phone photos arrive upright. Pass
`cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION` only if you specifically want raw sensor
orientation.

## Recipes

**Segment and save**

```python
img = cv2.imread("in.jpg")
res = fcm_gki.ModifiedFCM(4, use_gki=True, use_spatial=True, seed=0).fit(img)
cv2.imwrite("seg.png", fcm_gki.render_segmented(res, mark_centres=True))
```

**Cluster in Lab, display in BGR**

```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
res = fcm_gki.ModifiedFCM(4, use_gki=True).fit(lab)
seg = cv2.cvtColor(fcm_gki.render_segmented(res), cv2.COLOR_LAB2BGR)
```

**Cut out a cluster, find its contours**

```python
mask = res.mask(k)
cut = cv2.bitwise_and(img, img, mask=mask)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Fuzzy (soft) mask for blending or matting**

```python
alpha = cv2.GaussianBlur(res.membership(k), (5, 5), 0)[..., None]   # float32 in [0, 1]
blend = (img * alpha + background * (1 - alpha)).astype(np.uint8)
```

**Clean up the hard labels**

```python
mask = cv2.morphologyEx(res.mask(k), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
```

**Large images:** cost grows with pixels × clusters. Downscale first, then upscale the labels
with nearest-neighbour interpolation:

```python
small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
res = fcm_gki.ModifiedFCM(5, use_gki=True).fit(small)
labels = cv2.resize(res.labels.astype(np.uint8), img.shape[1::-1],
                    interpolation=cv2.INTER_NEAREST)
```

**Video frames:** seed each frame with the previous frame's centres so cluster k keeps the same
meaning from frame to frame:

```python
centres = None
for frame in frames:
    fcm = fcm_gki.ModifiedFCM(4, use_gki=True,
                              init=centres if centres is not None else "top_colours")
    res = fcm.fit(frame)
    centres = res.centers
```

## Towards a native OpenCV module

The library is built so that a C++ port stays mechanical:

- **Data:** the core is pure NumPy on `(H·W, K)` matrices, so each update maps directly to
  `cv::Mat` operations.
- **G<sub>ki</sub>:** the factor is a sum of shifted arrays over the window. In C++ this is a
  `cv::filter2D` per cluster, with the kernel 1/(d+1) and the centre set to 0, applied to the
  (1−U)<sup>m</sup>·D² plane. That is fast and border-aware with `BORDER_CONSTANT`.
- **API shape:** `fuzzy_cmeans` already mirrors `cv::kmeans`. A natural C++ signature is
  `double cv::ximgproc::fuzzyCMeans(InputArray image, int K, OutputArray labels, TermCriteria
  criteria, int attempts, OutputArray centers, OutputArray memberships, const
  FuzzyCMeansParams& params)`, where `ximgproc` in opencv_contrib is the usual home for
  segmentation algorithms.
- **Tests:** the pytest suite in `tests/` doubles as a reference; a C++ port should reproduce
  `labels` exactly on the clean test images, and the accuracy figures on the noisy ones.

Licensing matters for upstreaming: opencv_contrib uses Apache 2.0, while this project is GPL v3.
The copyright holder would need to relicense the C++ port before a contrib pull request.
