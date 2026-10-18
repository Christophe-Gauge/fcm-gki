# Porting notes: C# → Python

This port is based on `FuzzyCMeansClusteringGKI/CMeansAlgorithm.cs` and `Form1.cs`. Every
behavioural difference from the C# code is listed here, with the reason.

## Orientation ("rotation") bugs

| # | C# behaviour | Effect | Fix |
|---|---|---|---|
| 1 | In the G<sub>ki</sub> loop, `CalculateEuclideanDistance(w, h, ww, wh)` is passed to a function with signature `(X1, X2, Y1, Y2)`, so it computes √((x−y)² + (x′−y′)²) | neighbour weights depend on the diagonal position of the pixel, not its distance from the neighbour, so smoothing is skewed along one diagonal | weight is 1/(√(dx² + dy²) + 1) for the true offset |
| 2 | Points are generated column-major (outer loop over x), with index `x·H + y` | correct in C#, but a naive port to NumPy's row-major (H, W) arrays transposes the image | one row-major convention everywhere; flat index n ↔ (y, x) = divmod(n, W) |
| 3 | `System.Drawing.Bitmap` ignores the EXIF orientation tag | phone photos are processed and saved sideways | `cv2.imread` applies EXIF orientation; `fcm_gki.io.load_image` calls `ImageOps.exif_transpose` |

## Algorithm bugs

| # | C# behaviour | Fix |
|---|---|---|
| 4 | The constructor stores raw colour **distances** in `U` and normalises them, so on the first pass the *farthest* cluster gets the highest membership | proper FCM memberships from the start |
| 5 | In `Step()`, the denominator of the membership ratio uses `CalculateEuclideanDistance2`, the **spatial** distance to each centroid, while the numerator uses colour distance | colour distance (plus G) in both |
| 6 | The non-GKI path mixes unsquared numerator and denominator with exponent 1/(m−1), and both paths raise the *sum* of ratios to the power instead of each ratio. This is harmless only at m = 2 | standard formula, correct for any m |
| 7 | `if (top < 1.0) top = Eps;` clamps only the numerator, so the denominator could be 0 and produce ∞ and then U = 0 | all dissimilarities floored at 1e-10 |
| 8 | `RecalculateClusterIndexes` starts `min` and `max` at 0, then min-max rescales each row. One cluster is always forced to membership 0, and equal entries give 0/0 = NaN, which is then set to 0 so the row sums to 0 | `max(U − H, 0)` renormalised to sum 1, falling back to `U` if the row vanishes |
| 9 | The G<sub>ki</sub> loops run to `W − 1` and `H − 1`, skipping the last column and row (which keep G = 0) and excluding them as neighbours | whole image processed |
| 10 | The centroid position uses `PixelCount = 1` as its starting count, which biases every centre toward (0, 0); an empty cluster jumps to (0, 0) | true mean; empty clusters keep their previous position |
| 11 | Hard assignment uses `U[i, j] == p.ClusterIndex` (float equality with the row maximum), so ties paint a pixel into several clusters | `argmax` |
| 12 | Convergence: `|J_old − J_new| < 1e-5` on an objective that sums over every pixel, which essentially never triggers, so every run hits max iterations | stop on max \|ΔU\| < tol; `converged` flag reported |
| 13 | J is computed without G even when G<sub>ki</sub> is on | J includes G |
| 14 | The centroid colour is truncated with a `(byte)` cast, and gives NaN when the membership sum is 0 | rounded and clamped; zero sums guarded |
| 15 | Seeds are the K most frequent *exact* colours, which in noisy images are often near-duplicates and leave one region without a seed | top colours with a minimum separation; `kmeans++` and explicit seeds also available |

## Structural changes

- `CMeansAlgorithm` created a hidden `new Form1()` as a field (`blah`), which built a second,
  invisible window for every run. It was removed; the algorithm has no UI dependency.
- Progress and cancel go through `fit(callback=...)` instead of `BackgroundWorker`.
- Per-pixel `GetPixel` / `SetPixel` loops became vectorised NumPy.
- `ColourCount`, `ClusterImage` and `ClusterPoint` are replaced by arrays.
- The unused `G`, `H` fields, `maxDist`, `Dispose` and other dead code are dropped.

## Behaviour that is kept on purpose

- The spatial penalty H<sub>ik</sub> = s<sub>ik</sub> / (1 + Σ<sub>j</sub> s<sub>ij</sub>) and
  its subtraction from U, which is the paper's modification. Only the renormalisation changed
  (bug 8).
- Random initial cluster positions, and the "top colours" seeding as the default.
- Default parameters from the form: 5 clusters, 20 iterations, precision 1e-5, window 1, m = 2.
- Output files `segmented.png` and `ClusterN.png`, with red markers at the cluster centres when
  the spatial factor is on.

## Verification

On the repository's `test_noise_low.png` and `test_noise_high.png`, scored against
`test_orig.png`:

| Mode | low | high |
|---|---|---|
| plain | 0.957 | 0.876 |
| spatial | 0.971 | 0.920 |
| G<sub>ki</sub> | 1.000 | 0.998 |
| spatial + G<sub>ki</sub> | 1.000 | 0.999 |

`tests/test_core.py::test_bgr_in_bgr_out_and_no_transpose` checks orientation on a non-square
image: the labels must match pixel-for-pixel and the centre of gravity must land at the known
(x, y).
