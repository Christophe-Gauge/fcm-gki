# Algorithm

This page describes exactly what `fcm_gki.ModifiedFCM` computes. Notation:

| Symbol | Meaning |
|---|---|
| N | number of pixels (H × W) |
| K | number of clusters |
| x<sub>i</sub> | feature vector of pixel i (e.g. its B, G, R values) |
| v<sub>k</sub> | centre of cluster k in feature space |
| p<sub>i</sub> | position (x, y) of pixel i |
| c<sub>k</sub> | spatial centre of gravity of cluster k |
| u<sub>ik</sub> | membership of pixel i in cluster k; each pixel's memberships sum to 1 |
| m | fuzzifier, m > 1 (default 2) |

## 1. Standard Fuzzy C-Means

FCM minimises

$$J = \sum_{i=1}^{N}\sum_{k=1}^{K} u_{ik}^{m}\,\lVert x_i - v_k\rVert^2$$

by alternating two updates:

$$v_k = \frac{\sum_i u_{ik}^m\,x_i}{\sum_i u_{ik}^m}
\qquad
u_{ik} = \left[\sum_{j=1}^{K}\left(\frac{D_{ik}}{D_{ij}}\right)^{\frac{1}{m-1}}\right]^{-1}$$

where D<sub>ik</sub> = ‖x<sub>i</sub> − v<sub>k</sub>‖² (squared distance, hence the exponent
1/(m−1)). The implementation computes the membership in the equivalent normalised form
D<sub>ik</sub><sup>−1/(m−1)</sup> / Σ<sub>j</sub> D<sub>ij</sub><sup>−1/(m−1)</sup>. Each row is
first divided by its minimum, which avoids overflow, and D is floored at 1e-10, so a pixel that
exactly equals a centre gets membership ≈ 1 instead of a division by zero.

Plain FCM looks only at colour, so isolated noisy pixels are classified on their own and the
result is speckled.

## 2. Fuzzy local factor G<sub>ki</sub>

From Krinidis & Chatzis (FLICM). For each pixel i and cluster k:

$$G_{ki} = \sum_{\substack{j \in N_i\\ j \ne i}} \frac{1}{d_{ij} + 1}\,(1 - u_{jk})^m\,\lVert x_j - v_k\rVert^2$$

- N<sub>i</sub> is the (2r+1)×(2r+1) window around pixel i, where r is the `window` parameter.
- d<sub>ij</sub> is the spatial distance between pixels i and j (1 for edge neighbours, √2 for
  diagonals, and so on).
- Neighbours outside the image contribute nothing.

G<sub>ki</sub> is large when the neighbours of pixel i are far from cluster k in colour and do
not belong to it. It is added to the dissimilarity:

$$D_{ik} = \lVert x_i - v_k\rVert^2 + G_{ki}$$

and the membership formula above is unchanged. A noisy pixel inside a uniform region is
therefore pulled toward its neighbours' cluster. G is computed from the memberships of the
previous iteration and the freshly updated centres, as in FLICM. The objective reported in
`history` is Σ u<sup>m</sup>(‖x − v‖² + G).

The implementation builds G with one shifted, zero-padded array addition per window offset, so
the cost is O(N · K · (2r+1)²) with no per-pixel Python loop.

## 3. Spatial distance to cluster centre of gravity

The ISM 2010 modification. Each cluster has a centre of gravity c<sub>k</sub>: the mean position
of the pixels whose highest membership is k. Clusters with no pixels keep their previous
position. Initial positions are random.

With s<sub>ik</sub> = ‖p<sub>i</sub> − c<sub>k</sub>‖:

$$H_{ik} = \frac{s_{ik}}{1 + \sum_j s_{ij}},
\qquad
u'_{ik} = \frac{\max(u_{ik} - H_{ik},\,0)}{\sum_j \max(u_{ij} - H_{ij},\,0)}$$

Memberships of clusters whose centre of gravity is far from the pixel are reduced, then each row
is renormalised. If every entry of a row would be clipped to zero, the unpenalised row is kept.

**Assumption:** this term favours spatially compact regions. On a small object near the image
border it can grow a thin halo of the surrounding cluster into the object's cluster. Use it when
regions are blob-like; use G<sub>ki</sub> alone for thin or scattered structures.

## 4. Iteration

```
seed centres v (top colours / k-means++ / random); random positions c
U ← memberships from ‖x − v‖²            (spatial penalty applied if enabled)
repeat up to max_iter:
    v ← weighted means with weights u^m;   c ← centres of gravity of argmax labels
    D ← ‖x − v‖² (+ G if use_gki)
    U_new ← memberships(D)                 (spatial penalty applied if enabled)
    stop if max |U_new − U| < tol
    U ← U_new
labels ← argmax U
```

The stopping test uses the change in memberships because it does not depend on image size. The
original compared the change in J, a sum over every pixel, against 1e-5, which practically
never triggered.

## 5. Seeding

- **`top_colours`** (default, as in the original): the most frequent pixel values, skipping any
  candidate closer than 1/16 of the value range (16 levels for 8-bit images) to a value already
  chosen. If too few remain, the threshold is halved. If the image has fewer distinct values
  than K, K is reduced.
- **`kmeans++`**: D²-weighted sampling, as `cv2.KMEANS_PP_CENTERS` does. Recommended for float
  features (Lab, normalised data), where exact values rarely repeat.
- **`random`**: K distinct random pixels.
- **An array**: explicit (K, C) initial centres.

## 6. Choosing parameters

- **`window`:** 1 (3×3) is enough for salt-and-pepper noise; use 2–3 for heavier or blotchy
  noise, at the cost of softer boundaries and more time.
- **`fuzziness`:** 2 is standard. Values toward 1 make the result closer to k-means; larger
  values make memberships softer and convergence slower.
- **Colour space:** clustering in Lab (`cv2.COLOR_BGR2LAB`) often matches perceived colour better
  than BGR.
