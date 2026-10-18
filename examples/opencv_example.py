"""Segment an image with OpenCV I/O and use the results as masks.

    python examples/opencv_example.py examples/test_noise_high.png
"""

import sys

import cv2
import numpy as np

import fcm_gki

path = sys.argv[1] if len(sys.argv) > 1 else "examples/test_noise_high.png"
img = cv2.imread(path)                       # BGR uint8, EXIF orientation applied
if img is None:
    sys.exit(f"cannot read {path}")

# 1) Object API
fcm = fcm_gki.ModifiedFCM(n_clusters=4, use_spatial=True, use_gki=True, window=1, seed=1)
res = fcm.fit(img)
cv2.imwrite("segmented.png", fcm_gki.render_segmented(res, mark_centres=True))

# 2) cv2.kmeans-style API, here clustering in Lab instead of BGR
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-5)
compactness, labels, centers, res_lab = fcm_gki.fuzzy_cmeans(lab, 4, criteria, use_gki=True, seed=1)
seg_lab = centers[labels.ravel()].reshape(lab.shape).astype(np.uint8)
cv2.imwrite("segmented_lab.png", cv2.cvtColor(seg_lab, cv2.COLOR_LAB2BGR))

# 3) Use a cluster as a hard mask and as a soft (fuzzy) mask
k = 0
cut = cv2.bitwise_and(img, img, mask=res.mask(k))
soft = res.membership(k)[..., None]          # (H, W, 1) float32 in [0, 1]
faded = (img * soft).astype(np.uint8)
contours, _ = cv2.findContours(res.mask(k), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imwrite("cluster0_cut.png", cut)
cv2.imwrite("cluster0_soft.png", faded)

print(f"{res.iterations} iterations, converged={res.converged}, "
      f"cluster {k}: {int((res.labels == k).sum())} px, {len(contours)} contour(s)")
