"""Modified Fuzzy C-Means clustering with spatial and G_ki terms.

Works directly on OpenCV / NumPy image arrays::

    import cv2, fcm_gki
    img = cv2.imread("photo.jpg")                     # BGR uint8
    res = fcm_gki.ModifiedFCM(4, use_gki=True, use_spatial=True).fit(img)
    cv2.imwrite("seg.png", fcm_gki.render_segmented(res))
"""

__version__ = "1.0.0"

from .core import (  # noqa: E402
    TERM_CRITERIA_COUNT,
    TERM_CRITERIA_EPS,
    TERM_CRITERIA_MAX_ITER,
    FCMResult,
    ModifiedFCM,
    fuzzy_cmeans,
    render_clusters,
    render_segmented,
)

__all__ = [
    "ModifiedFCM", "FCMResult", "fuzzy_cmeans", "render_segmented", "render_clusters",
    "TERM_CRITERIA_COUNT", "TERM_CRITERIA_MAX_ITER", "TERM_CRITERIA_EPS", "__version__",
]
