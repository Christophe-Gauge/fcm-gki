"""Image loading/saving via Pillow (optional dependency: ``pip install fcm-gki[io]``).

OpenCV users don't need this module: ``cv2.imread`` already returns a BGR
uint8 array and applies EXIF orientation by default, and every function in
``fcm_gki.core`` accepts that array directly.
"""

from __future__ import annotations

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError as e:  # pragma: no cover
    raise ImportError("fcm_gki.io needs Pillow: pip install 'fcm-gki[io]'") from e

__all__ = ["load_image", "save_image"]


def load_image(path: str, max_dim: int | None = None, order: str = "RGB") -> np.ndarray:
    """Load an image as a uint8 array.

    order: "RGB" (H, W, 3), "BGR" (H, W, 3, OpenCV order) or "L" (H, W) grey.
    EXIF orientation is applied so phone photos are upright.
    max_dim: if set, downscale so the longest side is at most this.
    """
    order = order.upper()
    if order not in ("RGB", "BGR", "L"):
        raise ValueError("order must be 'RGB', 'BGR' or 'L'")
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L" if order == "L" else "RGB")
        if max_dim and max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8).copy()
    return arr[..., ::-1].copy() if order == "BGR" else arr


def save_image(arr: np.ndarray, path: str, order: str = "RGB") -> None:
    """Save an (H, W) or (H, W, 3) array; order says how its channels are arranged."""
    a = np.clip(np.rint(np.asarray(arr, dtype=np.float64)), 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 3 and order.upper() == "BGR":
        a = a[..., ::-1]
    Image.fromarray(np.ascontiguousarray(a)).save(path)
