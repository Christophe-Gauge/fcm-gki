"""Round-trip with OpenCV, skipped when cv2 isn't installed."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fcm_gki import ModifiedFCM, fuzzy_cmeans, render_segmented  # noqa: E402


def test_imread_imwrite_roundtrip(tmp_path):
    img = np.zeros((50, 70, 3), np.uint8)
    img[:, :35] = (255, 0, 0)    # BGR blue
    img[:, 35:] = (0, 0, 255)    # BGR red
    p = tmp_path / "in.png"
    cv2.imwrite(str(p), img)
    loaded = cv2.imread(str(p))
    res = ModifiedFCM(2, use_gki=True, seed=0).fit(loaded)
    out = tmp_path / "seg.png"
    assert cv2.imwrite(str(out), render_segmented(res))
    np.testing.assert_array_equal(cv2.imread(str(out)), img)


def test_lab_colour_space_and_cv2_criteria():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:20] = (40, 160, 40)
    img[20:] = (200, 60, 60)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    _, labels, centers, _ = fuzzy_cmeans(lab, 2, criteria, use_gki=True, seed=0)
    seg_lab = centers[labels.ravel()].reshape(lab.shape).astype(np.uint8)
    back = cv2.cvtColor(seg_lab, cv2.COLOR_LAB2BGR)
    assert np.abs(back.astype(int) - img).max() <= 3


def test_membership_as_soft_mask():
    img = np.zeros((30, 30, 3), np.uint8)
    img[:, 15:] = 255
    res = ModifiedFCM(2, seed=0).fit(img)
    soft = res.membership(int(np.argmax(res.centers[:, 0])))
    blurred = cv2.GaussianBlur(soft, (5, 5), 0)   # float32 maps work with cv2 directly
    assert blurred.dtype == np.float32 and blurred[:, 25:].min() > 0.9
