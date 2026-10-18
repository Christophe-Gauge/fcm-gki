"""Tests for fcm_gki.core. Run with: pytest"""

from itertools import permutations

import numpy as np
import pytest

import fcm_gki
from fcm_gki import ModifiedFCM, fuzzy_cmeans, render_clusters, render_segmented

BGR = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}


def quadrants(h=60, w=80) -> tuple[np.ndarray, np.ndarray]:
    """Four-colour BGR test image and its ground-truth labels."""
    img = np.zeros((h, w, 3), np.uint8)
    gt = np.zeros((h, w), int)
    for k, (ys, xs) in enumerate([(slice(0, h // 2), slice(0, w // 2)),
                                  (slice(0, h // 2), slice(w // 2, w)),
                                  (slice(h // 2, h), slice(0, w // 2)),
                                  (slice(h // 2, h), slice(w // 2, w))]):
        img[ys, xs] = list(BGR.values())[k]
        gt[ys, xs] = k
    return img, gt


def add_noise(img, frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    mask = rng.random(img.shape[:2]) < frac
    out[mask] = rng.integers(0, 256, (mask.sum(), 3))
    return out


def accuracy(labels, gt, k):
    return max((np.array(p)[labels] == gt).mean() for p in permutations(range(k)))


def test_clean_image_exact():
    img, gt = quadrants()
    res = ModifiedFCM(4, seed=0).fit(img)
    assert accuracy(res.labels, gt, 4) == 1.0
    assert res.labels.dtype == np.int32 and res.labels.shape == gt.shape
    np.testing.assert_allclose(res.U.sum(1), 1.0, atol=1e-9)


def test_bgr_in_bgr_out_and_no_transpose():
    # Non-square, asymmetric: a red block in the top-left of a blue image.
    img = np.zeros((60, 100, 3), np.uint8)
    img[:] = BGR["blue"]
    img[5:20, 10:40] = BGR["red"]
    res = ModifiedFCM(2, use_gki=True, seed=0).fit(img)
    seg = render_segmented(res)
    assert seg.shape == img.shape and seg.dtype == np.uint8
    red = int(np.argmax(res.centers[:, 2]))
    assert ((res.labels == red) == (img[..., 2] > 128)).all()
    # centre of gravity of the red block is x = 24.5, y = 12
    np.testing.assert_allclose(res.positions[red], (24.5, 12.0), atol=0.5)


def test_gki_beats_plain_on_noise():
    img, gt = quadrants(80, 100)
    noisy = add_noise(img, 0.3)
    plain = accuracy(ModifiedFCM(4, seed=1).fit(noisy).labels, gt, 4)
    gki = accuracy(ModifiedFCM(4, use_gki=True, seed=1).fit(noisy).labels, gt, 4)
    both = accuracy(ModifiedFCM(4, use_gki=True, use_spatial=True, seed=1).fit(noisy).labels, gt, 4)
    assert gki > plain and gki > 0.98 and both > 0.98


def test_greyscale_2d_input():
    g = np.zeros((40, 50), np.uint8)
    g[:, 25:] = 200
    res = ModifiedFCM(2, seed=0).fit(g)
    seg = render_segmented(res)
    assert seg.shape == g.shape
    assert set(np.unique(seg)) == {0, 200}


def test_float_features_kmeanspp():
    img, gt = quadrants()
    lab_like = img.astype(np.float32) / 255.0 * 100.0
    res = ModifiedFCM(4, init="kmeans++", seed=3).fit(lab_like)
    assert accuracy(res.labels, gt, 4) == 1.0


def test_explicit_init_centres():
    img, gt = quadrants()
    res = ModifiedFCM(4, init=np.array(list(BGR.values()), float)).fit(img)
    assert (res.labels == gt).all()


def test_fuzzy_cmeans_kmeans_style():
    img, gt = quadrants()
    criteria = (fcm_gki.TERM_CRITERIA_EPS + fcm_gki.TERM_CRITERIA_MAX_ITER, 20, 1e-5)
    comp, labels, centers, res = fuzzy_cmeans(img, 4, criteria, attempts=2, seed=0)
    assert labels.shape == (img.shape[0] * img.shape[1], 1) and labels.dtype == np.int32
    assert centers.shape == (4, 3) and centers.dtype == np.float32
    seg = centers[labels.ravel()].reshape(img.shape).astype(np.uint8)
    np.testing.assert_array_equal(seg, img)
    assert comp == pytest.approx(res.compactness)


def test_callback_can_cancel():
    img, _ = quadrants()
    calls = []
    res = ModifiedFCM(4, use_gki=True, max_iter=50, tol=0, seed=0).fit(
        add_noise(img), callback=lambda r, d: calls.append(d) or len(calls) >= 3)
    assert res.iterations == 3 and not res.converged


def test_memberships_masks_and_cluster_images():
    img, gt = quadrants()
    res = ModifiedFCM(4, seed=0).fit(img)
    m = res.memberships
    assert m.shape == (60, 80, 4) and m.dtype == np.float32
    assert res.mask(0).dtype == np.uint8 and set(np.unique(res.mask(0))) <= {0, 255}
    clusters = render_clusters(img, res)
    np.testing.assert_array_equal(sum(c.astype(int) for c in clusters), img)


def test_fewer_colours_than_clusters():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, 5:] = 255
    res = ModifiedFCM(5, seed=0).fit(img)
    assert len(res.centers) == 2


@pytest.mark.parametrize("bad", [dict(fuzziness=1.0), dict(window=0), dict(n_clusters=0)])
def test_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        ModifiedFCM(**{"n_clusters": 3, **bad})
