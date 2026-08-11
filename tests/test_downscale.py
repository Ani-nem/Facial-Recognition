"""Tests for detection downscaling and bbox rescaling."""

import numpy as np

from faceorg.embed import downscale, rescale_bboxes


def test_downscale_noop_when_disabled():
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    out, scale = downscale(img, 0)
    assert scale == 1.0
    assert out.shape == img.shape


def test_downscale_noop_when_already_small():
    img = np.zeros((800, 1000, 3), dtype=np.uint8)
    out, scale = downscale(img, 1400)
    assert scale == 1.0
    assert out.shape == img.shape


def test_downscale_reduces_longest_side():
    img = np.zeros((3000, 6000, 3), dtype=np.uint8)  # h, w
    out, scale = downscale(img, 1400)
    assert scale == 1400 / 6000
    assert max(out.shape[:2]) == 1400
    # aspect ratio preserved (within rounding)
    assert abs(out.shape[1] / out.shape[0] - 2.0) < 0.01


def test_rescale_bboxes_roundtrips_to_original_scale():
    # A face at (100,200,180,120) in a full image, detected on a half-size copy.
    scale = 0.5
    detected = [(50, 100, 90, 60)]  # half-scale coords
    restored = rescale_bboxes(detected, scale)
    assert restored == [(100, 200, 180, 120)]


def test_rescale_bboxes_noop_at_scale_one():
    boxes = [(1, 2, 3, 4)]
    assert rescale_bboxes(boxes, 1.0) is boxes
