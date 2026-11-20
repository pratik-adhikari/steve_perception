"""Appearance signature of a taught object: a colour histogram of its mask, to tell THIS object apart from
other objects the text prompt also finds (two mugs, a red and a tennis ball, ...).

    sig = appearance_signature(rgb, mask)            # list of floats, sums to 1
    s = similarity(sig, appearance_signature(rgb2, mask2))   # 0 (nothing alike) .. 1 (same colours)

Hue is meaningless for grey / white / black pixels, so the histogram has two parts:
  * chromatic pixels (S >= S_GREY, V >= V_DARK): H_BINS hue x S_BINS saturation bins
  * achromatic pixels (low S or dark): V_BINS brightness bins
A white mug and a grey mug differ in the brightness bins, a red and a blue mug in the hue bins.
Pure NumPy (OpenCV HSV convention: H 0..179, S and V 0..255), so it runs in the system Python
(detector) and the calibrator alike.
"""

import numpy as np

H_BINS, S_BINS, V_BINS = 18, 3, 4
S_GREY, V_DARK = 50, 40
SIZE = H_BINS * S_BINS + V_BINS


def rgb_to_hsv(rgb):
    """uint8 RGB -> uint8 HSV, OpenCV convention (H 0..179)."""
    try:
        import cv2
        return cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], np.uint8), cv2.COLOR_RGB2HSV)
    except ImportError:
        pass
    x = rgb[..., :3].astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm, gm = nz & (mx == r), nz & (mx == g) & (mx != r)
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return np.stack([np.round(h * 30) % 180, np.round(s * 255), np.round(mx * 255)], -1).astype(np.uint8)


def histogram(hsv_pixels):
    """N x 3 uint8 HSV pixels -> unnormalised histogram (SIZE,)."""
    p = np.asarray(hsv_pixels).reshape(-1, 3).astype(np.int32)
    h, s, v = p[:, 0], p[:, 1], p[:, 2]
    chroma = (s >= S_GREY) & (v >= V_DARK)
    hb = np.minimum(h[chroma] * H_BINS // 180, H_BINS - 1)
    sb = np.minimum((s[chroma] - S_GREY) * S_BINS // (256 - S_GREY), S_BINS - 1)
    vb = np.minimum(v[~chroma] * V_BINS // 256, V_BINS - 1)
    out = np.zeros(SIZE)
    np.add.at(out, hb * S_BINS + sb, 1.0)
    np.add.at(out, H_BINS * S_BINS + vb, 1.0)
    return out


def _smooth(hist):
    """Spread each chromatic bin a little to its hue neighbours (circular), so a slight hue shift under other
    lighting does not drop the similarity to zero."""
    c = hist[:H_BINS * S_BINS].reshape(H_BINS, S_BINS)
    c = 0.5 * c + 0.25 * np.roll(c, 1, axis=0) + 0.25 * np.roll(c, -1, axis=0)
    return np.concatenate([c.ravel(), hist[H_BINS * S_BINS:]])


def appearance_signature(rgb, mask, erode_px=2):
    """Normalised signature (list of SIZE floats) of the mask's pixels; None if the mask is (almost) empty.
    The rim is eroded away: there SAM's mask mixes in background pixels."""
    mask = np.asarray(mask, bool)
    if erode_px:
        try:
            import cv2
            k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
            inner = cv2.erode(mask.astype(np.uint8), k) > 0
            if inner.sum() >= 30:
                mask = inner
        except ImportError:
            pass
    if mask.sum() < 20:
        return None
    return normalise(histogram(rgb_to_hsv(rgb)[mask]))


def normalise(hist):
    hist = np.asarray(hist, float)
    t = hist.sum()
    return (hist / t).tolist() if t > 0 else None


def similarity(a, b):
    """Bhattacharyya coefficient of two signatures (lightly hue-smoothed): 1 = same colour distribution."""
    if a is None or b is None or len(a) != SIZE or len(b) != SIZE:
        return 0.0
    pa, pb = _smooth(np.asarray(a, float)), _smooth(np.asarray(b, float))
    pa, pb = pa / max(pa.sum(), 1e-12), pb / max(pb.sum(), 1e-12)
    return float(np.sqrt(pa * pb).sum())
