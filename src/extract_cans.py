"""Cut each can out of its studio photo -> RGBA PNG with clean, decontaminated edges.

Uses two rembg models: ISNet (crisp edges) + U2Net (solid interior), merged.
The label pixels themselves are never altered - only the background is removed.
"""
import glob, os, sys
import numpy as np
from PIL import Image, ImageFilter
from rembg import remove, new_session
import cv2

SRC = os.path.join(os.path.dirname(__file__), '..', 'assets', 'products')
DST = os.path.join(os.path.dirname(__file__), '..', 'assets', 'cutouts')
os.makedirs(DST, exist_ok=True)

s_isnet = new_session('isnet-general-use')
s_u2 = new_session('u2net')

for f in sorted(glob.glob(os.path.join(SRC, '*.png'))):
    name = os.path.basename(f)[:-4]
    im = Image.open(f).convert('RGB')
    rgb = np.asarray(im).astype(np.float32) / 255.0
    a_is = np.asarray(remove(im, session=s_isnet, only_mask=True)).astype(np.float32) / 255.0
    a_u2 = np.asarray(remove(im, session=s_u2, only_mask=True)).astype(np.float32) / 255.0

    # solid interior from u2net, eroded so its soft edge never leaks out
    core = (a_u2 > 0.5).astype(np.uint8)
    core = cv2.erode(core, np.ones((15, 15), np.uint8))
    # fill any holes in the interior
    h, w = core.shape
    ff = core.copy() * 255
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 128)
    core = ((ff != 128)).astype(np.float32)
    a = np.maximum(a_is, core)
    # keep only the largest component (drops reflection fragments / specks)
    bin_ = (a > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    keep = cv2.dilate((lab == big).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    a = a * keep
    # tighten edge by ~1px to kill background halo, then slight smoothing
    a = np.clip((a - 0.12) / 0.88, 0, 1)
    a = cv2.GaussianBlur(a, (3, 3), 0.7)

    # edge colour decontamination: pull interior colours outward into semi-transparent rim
    inner = (a > 0.985).astype(np.float32)
    inner = cv2.erode(inner, np.ones((5, 5), np.uint8))
    num = cv2.GaussianBlur(rgb * inner[..., None], (0, 0), 4)
    den = cv2.GaussianBlur(inner, (0, 0), 4)[..., None] + 1e-6
    fill = num / den
    edge = (a < 0.985)[..., None]
    out_rgb = np.where(edge, fill, rgb)

    ys, xs = np.where(a > 0.05)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 4
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(h, y1 + pad), min(w, x1 + pad)
    rgba = np.dstack([out_rgb, a])[y0:y1, x0:x1]
    Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(DST, name + '.png'))
    print(name, 'bbox', (x0, y0, x1, y1), 'size', (x1 - x0, y1 - y0))
