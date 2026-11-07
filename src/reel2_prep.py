"""Reel 2 prep: build a matte (can + props touching its base) for each product
photo and measure the can's geometry. The photo itself is never altered; the
matte only tells the renderer which pixels are "in front" so effects can pass
behind the can.

Outputs: assets/reel2/mattes/<key>.png and assets/reel2/geometry.json
"""
import os
import glob
import json
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

import sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REEL = sys.argv[1] if len(sys.argv) > 1 else "reel2"      # e.g. "reel3"
ROBUST = "--robust" in sys.argv   # measure the can as the solid column through the photo centre and keep
                                  # only props near its base in the matte (splashes touching the can are dropped)
SRC = os.path.join(ROOT, "assets", REEL, "products")
DST = os.path.join(ROOT, "assets", REEL, "mattes")
os.makedirs(DST, exist_ok=True)

s_u2 = new_session("u2net")
s_is = new_session("isnet-general-use")
geo = {}
for f in sorted(glob.glob(os.path.join(SRC, "*.png"))):
    key = os.path.basename(f)[:-4]
    im = Image.open(f).convert("RGB")
    h, w = im.size[1], im.size[0]
    a_u2 = np.asarray(remove(im, session=s_u2, only_mask=True)).astype(np.float32) / 255
    a_is = np.asarray(remove(im, session=s_is, only_mask=True)).astype(np.float32) / 255

    core = (a_u2 > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    core = (lab == big)
    rows = np.where(core.any(1))[0]
    y0, y1 = rows[0], rows[-1]
    mid = range(int(y0 + 0.30 * (y1 - y0)), int(y0 + 0.60 * (y1 - y0)))
    L = np.median([np.argmax(core[y]) for y in mid])
    R = np.median([w - 1 - np.argmax(core[y, ::-1]) for y in mid])
    cw = R - L
    # top: first row whose span covers most of the can width
    top = next(y for y in rows if core[y, int(L):int(R)].mean() > 0.55)
    # bottom: last row where the central part of the can is still solid
    c0, c1 = int(L + 0.3 * cw), int(R - 0.3 * cw)
    bottom = max(y for y in rows if core[y, c0:c1].mean() > 0.9)

    if ROBUST:
        # can sides = the two long straight vertical edges: signed horizontal gradient averaged over
        # the mid rows is coherent along a straight edge and cancels out on splashes/background
        gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F, 1, 0, ksize=3)
        prof = np.abs(gx[int(0.30 * h):int(0.70 * h)].mean(0))
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        xl = np.arange(int(0.20 * w), int(0.42 * w))
        xr = np.arange(int(0.58 * w), int(0.80 * w))
        if not (380 <= R - L <= 480):     # mask outline merged with props -> use the straight-edge estimate
            L = float(xl[np.argmax(prof[xl])])
            R = float(xr[np.argmax(prof[xr])])
        cw = R - L
        c0, c1 = int(L + 0.2 * cw), int(R - 0.2 * cw)
        solid = core[:, c0:c1].mean(1) > 0.9
        top = int(np.argmax(solid))
        bottom = int(len(solid) - 1 - np.argmax(solid[::-1]))
    # matte = u2net (sharpened) + isnet props in a band around the can base
    m = np.clip((a_u2 - 0.25) / 0.5, 0, 1)
    band = np.zeros((h, w), np.float32)
    by0, by1 = int(bottom - 300), int(min(h, bottom + 140))
    bx0, bx1 = int(max(0, L - 380)), int(min(w, R + 380))
    band[by0:by1, bx0:bx1] = 1
    band = cv2.GaussianBlur(band, (0, 0), 25)
    props = np.clip((a_is - 0.2) / 0.6, 0, 1) * band
    if ROBUST:   # above the base band keep only the can column (drops attached splashes / cups)
        col = np.zeros((h, w), np.float32)
        col[:, int(L) - 12:int(R) + 13] = 1
        keep_zone = np.maximum(col, (np.arange(h)[:, None] > bottom - 300).astype(np.float32))
        m = m * cv2.GaussianBlur(keep_zone, (0, 0), 4)
    m = np.maximum(m, props)
    # drop specks
    b = (m > 0.4).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    keep = np.zeros_like(b)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 1500:
            keep[lab == i] = 1
    keep = cv2.dilate(keep, np.ones((7, 7), np.uint8))
    m = m * keep
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(os.path.join(DST, key + ".png"))
    geo[key] = dict(left=float(L), right=float(R), top=int(top), bottom=int(bottom), cx=float((L + R) / 2), w=w, h=h)
    print(key, geo[key])

json.dump(geo, open(os.path.join(ROOT, "assets", REEL, "geometry.json"), "w"), indent=1)
