"""Reel 5 prep: product cutouts for a dark, futuristic environment.

For each Red Bull photo: take the matte (can + the fruit at its base), tighten the
edge, drop small detached blobs (splash droplets / loose ice), and decontaminate the
edge colours so no bright photo background fringes the can on a dark set. The label
pixels themselves are never altered.

Output: assets/reel5/cutouts/<key>.png (RGBA)
"""
import os
import glob
import json
import numpy as np
import cv2
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A5 = os.path.join(ROOT, "assets", "reel5")
OUT = os.path.join(A5, "cutouts")
os.makedirs(OUT, exist_ok=True)
GEO = json.load(open(os.path.join(A5, "geometry.json")))
CAN_ONLY = {"03_yellow_tropical"}

for f in sorted(glob.glob(os.path.join(A5, "products", "*.png"))):
    key = os.path.basename(f)[:-4]
    g = GEO[key]
    rgb = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255
    a = np.asarray(Image.open(os.path.join(A5, "mattes", key + ".png"))).astype(np.float32) / 255
    h, w = a.shape
    a = np.clip((a - 0.25) / 0.6, 0, 1)
    # fill holes inside the can's own silhouette (low-contrast labels), never beyond it
    box = np.zeros((h, w), np.uint8)
    box[int(g["top"]):int(g["bottom"]) + 1, int(g["left"]) - 2:int(g["right"]) + 3] = 1
    sil = ((a > 0.5) & (box > 0)).astype(np.uint8)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) & box
    ff = sil.copy() * 255
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 128)
    sil = ((ff != 128) & (box > 0)).astype(np.float32)
    # straight-sided body of the can (below the curved top rim, above the curved bottom rim) is always solid
    body = np.zeros((h, w), np.float32)
    ch_ = g["bottom"] - g["top"]
    body[int(g["top"] + 0.12 * ch_):int(g["bottom"]) - 22, int(g["left"]) + 3:int(g["right"]) - 2] = 1
    sil = np.maximum(sil, body)
    a = np.maximum(a, cv2.GaussianBlur(sil, (0, 0), 0.8))
    if key in CAN_ONLY:   # no fruit at the base, only photo splash/ice: keep the can alone
        a = a * cv2.GaussianBlur(cv2.dilate(sil, np.ones((5, 5), np.uint8)), (0, 0), 1.0)
    # drop small detached blobs (droplets, loose ice); keep the can and sizeable props
    b = (a > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    keep = np.zeros_like(b)
    can_lab = lab[int((g["top"] + g["bottom"]) / 2), int(g["cx"])]
    for i in range(1, n):
        if i == can_lab or st[i, cv2.CC_STAT_AREA] > 6000:
            keep[lab == i] = 1
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8)).astype(np.float32)
    a = a * cv2.GaussianBlur(keep, (0, 0), 1.0)
    a = cv2.GaussianBlur(a, (0, 0), 0.7)
    # edge colour decontamination: spread interior colours into the soft rim
    inner = cv2.erode((a > 0.98).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    num = cv2.GaussianBlur(rgb * inner[..., None], (0, 0), 5)
    den = cv2.GaussianBlur(inner, (0, 0), 5)[..., None] + 1e-6
    fill = num / den
    edge = (a < 0.98)[..., None]
    out = np.where(edge, fill, rgb)
    Image.fromarray((np.dstack([np.clip(out, 0, 1), a]) * 255 + 0.5).astype(np.uint8), "RGBA").save(os.path.join(OUT, key + ".png"))
    print(key, "kept components:", int(keep.max()), "coverage %.3f" % a.mean())
