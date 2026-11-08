"""Reel 5 effects library - the "energy universe" look (built new for this reel).

Product compositing (cutout can + lighting + reflection), electromagnetic rings, speed tunnel,
metallic shockwave, iris shutter, CRT collapse, hologram, particle assemble/dissolve, glitch,
racing grid + light trails, carbon fibre, brushed metal, hex grid, prism beams, tesla arcs,
wind-tunnel streamlines, zero-g particles, turbine, HUD, decoding text, motorsport price badge.
"""
import os
import json
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image, ImageDraw

import fx
from fx import lin, smooth, decay, ease_out_cubic, ease_in_cubic, ease_in_out, ease_out_expo, lerp, add_glow, draw_text, fit_size

W, H = 1080, 1920
ROOT = fx.ROOT
A5 = os.path.join(ROOT, "assets", "reel5")
GEO = json.load(open(os.path.join(A5, "geometry.json")))
SAFE_BOTTOM = 1626           # no product pixel below this line
BADGE_W, BADGE_H, BADGE_CY = 440, 108, 1712   # badge occupies y 1658-1766
CAN_SCREEN_H = 1150.0

cv2.setNumThreads(1)


def half():
    return np.zeros((H // 2, W // 2, 3), np.float32)


def glow_add(canvas, lay_half, core=1.0, g1=3.0, g2=12.0, k1=1.0, k2=1.2, gain=1.0):
    """Add a half-res light layer to the canvas with a two-radius glow."""
    g = lay_half * core + cv2.GaussianBlur(lay_half, (0, 0), g1) * k1 + cv2.GaussianBlur(lay_half, (0, 0), g2) * k2
    canvas += cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) * gain
    return canvas


def soft_blur(img, sigma):
    """Fast large blur (downsample -> blur -> upsample)."""
    if sigma < 0.6:
        return img
    if sigma < 4:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    s = 4
    small = cv2.resize(img, (W // s, H // s), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / s)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


# ================================================================ camera
def affine(z, fx_, fy_, ax, ay, roll=0.0):
    c, s = np.cos(np.radians(roll)) * z, np.sin(np.radians(roll)) * z
    return np.array([[c, -s, ax - (c * fx_ - s * fy_)], [s, c, ay - (s * fx_ + c * fy_)]], np.float64)


def apply(M, x, y):
    return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]


# ================================================================ product
class Can5:
    def __init__(self, key):
        im = np.asarray(Image.open(os.path.join(A5, "cutouts", key + ".png")).convert("RGBA")).astype(np.float32) / 255
        a, rgb = im[..., 3], im[..., :3]
        g = GEO[key]
        self.key = key
        self.h, self.w = a.shape
        self.cx, self.top, self.bot = g["cx"], g["top"], g["bottom"]
        self.left, self.right = g["left"], g["right"]
        self.R = (self.right - self.left) / 2
        rows = np.where(a.max(1) > 0.3)[0]
        cols = np.where(a.max(0) > 0.3)[0]
        self.pbot, self.ptop = int(rows[-1]), int(rows[0])
        self.pleft, self.pright = int(cols[0]), int(cols[-1])
        box = np.zeros_like(a)
        box[int(self.top) + 2:int(self.bot) - 1, int(self.left) + 1:int(self.right)] = 1
        canm = a * cv2.GaussianBlur(box, (0, 0), 3)
        u = np.clip((np.arange(self.w, dtype=np.float32) - self.cx) / self.R, -1, 1)
        th = np.broadcast_to(np.arcsin(u)[None, :], a.shape).astype(np.float32)
        dist = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        rim = (np.exp(-dist / 6.0) * canm).astype(np.float32)
        self.lv = {1: (np.dstack([rgb * a[..., None], a]).astype(np.float32), np.dstack([canm, th, rim]).astype(np.float32))}
        # particle samples (world coords + colours) for assemble/dissolve effects
        r = np.random.default_rng(len(key))
        ys, xs = np.where(a > 0.6)
        idx = r.choice(len(ys), 2600, replace=False)
        self.px, self.py = xs[idx].astype(np.float32), ys[idx].astype(np.float32)
        self.pcol = np.clip(rgb[ys[idx], xs[idx]] * 1.15 + 0.08, 0, 1)
        self.pdelay = r.uniform(0, 0.4, len(idx)).astype(np.float32)
        ang = r.uniform(0, 2 * np.pi, len(idx))
        dist_ = r.uniform(350, 1000, len(idx))
        self.pdx, self.pdy = (np.cos(ang) * dist_).astype(np.float32), (np.sin(ang) * dist_).astype(np.float32)

    def level(self, k):
        if k not in self.lv:
            rgba, aux = self.lv[1]
            up = cv2.resize(rgba, (self.w * k, self.h * k), interpolation=cv2.INTER_LANCZOS4)
            bl = cv2.GaussianBlur(up[..., :3], (0, 0), 1.2)
            up[..., :3] = np.clip(up[..., :3] + 0.55 * (up[..., :3] - bl), 0, None)
            up[..., 3] = np.clip(up[..., 3], 0, 1)
            self.lv[k] = (up, cv2.resize(aux, (self.w * k, self.h * k), interpolation=cv2.INTER_LINEAR))
        return self.lv[k]

    def wide_z(self):
        return CAN_SCREEN_H / (self.bot - self.top)

    def cam(self, zmul=1.0, dx=0.0, lift=0.0, roll=0.0, ax=540.0):
        """Default framing: lowest product pixel sits on SAFE_BOTTOM (clear of the price badge)."""
        z = self.wide_z() * zmul
        return affine(z, self.cx + dx, self.pbot, ax, SAFE_BOTTOM - lift, roll)

    def can_box(self, M):
        """Screen bbox of the can column (x0, y0, x1, y1)."""
        pts = [apply(M, x, y) for x in (self.left, self.right) for y in (self.top, self.bot)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def layer(self, M, expo=0.95, sweep=(), rim_l=None, rim_r=None, flash=0.0, flash_col=(1, 1, 1), tint=None,
              vgrad=0.0, alpha=1.0, pad=6):
        z = float(np.hypot(M[0, 0], M[1, 0]))
        k = 2 if z > 1.25 else 1
        rgba, aux = self.level(k)
        Mk = M.copy()
        Mk[:, :2] /= k
        hh, ww = rgba.shape[:2]
        cs = np.array([[0, 0], [ww, 0], [0, hh], [ww, hh]], np.float64) @ Mk[:, :2].T + Mk[:, 2]
        x0, y0 = int(max(0, np.floor(cs[:, 0].min()) - pad)), int(max(0, np.floor(cs[:, 1].min()) - pad))
        x1, y1 = int(min(W, np.ceil(cs[:, 0].max()) + pad)), int(min(H, np.ceil(cs[:, 1].max()) + pad))
        if x1 <= x0 or y1 <= y0:
            return None
        Mt = Mk.copy()
        Mt[0, 2] -= x0
        Mt[1, 2] -= y0
        size = (x1 - x0, y1 - y0)
        L = cv2.warpAffine(rgba, Mt, size, flags=cv2.INTER_CUBIC if k == 2 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        X = cv2.warpAffine(aux, Mt, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = np.clip(L[..., 3], 0, 1)
        rgb = np.clip(L[..., :3], 0, None) * expo
        canm, th, rim = X[..., 0], X[..., 1], X[..., 2]
        if tint is not None:
            rgb = rgb * np.array(tint, np.float32)
        if vgrad:
            yy = np.linspace(0, 1, size[1], dtype=np.float32)[:, None, None]
            rgb = rgb * (1 - vgrad * yy ** 2)
        for (pos, width, strength, col) in sweep:
            band = np.exp(-((th - pos) / width) ** 2) * strength * canm
            rgb = rgb + (band * a)[..., None] * np.array(col, np.float32)
        side = np.sin(th)
        if rim_l is not None:
            rgb = rgb + (rim * np.clip(-side, 0, 1) ** 2 * a)[..., None] * np.array(rim_l, np.float32)
        if rim_r is not None:
            rgb = rgb + (rim * np.clip(side, 0, 1) ** 2 * a)[..., None] * np.array(rim_r, np.float32)
        if flash > 0:
            rgb = rgb + a[..., None] * flash * np.array(flash_col, np.float32)
        return dict(x0=x0, y0=y0, rgb=rgb * alpha, a=a * alpha, M=M)


def blit(canvas, L, additive=False):
    if L is None:
        return
    h, w = L["a"].shape
    roi = canvas[L["y0"]:L["y0"] + h, L["x0"]:L["x0"] + w]
    if not additive:
        roi *= (1 - L["a"][..., None])
    roi += L["rgb"]


def reflection(canvas, L, floor_y, strength=0.22, fade=240.0):
    if L is None:
        return
    h, w = L["a"].shape
    top = int(2 * floor_y - (L["y0"] + h))
    rgb, a = L["rgb"][::-1], L["a"][::-1]
    ys = top + np.arange(h)
    f = np.clip(1 - (ys - floor_y) / fade, 0, 1) ** 2 * (ys >= floor_y) * strength
    rgb, a = rgb * f[:, None, None], a * f[:, None]
    rgb, a = cv2.GaussianBlur(rgb, (0, 0), 2.5), cv2.GaussianBlur(a, (0, 0), 2.5)
    y0c, y1c = max(0, top), min(H, top + h)
    if y1c <= y0c:
        return
    roi = canvas[y0c:y1c, L["x0"]:L["x0"] + w]
    roi *= (1 - a[y0c - top:y1c - top, :, None])
    roi += rgb[y0c - top:y1c - top]


# ================================================================ static textures
@lru_cache(maxsize=1)
def carbon():
    n = 22
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ci, cj = (xx // n).astype(int), (yy // n).astype(int)
    fx_, fy_ = (xx % n) / n, (yy % n) / n
    horiz = ((ci + cj) % 2 == 0)
    bulge = np.where(horiz, np.cos(np.pi * (fy_ - 0.5)), np.cos(np.pi * (fx_ - 0.5)))
    fibre = np.where(horiz, np.sin(xx * 1.9), np.sin(yy * 1.9)) * 0.08
    v = 0.035 + 0.06 * bulge ** 2 + 0.01 * fibre
    return np.dstack([v * 0.92, v * 0.96, v * 1.05]).astype(np.float32)


@lru_cache(maxsize=1)
def brushed():
    r = np.random.default_rng(55)
    n = r.random((H // 2, W // 2)).astype(np.float32)
    n = cv2.blur(n, (151, 1))
    n = (n - n.min()) / (n.max() - n.min())
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    return (0.05 + 0.05 * n).astype(np.float32)


@lru_cache(maxsize=1)
def hex_lines():
    lay = np.zeros((H // 2, W // 2), np.float32)
    rr = 26
    dx, dy = rr * 1.5, rr * np.sqrt(3)
    for i in range(-1, int(W / 2 / dx) + 2):
        for j in range(-1, int(H / 2 / dy) + 2):
            cx = i * dx
            cy = j * dy + (dy / 2 if i % 2 else 0)
            pts = np.array([[cx + rr * np.cos(a), cy + rr * np.sin(a)] for a in np.arange(6) * np.pi / 3], np.float32)
            cv2.polylines(lay, [(pts * 16).astype(np.int32)], True, 1.0, 1, cv2.LINE_AA, 4)
    return lay


@lru_cache(maxsize=4)
def noise_tex(seed, w=270, h=480):
    return fx.fractal_noise(seed, w, h, 5)


# ================================================================ environments (each returns a fresh canvas)
def env_base(col_center, col_edge, cy=900):
    return fx.radial_bg(tuple(col_center), tuple(col_edge), 540, cy, 1000, 1350, 1.3).copy()


def floor(canvas, floor_y, col, gloss=0.5):
    """Dark glossy floor below floor_y with a horizon glow line."""
    yy = np.arange(H, dtype=np.float32)
    f = np.clip((yy - floor_y) / 380, 0, 1)[:, None, None]
    canvas *= 1 - 0.55 * f
    line = np.exp(-((yy - floor_y) / 3.0) ** 2)[:, None, None] * 0.35 + np.exp(-((yy - floor_y) / 40.0) ** 2)[:, None, None] * 0.12
    canvas += line * np.array(col, np.float32) * gloss
    return canvas


def chamber(t, col, floor_y, par=0.0):
    """Dark brushed-metal chamber with an overhead light cone."""
    c = np.dstack([brushed()] * 3) * np.array([0.85, 0.92, 1.05], np.float32)
    if par:
        c = np.roll(c, int(par * 0.3), axis=1)
    xx, yy = fx.grid()
    cone = np.clip(1 - np.abs(xx - 540) / (120 + yy * 0.42), 0, 1) ** 1.6 * np.clip(1 - yy / 2000, 0, 1)
    c = c * (0.55 + 0.9 * cone[..., None]) + cone[..., None] * np.array(col, np.float32) * 0.10
    sweep = np.exp(-(((xx - (t * 420) % 2400 + 600) - (yy - 900) * 0.35) / 160) ** 2)
    c += sweep[..., None] * 0.035
    return floor(c, floor_y, col)


def carbon_wall(t, col, floor_y, par=0.0):
    c = carbon().copy()
    if par:
        c = np.roll(c, int(par * 0.35), axis=1)
    xx, yy = fx.grid()
    spot = np.exp(-(((xx - 540) / 520) ** 2 + ((yy - 820) / 760) ** 2))
    c = c * (0.6 + 2.6 * spot[..., None])
    band = np.exp(-(((xx + (yy - 960) * 0.5) - lerp(-600, 1800, (t * 0.35) % 1)) / 140) ** 2)
    c += band[..., None] * c * 2.0
    c += spot[..., None] * np.array(col, np.float32) * 0.06
    return floor(c, floor_y, col)


def hexgrid(t, col, floor_y, scan_y=None):
    c = env_base((0.10, 0.03, 0.04), (0.01, 0.005, 0.01))
    lay = hex_lines()
    yy = np.arange(H // 2, dtype=np.float32)[:, None]
    sy = (t * 520) % 1400 - 200 if scan_y is None else scan_y
    band = 0.25 + 1.6 * np.exp(-((yy * 2 - sy) / 120) ** 2)
    m = (lay * band)[..., None] * np.array(col, np.float32) * 0.5
    c = glow_add(c, m, core=1.0, g1=2, g2=8, k1=0.6, k2=0.4)
    return floor(c, floor_y, col)


def nebula(t, cols, floor_y):
    c = env_base((0.10, 0.03, 0.07), (0.008, 0.004, 0.012))
    n1 = cv2.resize(fx.scroll(noise_tex(501), t * 10, -t * 6), (W, H))
    n2 = cv2.resize(fx.scroll(noise_tex(502), -t * 8, t * 5), (W, H))
    c += np.clip(n1 - 0.45, 0, 1)[..., None] * np.array(cols[0], np.float32) * 0.5
    c += np.clip(n2 - 0.5, 0, 1)[..., None] * np.array(cols[1], np.float32) * 0.45
    return floor(c, floor_y, cols[0], 0.4)


def circuit(t, col, floor_y, speed=1.0):
    """Racing circuit: perspective grid floor rushing toward camera + horizon glow."""
    c = env_base((0.10, 0.02, 0.03), (0.01, 0.0, 0.005), cy=1000)
    hz = 1000.0
    lay = half()
    vx = 540
    for k in range(-14, 15):
        xb = vx + k * 190
        cv2.line(lay, (int(vx / 2 * 16), int(hz / 2 * 16)), (int(xb * 3 / 2 * 16), int(H / 2 * 16)),
                 tuple(float(v) * 0.35 for v in col), 1, cv2.LINE_AA, 4)
    for k in range(14):
        zz = ((k / 14.0) - t * 0.55 * speed) % 1.0 + 0.04
        y = hz + 70 / zz
        if y < H:
            b = float(np.clip(1.3 - zz, 0, 1))
            cv2.line(lay, (0, int(y / 2 * 16)), (int(W / 2 * 16), int(y / 2 * 16)), tuple(float(v) * 0.5 * b for v in col), 1, cv2.LINE_AA, 4)
    yy = np.arange(H // 2, dtype=np.float32)[:, None, None]
    lay *= (yy * 2 > hz)
    c = glow_add(c, lay, core=1.0, g1=1.5, g2=6, k1=0.5, k2=0.5)
    yyf = np.arange(H, dtype=np.float32)[:, None, None]
    c += np.exp(-((yyf - hz) / 60) ** 2) * np.array(col, np.float32) * 0.25
    return floor(c, floor_y, col, 0.3)


def light_trails(canvas, t, cols, seed=7, n=9, y_range=(1080, 1580), speed=1.0, density=1.0, inten=1.0):
    """Long-exposure racing light trails sweeping across behind the product."""
    r = np.random.default_rng(seed)
    lay = half()
    for i in range(int(n * density)):
        yb = r.uniform(*y_range)
        cur = r.uniform(-90, 90)
        v = r.uniform(0.7, 1.3) * speed
        ph = r.uniform(0, 1.6)
        L = r.uniform(0.35, 0.7)
        col = cols[i % len(cols)]
        direction = 1 if i % 2 == 0 else -1
        head = (t * v * 0.9 + ph) % 1.6 - 0.3
        ss = np.linspace(head - L, head, 40)
        ss = ss[(ss > -0.1) & (ss < 1.1)]
        if len(ss) < 2:
            continue
        x = -150 + ss * (W + 300) if direction > 0 else W + 150 - ss * (W + 300)
        y = yb - cur * np.sin(np.pi * ss)
        pts = (np.stack([x, y], 1) / 2 * 16).astype(np.int32)
        for j in range(len(pts) - 1):
            b = (j + 1) / len(pts)
            cv2.line(lay, tuple(pts[j]), tuple(pts[j + 1]), tuple(float(v_) * b * inten for v_ in col), max(1, int(1 + 2.5 * b)), cv2.LINE_AA, 4)
    return glow_add(canvas, lay, core=1.2, g1=2, g2=9, k1=0.8, k2=0.9)


def speed_tunnel(canvas, travel, stretch, cols, inten=1.0, cx=540, cy=900, seed=5, n=240, ribs=True):
    """Supersonic tunnel of light streaks rushing past the camera."""
    r = np.random.default_rng(seed)
    lay = half()
    ang = r.uniform(0, 2 * np.pi, n)
    ph = r.uniform(0, 1, n)
    ci = r.integers(0, len(cols), n)
    for a, p, k in zip(ang, ph, ci):
        zz = (p - travel) % 1.0 + 0.03
        rr = 80 / zz
        rt = 80 / (zz + 0.012 + 0.05 * stretch)
        if rt > 1600:
            continue
        b = float(np.clip(1.2 - zz, 0, 1)) * inten
        x0_, y0_ = cx + np.cos(a) * rt, cy + np.sin(a) * rt
        x1_, y1_ = cx + np.cos(a) * min(rr, 1700), cy + np.sin(a) * min(rr, 1700)
        cv2.line(lay, (int(x0_ / 2 * 16), int(y0_ / 2 * 16)), (int(x1_ / 2 * 16), int(y1_ / 2 * 16)),
                 tuple(float(v) * b for v in cols[k]), 1, cv2.LINE_AA, 4)
    if ribs:
        for j in range(9):
            zz = (j / 9 - travel * 0.5) % 1.0 + 0.05
            rr = 110 / zz
            if rr > 1500:
                continue
            pts = np.array([[cx + rr * np.cos(a), cy + rr * 1.15 * np.sin(a)] for a in np.arange(8) * np.pi / 4 + np.pi / 8], np.float32)
            b = float(np.clip(1.1 - zz, 0, 1)) * 0.35 * inten
            cv2.polylines(lay, [(pts / 2 * 16).astype(np.int32)], True, tuple(float(v) * b for v in cols[0]), 1, cv2.LINE_AA, 4)
    return glow_add(canvas, lay, core=1.3, g1=2, g2=10, k1=0.8, k2=1.0)


def horizontal_streaks(canvas, t, speed, cols, inten=1.0, seed=3, n=70, y_range=(0, H), length=300):
    r = np.random.default_rng(seed)
    lay = half()
    for i in range(n):
        y = r.uniform(*y_range)
        v = r.uniform(0.6, 1.4) * speed
        ph = r.uniform(0, 1)
        x = W + 200 - ((t * v * 900 + ph * 2400) % 2400)
        L = length * r.uniform(0.5, 1.5) * (0.5 + speed)
        col = cols[i % len(cols)]
        cv2.line(lay, (int(x / 2 * 16), int(y / 2 * 16)), (int((x + L) / 2 * 16), int(y / 2 * 16)), tuple(float(v_) * inten for v_ in col), 1, cv2.LINE_AA, 4)
    return glow_add(canvas, lay, core=1.2, g1=1.5, g2=6, k1=0.6, k2=0.6)


def rings(specs, t):
    """Electromagnetic rings: returns (back, front) full-res additive layers (front half drawn over the can)."""
    back, front = half(), half()
    for sp in specs:
        cx, cy, rx = sp["cx"] / 2, sp["cy"] / 2, sp["rx"] / 2
        tilt, rot, col, inten = sp.get("tilt", 0.22), sp.get("rot", 0.0), np.array(sp["col"], np.float32), sp.get("inten", 1.0)
        phis = np.linspace(0, 2 * np.pi, 240)
        cr, sr = np.cos(rot), np.sin(rot)
        x = rx * np.cos(phis)
        y = rx * tilt * np.sin(phis)
        X, Y = cx + x * cr - y * sr, cy + x * sr + y * cr
        P = (np.stack([X, Y], 1) * 16).astype(np.int32)
        for j in range(len(P) - 1):
            tgt = back if np.sin(phis[j]) < 0 else front
            cv2.line(tgt, tuple(P[j]), tuple(P[j + 1]), tuple(float(v) * 0.55 * inten for v in col), sp.get("width", 1), cv2.LINE_AA, 4)
        npart = sp.get("n", 16)
        for k in range(npart):
            ph0 = sp.get("phase", 0.0) + 2 * np.pi * k / npart + t * sp.get("speed", 3.0)
            seg = np.linspace(ph0 - 0.45, ph0, 10)
            xs, ys = rx * np.cos(seg), rx * tilt * np.sin(seg)
            Xs, Ys = cx + xs * cr - ys * sr, cy + xs * sr + ys * cr
            Ps = (np.stack([Xs, Ys], 1) * 16).astype(np.int32)
            for j in range(len(Ps) - 1):
                tgt = back if np.sin(seg[j]) < 0 else front
                b = (j + 1) / len(Ps)
                cv2.line(tgt, tuple(Ps[j]), tuple(Ps[j + 1]), tuple(float(v) * 1.6 * b * inten for v in np.minimum(1, col + 0.35)), 2, cv2.LINE_AA, 4)

    def full(lay):
        g = lay * 1.2 + cv2.GaussianBlur(lay, (0, 0), 2.5) * 0.9 + cv2.GaussianBlur(lay, (0, 0), 10) * 1.1
        return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    return full(back), full(front)


def shockwave(canvas, cx, cy, R, width, amp, col, inten=1.0):
    """Expanding metallic shockwave: refractive distortion band + chromatic bright edge."""
    if R <= 1 or inten <= 0.01:
        return canvas
    xx, yy = fx.grid()
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    d = amp * np.exp(-((r - R) / width) ** 2)
    canvas = cv2.remap(canvas, (xx - dx / r * d).astype(np.float32), (yy - dy / r * d).astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    edge = np.stack([np.exp(-((r - R - o) / (width * 0.16)) ** 2) for o in (width * 0.12, 0.0, -width * 0.12)], -1)
    canvas += edge * (np.array(col, np.float32) * 0.6 + 0.5) * inten
    return canvas


def metal_debris(canvas, cx, cy, age, seed=9, n=70, col=(0.85, 0.9, 1.0)):
    if age < 0 or age > 0.8:
        return
    r = np.random.default_rng(seed)
    lay = half()
    for _ in range(n):
        a = r.uniform(0, 2 * np.pi)
        v = r.uniform(600, 1900)
        p1 = (cx + np.cos(a) * v * age, cy + np.sin(a) * v * age * 0.8)
        p0 = (cx + np.cos(a) * v * max(0, age - 0.05), cy + np.sin(a) * v * max(0, age - 0.05) * 0.8)
        b = float(1 - age / 0.8)
        cv2.line(lay, (int(p0[0] / 2 * 16), int(p0[1] / 2 * 16)), (int(p1[0] / 2 * 16), int(p1[1] / 2 * 16)), tuple(float(c) * b for c in col), 1, cv2.LINE_AA, 4)
    glow_add(canvas, lay, core=1.4, g1=1.5, g2=5, k1=0.7, k2=0.5)


def iris(canvas, ap, rot=0.0):
    """Mechanical 8-blade iris: ap=1 open, ap=0 closed."""
    if ap >= 0.999:
        return canvas
    rr = max(0.0, ap) * 1250
    m = np.zeros((H, W), np.float32)
    pts = np.array([[540 + rr * np.cos(a + rot), 960 + rr * np.sin(a + rot)] for a in np.arange(8) * np.pi / 4], np.float32)
    if rr > 1:
        cv2.fillPoly(m, [(pts * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    plate = np.dstack([brushed() * 0.9] * 3) * np.array([0.9, 0.95, 1.05], np.float32)
    canvas = canvas * m[..., None] + plate * (1 - m[..., None])
    lay = np.zeros((H, W), np.float32)
    for k in range(8):
        p = pts[k]
        q = p + np.array([np.cos(np.pi / 4 * k + rot + np.pi / 2), np.sin(np.pi / 4 * k + rot + np.pi / 2)]) * 1500
        cv2.line(lay, (int(p[0] * 16), int(p[1] * 16)), (int(q[0] * 16), int(q[1] * 16)), 1.0, 2, cv2.LINE_AA, 4)
    if rr > 1:
        cv2.polylines(lay, [(pts * 16).astype(np.int32)], True, 1.0, 3, cv2.LINE_AA, 4)
    canvas += (lay * 0.35 + cv2.GaussianBlur(lay, (0, 0), 4) * 0.4)[..., None] * np.array([0.75, 0.85, 1.0], np.float32)
    return canvas


def crt(canvas, k):
    """CRT collapse: k=0 normal, k=1 collapsed to a bright dot (via a horizontal line)."""
    if k <= 0:
        return canvas
    sy = max(0.003, 1 - ease_in_cubic(min(1, k / 0.7)))
    sx = 1 - ease_in_cubic(max(0, (k - 0.7) / 0.3)) * 0.995
    M = np.float32([[sx, 0, 540 * (1 - sx)], [0, sy, 960 * (1 - sy)]])
    out = cv2.warpAffine(canvas * (1 + 3 * k), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    add_glow(out, 540, 960, 140 + 500 * sx, (0.85, 0.95, 1.0), 0.9 * k, sx=1.0 * sx + 0.05, sy=0.05)
    return out


def glitch(canvas, k, fi, seed=0):
    if k <= 0.01:
        return canvas
    r = np.random.default_rng(seed * 997 + fi // 2)
    out = canvas.copy()
    for _ in range(int(4 + 26 * k)):
        y0 = int(r.integers(0, H - 20))
        hgt = int(r.integers(10, 120))
        out[y0:y0 + hgt] = np.roll(out[y0:y0 + hgt], int(r.uniform(-200, 200) * k), axis=1)
    sh = int(30 * k)
    out[..., 0] = np.roll(out[..., 0], sh, axis=1)
    out[..., 2] = np.roll(out[..., 2], -sh, axis=1)
    return out


def hologram(L, t, col, solid_y=None):
    """Turn a can layer into a hologram (edges + scanlines); rows above solid_y stay solid."""
    rgb, a = L["rgb"], L["a"]
    lum = (rgb / np.maximum(a[..., None], 1e-3)).mean(2) * a
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(gx * gx + gy * gy) * 2.2, 0, 1)
    h = a.shape[0]
    ys = np.arange(h, dtype=np.float32) + L["y0"]
    scan = 0.55 + 0.45 * (np.sin(ys * np.pi / 3 + t * 40) > 0)
    flick = 0.85 + 0.15 * np.sin(t * 97)
    holo = (edge * 1.3 + lum * 0.35)[..., None] * np.array(col, np.float32) * (scan[:, None, None] * flick)
    if solid_y is None:
        return dict(L, rgb=holo * a[..., None], a=a * 0.0), None
    m = np.clip((solid_y - ys) / 10, 0, 1)[:, None]
    solid = dict(L, rgb=rgb * m[..., None], a=a * m)
    ghost = dict(L, rgb=holo * (a * (1 - m))[..., None], a=a * 0.0)
    return ghost, solid


def particles_draw(canvas, xs, ys, cols, alpha, size=2.2):
    lay = np.zeros((H, W, 3), np.float32)
    ok = (xs > -10) & (xs < W + 10) & (ys > -10) & (ys < H + 10)
    for x, y, c in zip(xs[ok], ys[ok], cols[ok]):
        cv2.circle(lay, (int(x * 16), int(y * 16)), int(size * 16), tuple(float(v) for v in c), -1, cv2.LINE_AA, 4)
    g = lay + cv2.GaussianBlur(lay, (0, 0), 3) * 0.8
    canvas += g * alpha


def can_particles(can, M, k, mode="assemble", swirl=0.0):
    """Particle positions on screen for assemble (k 0->1 converge) or dissolve (k 0->1 fly away)."""
    tx, ty = apply(M, can.px, can.py)
    if mode == "assemble":
        kk = np.clip((k - can.pdelay) / 0.6, 0, 1)
        e = 1 - (1 - kk) ** 3
        ang = swirl * (1 - e) * 2.5
        dx, dy = can.pdx * (1 - e), can.pdy * (1 - e)
        dx, dy = dx * np.cos(ang) - dy * np.sin(ang), dx * np.sin(ang) + dy * np.cos(ang)
        return tx + dx, ty + dy, (kk > 0)
    kk = np.clip((k - can.pdelay) / 0.6, 0, 1)
    e = kk ** 2
    return tx + can.pdx * 0.7 * e, ty + can.pdy * 0.5 * e - 500 * e, (kk > 0) & (kk < 1)


def dissolve_mask(L, k, seed=301):
    h, w = L["a"].shape
    n = cv2.resize(noise_tex(seed), (w, h), interpolation=cv2.INTER_LINEAR)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    m = np.clip((n - k * 1.2 + 0.1) / 0.1, 0, 1)
    edge = np.exp(-((n - k * 1.2 + 0.05) / 0.03) ** 2)
    return m, edge


def tesla_arcs(canvas, t, fi, nodes, col, inten=1.0, n_arcs=3, seed=0, targets=()):
    core = np.zeros((H, W), np.float32)
    glow = np.zeros((H // 2, W // 2), np.float32)
    r = np.random.default_rng(seed * 131 + fi // 2)
    pairs = [(nodes[0], nodes[1])] * n_arcs + [(nodes[i % 2], tg) for i, tg in enumerate(targets)]
    for (p0, p1) in pairs:
        pts = fx.bolt_points(p0, p1, r, disp=0.18, depth=7)
        P = (pts * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(core, [P], False, 1.0, 2, cv2.LINE_AA, 4)
        cv2.polylines(glow, [(pts * 8).astype(np.int32).reshape(-1, 1, 2)], False, 1.0, 2, cv2.LINE_AA, 4)
    g = cv2.resize(cv2.GaussianBlur(glow, (0, 0), 4) * 1.2 + cv2.GaussianBlur(glow, (0, 0), 14) * 1.5, (W, H))
    canvas += (g[..., None] * np.array(col, np.float32) + cv2.GaussianBlur(core, (0, 0), 0.7)[..., None] * 0.9) * inten
    for nd in nodes:
        add_glow(canvas, nd[0], nd[1], 26, (1, 1, 1), 0.9 * inten)
        add_glow(canvas, nd[0], nd[1], 120, col, 0.5 * inten)


def streamlines(canvas, t, cx, cy, rx, ry, col, inten=1.0, speed=1.0, n=34):
    """Wind-tunnel laser streamlines flowing around the can silhouette (moving dashes)."""
    lay = half()
    xs = np.arange(-40, W + 40, 10, dtype=np.float32)
    for j in range(n):
        y0 = 260 + j * (1380 / n)
        dyc = y0 - cy
        near = np.clip(1 - abs(dyc) / (ry * 1.3), 0, 1)
        defl = np.sign(dyc + 1e-3) * (rx * 0.9) * np.exp(-((xs - cx) / (rx * 1.6)) ** 2) * near
        ys = y0 + defl
        ph = (j * 0.37) % 1
        on = ((xs / 260 - t * 2.4 * speed + ph) % 1.0) < 0.6
        P = (np.stack([xs, ys], 1) / 2 * 16).astype(np.int32)
        for k in range(len(P) - 1):
            if on[k]:
                cv2.line(lay, tuple(P[k]), tuple(P[k + 1]), tuple(float(v) * 0.6 * inten for v in col), 1, cv2.LINE_AA, 4)
    return glow_add(canvas, lay, core=1.0, g1=1.5, g2=6, k1=0.6, k2=0.5)


def float_motes(canvas, t, seed, n, col, inten=0.6, rmin=2, rmax=7, drift=12):
    r = np.random.default_rng(seed)
    for i in range(n):
        x0, y0, ph = r.uniform(0, W), r.uniform(0, H), r.uniform(0, 6.28)
        d = r.uniform(0.3, 1.0)
        x = (x0 + drift * np.sin(t * 0.6 + ph) * d) % W
        y = (y0 - t * 18 * d) % H
        add_glow(canvas, x, y, r.uniform(rmin, rmax) * (0.6 + d), col, inten * (0.5 + 0.5 * np.sin(t * 2 + ph)) * d)


def turbine(canvas, cx, cy, R, angle, spin, heat, col):
    """Jet-turbine fan behind the can: rotating blades, rotational blur at speed, heat glow."""
    s = 0.5
    lay = np.zeros((H // 2, W // 2, 3), np.float32)
    nb = 16
    passes = 1 + int(min(6, spin * 3))
    for pss in range(passes):
        ang0 = angle - pss * spin * 0.05
        for b in range(nb):
            a0 = ang0 + 2 * np.pi * b / nb
            pts = []
            for rr in np.linspace(0.22, 1.0, 8):
                pts.append((cx + np.cos(a0 + rr * 0.55) * R * rr, cy + np.sin(a0 + rr * 0.55) * R * rr))
            for rr in np.linspace(1.0, 0.22, 8):
                pts.append((cx + np.cos(a0 + rr * 0.55 + 0.22) * R * rr, cy + np.sin(a0 + rr * 0.55 + 0.22) * R * rr))
            P = (np.array(pts) * s * 16).astype(np.int32)
            shade = 0.17 + 0.10 * np.cos(a0 * 2)
            cv2.fillPoly(lay, [P], (shade * 0.95, shade, shade * 1.08), cv2.LINE_AA, 4)
    lay /= passes
    cv2.circle(lay, (int(cx * s * 16), int(cy * s * 16)), int(R * 0.22 * s * 16), (0.16, 0.16, 0.18), -1, cv2.LINE_AA, 4)
    cv2.circle(lay, (int(cx * s * 16), int(cy * s * 16)), int(R * s * 16), (0.25, 0.27, 0.3), 2, cv2.LINE_AA, 4)
    full = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
    m = (full.max(2) > 0.01).astype(np.float32)[..., None]
    canvas[:] = canvas * (1 - m * 0.85) + full
    add_glow(canvas, cx, cy, R * 0.9, col, 0.8 * heat)
    add_glow(canvas, cx, cy, R * 0.35, (1.0, 0.85, 0.5), 0.6 * heat)


def heat_shimmer(canvas, t, strength, y_from=0):
    if strength <= 0.01:
        return canvas
    n = cv2.resize(fx.scroll(noise_tex(707), 0, -t * 160), (W, H)) - 0.5
    xx, yy = fx.grid()
    mask = np.clip((yy - y_from) / 400, 0, 1)
    return cv2.remap(canvas, xx + n * 14 * strength * mask, yy + n * 6 * strength * mask, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def prism_beams(canvas, t, inten=1.0, offset=0.0):
    xx, yy = fx.grid()
    for k, col in enumerate(((1, 0.1, 0.25), (0.2, 1, 0.5), (0.25, 0.35, 1))):
        pos = ((t * 0.45 + k * 0.11 + offset) % 1.4 - 0.2) * 1800 - 300
        band = np.exp(-(((xx - (yy - 960) * 0.55) - pos) / 90) ** 2)
        canvas += band[..., None] * np.array(col, np.float32) * 0.22 * inten
    return canvas


def rgb_split(L, px):
    if px < 0.5 or L is None:
        return L
    rgb = L["rgb"].copy()
    rgb[..., 0] = np.roll(rgb[..., 0], int(px), axis=1)
    rgb[..., 2] = np.roll(rgb[..., 2], -int(px), axis=1)
    return dict(L, rgb=rgb)


# ================================================================ HUD + type
GLYPHS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789#%+/"


def decode(text, k, fi, seed=0):
    """Sci-fi decode: resolved prefix, a run of random glyphs, rest blank."""
    n = len(text)
    res = int(np.clip(k, 0, 1) * n + 1e-6)
    r = np.random.default_rng(seed * 1000 + fi // 3)
    out = []
    for i, ch in enumerate(text):
        if i < res or ch == " ":
            out.append(ch)
        elif i < res + 5:
            out.append(GLYPHS[int(r.integers(len(GLYPHS)))])
        else:
            out.append(" ")
    return "".join(out)


def text_at(canvas, text, font, size, x_left, cy, **kw):
    if not text.strip():
        return
    w = fx.font(font, size).getlength(text) + kw.get("tracking", 0) * size * (len(text) - 1)
    draw_text(canvas, text, font, size, x_left + w / 2, cy, **kw)


def text_width(text, font, size, tracking=0.0):
    f = fx.font(font, size)
    return sum(f.getlength(c) for c in text) + tracking * size * (len(text) - 1) if tracking else f.getlength(text)


CHROME = ((1.0, 1.0, 1.0), (0.62, 0.68, 0.78))


def hud_title(canvas, edition, flavour, k, fi, accent, alpha=1.0, idx=0, total=13):
    """Edition line + chrome flavour name, decoding in; counter at left."""
    if alpha <= 0.01 or k <= 0:
        return
    fs = fit_size(flavour, "InterDisplay-Black.otf", 860, 92)
    es = 30
    ew = text_width(edition, "InterDisplay-SemiBold.otf", es, 0.32)
    fw = text_width(flavour, "InterDisplay-Black.otf", fs)
    e_txt = decode(edition, k * 1.6, fi, seed=idx)
    f_txt = decode(flavour, k * 1.2 - 0.15, fi, seed=idx + 50)
    text_at(canvas, e_txt, "InterDisplay-SemiBold.otf", es, 540 - ew / 2, 168, alpha=alpha, tracking=0.32,
            color=tuple(min(1, c * 0.5 + 0.55) for c in accent), shadow=0.5)
    text_at(canvas, f_txt, "InterDisplay-Black.otf", fs, 540 - fw / 2, 168 + 34 + fs * 0.5, alpha=alpha, grad=CHROME,
            glow=accent, glow_r=16, glow_s=0.45, shadow=0.6)
    # rule with ticks under the title
    y = int(168 + 34 + fs + 18)
    kk = ease_out_cubic(min(1, k * 1.4))
    hw = int(300 * kk)
    lay = np.zeros((H, W), np.float32)
    cv2.line(lay, (540 - hw, y), (540 + hw, y), 1.0, 1, cv2.LINE_AA)
    for xx_ in (540 - hw, 540 + hw):
        cv2.line(lay, (xx_, y - 6), (xx_, y + 6), 1.0, 2, cv2.LINE_AA)
    canvas += (lay * 0.6 + cv2.GaussianBlur(lay, (0, 0), 3) * 0.6)[..., None] * np.array(accent, np.float32) * alpha
    text_at(canvas, f"{idx + 1:02d}/{total:02d}", "JetBrainsMono-Bold.ttf", 26, 60, 116, alpha=alpha * 0.9, color=(0.85, 0.9, 1.0))
    text_at(canvas, "ENERGY", "JetBrainsMono-Medium.ttf", 20, 870, 116, alpha=alpha * 0.8, color=(0.7, 0.8, 0.95), tracking=0.2)
    lvl = (idx + 1) / total
    bar = np.zeros((H, W), np.float32)
    for sgm in range(10):
        x0 = 870 + sgm * 15
        if sgm / 10 < lvl:
            cv2.rectangle(bar, (x0, 134), (x0 + 10, 142), 1.0, -1)
        else:
            cv2.rectangle(bar, (x0, 134), (x0 + 10, 142), 0.25, -1)
    canvas += bar[..., None] * np.array(accent, np.float32) * alpha * 0.9


def brackets(canvas, box, k, accent, alpha=1.0, grow=40):
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = box
    e = 1 - ease_out_expo(k)
    x0, y0, x1, y1 = x0 - 24 - grow * e, y0 - 24 - grow * e, x1 + 24 + grow * e, y1 + 24 + grow * e
    y1 = min(y1, SAFE_BOTTOM + 10)
    L = 44
    lay = np.zeros((H, W), np.float32)
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(lay, (int(cx), int(cy)), (int(cx + sx * L), int(cy)), 1.0, 3, cv2.LINE_AA)
        cv2.line(lay, (int(cx), int(cy)), (int(cx), int(cy + sy * L)), 1.0, 3, cv2.LINE_AA)
    canvas += (lay * 0.8 + cv2.GaussianBlur(lay, (0, 0), 4) * 0.7)[..., None] * np.array(accent, np.float32) * alpha * min(1, k * 3)


# ================================================================ price badge (every frame)
@lru_cache(maxsize=1)
def badge_layers():
    S = 3
    w, h = BADGE_W * S, BADGE_H * S
    ch = 24 * S
    poly = [(ch, 0), (w - 1, 0), (w - 1, h - 1 - ch), (w - 1 - ch, h - 1), (0, h - 1), (0, ch)]
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(poly, fill=255)
    m = np.asarray(im).astype(np.float32) / 255
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    rgb = np.array([0.15, 0.17, 0.21], np.float32) * (1 - yy) + np.array([0.04, 0.05, 0.07], np.float32) * yy
    cf = cv2.resize(carbon()[:h // 4 * 2, :w // 4 * 2], (w, h)) * 0.9
    rgb = rgb + cf
    rgb = np.broadcast_to(rgb, (h, w, 3)).copy()
    # chrome inner border
    inset = 7 * S
    poly_in = [(ch + inset * 0.6, inset), (w - 1 - inset, inset), (w - 1 - inset, h - 1 - ch - inset * 0.4),
               (w - 1 - ch - inset * 0.4, h - 1 - inset), (inset, h - 1 - inset), (inset, ch + inset * 0.6)]
    bi = Image.new("L", (w, h), 0)
    ImageDraw.Draw(bi).line(poly_in + [poly_in[0]], fill=255, width=2 * S)
    bm = np.asarray(bi).astype(np.float32)[..., None] / 255
    chrome = np.array([0.95, 0.97, 1.0], np.float32) * (1 - yy) + np.array([0.5, 0.56, 0.66], np.float32) * yy
    rgb = rgb * (1 - bm) + chrome * bm
    # accent light strips: blue left, red right
    rgb[ch + 4 * S:h - 16 * S, 14 * S:18 * S] = np.array([0.3, 0.65, 1.0], np.float32) * 1.3
    rgb[16 * S:h - ch - 4 * S, w - 18 * S:w - 14 * S] = np.array([1.0, 0.18, 0.22], np.float32) * 1.2

    def mask(txt, font, size, tracking=0.0):
        f = fx.font(font, size)
        adv = [f.getlength(c) for c in txt]
        tot = sum(adv) + tracking * size * (len(txt) - 1)
        bb = f.getbbox(txt)
        im_ = Image.new("L", (int(tot) + 8, bb[3] + 8), 0)
        d = ImageDraw.Draw(im_)
        x = 4
        for c, a_ in zip(txt, adv):
            d.text((x, 4), c, font=f, fill=255)
            x += a_ + tracking * size
        a_ = np.asarray(im_).astype(np.float32) / 255
        ys, xs = np.where(a_.max(1) > 0.02)[0], np.where(a_.max(0) > 0.02)[0]
        return a_[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]

    def paste(mk, cx, cy, col):
        mh, mw = mk.shape
        x0, y0 = int(cx - mw / 2), int(cy - mh / 2)
        c = col(mh, mw) if callable(col) else np.broadcast_to(np.array(col, np.float32), (mh, mw, 3))
        sh = cv2.GaussianBlur(mk, (0, 0), 2 * S) * 0.6
        rs = rgb[y0 + 2 * S:y0 + 2 * S + mh, x0 + S:x0 + S + mw]
        rs *= (1 - sh[:rs.shape[0], :rs.shape[1], None])
        reg = rgb[y0:y0 + mh, x0:x0 + mw]
        reg[:] = reg * (1 - mk[..., None]) + c * mk[..., None]

    k = 1.0
    while True:
        m2 = mask("2 FOR", "InterDisplay-Black.otf", int(44 * S * k))
        mp = mask("$5.99", "InterDisplay-Black.otf", int(58 * S * k))
        gap = int(16 * S * k)
        if m2.shape[1] + gap + mp.shape[1] <= w - 70 * S and mp.shape[0] <= 0.5 * h:
            break
        k *= 0.96
    row_cy = h * 0.40
    xl = w / 2 - (m2.shape[1] + gap + mp.shape[1]) / 2
    paste(m2, xl + m2.shape[1] / 2, row_cy + (mp.shape[0] - m2.shape[0]) * 0.5 * 0, (0.78, 0.84, 0.92))
    chrome_txt = lambda mh, mw: (np.array([1.0, 1.0, 1.0], np.float32) * (1 - np.linspace(0, 1, mh, dtype=np.float32)[:, None, None])
                                 + np.array([0.72, 0.8, 0.92], np.float32) * np.linspace(0, 1, mh, dtype=np.float32)[:, None, None]) * np.ones((1, mw, 1), np.float32)
    paste(mp, xl + m2.shape[1] + gap + mp.shape[1] / 2, row_cy, chrome_txt)
    sub = mask("BEFORE TAXES", "InterDisplay-Bold.otf", 19 * S, tracking=0.34)
    paste(sub, w / 2, h * 0.79, (0.55, 0.78, 1.0))
    rgb = np.clip(rgb, 0, 1.4)
    out_rgb = cv2.resize(rgb * m[..., None], (BADGE_W, BADGE_H), interpolation=cv2.INTER_AREA)
    out_a = cv2.resize(m, (BADGE_W, BADGE_H), interpolation=cv2.INTER_AREA)
    pad = 36
    shp = np.zeros((BADGE_H + 2 * pad, BADGE_W + 2 * pad), np.float32)
    shp[pad:pad + BADGE_H, pad:pad + BADGE_W] = out_a
    return out_rgb.astype(np.float32), out_a.astype(np.float32), cv2.GaussianBlur(shp, (0, 0), 10), cv2.GaussianBlur(shp, (0, 0), 18), pad


def draw_badge(img, t, beat=0.46875, t0=0.0):
    rgb, a, shadow, glow, pad = badge_layers()
    x0, y0 = int(540 - BADGE_W / 2), int(BADGE_CY - BADGE_H / 2)
    gy0, gx0 = y0 - pad, x0 - pad
    gh, gw = shadow.shape
    img[gy0 + 6:gy0 + 6 + gh, gx0:gx0 + gw] *= (1 - 0.65 * shadow)[..., None]
    pulse = np.exp(-((t - t0) % beat) / 0.09)
    img[gy0:gy0 + gh, gx0:gx0 + gw] += glow[..., None] * np.array([0.3, 0.55, 1.0], np.float32) * (0.16 + 0.12 * pulse)
    roi = img[y0:y0 + BADGE_H, x0:x0 + BADGE_W]
    roi *= (1 - a[..., None])
    roi += rgb
    ph = ((t - t0) % (8 * beat)) / 0.55                      # light sweep every 2 bars
    if ph < 1:
        xx = np.arange(BADGE_W, dtype=np.float32)[None, :] / BADGE_W
        yy = np.arange(BADGE_H, dtype=np.float32)[:, None] / BADGE_H
        band = np.exp(-((xx + 0.3 * yy - lerp(-0.3, 1.3, ease_in_out(ph))) / 0.05) ** 2) * a * 0.35
        roi += band[..., None]
    led = 0.4 + 0.6 * pulse                                  # LED strips pulse on the beat
    S = 1
    roi[30:BADGE_H - 16, 14:18] += np.array([0.2, 0.45, 1.0], np.float32) * 0.5 * led
    return img
