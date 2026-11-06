"""Reel 2 renderer - Crush & Fanta line-up for Ezy Pzy Convenience.

    python reel2_render.py --test 3.5,8.0     # preview stills -> output/build2/test/
    python reel2_render.py --all              # render every frame -> output/build2/frames/

Each product photo is used as a 2.5D "plate": the photo itself is never
altered, a matte (can + props touching its base) lets effects pass BEHIND the
can, and foreground effects/camera moves/transitions are layered on top.
Exactly one product is visible in any frame.

    0.0 - 3.0   Intro: dark stage, light sweeps reveal Crush Pineapple, text slams
    3.0 - 6.0   Crush Pineapple        golden liquid explosion, sun rays
    6.0 - 9.0   Crush Big Troppa Punch neon liquid wave reveal, retro sun, 90s shapes, VHS glitch
    9.0 - 12.0  Crush Peach            whip-pan in, fizz bubbles, can crack bubble stream
   12.0 - 15.0  Fanta Strawberry       crimson smoke, orbiting embers, smoke wipe out
   15.0 - 18.0  Fanta Pineapple        lightning strikes, electric arcs, sparks, zoom-through
   18.0 - 21.0  Fanta Mango            tumbling ice, cold mist, mango juice drops, vertical whip
   21.0 - 24.0  Fanta Crimson Sour Cherry  dark fire, embers, cherry splash, fire burst out
   24.0 - 28.5  Ezy Pzy Convenience end card
"""
import os
import json
import time
import argparse
from multiprocessing import Pool
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image

import timeline2 as TL
from timeline2 import W, H, FPS
import fx
from fx import (clamp, lin, smooth, keys, decay, ease_out_cubic, ease_in_cubic, ease_in_out, ease_out_expo,
                ease_in_expo, lerp, Cam, add_glow, render_liquid, draw_text, fit_size, hblur, vblur, zoom_blur,
                finish, draw_sprite, sphere_sprite, draw_ice, fractal_noise, scroll, smoothstep_arr, blend_over)
import render as R

ROOT = fx.ROOT
A2 = os.path.join(ROOT, "assets", "reel2")
BUILD = os.path.join(ROOT, "output", "build2")
FRAMES = os.path.join(BUILD, "frames")
GEO = json.load(open(os.path.join(A2, "geometry.json")))
Z0 = 1.15      # photo -> screen scale for the hero framing


# =========================================================== the photo plate
class Plate:
    def __init__(self, key):
        self.key = key
        img = np.asarray(Image.open(os.path.join(A2, "products", key + ".png")).convert("RGB")).astype(np.float32) / 255
        a = np.asarray(Image.open(os.path.join(A2, "mattes", key + ".png"))).astype(np.float32) / 255
        g = GEO[key]
        self.h, self.w = a.shape
        self.cx, self.top, self.bot = g["cx"], g["top"], g["bottom"]
        self.left, self.right = g["left"], g["right"]
        self.Rc = (g["right"] - g["left"]) / 2
        box = np.zeros_like(a)
        box[int(self.top) + 10:int(self.bot) - 10, int(self.left) + 4:int(self.right) - 4] = 1
        box = cv2.GaussianBlur(box, (0, 0), 6)
        canm = a * box
        xs = np.arange(self.w, dtype=np.float32)
        u = np.clip((xs - self.cx) / self.Rc, -1, 1)
        th = np.broadcast_to(np.arcsin(u)[None, :], a.shape)
        dist = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        rim = np.exp(-dist / 7.0) * a * box
        self.lv = {1: (np.dstack([img, a]).astype(np.float32),
                       np.dstack([canm, th / np.pi + 0.5, rim, np.zeros_like(a)]).astype(np.float32))}
        # blurred backdrop that fills the 9:16 frame beyond the photo's edges
        s = max(W / self.w, H / self.h) * 1.12
        bd = cv2.resize(img, (int(self.w * s), int(self.h * s)), interpolation=cv2.INTER_AREA)
        y0, x0 = (bd.shape[0] - H) // 2, (bd.shape[1] - W) // 2
        bd = bd[y0:y0 + H, x0:x0 + W]
        small = cv2.resize(bd, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        self.backdrop = (cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC) * 0.55).astype(np.float32)

    def level(self, k):
        if k not in self.lv:
            rgba, aux = self.lv[1]
            up = cv2.resize(rgba, (self.w * k, self.h * k), interpolation=cv2.INTER_LANCZOS4)
            bl = cv2.GaussianBlur(up[..., :3], (0, 0), 1.3)
            up[..., :3] = np.clip(up[..., :3] + 0.6 * (up[..., :3] - bl), 0, 1)
            ax = cv2.resize(aux, (self.w * k, self.h * k), interpolation=cv2.INTER_LINEAR)
            self.lv[k] = (up, ax)
        return self.lv[k]

    def render(self, cam, behind=None, amb_bg=1.0, amb_can=1.0, amb_props=None, sweep=None, spec=None, spec_s=0.0,
               spec_w=0.3, rim_l=(0, 0, 0), rim_r=(0, 0, 0), backdrop_shift=(0, 0), protect=True, hero_gain=1.0):
        """Composite: backdrop -> photo (graded) -> behind-effects -> hero (lit)."""
        k = 2 if cam.z > 1.45 else 1
        rgba, aux = self.level(k)
        sc = cam.z / k
        M = np.float32([[sc, 0, cam.ax - cam.fx * cam.z], [0, sc, cam.ay - cam.fy * cam.z]])
        flags = cv2.INTER_CUBIC if k == 2 else cv2.INTER_LINEAR
        ps = cv2.warpAffine(rgba, M, (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT)
        xs = cv2.warpAffine(aux, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb = np.clip(ps[..., :3], 0, 1)
        m = np.clip(ps[..., 3], 0, 1)
        # feathered plate edges (photo -> backdrop)
        x_l, y_t = cam.p(0, 0)
        x_r, y_b = cam.p(self.w, self.h)
        fe = 110 * cam.z / Z0
        yy = np.arange(H, dtype=np.float32)
        xx = np.arange(W, dtype=np.float32)
        fy = smoothstep_arr(y_t, y_t + fe, yy) * smoothstep_arr(y_b, y_b - fe, yy)
        fxv = smoothstep_arr(x_l, x_l + fe, xx) * smoothstep_arr(x_r, x_r - fe, xx)
        feather = fy[:, None] * fxv[None, :]
        bd = self.backdrop
        if backdrop_shift != (0, 0):
            bd = np.roll(bd, (int(backdrop_shift[1]), int(backdrop_shift[0])), axis=(0, 1))
        canvas = bd * amb_bg
        canvas = canvas * (1 - feather[..., None]) + rgb * (amb_bg * feather[..., None])
        if behind is not None:
            base = canvas.copy() if protect else None
            behind(canvas)
            if protect:
                # restore the photo's tabletop band (floor + props) over behind-effects
                y0 = cam.p(0, self.bot - 330)[1]
                y1 = cam.p(0, self.bot - 150)[1]
                p = smoothstep_arr(y0, y1, yy)[:, None, None]
                canvas = canvas * (1 - p) + base * p
        canm = xs[..., 0]
        th = (xs[..., 1] - 0.5) * np.pi
        rim = xs[..., 2]
        ap = amb_can if amb_props is None else amb_props
        light = ap + (amb_can - ap) * canm
        if sweep:
            for (pos, width, power) in sweep:
                light = light + power * np.exp(-((th - pos) / width) ** 2) * canm
        lit = rgb * (light * hero_gain)[..., None]
        if spec is not None and spec_s > 0:
            band = np.exp(-((th - spec) / spec_w) ** 2) * spec_s * canm
            lit = 1 - (1 - np.clip(lit, 0, 1)) * (1 - band[..., None])
        rl, rr = np.array(rim_l, np.float32), np.array(rim_r, np.float32)
        if rl.any() or rr.any():
            side = np.sin(th)
            lit = lit + rim[..., None] * ((np.clip(-side, 0, 1) ** 3)[..., None] * rl + (np.clip(side, 0, 1) ** 3)[..., None] * rr)
        mm = (m * feather)[..., None]
        return canvas * (1 - mm) + lit * mm

    def hero_cam(self, z=Z0, dx=0.0, dy=0.0, fy_off=0.0):
        """Default framing: can top lands at ~y=390 on screen."""
        return Cam(z=z, fx=self.cx + dx, fy=self.top + 610 / Z0 + fy_off + dy, ax=540, ay=1000)


_plates = {}


def plate(i):
    if i not in _plates:
        _plates[i] = Plate(TL.PRODUCTS[i]["key"])
    return _plates[i]


# close-up targets in photo coordinates (from, to) - labels/logos on each can
CLOSEUPS = [
    ((560, 540), (600, 790)),     # crush logo -> "Pineapple"
    ((560, 860), (560, 470)),     # 90s / BIG TROPPA PUNCH -> crush logo
    ((570, 560), (640, 780)),     # crush logo -> "Peach"
    ((560, 380), (560, 720)),     # FANTA -> artwork
    ((560, 420), (560, 760)),     # FANTA -> helmet
    ((580, 760), (560, 470)),     # mango art -> FANTA
    ((560, 440), (560, 860)),     # FANTA -> knight
]


def scene_cam(i, t, z_wide=Z0):
    """Shared shot grammar: wide (push-in) -> close-up (pan) -> medium (zoom out)."""
    p = plate(i)
    s = TL.DRINK_STARTS[i]
    u = t - s
    if u < TL.CLOSE:
        c = p.hero_cam(z=z_wide * (1 + 0.055 * ease_in_out(u / TL.CLOSE)), dx=14 * np.sin(u * 1.7 + i))
    elif u < TL.MEDIUM:
        x = lin(u, TL.CLOSE, TL.MEDIUM)
        (ax_, ay_), (bx_, by_) = CLOSEUPS[i]
        c = Cam(z=2.05 + 0.1 * x + 0.06 * decay(u, TL.CLOSE, 0.1), fx=lerp(ax_, bx_, ease_in_out(x)), fy=lerp(ay_, by_, ease_in_out(x)))
    else:
        z = keys(u, [(TL.MEDIUM, 1.62), (TL.MEDIUM + 0.25, Z0 * 1.08), (3.0, Z0 * 1.04)], ease_out_expo)
        c = p.hero_cam(z=z, dx=-10 * np.sin(u * 2 + i))
    return c


CHIP_TEXT = "NOW AT EZY PZY CONVENIENCE"   # store line under each product name


def tag(canvas, t, t_in, t_out, name, glow):
    if t < t_in or t > t_out + 0.3:
        return
    a_in = ease_out_cubic(lin(t, t_in, t_in + 0.3))
    a_out = 1 - ease_in_cubic(lin(t, t_out, t_out + 0.22))
    a = a_in * a_out
    size = fit_size(name, "InterDisplay-BlackItalic.otf", 900, 92)
    yoff = (1 - a_in) * 26 - (1 - a_out) * 34
    draw_text(canvas, name, "InterDisplay-BlackItalic.otf", size, 540 - (1 - a_in) * 50, 222 + yoff, alpha=a,
              reveal=ease_out_cubic(lin(t, t_in, t_in + 0.35)), glow=glow, glow_r=22, glow_s=0.9, shadow=0.55,
              blur=(1 - a_out) * 8)
    draw_text(canvas, CHIP_TEXT, "InterDisplay-SemiBold.otf", 30, 540, 222 + size * 0.6 + 20 + yoff,
              alpha=a * ease_out_cubic(lin(t, t_in + 0.1, t_in + 0.4)), tracking=0.22, shadow=0.5,
              glow=glow, glow_r=10, glow_s=0.35)


# =========================================================== extra effects
@lru_cache(maxsize=4)
def bubble_sprite(tint=(1.0, 1.0, 1.0), n=192):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
    inside = np.clip((1 - r) * c / 1.5, 0, 1)
    a = (0.06 + 0.75 * smoothstep_arr(0.72, 1.0, r)) * inside
    hl = np.exp(-(((x - c * 0.6) / (c * 0.22)) ** 2 + ((y - c * 0.58) / (c * 0.16)) ** 2))
    hl2 = np.exp(-(((x - c * 1.45) / (c * 0.12)) ** 2 + ((y - c * 1.42) / (c * 0.08)) ** 2)) * 0.5
    a = np.clip(a + (hl + hl2) * inside, 0, 1)
    col = np.array(tint, np.float32)
    return np.dstack([col * a[..., None], a]).astype(np.float32)


def smoke(canvas, t, density, color, seed=81, sx=40, sy=-60, contrast=2.0):
    n1 = scroll(fractal_noise(seed, 540, 960, 6), t * sx, t * sy)
    n2 = scroll(fractal_noise(seed + 1, 540, 960, 6), -t * sx * 0.7, t * sy * 1.5)
    n = np.clip((n1 * 0.55 + n2 * 0.45 - 0.3) * contrast, 0, 1)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.clip(n * density, 0, 1)[..., None]
    canvas *= 1 - a
    canvas += a * np.array(color, np.float32)


def embers(canvas, t, seed, n, cam, center, color=(1.0, 0.45, 0.1), mode="rise", orbit_r=420, front=None, inten=1.0):
    r = np.random.default_rng(seed)
    for i in range(n):
        ph = r.uniform(0, 6.28)
        if mode == "orbit":
            om = r.uniform(0.7, 1.4)
            th = ph + om * t
            depth = np.sin(th)
            if front is not None and (depth > 0) != front:
                continue
            x = center[0] + orbit_r * r.uniform(0.75, 1.25) * np.cos(th)
            y = center[1] + r.uniform(-520, 520) - 60 * t + 30 * np.sin(t * 2 + ph)
            sz = r.uniform(5, 11) * (1 + 0.35 * depth)
        else:
            x = center[0] + r.uniform(-650, 650) + 40 * np.sin(t * 2.3 + ph)
            y = center[1] + 700 - ((t * r.uniform(150, 380) + r.uniform(0, 1500)) % 1500)
            sz = r.uniform(4, 10)
        flick = 0.55 + 0.45 * np.sin(t * 17 + ph * 5)
        sx_, sy_ = cam.p(x, y)
        add_glow(canvas, sx_, sy_, cam.s(sz), (1.0, 0.95, 0.8), 0.9 * flick * inten)
        add_glow(canvas, sx_, sy_, cam.s(sz * 4), color, 0.35 * flick * inten)


def fire(canvas, t, intensity, y_base, height, seed=91, cover=0.0):
    """Procedural flames rising from y_base (screen px). cover>0 floods the frame."""
    s = 4
    h, w = H // s, W // s
    n1 = scroll(fractal_noise(seed, w, h, 5), 0, -t * 90)
    n2 = scroll(fractal_noise(seed + 1, w, h, 4), t * 8, -t * 150)
    yy = np.arange(h, dtype=np.float32)[:, None] * s
    grad = np.clip((yy - (y_base - height)) / height, 0, 1) ** 1.3
    grad = np.clip(grad + cover, 0, 1.6)
    v = np.clip((n1 * 0.6 + n2 * 0.4) * 1.5 * grad - 0.35, 0, 1) * intensity
    v = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 1.2)
    c = np.stack([np.clip(v * 3.0, 0, 1), np.clip(v * 3.0 - 1.0, 0, 1) * 0.85, np.clip(v * 3.0 - 2.1, 0, 1) * 0.7], -1)
    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR)
    canvas += c * 1.15


def retro_sun(canvas, cam, cx, cy, r, t, inten=1.0):
    s = 0.5
    hh, ww = H // 2, W // 2
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    sx, sy = cam.p(cx, cy)
    rr = cam.s(r)
    d = np.sqrt((xx / s - sx) ** 2 + (yy / s - sy) ** 2)
    disc = np.clip((rr - d) / 3, 0, 1)
    vy = np.clip((yy / s - (sy - rr)) / (2 * rr), 0, 1)
    col = np.stack([np.ones_like(vy), 0.72 - 0.55 * vy, 0.12 + 0.55 * vy], -1)
    stripes = np.ones_like(vy)
    lower = (yy / s) > sy
    k = ((yy / s - sy) / (rr * 0.12) + t * 0.8) % 1.0
    stripes[lower] = (k[lower] > (0.25 + 0.5 * vy[lower])).astype(np.float32)
    lay = col * (disc * stripes)[..., None]
    glow = cv2.GaussianBlur(disc, (0, 0), 25)[..., None] * np.array([1.0, 0.35, 0.6], np.float32)
    out = cv2.resize(lay * 0.85 + glow * 0.6, (W, H), interpolation=cv2.INTER_LINEAR) * inten
    canvas[:] = 1 - (1 - np.clip(canvas, 0, 1)) * (1 - np.clip(out, 0, 1))


def memphis(canvas, cam, t, seed, n=16, front=False, scale=1.0):
    r = np.random.default_rng(seed)
    cols = [(1.0, 0.25, 0.65), (0.15, 0.9, 1.0), (1.0, 0.9, 0.2)]
    lay = np.zeros((H // 2, W // 2, 3), np.float32)
    for i in range(n):
        kind = i % 3
        x0, y0 = r.uniform(0, 1122), r.uniform(80, 1250)
        if not front and abs(x0 - 560) < 230:
            x0 += 520 * np.sign(x0 - 560 + 1e-3)
        ang = r.uniform(0, 6.28) + t * r.uniform(-1.2, 1.2)
        sz = r.uniform(40, 85) * scale
        x = x0 + 25 * np.sin(t * 1.3 + i)
        y = y0 - 50 * t
        c = cols[i % 3]
        sx, sy = cam.p(x, y)
        S = cam.s(sz) / 2
        if kind == 0:   # zigzag
            pts = np.array([[(-1 + k * 0.5) * S, (0.35 if k % 2 else -0.35) * S] for k in range(5)], np.float32)
        elif kind == 1:  # triangle
            pts = np.array([[np.cos(a) * S, np.sin(a) * S] for a in (0, 2.094, 4.189, 0)], np.float32)
        else:           # squiggle
            pts = np.array([[(-1 + k / 6) * S, np.sin(k * 1.4) * 0.3 * S] for k in range(13)], np.float32)
        Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        P = ((pts @ Rm.T + [sx, sy]) / 2 * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(lay, [P], False, c, max(2, int(cam.s(5) / 2)), cv2.LINE_AA, 4)
    g = cv2.GaussianBlur(lay, (0, 0), 5)
    canvas += cv2.resize(lay * 0.95 + g * 0.9, (W, H), interpolation=cv2.INTER_LINEAR)


def vhs(img, amount, seed):
    if amount <= 0.01:
        return img
    r = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(int(10 * amount) + 2):
        y0 = r.integers(0, H - 60)
        hgt = r.integers(8, 70)
        out[y0:y0 + hgt] = np.roll(out[y0:y0 + hgt], int(r.uniform(-60, 60) * amount), axis=1)
    sh = int(10 * amount)
    out[..., 0] = np.roll(out[..., 0], sh, axis=1)
    out[..., 2] = np.roll(out[..., 2], -sh, axis=1)
    scan = (np.arange(H) % 4 < 2).astype(np.float32)[:, None, None]
    out = out * (1 - 0.18 * amount * scan)
    return out


def palette_hue(t, cols):
    cols = np.array(cols, np.float32)

    def fn(h, w):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pos = (xx / w * 0.9 + yy / h * 0.35 + t * 0.5) % 1.0 * len(cols)
        i0 = np.floor(pos).astype(int) % len(cols)
        i1 = (i0 + 1) % len(cols)
        f = (pos - np.floor(pos))[..., None]
        f = f * f * (3 - 2 * f)
        return cols[i0] * (1 - f) + cols[i1] * f
    return fn


NEON = [(1.0, 0.2, 0.62), (1.0, 0.55, 0.1), (1.0, 0.92, 0.2), (0.1, 0.85, 0.95)]


def neon_wave(canvas, t, t_rise, t_cover, t_clear):
    """Liquid band moving upward: covers the frame at t_cover, clears by t_clear."""
    if t < t_rise or t > t_clear + 0.4:
        return
    xs = np.linspace(-60, W + 60, 48)
    if t <= t_cover:
        k = ease_in_cubic(lin(t, t_rise, t_cover)) * 0.75 + 0.25 * lin(t, t_rise, t_cover)
        ub, lb = lerp(2050, -420, k), 2600
    else:
        ub = -420 - 900 * (t - t_cover)
        lb = lerp(2250, -520, ease_in_out(lin(t, t_cover, t_clear)))
    up = ub + 95 * np.sin(xs * 0.006 + t * 7) + 60 * np.sin(xs * 0.017 - t * 9) - (xs - 540) * 0.22
    lo = lb + 80 * np.sin(xs * 0.008 - t * 6) + 50 * np.sin(xs * 0.02 + t * 8) + (xs - 540) * 0.18
    poly = np.concatenate([np.stack([xs, up], 1), np.stack([xs[::-1], lo[::-1]], 1)])
    rr = np.random.default_rng(171)
    discs = []
    for i in range(70):
        x = rr.uniform(-40, W + 40)
        life = (t * 2.2 + rr.uniform(0, 1)) % 1.0
        discs.append((x + 30 * np.sin(i), np.interp(x, xs, up) - 30 - 320 * life, rr.uniform(6, 20) * (1 - life)))
    if t > t_cover:
        for i in range(26):
            x = rr.uniform(0, W)
            yl = np.interp(x, xs, lo)
            L = rr.uniform(80, 360) * smooth(lin(t, t_cover, t_cover + 0.25))
            for j in range(9):
                f = j / 8
                discs.append((x + 6 * np.sin(j + i), yl + L * f, rr.uniform(10, 22) * (1 - 0.55 * f)))
            dt_ = max(0.0, t - t_cover - 0.05 - rr.uniform(0, 1) * 0.3)
            discs.append((x, yl + L + 60 + 1800 * dt_ * dt_, rr.uniform(8, 16)))
    rip = scroll(R.smooth_ripple(), t * 160, -t * 420)
    render_liquid(canvas, discs=discs, polys=[poly], hue_fn=palette_hue(t, NEON), sigma=3.4, thr=(0.3, 0.52),
                  bump=30, ripple=rip, opacity=1.0, spec_power=50)


def bolts(canvas, cam, t, strikes, p, color=(0.45, 0.85, 1.0), big=()):
    core = np.zeros((H, W), np.float32)
    glow = np.zeros((H // 2, W // 2), np.float32)
    for idx, s in enumerate(strikes):
        age = t - s
        if age < 0 or age > 0.35:
            continue
        fr = int(age * FPS)
        isbig = s in big
        if fr in (0, 1, 2, 4, 5) or (isbig and fr < 9 and fr != 6):
            rb = np.random.default_rng(2000 + idx * 31 + fr // 2)
            for b in range(4 if isbig else 2):
                side = -1 if (b + idx) % 2 == 0 else 1
                p0 = cam.p(p.cx + side * rb.uniform(150, 520), -250)
                p1 = cam.p(p.cx + side * rb.uniform(p.Rc + 10, p.Rc + 140), p.bot - rb.uniform(150, 500))
                fx.draw_bolt(core, glow, p0, p1, rb, width=3 if isbig else 2, branches=4 if isbig else 2)
    fx.composite_electric(canvas, core, glow, color, 1.0)


def sparks(canvas, cam, t, events, origin, color=(1.0, 0.85, 0.35), n=40, seed=300):
    lay = np.zeros((H, W), np.float32)
    for i, s in enumerate(events):
        a = t - s
        if a < 0 or a > 0.75:
            continue
        rs = np.random.default_rng(seed + i)
        for j in range(n):
            ox, oy = origin[0] + rs.uniform(-260, 260), origin[1] - rs.uniform(0, 80)
            ang = rs.uniform(-np.pi * 0.95, -np.pi * 0.05)
            v = rs.uniform(700, 2300)
            pts = []
            for dt in (a, a - 0.035):
                pts.append(cam.p(ox + np.cos(ang) * v * dt, oy + np.sin(ang) * v * dt + 1600 * dt * dt))
            cv2.line(lay, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), float(1 - a / 0.75), 2, cv2.LINE_AA)
    if lay.any():
        g = cv2.GaussianBlur(lay, (0, 0), 1.0) + cv2.GaussianBlur(lay, (0, 0), 5) * 1.2
        canvas += g[..., None] * np.array(color, np.float32)


# =========================================================== scenes
def intro_and_s1(t, fi):
    p = plate(0)
    post = dict(flash=0.0, ca=0.0)
    pre = t < TL.DROP
    u = t - TL.DROP
    if pre:
        cam = p.hero_cam(z=keys(t, [(0, 0.98), (2.95, 1.08)]), fy_off=-120)
        cam.ay = 1150 - 40 * smooth(t / 3)
    else:
        cam = scene_cam(0, t)
        cam.z += 0.09 * decay(t, TL.DROP, 0.16)
    amp = 26 * decay(t, TL.DROP, 0.13) + 8 * decay(t, 4.5, 0.1) + sum(7 * decay(t, w, 0.08) for w, _ in TL.INTRO_WORDS)
    dx, dy = R.shake(t, amp)
    cam.ax += dx
    cam.ay += dy
    gold = (1.0, 0.80, 0.14)
    cxp, cyp = p.cx, p.top + 480

    def behind(cv):
        if pre:
            g = ease_in_expo(lin(t, 2.2, 3.0))
            if g > 0:
                sx, sy = cam.p(cxp, cyp)
                add_glow(cv, sx, sy, 200 + 900 * g, (1.0, 0.75, 0.2), 0.9 * g)
                add_glow(cv, sx, sy, 80 + 300 * g, (1.0, 1.0, 0.85), 0.8 * g)
            R.dust(cv, t, 5, 70, (1.0, 0.9, 0.6), inten=0.5)
            return
        R.god_rays(cv, t, *cam.p(cxp, cyp), (1.0, 0.85, 0.45), 0.22, n=12, speed=0.05)
        R.bokeh(cv, t, 3, 20, [(1.0, 0.85, 0.3), (1.0, 0.95, 0.6), (1.0, 0.7, 0.2)], cam, inten=0.10)
        sim = R.s1_sim(u)
        streams, drops, red, front = R.SPLASH
        discs = R.splash_prims(sim, cxp, cyp, streams, drops)
        discs.append((cxp, cyp, 260 * ease_out_cubic(lin(u, 0, 0.12)) * (1 - 0.6 * lin(u, 0.1, 1.2))))
        sd = R.cam_discs(cam, discs)
        render_liquid(cv, discs=[d for d in sd if d[2] > 4], color=gold, dark=(0.55, 0.28, 0.0), sigma=4.0,
                      thr=(0.30, 0.52), bump=40)
        for (x, y, r) in sd:
            if 1.0 < r <= 4:
                draw_sprite(cv, sphere_sprite((1.0, 0.85, 0.25)), x, y, r * 2.2)
        sx, sy = cam.p(cxp, cyp)
        R.shock_ring(cv, sx, sy, 1700 * ease_out_expo(lin(u, 0, 0.7)), 46 * (1 - lin(u, 0, 0.7)) + 6, (1.0, 0.9, 0.6), 1.2 * (1 - lin(u, 0, 0.7)))

    if pre:
        rev = lin(t, 0.15, 2.9)
        sweep = [(keys(t, [(0.4, -1.9), (1.8, 1.9)]), 0.42, 1.2 * smooth(lin(t, 0.35, 0.6)) * (1 - smooth(lin(t, 1.65, 1.9)))),
                 (keys(t, [(1.8, 1.9), (2.8, -1.9)]), 0.5, 1.1 * smooth(lin(t, 1.75, 1.95)))]
        rs = 0.15 + 1.0 * smooth(rev)
        canvas = p.render(cam, behind, amb_bg=0.05 + 0.18 * ease_in_cubic(rev), amb_can=0.03 + 0.3 * ease_in_cubic(rev) + 0.5 * ease_in_expo(lin(t, 2.6, 3.0)),
                          amb_props=0.04 + 0.2 * ease_in_cubic(rev), sweep=sweep,
                          rim_l=(1.0 * rs * 0.6, 0.8 * rs * 0.6, 0.3 * rs * 0.6), rim_r=(1.0 * rs, 0.85 * rs, 0.35 * rs), protect=False)
        lines = [("THE ULTIMATE", 760, 290), ("REFRESHMENT", 960, 418), ("IS HERE.", 560, 540)]
        fonts = ["InterDisplay-Black.otf", "InterDisplay-BlackItalic.otf", "InterDisplay-Black.otf"]
        explode = ease_in_cubic(lin(t, 2.8, 3.0))
        for i, ((t0, _), (txt, mw, y)) in enumerate(zip(TL.INTRO_WORDS, lines)):
            if t < t0:
                continue
            k = ease_out_expo(lin(t, t0, t0 + 0.22))
            size = fit_size(txt, fonts[i], mw, 190)
            grad = ((1, 1, 1), (1.0, 0.86, 0.35)) if i == 1 else None
            draw_text(canvas, txt, fonts[i], size, 540, y, alpha=k * (1 - explode), scale=lerp(1.45, 1.0, k) * (1 + 0.5 * explode),
                      blur=(1 - k) * 14 + explode * 12, glow=(1.0, 0.75, 0.2), glow_r=26, glow_s=0.7 + 0.6 * decay(t, t0, 0.15),
                      grad=grad, shadow=0.4)
        post["flash"] = 0.4 * ease_in_expo(lin(t, 2.75, 3.0)) + sum(0.1 * decay(t, w, 0.05) for w, _ in TL.INTRO_WORDS)
        post["ca"] = sum(6 * decay(t, w, 0.1) for w, _ in TL.INTRO_WORDS)
        return canvas, post

    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, 3, 6)), spec_s=0.16, rim_r=(0.5, 0.42, 0.15), rim_l=(0.3, 0.28, 0.2))
    # foreground golden drops (depth of field)
    fd = []
    for (x0, y0, r0, vx, vy) in R.SPLASH[3]:
        k = ease_out_cubic(lin(u, 0, 0.4))
        sim = R.s1_sim(u)
        x = cxp + (x0 - 540) * (0.3 + 0.7 * k) + vx * sim * 3
        y = cyp + (y0 - 960) * (0.3 + 0.7 * k) + vy * sim * 3 + 400 * sim * sim
        fd.append((x, y, r0 * (0.4 + 0.6 * k) * (1 + 0.6 * sim)))
    render_liquid(canvas, discs=R.cam_discs(cam, fd), color=gold, dark=(0.6, 0.32, 0.0), sigma=3.0, thr=(0.3, 0.52),
                  bump=30, blur_out=6.0 if u < TL.CLOSE else 10.0, opacity=0.85)
    can_open_fx(canvas, cam, p, t, 4.5, (1.0, 0.95, 0.75))
    tag(canvas, t, 3.25, 4.3, TL.PRODUCTS[0]["name"], (1.0, 0.75, 0.15))
    neon_wave(canvas, t, 5.6, 6.0, 6.45)
    post["flash"] = 0.9 * decay(t, TL.DROP, 0.06) + 0.25 * decay(t, 4.5, 0.05) + 0.25 * decay(t, 5.25, 0.05)
    post["ca"] = 14 * decay(t, TL.DROP, 0.18) + 5 * decay(t, 4.5, 0.1)
    return canvas, post


def can_open_fx(canvas, cam, p, t, t0, tint):
    if t < t0 or t > t0 + 1.2:
        return
    v = t - t0
    tx, ty = cam.p(p.cx, p.top + 10)
    add_glow(canvas, tx, ty - cam.s(50) * ease_out_cubic(v / 0.6), cam.s(240) * ease_out_cubic(lin(v, 0, 0.5)), tint, 0.5 * decay(t, t0, 0.35), sy=0.6)
    r = np.random.default_rng(int(t0 * 10))
    for i in range(46):
        st = r.uniform(0, 0.5)
        if v < st:
            continue
        vv = v - st
        x, y = cam.p(p.cx + r.uniform(-150, 150) + 25 * np.sin(vv * 9 + i), p.top - r.uniform(160, 420) * vv - 10)
        draw_sprite(canvas, bubble_sprite(), x, y, cam.s(r.uniform(9, 20)), alpha=0.9 * clamp(1.4 - vv))


def scene2(t, fi):          # Crush Big Troppa Punch
    i, s = 1, 6.0
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    cam = scene_cam(i, t)
    if t < 6.45:
        cam.z *= lerp(1.18, 1.0, ease_out_cubic(lin(t, 6.0, 6.45)))
    shift = 0.0
    if t >= s + TL.EXIT:
        k = ease_in_cubic(lin(t, s + TL.EXIT, 8.96))
        shift = -1500 * k
    cam.ax += shift
    dx, dy = R.shake(t, 10 * decay(t, s, 0.12) + 8 * decay(t, s + TL.CLOSE, 0.1))
    cam.ax += dx
    cam.ay += dy
    u = t - s

    def behind(cv):
        retro_sun(cv, cam, p.cx, p.top + 330, 430, t, 0.8)
        memphis(cv, cam, u, 21, 18, scale=1.45)
        R.bokeh(cv, t, 8, 18, [(1.0, 0.3, 0.7), (0.2, 0.9, 1.0), (1.0, 0.9, 0.3)], cam, inten=0.11)
    canvas = p.render(cam, behind, spec=lerp(0.6, -0.6, lin(t, s, s + 3)), spec_s=0.16, rim_l=(0.9, 0.25, 0.6), rim_r=(0.2, 0.8, 1.0),
                      backdrop_shift=(shift * 0.5, 0))
    rr = np.random.default_rng(5)
    drops = []
    for j in range(46):
        x0, y0 = rr.uniform(0, 1122), rr.uniform(-100, 1400)
        drops.append((x0 + 20 * np.sin(u * 2 + j), (y0 + rr.uniform(50, 160) * u) % 1500 - 80, rr.uniform(6, 15)))
    render_liquid(canvas, discs=R.cam_discs(cam, drops), hue_fn=palette_hue(t, NEON), sigma=2.6, thr=(0.3, 0.52), bump=30,
                  opacity=0.9 * smooth(lin(t, 6.1, 6.5)))
    memphis(canvas, cam, u, 33, 6, front=True, scale=1.0)
    tag(canvas, t, 6.35, 7.35, TL.PRODUCTS[i]["name"], (1.0, 0.3, 0.7))
    neon_wave(canvas, t, 5.6, 6.0, 6.45)
    if shift:
        canvas = hblur(canvas, 10 + 210 * lin(t, s + TL.EXIT, 9.0) ** 2)
    g = 1.0 * decay(t, s + TL.CLOSE, 0.07) + 0.8 * decay(t, s + TL.MEDIUM, 0.06)
    canvas = vhs(canvas, g, fi)
    post["flash"] = 0.3 * decay(t, s, 0.08) + 0.5 * ease_in_expo(lin(t, 8.85, 9.0))
    post["ca"] = 6 * decay(t, s + TL.CLOSE, 0.1) + 10 * lin(t, s + TL.EXIT, 9.0)
    return canvas, post


def scene3(t, fi):          # Crush Peach - fizz bubbles
    i, s = 2, 9.0
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    cam = scene_cam(i, t)
    u = t - s
    shift, zb = 0.0, 0.0
    if u < 0.27:
        k = ease_out_cubic(lin(u, 0, 0.27))
        shift = 1500 * (1 - k)
    if u >= TL.EXIT:
        x = ease_in_cubic(lin(u, TL.EXIT, 3.0))
        cam.z *= np.exp(np.log(3.2) * x)
        zb = 0.26 * x
    cam.ax += shift
    dx, dy = R.shake(t, 9 * decay(t, s + TL.CLOSE, 0.1) + 10 * decay(t, 10.5, 0.1))
    cam.ax += dx
    cam.ay += dy
    peach = (1.0, 0.62, 0.48)

    def behind(cv):
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (1.0, 0.8, 0.7), 0.16, n=10, speed=0.04)
        R.bokeh(cv, t, 12, 22, [(1.0, 0.7, 0.6), (1.0, 0.85, 0.75), (1.0, 0.55, 0.45)], cam, inten=0.12)
        rb = np.random.default_rng(41)
        for j in range(40):
            x0 = rb.uniform(0, 1122)
            if abs(x0 - p.cx) < p.Rc:
                continue
            y = 1500 - ((u * rb.uniform(200, 420) + rb.uniform(0, 1600)) % 1600)
            x, y_ = cam.p(x0 + 18 * np.sin(u * 4 + j), y)
            draw_sprite(cv, bubble_sprite(), x, y_, cam.s(rb.uniform(12, 30)), blur=1.5, alpha=0.7)
    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, s, s + 3)), spec_s=0.16, rim_r=(0.6, 0.4, 0.35), rim_l=(0.5, 0.35, 0.3),
                      backdrop_shift=(shift * 0.5, 0))
    # foreground bubbles: mid (sharp) + big (blurred)
    rb = np.random.default_rng(42)
    for j in range(44):
        big = j < 7
        x0 = rb.uniform(-50, 1170)
        if not big and abs(x0 - p.cx) < p.Rc * 0.8:
            x0 = p.cx + np.sign(x0 - p.cx + 1e-3) * rb.uniform(p.Rc * 0.85, p.Rc * 1.4)
        sp = rb.uniform(250, 520) * (1.6 if big else 1.0)
        y = 1550 - ((u * sp + rb.uniform(0, 1700)) % 1700)
        x, y_ = cam.p(x0 + 22 * np.sin(u * 3 + j), y)
        sz = rb.uniform(90, 160) if big else rb.uniform(22, 56)
        draw_sprite(canvas, bubble_sprite(), x, y_, cam.s(sz), blur=7 if big else 0.5, alpha=0.85 if big else 1.0)
    if u < 0.5:   # entry burst
        for j in range(30):
            a = rb.uniform(0, 6.28)
            d = 900 * ease_out_cubic(u / 0.5)
            x, y_ = cam.p(p.cx + np.cos(a) * d * rb.uniform(0.4, 1), p.top + 520 + np.sin(a) * d * rb.uniform(0.4, 1))
            draw_sprite(canvas, bubble_sprite(), x, y_, cam.s(rb.uniform(20, 50)), alpha=1 - u / 0.5)
    can_open_fx(canvas, cam, p, t, 10.5, (1.0, 0.85, 0.8))
    if u >= TL.EXIT:   # bubble swarm rushing at the camera
        k = lin(u, TL.EXIT, 3.0)
        for j in range(40):
            a = rb.uniform(0, 6.28)
            d = rb.uniform(50, 700) * (1 + 3 * k * k)
            x, y_ = 540 + np.cos(a) * d, 960 + np.sin(a) * d
            draw_sprite(canvas, bubble_sprite(), x, y_, rb.uniform(30, 90) * (1 + 6 * k * k), blur=2 + 10 * k, alpha=0.9)
    tag(canvas, t, 9.3, 10.35, TL.PRODUCTS[i]["name"], peach)
    if shift:
        canvas = hblur(canvas, 220 * (1 - ease_out_cubic(lin(u, 0, 0.27))))
    if zb:
        canvas = zoom_blur(canvas, zb, *cam.p(p.cx, p.top + 620))
    post["flash"] = 0.4 * decay(t, s, 0.08) + 0.25 * decay(t, s + TL.CLOSE, 0.05) + ease_in_expo(lin(u, TL.EXIT + 0.08, 3.0))
    post["flash_color"] = (1.0, 0.96, 0.94)
    post["ca"] = 8 * (1 - lin(u, 0, 0.3)) + 12 * lin(u, TL.EXIT, 3.0)
    return canvas, post


def scene4(t, fi):          # Fanta Strawberry - crimson smoke + embers
    i, s = 3, 12.0
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    cam = scene_cam(i, t)
    u = t - s
    zb = 0.0
    if u < 0.35:
        x = ease_out_expo(lin(u, 0, 0.35))
        cam.z *= lerp(1.6, 1.0, x)
        zb = 0.25 * (1 - x)
    dx, dy = R.shake(t, 10 * decay(t, s, 0.1) + 8 * decay(t, s + TL.CLOSE, 0.1))
    cam.ax += dx
    cam.ay += dy
    red = (1.0, 0.18, 0.15)
    center = (p.cx, p.top + 560)

    def behind(cv):
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 250), (1.0, 0.35, 0.3), 0.18, n=12, speed=-0.05, sharp=12)
        smoke(cv, t, 0.55, (0.55, 0.05, 0.08), seed=81, sx=35, sy=-45)
        embers(cv, u, 7, 40, cam, center, color=red, mode="orbit", front=False, inten=0.9)
    canvas = p.render(cam, behind, spec=lerp(0.6, -0.6, lin(t, s, s + 3)), spec_s=0.16, rim_l=(0.9, 0.2, 0.2), rim_r=(1.0, 0.5, 0.4))
    embers(canvas, u, 7, 40, cam, center, color=red, mode="orbit", front=True)
    embers(canvas, u, 8, 30, cam, center, color=(1.0, 0.4, 0.3), mode="rise", inten=0.7)
    # low smoke wisps in front + the smoke wipe out
    wipe = ease_in_cubic(lin(u, 2.4, 2.97))
    yy = np.arange(H, dtype=np.float32)
    low = np.clip((yy - 1300) / 500, 0, 1)
    dens = np.clip(low * 0.5 + wipe * 3.2, 0, 4)[:, None]
    n1 = scroll(fractal_noise(85, 540, 960, 6), u * 50, -u * 160)
    n = cv2.resize(np.clip((n1 - 0.25) * 2.2, 0, 1), (W, H))
    a = np.clip(n * dens + wipe ** 1.5 * 1.3, 0, 1)[..., None]
    canvas = canvas * (1 - a) + a * np.array([0.35, 0.02, 0.05], np.float32)
    tag(canvas, t, 12.3, 13.35, TL.PRODUCTS[i]["name"], (1.0, 0.2, 0.2))
    if zb:
        canvas = zoom_blur(canvas, zb, 540, 960)
    post["flash"] = (1 - ease_out_cubic(lin(u, 0, 0.25))) * 0.9 + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.25 * decay(t, s + TL.MEDIUM, 0.05)
    post["flash_color"] = (1.0, 0.96, 0.94)
    post["ca"] = 6 * decay(t, s + TL.CLOSE, 0.1)
    return canvas, post


def scene5(t, fi):          # Fanta Pineapple - lightning
    i, s = 4, 15.0
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    cam = scene_cam(i, t)
    u = t - s
    zb = 0.0
    cam.z += sum(0.035 * decay(t, st, 0.12) for st in TL.STRIKES_S5 if t >= st)
    if u >= TL.EXIT:
        x = ease_in_cubic(lin(u, TL.EXIT, 3.0))
        cam.z *= np.exp(np.log(3.5) * x)
        zb = 0.28 * x
    amp = sum((20 if st == 15.0 else 10) * decay(t, st, 0.1) for st in TL.STRIKES_S5 if t >= st)
    dx, dy = R.shake(t, amp, 3.0)
    cam.ax += dx
    cam.ay += dy
    last = max([st for st in TL.STRIKES_S5 if st <= t], default=None)
    fl = np.exp(-(t - last) / 0.07) * (1.0 if last == 15.0 else 0.6) if last is not None else 0.0

    def behind(cv):
        cv *= 0.8 + 0.6 * fl
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (1.0, 0.9, 0.5), 0.15 + 0.3 * fl, n=16, speed=0.12, sharp=14)
        bolts(cv, cam, t, TL.STRIKES_S5, p, color=(0.55, 0.9, 1.0), big=(15.0,))
        # smoke from scene 4 clearing
        if u < 0.4:
            smoke(cv, t, 2.5 * (1 - ease_out_cubic(u / 0.4)), (0.35, 0.02, 0.05), seed=85)
    canvas = p.render(cam, behind, amb_bg=0.85, spec=lerp(-0.6, 0.6, lin(t, s, s + 3)), spec_s=0.16,
                      rim_l=tuple(np.array([0.4, 0.8, 1.0]) * (0.5 + 1.2 * fl)), rim_r=tuple(np.array([1.0, 0.9, 0.4]) * (0.5 + 0.8 * fl)),
                      hero_gain=1.0 + 0.1 * fl)
    # electric arcs crawling on the can edges
    core = np.zeros((H, W), np.float32)
    glow = np.zeros((H // 2, W // 2), np.float32)
    ra = np.random.default_rng(int(t * FPS) // 2 + 500)
    for k in range(4):
        side = ra.choice([-1, 1])
        xe = p.cx + side * (p.Rc + ra.uniform(-6, 12))
        ya = ra.uniform(p.top + 120, p.bot - 200)
        fx.draw_bolt(core, glow, cam.p(xe, ya), cam.p(xe + side * ra.uniform(0, 25), ya + ra.uniform(90, 240) * ra.choice([-1, 1])), ra, width=2, branches=1)
    fx.composite_electric(canvas, core, glow, (0.5, 0.85, 1.0), 0.7 + 0.4 * fl)
    sparks(canvas, cam, t, TL.STRIKES_S5, (p.cx, p.bot - 60), n=40)
    R.dust(canvas, t, 15, 40, (1.0, 0.9, 0.4), inten=0.5, vy=-60)
    tag(canvas, t, 15.3, 16.35, TL.PRODUCTS[i]["name"], (1.0, 0.8, 0.15))
    if zb:
        canvas = zoom_blur(canvas, zb, *cam.p(560, 700))
    post["flash"] = 0.5 * fl + 0.25 * decay(t, s + TL.CLOSE, 0.05) + ease_in_expo(lin(u, TL.EXIT + 0.08, 3.0))
    post["flash_color"] = (0.95, 0.98, 1.0)
    post["ca"] = 8 * fl + 12 * lin(u, TL.EXIT, 3.0)
    return canvas, post


ICE6 = []
_r = np.random.default_rng(66)
_lx = [90, 1040, 170, 960, 1010]
for _i, _th in enumerate(TL.ICE_HITS_S6):
    ICE6.append(dict(x=_lx[_i], th=_th, fall=_r.uniform(0.4, 0.55), size=_r.uniform(120, 160), rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3.5, 3.5, 3)))
for _i in range(6):
    ICE6.append(dict(x=_r.uniform(40, 1080), t0=18.0 + _r.uniform(0, 2.6), size=_r.uniform(60, 90) if _i < 4 else _r.uniform(260, 340),
                     rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3, 3, 3), vy=_r.uniform(500, 800), fore=_i >= 4))


def scene6(t, fi):          # Fanta Mango - ice + cold mist
    i, s = 5, 18.0
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    cam = scene_cam(i, t)
    u = t - s
    zb, shift_y = 0.0, 0.0
    if u < 0.35:
        x = ease_out_expo(lin(u, 0, 0.35))
        cam.z *= lerp(1.7, 1.0, x)
        zb = 0.28 * (1 - x)
    if u >= TL.EXIT:
        shift_y = 2200 * ease_in_cubic(lin(u, TL.EXIT, 2.97))
    cam.ay += shift_y
    dx, dy = R.shake(t, 8 * decay(t, s + TL.CLOSE, 0.1) + 10 * decay(t, 19.5, 0.1) + sum(5 * decay(t, h, 0.08) for h in TL.ICE_HITS_S6))
    cam.ax += dx
    cam.ay += dy

    def ice_state(o):
        if "th" in o:
            ts = o["th"] - o["fall"]
            land = p.bot - o["size"] * 0.35
            if t < ts:
                return None
            if t < o["th"]:
                f = (t - ts) / o["fall"]
                return o["x"], lerp(-250, land, f * f), o["rot0"] + o["om"] * (t - ts)
            d = t - o["th"]
            return o["x"], land - o["size"] * 0.4 * abs(np.sin(d * 9)) * np.exp(-d / 0.18), o["rot0"] + o["om"] * (o["fall"] + 0.25 * (1 - np.exp(-d / 0.25)))
        if t < o["t0"]:
            return None
        d = t - o["t0"]
        y = -400 + o["vy"] * d + 120 * d * d
        return (o["x"], y, o["rot0"] + o["om"] * d) if y < 1800 else None

    def behind(cv):
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 250), (1.0, 0.85, 0.4), 0.14, n=9, speed=0.03, sharp=6)
        R.mist_layer(cv, t, np.full((H, W), 0.22, np.float32), color=(0.85, 0.97, 0.9), seed=51, sx=25, sy=-8)
        rr = np.random.default_rng(9)
        drops = []
        for j in range(26):
            x0 = rr.uniform(0, 1122)
            if abs(x0 - p.cx) < p.Rc:
                x0 += np.sign(x0 - p.cx + 1e-3) * p.Rc
            drops.append((x0 + 20 * np.sin(u * 2 + j), (rr.uniform(-100, 1300) + rr.uniform(40, 120) * u) % 1450 - 60, rr.uniform(8, 18)))
        render_liquid(cv, discs=R.cam_discs(cam, drops), color=(1.0, 0.62, 0.08), dark=(0.6, 0.25, 0.0), sigma=2.8, thr=(0.3, 0.52), bump=30)
        for o in ICE6:
            if "th" not in o and not o["fore"]:
                st = ice_state(o)
                if st:
                    x, y = cam.p(st[0], st[1])
                    draw_ice(cv, x, y, cam.s(o["size"]), *st[2], blur=2.5, alpha=0.85)
    canvas = p.render(cam, behind, spec=lerp(0.6, -0.6, lin(t, s, s + 3)), spec_s=0.18, rim_l=(0.5, 0.9, 0.85), rim_r=(1.0, 0.8, 0.4),
                      backdrop_shift=(0, shift_y * 0.5))
    # vapour from the top + cold fog at the base
    xx, yy = fx.grid()
    tx, ty = cam.p(p.cx, p.top)
    burst = ease_out_cubic(lin(t, 19.5, 20.2)) * decay(t, 19.8, 0.8) if t >= 19.5 else 0.0
    vap = np.clip(1 - np.abs(xx - tx) / cam.s(320), 0, 1) ** 1.5 * np.clip(1 - np.abs(yy - (ty - cam.s(140))) / cam.s(320), 0, 1) * (0.3 + 1.1 * burst)
    fy_ = cam.p(0, p.bot)[1]
    fog = np.clip(1 - np.abs(yy - fy_) / cam.s(240), 0, 1) ** 1.2 * 0.4
    R.mist_layer(canvas, t, vap + fog, color=(0.85, 0.97, 0.95), seed=61, sx=40, sy=-70)
    for o in ICE6:
        st = ice_state(o)
        if st is None:
            continue
        x, y = cam.p(st[0], st[1])
        if "th" in o:
            draw_ice(canvas, x, y, cam.s(o["size"]), *st[2], blur=0.8)
        elif o["fore"]:
            draw_ice(canvas, x, y, cam.s(o["size"]), *st[2], blur=7.0, alpha=0.9)
    r = np.random.default_rng(77)
    for j in range(80):
        x0, y0 = r.uniform(0, W), r.uniform(0, H)
        ph = r.uniform(0, 6.28)
        k = max(0.0, np.sin(t * 3.1 + ph * 5)) ** 6
        if k > 0.03:
            add_glow(canvas, x0, (y0 + 35 * u) % H, r.uniform(4, 9), (0.9, 1.0, 0.95), 0.9 * k)
    tag(canvas, t, 18.3, 19.35, TL.PRODUCTS[i]["name"], (0.4, 0.95, 0.3))
    if zb:
        canvas = zoom_blur(canvas, zb, 540, 960)
    if shift_y:
        canvas = vblur(canvas, 10 + 240 * lin(u, TL.EXIT, 3.0) ** 2)
    post["flash"] = (1 - ease_out_cubic(lin(u, 0, 0.25))) * 0.9 + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, 19.5, 0.08)
    post["flash"] += 0.5 * ease_in_expo(lin(u, 2.85, 3.0))
    post["flash_color"] = (0.95, 1.0, 0.97)
    post["ca"] = 5 * decay(t, s + TL.CLOSE, 0.1) + 10 * lin(u, TL.EXIT, 3.0)
    return canvas, post


def scene7(t, fi):          # Fanta Crimson Sour Cherry - dark fire
    i, s = 6, 21.0
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    cam = scene_cam(i, t)
    u = t - s
    shift_y = 0.0
    if u < 0.27:
        shift_y = -2200 * (1 - ease_out_cubic(lin(u, 0, 0.27)))
    cam.ay += shift_y
    dx, dy = R.shake(t, 22 * decay(t, s, 0.13) + 8 * decay(t, s + TL.CLOSE, 0.1) + 10 * decay(t, 23.55, 0.15))
    cam.ax += dx
    cam.ay += dy
    cxp, cyp = p.cx, p.top + 520
    burst = ease_in_cubic(lin(u, 2.55, 3.0))

    def behind(cv):
        cv *= 0.6
        fire(cv, t, 1.35, cam.p(0, p.bot - 280)[1], cam.s(1350), seed=91)
        smoke(cv, t, 0.35, (0.25, 0.02, 0.03), seed=95, sx=20, sy=-80)
        sim = R.s1_sim(u)
        streams, drops, red, front = R.SPLASH
        sd = R.cam_discs(cam, R.splash_prims(sim * 0.9, cxp, cyp, streams[::2], drops[::2]))
        render_liquid(cv, discs=[d for d in sd if d[2] > 4], color=(0.75, 0.02, 0.08), dark=(0.25, 0.0, 0.02), sigma=4.0,
                      thr=(0.30, 0.52), bump=40)
        sx, sy = cam.p(cxp, cyp)
        R.shock_ring(cv, sx, sy, 1600 * ease_out_expo(lin(u, 0, 0.6)), 40 * (1 - lin(u, 0, 0.6)) + 6, (1.0, 0.45, 0.2), 1.1 * (1 - lin(u, 0, 0.6)))
    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, s, s + 3)), spec_s=0.15, rim_l=(1.0, 0.3, 0.1), rim_r=(1.0, 0.6, 0.2))
    embers(canvas, u, 17, 60, cam, (p.cx, p.top + 600), color=(1.0, 0.35, 0.05), mode="rise", inten=1.0)
    if burst > 0:
        fire(canvas, t, 1.2, H + 50, 600 + 2200 * burst, seed=97, cover=burst * 1.2)
    tag(canvas, t, 21.3, 22.35, TL.PRODUCTS[i]["name"], (1.0, 0.25, 0.1))
    if shift_y:
        canvas = vblur(canvas, 240 * (1 - ease_out_cubic(lin(u, 0, 0.27))))
    post["flash"] = 0.5 * decay(t, s, 0.07) + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.25 * decay(t, s + TL.MEDIUM, 0.05)
    post["flash"] += ease_in_expo(lin(u, 2.75, 3.0)) * 0.95
    post["flash_color"] = (1.0, 0.9, 0.75)
    post["ca"] = 10 * decay(t, s, 0.15) + 6 * decay(t, s + TL.CLOSE, 0.1)
    return canvas, post


LEAKS2 = [((1.0, 0.75, 0.15), 120, 380), ((1.0, 0.25, 0.6), 980, 520), ((1.0, 0.2, 0.15), 160, 1500), ((0.4, 0.95, 0.3), 950, 1650)]


def render_frame(fi):
    t = fi / FPS
    if t < 6.0:
        canvas, post = intro_and_s1(t, fi)
    elif t < 9.0:
        canvas, post = scene2(t, fi)
    elif t < 12.0:
        canvas, post = scene3(t, fi)
    elif t < 15.0:
        canvas, post = scene4(t, fi)
    elif t < 18.0:
        canvas, post = scene5(t, fi)
    elif t < 21.0:
        canvas, post = scene6(t, fi)
    elif t < TL.END_START:
        canvas, post = scene7(t, fi)
    else:
        canvas, post = R.end_card(t, fi, T=TL, leaks=LEAKS2)
    img = finish(canvas, fi, flash=min(1.0, post.get("flash", 0)), flash_color=post.get("flash_color", (1, 1, 1)),
                 ca=post.get("ca", 0), grain=post.get("grain", 0.014))
    if "fade" in post:
        img *= post["fade"]
    return (img * 255 + 0.5).astype(np.uint8)


def _work(fi):
    path = os.path.join(FRAMES, f"{fi:05d}.png")
    if os.path.exists(path):
        return fi
    img = render_frame(fi)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.test:
        out = os.path.join(BUILD, "test")
        os.makedirs(out, exist_ok=True)
        for ts in args.test.split(","):
            t = float(ts)
            t0 = time.time()
            img = render_frame(int(round(t * FPS)))
            cv2.imwrite(os.path.join(out, f"t{t:06.2f}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"t={t:.2f} {time.time() - t0:.2f}s", flush=True)
        return
    os.makedirs(FRAMES, exist_ok=True)
    n = int(round(TL.DURATION * FPS))
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for k, _ in enumerate(pool.imap_unordered(_work, range(n), chunksize=2)):
            if (k + 1) % 30 == 0:
                print(f"{k + 1}/{n} frames  {time.time() - t0:.0f}s", flush=True)
    print("render done", time.time() - t0)


if __name__ == "__main__":
    main()
