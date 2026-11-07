"""Reel 3 renderer - Alani Nu x9, "2 FOR $5.99 BEFORE TAXES" offer, for Ezy Pzy Convenience.

    python reel3_render.py --test 1.5,4.0     # preview stills -> output/build3/test/
    python reel3_render.py --all              # render every frame (60 fps) -> output/build3/frames/

Each product photo is a 2.5D plate (photo untouched; a matte lets effects pass
BEHIND the can). Exactly one product is visible in any frame. The price label is
composited last on every frame, in the left margin; every shot keeps the can clear
of it (close-ups are framed right of centre, all whips are vertical).

    0.00 -  3.00  Intro: dark stage, light sweeps reveal Cotton Candy, "ICE COLD. / UNBEATABLE / DEALS."
    3.00 -  5.25  Cotton Candy        pink + teal liquid explosion          -> purple/teal liquid wave
    5.25 -  7.50  Purple Cotton Candy carnival string lights, confetti      -> vertical whip up
    7.50 -  9.75  Witch's Brew        green cauldron bubbles, purple smoke  -> green smoke wipe
    9.75 - 12.00  Hawaiian Shaved Ice tumbling ice, cold mist              -> vertical whip down
   12.00 - 14.25  Orange Kiss         sunburst, orange/pink splash, fizz    -> sun-flare burst
   14.25 - 16.50  Cherry Slush        teal lightning on the beat, sparks    -> blue liquid wave
   16.50 - 18.75  Breezeberry         falling blue droplets, frost          -> hyperspace warp flash
   18.75 - 21.00  Cosmic Stardust     warp streaks -> starfield + nebula    -> vertical whip up
   21.00 - 23.25  Watermelon Wave     pink/teal liquid burst, can crack     -> flash + whip
   23.25 - 27.75  End card: store logo, address, 2 FOR $5.99 BEFORE TAXES, call to action
"""
import os
import json
import time
import argparse
from multiprocessing import Pool
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image, ImageDraw

import timeline3 as TL
from timeline3 import W, H, FPS
import fx
from fx import (clamp, lin, smooth, keys, decay, ease_out_cubic, ease_in_cubic, ease_in_out, ease_out_expo,
                ease_in_expo, ease_out_back, lerp, Cam, add_glow, render_liquid, draw_text, fit_size, hblur, vblur,
                zoom_blur, finish, draw_sprite, sphere_sprite, draw_ice, fractal_noise, scroll, smoothstep_arr,
                blend_over, blend_add, radial_bg)
import render as R
import reel2_render as R2
import reel2_two_locations as TLOC

ROOT = fx.ROOT
A3 = os.path.join(ROOT, "assets", "reel3")
GEO3 = json.load(open(os.path.join(A3, "geometry.json")))
BUILD = os.path.join(ROOT, "output", "build3")
FRAMES = os.path.join(BUILD, "frames")

# ------------------------------------------------------------- layout guards
BADGE_W, BADGE_H = 200, 224
BADGE_CX, BADGE_CY = 16 + BADGE_W / 2, 900          # left margin, mid height (inside IG safe area)
CAN_H = 1200.0                                       # wide shot: can height on screen
CAN_CY = 1000.0                                      # wide shot: can centre (top at 400, bottom at 1600)
CLOSE_W = 750.0                                      # close-up: can width on screen
CLOSE_CX = 625.0                                     # close-up: can centre x (left edge ~250 > badge edge 216)
ENTRY_ZOOM = 1.33                                    # max zoom-out entrance (keeps centred can clear of the badge)

_plates = {}


def plate(i):
    if i not in _plates:
        _plates[i] = R2.Plate(TL.PRODUCTS[i]["key"], base=A3, geo=GEO3)
    return _plates[i]


def zw(p):
    return CAN_H / (p.bot - p.top)


def wide_cam(p, zmul=1.0, dx=0.0, dy=0.0, ay=CAN_CY):
    return Cam(z=zw(p) * zmul, fx=p.cx + dx, fy=(p.top + p.bot) / 2 + dy, ax=540, ay=ay)


# close-up pans, as fractions of can height (0 = top of can)
CLOSE_PATH = [(0.30, 0.52), (0.48, 0.30), (0.30, 0.55), (0.52, 0.30), (0.30, 0.50),
              (0.50, 0.30), (0.32, 0.52), (0.52, 0.32), (0.30, 0.50)]


def scene_cam(i, t):
    p = plate(i)
    s = TL.DRINK_STARTS[i]
    u = t - s
    if u < TL.CLOSE:
        return wide_cam(p, 1 + 0.05 * ease_in_out(u / TL.CLOSE), dx=8 * np.sin(u * 1.9 + i))
    if u < TL.MEDIUM:
        x = lin(u, TL.CLOSE, TL.MEDIUM)
        zc = CLOSE_W / (p.right - p.left) * (1 + 0.035 * x + 0.025 * decay(u, TL.CLOSE, 0.1))
        a, b = CLOSE_PATH[i]
        fy = p.top + (p.bot - p.top) * lerp(a, b, ease_in_out(x))
        return Cam(z=zc, fx=p.cx - (CLOSE_CX - 540) / zc, fy=fy, ax=540, ay=960)
    zm = keys(u, [(TL.MEDIUM, 1.28), (TL.MEDIUM + 0.22, 1.05), (TL.SCENE, 1.03)], ease_out_expo)
    return wide_cam(p, zm, dx=-6 * np.sin(u * 2 + i))


def shaken(cam, t, amp, seed=0.0):
    dx, dy = R.shake(t, min(amp, 16), seed)
    cam.ax += dx * 0.5          # mostly vertical shake: never pushes the can into the badge margin
    cam.ay += dy
    return cam


def tag(canvas, t, t_in, t_out, name, glow):
    if t < t_in or t > t_out + 0.3:
        return
    a_in = ease_out_cubic(lin(t, t_in, t_in + 0.28))
    a_out = 1 - ease_in_cubic(lin(t, t_out, t_out + 0.22))
    a = a_in * a_out
    size = fit_size(name, "InterDisplay-BlackItalic.otf", 900, 90)
    yoff = (1 - a_in) * 26 - (1 - a_out) * 34
    draw_text(canvas, name, "InterDisplay-BlackItalic.otf", size, 540 - (1 - a_in) * 50, 222 + yoff, alpha=a,
              reveal=ease_out_cubic(lin(t, t_in, t_in + 0.32)), glow=glow, glow_r=22, glow_s=0.9, shadow=0.55,
              blur=(1 - a_out) * 8)
    draw_text(canvas, "ALANI NU ENERGY", "InterDisplay-SemiBold.otf", 30, 540, 222 + size * 0.6 + 20 + yoff,
              alpha=a * ease_out_cubic(lin(t, t_in + 0.1, t_in + 0.38)), tracking=0.22, shadow=0.5,
              glow=glow, glow_r=10, glow_s=0.35)


# ------------------------------------------------------------- price label
@lru_cache(maxsize=1)
def badge_layers():
    """Retail price tag, drawn once at 3x then downsampled. Returns premultiplied RGB, alpha, shadow."""
    S = 3
    w, h = BADGE_W * S, BADGE_H * S
    r = 22 * S
    body = Image.new("L", (w, h), 0)
    ImageDraw.Draw(body).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    m = np.asarray(body).astype(np.float32) / 255
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    rgb = np.array([0.93, 0.13, 0.17], np.float32) * (1 - yy) + np.array([0.60, 0.03, 0.08], np.float32) * yy
    rgb = np.broadcast_to(rgb, (h, w, 3)).copy()
    # bottom strip for BEFORE TAXES
    strip_y0 = int(h * 0.735)
    rgb[strip_y0:] = np.array([0.30, 0.01, 0.05], np.float32)
    # gold inner border
    border = Image.new("L", (w, h), 0)
    ImageDraw.Draw(border).rounded_rectangle((7 * S, 7 * S, w - 1 - 7 * S, h - 1 - 7 * S), r - 6 * S, outline=255, width=3 * S)
    bm = np.asarray(border).astype(np.float32)[..., None] / 255
    gold = np.array([1.0, 0.82, 0.36], np.float32)
    rgb = rgb * (1 - bm) + gold * bm
    # glossy top highlight
    gl = np.clip(1 - yy[..., 0] / 0.42, 0, 1) ** 2 * 0.16
    rgb = rgb + gl[..., None]

    def text_mask(txt, font, size, tracking=0.0):
        f = fx.font(font, size)
        adv = [f.getlength(c) for c in txt]
        total = sum(adv) + tracking * size * (len(txt) - 1)
        bb = f.getbbox(txt)
        im = Image.new("L", (int(total) + 8, bb[3] + 8), 0)
        d = ImageDraw.Draw(im)
        x = 4
        for c, a_ in zip(txt, adv):
            d.text((x, 4), c, font=f, fill=255)
            x += a_ + tracking * size
        a_ = np.asarray(im).astype(np.float32) / 255
        ys = np.where(a_.max(1) > 0.02)[0]
        return a_[ys[0]:ys[-1] + 1]

    def paste(mask, cx, cy, color):
        nonlocal rgb
        mh, mw = mask.shape
        x0, y0 = int(cx - mw / 2), int(cy - mh / 2)
        reg = rgb[y0:y0 + mh, x0:x0 + mw]
        if callable(color):
            col = color(mh, mw)
        else:
            col = np.broadcast_to(np.array(color, np.float32), (mh, mw, 3))
        # drop shadow
        sh = cv2.GaussianBlur(mask, (0, 0), 2.5 * S) * 0.55
        rs = rgb[y0 + 2 * S:y0 + 2 * S + mh, x0 + S:x0 + S + mw]
        rs *= (1 - sh[:rs.shape[0], :rs.shape[1], None])
        reg[:] = reg * (1 - mask[..., None]) + col * mask[..., None]

    inner_w = (BADGE_W - 30) * S
    s_2for = fit_size("2 FOR", "InterDisplay-Black.otf", inner_w, 46 * S)
    s_price = fit_size("$5.99", "InterDisplay-Black.otf", inner_w, 74 * S)
    s_sub = fit_size("BEFORE TAXES", "InterDisplay-Bold.otf", inner_w, 20 * S, tracking=0.06)
    paste(text_mask("2 FOR", "InterDisplay-Black.otf", s_2for), w / 2, h * 0.205, (1, 1, 1))
    price_grad = lambda mh, mw: (np.array([1.0, 0.97, 0.70], np.float32) * (1 - np.linspace(0, 1, mh, dtype=np.float32)[:, None, None])
                                 + np.array([1.0, 0.78, 0.22], np.float32) * np.linspace(0, 1, mh, dtype=np.float32)[:, None, None]) * np.ones((1, mw, 1), np.float32)
    paste(text_mask("$5.99", "InterDisplay-Black.otf", s_price), w / 2, h * 0.47, price_grad)
    paste(text_mask("BEFORE TAXES", "InterDisplay-Bold.otf", s_sub, tracking=0.06), w / 2, (strip_y0 + h - 7 * S) / 2, (1, 1, 1))
    rgb = np.clip(rgb, 0, 1)
    out_rgb = cv2.resize(rgb * m[..., None], (BADGE_W, BADGE_H), interpolation=cv2.INTER_AREA)
    out_a = cv2.resize(m, (BADGE_W, BADGE_H), interpolation=cv2.INTER_AREA)
    pad = 40
    sh = np.zeros((BADGE_H + 2 * pad, BADGE_W + 2 * pad), np.float32)
    sh[pad:pad + BADGE_H, pad:pad + BADGE_W] = out_a
    shadow = cv2.GaussianBlur(sh, (0, 0), 12)
    glow = cv2.GaussianBlur(sh, (0, 0), 22)
    return out_rgb.astype(np.float32), out_a.astype(np.float32), shadow, glow, pad


def draw_badge(img, t):
    """Composite the price label onto the finished frame (every frame, never faded)."""
    rgb, a, shadow, glow, pad = badge_layers()
    beat_hits = [w for w, _ in TL.INTRO_WORDS] + [TL.DROP + k * TL.BAR for k in range(40)]
    pulse = 1 + max(0.045 * decay(t, h, 0.12) for h in beat_hits if t >= h) if t >= beat_hits[0] else 1.0
    sc = pulse
    if abs(sc - 1) > 1e-3:
        rgb = cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        shadow = cv2.resize(shadow, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        glow = cv2.resize(glow, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    bh, bw = a.shape
    x0, y0 = int(round(BADGE_CX - bw / 2)), int(round(BADGE_CY - bh / 2))
    sh_h, sh_w = shadow.shape

    def clip_op(layer, x0_, y0_, fn):
        lh, lw = layer.shape[:2]
        cx0, cy0 = max(0, x0_), max(0, y0_)
        cx1, cy1 = min(W, x0_ + lw), min(H, y0_ + lh)
        if cx1 > cx0 and cy1 > cy0:
            fn(img[cy0:cy1, cx0:cx1], layer[cy0 - y0_:cy1 - y0_, cx0 - x0_:cx1 - x0_])

    gx0, gy0 = int(round(BADGE_CX - sh_w / 2)), int(round(BADGE_CY - sh_h / 2))

    def mul(roi_, l_):
        roi_ *= (1 - 0.6 * l_)[..., None]

    def add(roi_, l_):
        roi_ += l_[..., None] * np.array([1.0, 0.25, 0.2], np.float32) * (0.22 + 0.25 * (pulse - 1) / 0.045)
    clip_op(shadow, gx0, gy0 + 8, mul)
    clip_op(glow, gx0, gy0, add)
    roi = img[y0:y0 + bh, x0:x0 + bw]
    roi *= (1 - a[..., None])
    roi += rgb
    # shine sweep every 3 s
    ph = (t % 3.0) / 0.6
    if ph < 1:
        xx = np.arange(bw, dtype=np.float32)[None, :] / bw
        yy = np.arange(bh, dtype=np.float32)[:, None] / bh
        band = np.exp(-((xx + 0.4 * yy - lerp(-0.4, 1.4, ease_in_out(ph))) / 0.07) ** 2) * a * 0.32
        roi += band[..., None]
    return img


# ------------------------------------------------------------- shared effects
def splash_two_tone(cv, cam, u, origin, col_a, col_b, dark_a, dark_b, scale_sim=1.0, core=True):
    """Explosive liquid splash; alternate streams in two colours."""
    sim = R.s1_sim(u) * scale_sim
    streams, drops, red, front = R.SPLASH
    ox, oy = origin
    for k, (col, dark) in enumerate(((col_a, dark_a), (col_b, dark_b))):
        st = streams[k::2]
        dr = drops[k::2]
        discs = R.splash_prims(sim, ox, oy, st, dr)
        if core and k == 0:
            discs.append((ox, oy, 230 * ease_out_cubic(lin(u, 0, 0.12)) * (1 - 0.6 * lin(u, 0.1, 1.2))))
        sd = R.cam_discs(cam, discs)
        render_liquid(cv, discs=[d for d in sd if d[2] > 4], color=col, dark=dark, sigma=4.0, thr=(0.30, 0.52), bump=40)
        for (x, y, r) in sd:
            if 1.0 < r <= 4:
                draw_sprite(cv, sphere_sprite(tuple(col)), x, y, r * 2.2)
    sx, sy = cam.p(ox, oy)
    R.shock_ring(cv, sx, sy, 1700 * ease_out_expo(lin(u, 0, 0.7)), 46 * (1 - lin(u, 0, 0.7)) + 6,
                 tuple(np.clip(np.array(col_a) + 0.3, 0, 1)), 1.1 * (1 - lin(u, 0, 0.7)))


def fg_drops(canvas, cam, u, origin, col, dark, blur=7.0):
    fd = []
    sim = R.s1_sim(u)
    for (x0, y0, r0, vx, vy) in R.SPLASH[3]:
        k = ease_out_cubic(lin(u, 0, 0.4))
        x = origin[0] + (x0 - 540) * (0.3 + 0.7 * k) + vx * sim * 3
        y = origin[1] + (y0 - 960) * (0.3 + 0.7 * k) + vy * sim * 3 + 400 * sim * sim
        fd.append((x, y, r0 * (0.4 + 0.6 * k) * (1 + 0.6 * sim)))
    render_liquid(canvas, discs=R.cam_discs(cam, fd), color=col, dark=dark, sigma=3.0, thr=(0.3, 0.52), bump=30,
                  blur_out=blur, opacity=0.85)


def palette_hue(t, cols):
    return R2.palette_hue(t, cols)


def liquid_wave(canvas, t, t_rise, t_cover, t_clear, pal):
    """Liquid band rising through the frame: covers everything at t_cover, clears by t_clear."""
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
    rr = np.random.default_rng(271)
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
    render_liquid(canvas, discs=discs, polys=[poly], hue_fn=palette_hue(t, pal), sigma=3.4, thr=(0.3, 0.52),
                  bump=30, ripple=rip, opacity=1.0, spec_power=50)


PAL_PURPLE = [(0.62, 0.28, 1.0), (0.2, 0.9, 0.92), (0.85, 0.35, 1.0), (0.3, 0.75, 1.0)]
PAL_BLUE = [(0.1, 0.45, 1.0), (0.2, 0.85, 1.0), (0.05, 0.3, 0.85), (0.45, 0.95, 1.0)]


def vwhip_out(u, direction):
    """Exit whip: returns (shift_y, blur) for the last 0.27 s of a scene. direction -1 = up, +1 = down."""
    if u < TL.EXIT:
        return 0.0, 0.0
    k = ease_in_cubic(lin(u, TL.EXIT, TL.SCENE - 0.02))
    return direction * 2200 * k, 10 + 240 * k


def vwhip_in(u, direction):
    """Entry whip: content arrives from below (direction +1) or above (-1)."""
    if u >= 0.27:
        return 0.0, 0.0
    k = 1 - ease_out_cubic(u / 0.27)
    return direction * 2200 * k, 240 * k


def bubbles(cv, cam, u, p, seed, n, tint=(1, 1, 1), smin=12, smax=32, spd=(200, 420), avoid_can=True, blur=1.2, alpha=0.75):
    rb = np.random.default_rng(seed)
    for j in range(n):
        x0 = rb.uniform(0, 1122)
        if avoid_can and abs(x0 - p.cx) < p.Rc:
            x0 = p.cx + np.sign(x0 - p.cx + 1e-3) * rb.uniform(p.Rc, p.Rc * 2.2)
        y = 1500 - ((u * rb.uniform(*spd) + rb.uniform(0, 1700)) % 1700)
        x, y_ = cam.p(x0 + 18 * np.sin(u * 4 + j), y)
        draw_sprite(cv, R2.bubble_sprite(tuple(tint)), x, y_, cam.s(rb.uniform(smin, smax)), blur=blur, alpha=alpha)


def frost(canvas, t, u, seed=77, n=80, col=(0.9, 1.0, 1.0)):
    r = np.random.default_rng(seed)
    for j in range(n):
        x0, y0 = r.uniform(0, W), r.uniform(0, H)
        ph = r.uniform(0, 6.28)
        k = max(0.0, np.sin(t * 3.1 + ph * 5)) ** 6
        if k > 0.03:
            y = (y0 + 35 * u) % H
            add_glow(canvas, x0, y, r.uniform(4, 9), col, 0.9 * k)
            add_glow(canvas, x0, y, r.uniform(14, 24), col, 0.16 * k, sx=1.6, sy=0.12)


# ------------------------------------------------------------- scene 1 (intro + Cotton Candy)
PINK, PINK_D = (1.0, 0.36, 0.72), (0.55, 0.05, 0.3)
TEAL, TEAL_D = (0.25, 0.92, 0.88), (0.02, 0.4, 0.42)


def scene1(t, fi):
    i, s = 0, TL.DRINK_STARTS[0]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    pre = t < TL.DROP
    u = t - s
    if pre:
        cam = wide_cam(p, keys(t, [(0, 0.92), (2.95, 1.0)]), ay=CAN_CY + 250 - 40 * smooth(t / 3))
    else:
        cam = scene_cam(i, t)
        if u < TL.CLOSE:
            cam.z *= 1 + 0.07 * decay(t, TL.DROP, 0.16)
    amp = 24 * decay(t, TL.DROP, 0.13) + 8 * decay(t, TL.CAN_OPENS[0], 0.1) + sum(7 * decay(t, w, 0.08) for w, _ in TL.INTRO_WORDS)
    shaken(cam, t, amp)
    origin = (p.cx, p.top + (p.bot - p.top) * 0.42)

    def behind(cv):
        if pre:
            g = ease_in_expo(lin(t, 2.2, 3.0))
            if g > 0:
                sx, sy = cam.p(*origin)
                add_glow(cv, sx, sy, 200 + 900 * g, PINK, 0.85 * g)
                add_glow(cv, sx, sy, 80 + 300 * g, (1.0, 0.95, 0.98), 0.8 * g)
            R.dust(cv, t, 5, 70, (1.0, 0.85, 0.95), inten=0.5)
            return
        R.god_rays(cv, t, *cam.p(*origin), (1.0, 0.6, 0.85), 0.2, n=12, speed=0.05)
        R.bokeh(cv, t, 3, 20, [PINK, TEAL, (1.0, 0.8, 0.95)], cam, inten=0.10)
        splash_two_tone(cv, cam, u, origin, PINK, TEAL, PINK_D, TEAL_D)

    if pre:
        rev = lin(t, 0.15, 2.9)
        sweep = [(keys(t, [(0.4, -1.9), (1.8, 1.9)]), 0.42, 1.2 * smooth(lin(t, 0.35, 0.6)) * (1 - smooth(lin(t, 1.65, 1.9)))),
                 (keys(t, [(1.8, 1.9), (2.8, -1.9)]), 0.5, 1.1 * smooth(lin(t, 1.75, 1.95)))]
        rs = 0.15 + 1.0 * smooth(rev)
        canvas = p.render(cam, behind, amb_bg=0.05 + 0.18 * ease_in_cubic(rev),
                          amb_can=0.03 + 0.3 * ease_in_cubic(rev) + 0.5 * ease_in_expo(lin(t, 2.6, 3.0)),
                          amb_props=0.04 + 0.2 * ease_in_cubic(rev), sweep=sweep,
                          rim_l=tuple(np.array(TEAL) * rs * 0.8), rim_r=tuple(np.array(PINK) * rs), protect=False)
        lines = [("ICE COLD.", 720, 290), ("UNBEATABLE", 960, 412), ("DEALS.", 620, 534)]
        fonts = ["InterDisplay-Black.otf", "InterDisplay-BlackItalic.otf", "InterDisplay-Black.otf"]
        explode = ease_in_cubic(lin(t, 2.8, 3.0))
        for k, ((t0, _), (txt, mw, y)) in enumerate(zip(TL.INTRO_WORDS, lines)):
            if t < t0:
                continue
            kk = ease_out_expo(lin(t, t0, t0 + 0.22))
            size = fit_size(txt, fonts[k], mw, 190)
            grad = ((1, 1, 1), (1.0, 0.62, 0.86)) if k == 1 else (((0.85, 1.0, 1.0), (0.45, 0.95, 0.92)) if k == 0 else None)
            draw_text(canvas, txt, fonts[k], size, 540, y, alpha=kk * (1 - explode), scale=lerp(1.45, 1.0, kk) * (1 + 0.5 * explode),
                      blur=(1 - kk) * 14 + explode * 12, glow=PINK if k != 0 else TEAL, glow_r=26,
                      glow_s=0.7 + 0.6 * decay(t, t0, 0.15), grad=grad, shadow=0.4)
        post["flash"] = 0.4 * ease_in_expo(lin(t, 2.75, 3.0)) + sum(0.1 * decay(t, w, 0.05) for w, _ in TL.INTRO_WORDS)
        post["ca"] = sum(6 * decay(t, w, 0.1) for w, _ in TL.INTRO_WORDS)
        return canvas, post

    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, s, s + TL.SCENE)), spec_s=0.16, rim_l=(0.2, 0.6, 0.6), rim_r=(0.6, 0.25, 0.45))
    fg_drops(canvas, cam, u, origin, PINK, PINK_D, blur=6.0 if u < TL.CLOSE else 10.0)
    R2.can_open_fx(canvas, cam, p, t, TL.CAN_OPENS[0], (1.0, 0.92, 0.97))
    tag(canvas, t, s + 0.22, s + 0.98, TL.PRODUCTS[i]["name"], PINK)
    liquid_wave(canvas, t, s + TL.SCENE - 0.4, s + TL.SCENE, s + TL.SCENE + 0.45, PAL_PURPLE)
    post["flash"] = 0.9 * decay(t, TL.DROP, 0.06) + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.25 * decay(t, s + TL.MEDIUM, 0.05)
    post["ca"] = 14 * decay(t, TL.DROP, 0.18) + 5 * decay(t, s + TL.CLOSE, 0.1)
    return canvas, post


# ------------------------------------------------------------- scene 2 Purple Cotton Candy (carnival)
def carnival_lights(cv, cam, t, p):
    cols = [(1.0, 0.75, 0.3), (0.85, 0.45, 1.0), (0.35, 0.95, 0.95), (1.0, 0.5, 0.8)]
    strands = [(-80, 140, 520, 330, 120), (600, 90, 1200, 260, 110), (-60, 520, 380, 640, 90), (720, 480, 1190, 600, 90)]
    for si, (x0, y0, x1, y1, sag) in enumerate(strands):
        n = 13
        for k in range(n):
            f = k / (n - 1)
            x = lerp(x0, x1, f)
            y = lerp(y0, y1, f) + sag * 4 * f * (1 - f)
            if abs(x - p.cx) < p.Rc + 30 and p.top - 40 < y:
                continue
            chase = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(t * 9 - k * 1.3 + si))
            sx, sy = cam.p(x, y)
            c = cols[(k + si) % len(cols)]
            add_glow(cv, sx, sy, cam.s(9), (1, 1, 0.95), 0.9 * chase)
            add_glow(cv, sx, sy, cam.s(34), c, 0.45 * chase)


def confetti(canvas, t, u, seed=55, n=46):
    r = np.random.default_rng(seed)
    cols = [(0.75, 0.4, 1.0), (0.3, 0.92, 0.9), (1.0, 0.85, 0.3), (1.0, 0.45, 0.75)]
    lay = np.zeros((H, W, 3), np.float32)
    for j in range(n):
        x = (r.uniform(0, W) + 60 * np.sin(u * r.uniform(1, 2.5) + j)) % W
        y = (r.uniform(-200, H) + u * r.uniform(220, 420)) % (H + 200) - 100
        ang = u * r.uniform(3, 8) + j
        L, Wd = r.uniform(14, 26), r.uniform(6, 11)
        flip = abs(np.cos(u * r.uniform(4, 9) + j))
        pts = np.array([[-L, -Wd * flip], [L, -Wd * flip], [L, Wd * flip], [-L, Wd * flip]], np.float32)
        Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        P = ((pts @ Rm.T + [x, y]) * 16).astype(np.int32)
        cv2.fillPoly(lay, [P], cols[j % 4], cv2.LINE_AA, 4)
    a = np.clip(lay.max(2), 0, 1)
    blend_over(canvas, lay * 0.95, a * 0.95)


def scene2(t, fi):
    i = 1
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    if u < 0.45:
        cam.z *= lerp(1.15, 1.0, ease_out_cubic(u / 0.45))
    sy_out, vb = vwhip_out(u, -1)
    cam.ay += sy_out
    shaken(cam, t, 10 * decay(t, s, 0.12) + 8 * decay(t, s + TL.CLOSE, 0.1))

    def behind(cv):
        carnival_lights(cv, cam, t, p)
        R.bokeh(cv, t, 8, 18, [(0.8, 0.4, 1.0), (0.3, 0.9, 0.95), (1.0, 0.8, 0.4)], cam, inten=0.11)
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (0.8, 0.55, 1.0), 0.14, n=10, speed=-0.04)
    canvas = p.render(cam, behind, spec=lerp(0.6, -0.6, lin(t, s, s + TL.SCENE)), spec_s=0.16, rim_l=(0.6, 0.3, 0.9), rim_r=(0.25, 0.8, 0.85),
                      backdrop_shift=(0, sy_out * 0.5))
    confetti(canvas, t, u)
    tag(canvas, t, s + 0.3, s + 0.98, TL.PRODUCTS[i]["name"], (0.7, 0.4, 1.0))
    liquid_wave(canvas, t, s - 0.4, s, s + 0.45, PAL_PURPLE)
    if vb > 1:
        canvas = vblur(canvas, vb)
    post["flash"] = 0.3 * decay(t, s, 0.08) + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.25 * decay(t, s + TL.MEDIUM, 0.05)
    post["ca"] = 6 * decay(t, s + TL.CLOSE, 0.1) + 8 * lin(u, TL.EXIT, TL.SCENE)
    return canvas, post


# ------------------------------------------------------------- scene 3 Witch's Brew
GREEN = (0.45, 1.0, 0.25)


def scene3(t, fi):
    i = 2
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    sy_in, vb_in = vwhip_in(u, +1)
    cam.ay += sy_in
    shaken(cam, t, 8 * decay(t, s + TL.CLOSE, 0.1))

    def behind(cv):
        sx, sy = cam.p(p.cx, p.bot)
        add_glow(cv, sx, sy, cam.s(700), GREEN, 0.32, sy=0.55)
        R2.smoke(cv, t, 0.45, (0.25, 0.08, 0.35), seed=81, sx=30, sy=-50)
        bubbles(cv, cam, u, p, 41, 34, tint=(0.6, 1.0, 0.45), smin=14, smax=34, spd=(260, 520))
        R2.embers(cv, u, 9, 34, cam, (p.cx, p.top + 560), color=(0.7, 0.3, 1.0), mode="orbit", front=False, inten=0.9)
    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, s, s + TL.SCENE)), spec_s=0.15, rim_l=(0.6, 0.3, 0.9), rim_r=(0.4, 1.0, 0.3),
                      backdrop_shift=(0, sy_in * 0.5))
    R2.embers(canvas, u, 9, 34, cam, (p.cx, p.top + 560), color=(0.7, 0.3, 1.0), mode="orbit", front=True)
    bubbles(canvas, cam, u, p, 42, 14, tint=(0.6, 1.0, 0.45), smin=24, smax=60, spd=(300, 560), blur=0.6, alpha=0.9)
    # green smoke wipe out
    wipe = ease_in_cubic(lin(u, TL.SCENE - 0.42, TL.SCENE - 0.02))
    yy = np.arange(H, dtype=np.float32)
    low = np.clip((yy - 1320) / 500, 0, 1)
    dens = np.clip(low * 0.45 + wipe * 3.2, 0, 4)[:, None]
    n1 = scroll(fractal_noise(87, 540, 960, 6), u * 50, -u * 170)
    n = cv2.resize(np.clip((n1 - 0.25) * 2.2, 0, 1), (W, H))
    a = np.clip(n * dens + wipe ** 1.5 * 1.3, 0, 1)[..., None]
    canvas = canvas * (1 - a) + a * np.array([0.10, 0.22, 0.06], np.float32)
    tag(canvas, t, s + 0.3, s + 0.98, TL.PRODUCTS[i]["name"], GREEN)
    if vb_in > 1:
        canvas = vblur(canvas, vb_in)
    post["flash"] = 0.35 * decay(t, s, 0.08) + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, s + TL.MEDIUM, 0.05)
    post["ca"] = 8 * (1 - lin(u, 0, 0.3)) + 5 * decay(t, s + TL.CLOSE, 0.1)
    return canvas, post


# ------------------------------------------------------------- scene 4 Hawaiian Shaved Ice
ICE4 = []
_r = np.random.default_rng(44)
_lx = [120, 1010, 190, 950, 1050]
for _k, _th in enumerate(TL.ICE_HITS_S4):
    ICE4.append(dict(x=_lx[_k], th=_th, fall=_r.uniform(0.38, 0.5), size=_r.uniform(115, 150), rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3.5, 3.5, 3)))
for _k in range(6):
    ICE4.append(dict(x=_r.uniform(820, 1100) if _k % 2 else _r.uniform(240, 330), t0=TL.DRINK_STARTS[3] + _r.uniform(0, 1.9),
                     size=_r.uniform(60, 90), rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3, 3, 3), vy=_r.uniform(500, 800)))


def ice_state(o, t, p):
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


def scene4(t, fi):
    i = 3
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    sy_out, vb = vwhip_out(u, +1)
    cam.ay += sy_out
    shaken(cam, t, 8 * decay(t, s + TL.CLOSE, 0.1) + sum(5 * decay(t, h, 0.08) for h in TL.ICE_HITS_S4))

    def behind(cv):
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 250), (0.75, 0.9, 1.0), 0.12, n=9, speed=0.03, sharp=6)
        R.bokeh(cv, t, 14, 16, [(1.0, 0.3, 0.6), (0.4, 0.9, 1.0), (0.8, 0.4, 1.0)], cam, inten=0.10)
        R.mist_layer(cv, t, np.full((H, W), 0.2, np.float32), color=(0.85, 0.95, 1.0), seed=51, sx=25, sy=-8)
        for o in ICE4:
            if "th" not in o:
                st = ice_state(o, t, p)
                if st:
                    x, y = cam.p(st[0], st[1])
                    draw_ice(cv, x, y, cam.s(o["size"]), *st[2], blur=2.5, alpha=0.85)
    canvas = p.render(cam, behind, spec=lerp(0.6, -0.6, lin(t, s, s + TL.SCENE)), spec_s=0.18, rim_l=(0.5, 0.85, 1.0), rim_r=(1.0, 0.4, 0.6),
                      backdrop_shift=(0, sy_out * 0.5))
    xx, yy = fx.grid()
    fy_ = cam.p(0, p.bot)[1]
    fog = np.clip(1 - np.abs(yy - fy_) / cam.s(240), 0, 1) ** 1.2 * 0.42
    R.mist_layer(canvas, t, fog, color=(0.88, 0.96, 1.0), seed=61, sx=40, sy=-70)
    for o in ICE4:
        if "th" in o:
            st = ice_state(o, t, p)
            if st:
                x, y = cam.p(st[0], st[1])
                draw_ice(canvas, x, y, cam.s(o["size"]), *st[2], blur=0.8)
    frost(canvas, t, u)
    # smoke from scene 3 clearing
    if u < 0.42:
        R2.smoke(canvas, t, 3.0 * (1 - ease_out_cubic(u / 0.42)), (0.10, 0.22, 0.06), seed=87)
    tag(canvas, t, s + 0.32, s + 0.98, TL.PRODUCTS[i]["name"], (0.4, 0.85, 1.0))
    if vb > 1:
        canvas = vblur(canvas, vb)
    post["flash"] = 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, s + TL.MEDIUM, 0.05)
    post["flash_color"] = (0.95, 0.98, 1.0)
    post["ca"] = 5 * decay(t, s + TL.CLOSE, 0.1) + 8 * lin(u, TL.EXIT, TL.SCENE)
    return canvas, post


# ------------------------------------------------------------- scene 5 Orange Kiss
ORANGE, ORANGE_D = (1.0, 0.55, 0.08), (0.6, 0.22, 0.0)
CORAL, CORAL_D = (1.0, 0.35, 0.45), (0.55, 0.06, 0.15)


def scene5(t, fi):
    i = 4
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    sy_in, vb_in = vwhip_in(u, -1)
    cam.ay += sy_in
    shaken(cam, t, 16 * decay(t, s + 0.05, 0.12) + 8 * decay(t, s + TL.CLOSE, 0.1))
    origin = (p.cx, p.top + (p.bot - p.top) * 0.42)
    flare = ease_in_cubic(lin(u, TL.SCENE - 0.35, TL.SCENE))

    def behind(cv):
        R.god_rays(cv, t, *cam.p(p.cx - 250, p.top - 50), (1.0, 0.75, 0.35), 0.28, n=14, speed=0.04, sharp=8)
        add_glow(cv, *cam.p(p.cx - 300, p.top), cam.s(700), (1.0, 0.65, 0.25), 0.3)
        splash_two_tone(cv, cam, max(0.0, u - 0.05), origin, ORANGE, CORAL, ORANGE_D, CORAL_D, core=False)
        bubbles(cv, cam, u, p, 51, 26, tint=(1.0, 0.85, 0.6), smin=10, smax=26, spd=(240, 460))
    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, s, s + TL.SCENE)), spec_s=0.16, rim_l=(1.0, 0.6, 0.25), rim_r=(1.0, 0.4, 0.5),
                      backdrop_shift=(0, sy_in * 0.5))
    fg_drops(canvas, cam, max(0.0, u - 0.05), origin, ORANGE, ORANGE_D, blur=6.0 if u < TL.CLOSE else 10.0)
    bubbles(canvas, cam, u, p, 52, 10, tint=(1.0, 0.9, 0.7), smin=22, smax=50, spd=(300, 520), blur=0.5, alpha=0.95)
    if flare > 0:   # sun-flare burst transition
        sx, sy = cam.p(p.cx - 280, p.top - 40)
        add_glow(canvas, sx, sy, 300 + 1700 * flare, (1.0, 0.8, 0.45), 1.6 * flare)
        add_glow(canvas, sx, sy, 900 + 1200 * flare, (1.0, 0.95, 0.8), 1.2 * flare, sx=1.0, sy=0.06)
    tag(canvas, t, s + 0.3, s + 0.98, TL.PRODUCTS[i]["name"], ORANGE)
    if vb_in > 1:
        canvas = vblur(canvas, vb_in)
    post["flash"] = 0.3 * decay(t, s, 0.08) + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, s + TL.MEDIUM, 0.05) + ease_in_expo(flare) * 0.95
    post["flash_color"] = (1.0, 0.93, 0.8)
    post["ca"] = 8 * (1 - lin(u, 0, 0.3)) + 5 * decay(t, s + TL.CLOSE, 0.1)
    return canvas, post


# ------------------------------------------------------------- scene 6 Cherry Slush (teal lightning)
def scene6(t, fi):
    i = 5
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    cam.z *= 1 + sum(0.03 * decay(t, st, 0.12) for st in TL.STRIKES_S6 if t >= st) if u >= TL.MEDIUM or u < TL.CLOSE else 1.0
    amp = sum((16 if st == s else 9) * decay(t, st, 0.1) for st in TL.STRIKES_S6 if t >= st)
    shaken(cam, t, amp, 3.0)
    last = max([st for st in TL.STRIKES_S6 if st <= t], default=None)
    fl = np.exp(-(t - last) / 0.07) * (1.0 if last == s else 0.6) if last is not None else 0.0

    def behind(cv):
        cv *= 0.8 + 0.6 * fl
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (0.5, 1.0, 0.95), 0.12 + 0.3 * fl, n=16, speed=0.12, sharp=14)
        R2.bolts(cv, cam, t, TL.STRIKES_S6, p, color=(0.35, 1.0, 0.95), big=(s,))
    canvas = p.render(cam, behind, amb_bg=0.85, spec=lerp(0.6, -0.6, lin(t, s, s + TL.SCENE)), spec_s=0.16,
                      rim_l=tuple(np.array([0.3, 1.0, 0.95]) * (0.5 + 1.2 * fl)), rim_r=tuple(np.array([1.0, 0.3, 0.35]) * (0.5 + 0.8 * fl)),
                      hero_gain=1.0 + 0.1 * fl)
    R2.sparks(canvas, cam, t, TL.STRIKES_S6, (p.cx, p.bot - 60), color=(1.0, 0.4, 0.45), n=38)
    R.dust(canvas, t, 16, 34, (1.0, 0.5, 0.55), inten=0.45, vy=-60)
    tag(canvas, t, s + 0.3, s + 0.98, TL.PRODUCTS[i]["name"], (1.0, 0.25, 0.3))
    liquid_wave(canvas, t, s + TL.SCENE - 0.4, s + TL.SCENE, s + TL.SCENE + 0.45, PAL_BLUE)
    post["flash"] = (1 - ease_out_cubic(lin(u, 0, 0.22))) * 0.95 + 0.45 * fl + 0.25 * decay(t, s + TL.CLOSE, 0.05)
    post["flash_color"] = (1.0, 0.95, 0.85) if u < 0.25 else (0.92, 1.0, 1.0)
    post["ca"] = 8 * fl
    return canvas, post


# ------------------------------------------------------------- scene 7 Breezeberry
BLUE, BLUE_D = (0.15, 0.55, 1.0), (0.02, 0.15, 0.5)


def warp_streaks(canvas, k, cx=540, cy=900, seed=5, col=(0.85, 0.8, 1.0), outward=True):
    """Radial hyperspace streaks (overlay only - the can is not scaled)."""
    if k <= 0.01:
        return
    r = np.random.default_rng(seed)
    lay = np.zeros((H // 2, W // 2), np.float32)
    for j in range(160):
        a = r.uniform(0, 6.28)
        d0 = r.uniform(60, 1100)
        L = (80 + 900 * k) * r.uniform(0.4, 1.0)
        d0 = d0 + (600 * k * r.uniform(0.5, 1.5) if outward else 0)
        p0 = (cx + np.cos(a) * d0, cy + np.sin(a) * d0)
        p1 = (cx + np.cos(a) * (d0 + L), cy + np.sin(a) * (d0 + L))
        cv2.line(lay, (int(p0[0] / 2), int(p0[1] / 2)), (int(p1[0] / 2), int(p1[1] / 2)), float(r.uniform(0.5, 1)), 1, cv2.LINE_AA)
    g = cv2.GaussianBlur(lay, (0, 0), 1.0) + cv2.GaussianBlur(lay, (0, 0), 4) * 0.8
    canvas += cv2.resize(g, (W, H))[..., None] * np.array(col, np.float32) * min(1.0, k * 1.6)


def scene7(t, fi):
    i = 6
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    shaken(cam, t, 8 * decay(t, s + TL.CLOSE, 0.1))
    warp = ease_in_cubic(lin(u, TL.SCENE - 0.4, TL.SCENE))

    def behind(cv):
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 250), (0.5, 0.8, 1.0), 0.16, n=10, speed=-0.03)
        rr = np.random.default_rng(19)
        drops = []
        for j in range(30):
            x0 = rr.uniform(0, 1122)
            if abs(x0 - p.cx) < p.Rc:
                x0 += np.sign(x0 - p.cx + 1e-3) * p.Rc
            drops.append((x0 + 20 * np.sin(u * 2 + j), (rr.uniform(-100, 1300) + rr.uniform(90, 200) * u) % 1450 - 60, rr.uniform(8, 18)))
        render_liquid(cv, discs=R.cam_discs(cam, drops), color=BLUE, dark=BLUE_D, sigma=2.8, thr=(0.3, 0.52), bump=30)
        R.mist_layer(cv, t, np.full((H, W), 0.16, np.float32), color=(0.7, 0.85, 1.0), seed=71, sx=20, sy=-10)
    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, s, s + TL.SCENE)), spec_s=0.18, rim_l=(0.3, 0.6, 1.0), rim_r=(0.3, 0.95, 1.0))
    rr = np.random.default_rng(23)
    fdrops = [(rr.uniform(0, 1122), (rr.uniform(-100, 1300) + rr.uniform(160, 280) * u) % 1450 - 60, rr.uniform(16, 30)) for _ in range(8)]
    render_liquid(canvas, discs=R.cam_discs(cam, fdrops), color=(0.3, 0.7, 1.0), dark=BLUE_D, sigma=3.0, thr=(0.3, 0.52), bump=30,
                  blur_out=8.0, opacity=0.85)
    frost(canvas, t, u, seed=78, n=70, col=(0.8, 0.92, 1.0))
    warp_streaks(canvas, warp, col=(0.8, 0.75, 1.0))
    tag(canvas, t, s + 0.45, s + 0.98, TL.PRODUCTS[i]["name"], (0.3, 0.7, 1.0))
    liquid_wave(canvas, t, s - 0.4, s, s + 0.45, PAL_BLUE)
    post["flash"] = 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, s + TL.MEDIUM, 0.05) + ease_in_expo(warp) * 0.95
    post["flash_color"] = (0.9, 0.85, 1.0)
    post["ca"] = 5 * decay(t, s + TL.CLOSE, 0.1) + 12 * warp
    return canvas, post


# ------------------------------------------------------------- scene 8 Cosmic Stardust
@lru_cache(maxsize=1)
def nebula():
    n1 = fractal_noise(201, 540, 960, 6)
    n2 = fractal_noise(202, 540, 960, 6)
    a = np.clip((n1 - 0.45) * 2.4, 0, 1)
    b = np.clip((n2 - 0.5) * 2.4, 0, 1)
    img = a[..., None] * np.array([0.55, 0.2, 0.85], np.float32) + b[..., None] * np.array([0.1, 0.75, 0.8], np.float32)
    return cv2.resize(cv2.GaussianBlur(img, (0, 0), 6), (W, H))


def starfield(cv, t, n=170, seed=33):
    r = np.random.default_rng(seed)
    for j in range(n):
        x, y = r.uniform(0, W), r.uniform(0, H)
        ph = r.uniform(0, 6.28)
        k = 0.4 + 0.6 * max(0.0, np.sin(t * r.uniform(1.5, 4) + ph))
        sz = r.uniform(2, 5)
        add_glow(cv, x, y, sz, (1, 1, 1), 0.8 * k)
        if j % 9 == 0:
            add_glow(cv, x, y, sz * 7, (0.8, 0.85, 1.0), 0.2 * k, sx=1.0, sy=0.08)


def scene8(t, fi):
    i = 7
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    sy_out, vb = vwhip_out(u, -1)
    cam.ay += sy_out
    shaken(cam, t, 8 * decay(t, s + TL.CLOSE, 0.1))
    warp_in = 1 - ease_out_cubic(lin(u, 0, 0.5))

    def behind(cv):
        cv += nebula() * 0.28 * (0.8 + 0.2 * np.sin(t * 0.8))
        starfield(cv, t)
        R2.embers(cv, u, 12, 30, cam, (p.cx, p.top + 560), color=(0.5, 0.9, 1.0), mode="orbit", front=False, inten=0.8)
    canvas = p.render(cam, behind, spec=lerp(0.6, -0.6, lin(t, s, s + TL.SCENE)), spec_s=0.16, rim_l=(0.7, 0.4, 1.0), rim_r=(0.3, 0.95, 0.95),
                      backdrop_shift=(0, sy_out * 0.5))
    R2.embers(canvas, u, 12, 30, cam, (p.cx, p.top + 560), color=(0.5, 0.9, 1.0), mode="orbit", front=True, inten=0.8)
    warp_streaks(canvas, warp_in, col=(0.85, 0.8, 1.0))
    tag(canvas, t, s + 0.4, s + 0.98, TL.PRODUCTS[i]["name"], (0.75, 0.45, 1.0))
    if vb > 1:
        canvas = vblur(canvas, vb)
    post["flash"] = (1 - ease_out_cubic(lin(u, 0, 0.24))) * 0.95 + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, s + TL.MEDIUM, 0.05)
    post["flash_color"] = (0.9, 0.85, 1.0)
    post["ca"] = 12 * warp_in + 5 * decay(t, s + TL.CLOSE, 0.1) + 8 * lin(u, TL.EXIT, TL.SCENE)
    return canvas, post


# ------------------------------------------------------------- scene 9 Watermelon Wave
WM_PINK, WM_PINK_D = (1.0, 0.3, 0.5), (0.55, 0.03, 0.15)
WM_TEAL, WM_TEAL_D = (0.25, 0.9, 0.8), (0.02, 0.38, 0.33)


def scene9(t, fi):
    i = 8
    s = TL.DRINK_STARTS[i]
    p = plate(i)
    post = dict(flash=0.0, ca=0.0)
    u = t - s
    cam = scene_cam(i, t)
    sy_in, vb_in = vwhip_in(u, +1)
    sy_out, vb_out = vwhip_out(u, -1)
    cam.ay += sy_in + sy_out
    shaken(cam, t, 16 * decay(t, s + 0.05, 0.13) + 8 * decay(t, TL.CAN_OPENS[1], 0.1))
    origin = (p.cx, p.top + (p.bot - p.top) * 0.62)

    def behind(cv):
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (1.0, 0.55, 0.7), 0.16, n=12, speed=0.05)
        R.bokeh(cv, t, 21, 16, [WM_PINK, WM_TEAL, (0.6, 1.0, 0.4)], cam, inten=0.1)
        splash_two_tone(cv, cam, max(0.0, u - 0.05), origin, WM_PINK, WM_TEAL, WM_PINK_D, WM_TEAL_D, core=False)
        bubbles(cv, cam, u, p, 61, 22, tint=(1.0, 0.85, 0.9), smin=10, smax=26, spd=(240, 460))
    canvas = p.render(cam, behind, spec=lerp(-0.6, 0.6, lin(t, s, s + TL.SCENE)), spec_s=0.16, rim_l=(0.3, 0.9, 0.8), rim_r=(1.0, 0.35, 0.55),
                      backdrop_shift=(0, (sy_in + sy_out) * 0.5))
    fg_drops(canvas, cam, max(0.0, u - 0.05), origin, WM_PINK, WM_PINK_D, blur=6.0 if u < TL.CLOSE else 10.0)
    R2.can_open_fx(canvas, cam, p, t, TL.CAN_OPENS[1], (1.0, 0.9, 0.95))
    tag(canvas, t, s + 0.3, s + 0.98, TL.PRODUCTS[i]["name"], WM_PINK)
    vb = max(vb_in, vb_out)
    if vb > 1:
        canvas = vblur(canvas, vb)
    post["flash"] = 0.3 * decay(t, s, 0.08) + 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, s + TL.MEDIUM, 0.05)
    post["flash"] += 0.85 * ease_in_expo(lin(u, TL.SCENE - 0.15, TL.SCENE))
    post["ca"] = 8 * (1 - lin(u, 0, 0.3)) + 5 * decay(t, s + TL.CLOSE, 0.1) + 8 * lin(u, TL.EXIT, TL.SCENE)
    return canvas, post


# ------------------------------------------------------------- end card
LOGO_W = 800
ADDR = ("153 Great Northern Rd,", "Sault Ste. Marie, ON P6B 4Y9")
CTA1, CTA2 = "STOP BY & GRAB", "YOUR FAVOURITES!"
LEAKS3 = [((1.0, 0.36, 0.72), 120, 380), ((0.25, 0.9, 0.9), 980, 520), ((0.62, 0.3, 1.0), 160, 1500), ((1.0, 0.55, 0.15), 950, 1650)]
LOGO_CY, ADDR_CY, OFFER_CY, SUB_CY, CTA_CY = 520, 812, 1000, 1094, 1290
ROW_CX = 600      # rows beside the price label are centred right of the label column


@lru_cache(maxsize=1)
def logo_card():
    img = np.asarray(Image.open(TLOC.LOGO).convert("RGB")).astype(np.float32) / 255
    h0, w0 = img.shape[:2]
    cw, ch = LOGO_W, int(round(h0 * LOGO_W / w0))
    pic = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
    pad = 80
    m = Image.new("L", (cw * 4 + 2 * pad * 4, ch * 4 + 2 * pad * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((pad * 4, pad * 4, (pad + cw) * 4 - 1, (pad + ch) * 4 - 1), 24 * 4, fill=255)
    m = cv2.resize(np.asarray(m).astype(np.float32) / 255, (cw + 2 * pad, ch + 2 * pad), interpolation=cv2.INTER_AREA)
    rgb = np.zeros((m.shape[0], m.shape[1], 3), np.float32)
    rgb[pad:pad + ch, pad:pad + cw] = pic
    edge = np.clip((cv2.GaussianBlur(m, (0, 0), 1.2) - cv2.GaussianBlur(cv2.erode(m, np.ones((5, 5), np.uint8)), (0, 0), 1.2)) * 2, 0, 1)
    return dict(rgb=rgb, a=m, edge=edge, shadow=cv2.GaussianBlur(m, (0, 0), 26), glow=cv2.GaussianBlur(m, (0, 0), 40),
                w=m.shape[1], h=m.shape[0], ch=ch)


def end_card(t, fi):
    post = dict(flash=0.0, ca=0.0, grain=0.014)
    e = t - TL.END_START
    shift = 1800 * (1 - ease_out_cubic(lin(t, TL.END_START, TL.END_START + 0.28)))
    canvas = radial_bg((0.08, 0.05, 0.13), (0.005, 0.004, 0.014), 540, 860, 1000, 1300, 1.2).copy()
    for k, (col, x, y) in enumerate(LEAKS3):
        add_glow(canvas, x + 90 * np.sin(e * 0.5 + k), y + 70 * np.cos(e * 0.4 + k * 2) + shift, 620, col, 0.15)
    R.god_rays(canvas, t, 540, LOGO_CY + shift, R.GOLD, 0.10 + 0.1 * decay(t, TL.END_LOGO, 0.5), n=18, speed=0.03, sharp=8)
    add_glow(canvas, 540, LOGO_CY + shift, 820, R.GOLD, 0.15 * smooth(lin(t, TL.END_LOGO, TL.END_LOGO + 0.6)))
    R.bokeh(canvas, t, 31, 16, [(1.0, 0.36, 0.72), (0.25, 0.9, 0.9), (0.62, 0.3, 1.0), R.GOLD], inten=0.08, rmin=30, rmax=90, speed=14)
    R.dust(canvas, t, 21, 150, (1.0, 0.85, 0.5), inten=0.7, rmin=2, rmax=6, vy=-40)
    yo = shift
    # logo picture (unaltered) on a card
    L = logo_card()
    k = ease_out_expo(lin(t, TL.END_LOGO, TL.END_LOGO + 0.55))
    if k > 0:
        punch = 1 + 0.03 * decay(t, TL.END_FINAL, 0.25) if t >= TL.END_FINAL else 1.0
        sc = lerp(1.12, 1.0, k) * (1 + 0.01 * np.sin(e * 0.9)) * punch
        cy = LOGO_CY + yo + 5 * np.sin(e * 1.1)
        rv = ease_in_out(lin(t, TL.END_LOGO, TL.END_LOGO + 0.42))
        xs = (np.arange(L["w"], dtype=np.float32) - L["w"] / 2) / (LOGO_W / 2)
        reveal = np.clip((rv * 1.08 - np.abs(xs)) / 0.06, 0, 1)[None, :]
        M = np.float32([[sc, 0, 540 - sc * L["w"] / 2], [0, sc, cy - sc * L["h"] / 2]])
        warp = lambda a, fl=cv2.INTER_LINEAR: cv2.warpAffine(a, M, (W, H), flags=fl, borderMode=cv2.BORDER_CONSTANT)
        a_card = L["a"] * reveal
        canvas *= 1 - np.roll(warp(L["shadow"] * reveal), 20, axis=0)[..., None] * 0.7 * k
        canvas += warp(L["glow"] * reveal)[..., None] * np.array([1.0, 0.92, 0.85], np.float32) * 0.32 * k
        lay = np.clip(warp(np.dstack([L["rgb"] * a_card[..., None], a_card]), cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC), 0, None)
        if k < 0.98:
            lay = cv2.GaussianBlur(lay, (0, 0), 0.3 + 8 * (1 - k))
        blend_over(canvas, lay[..., :3] * k, lay[..., 3] * k)
        canvas += warp(L["edge"] * reveal)[..., None] * np.array(R.GOLD, np.float32) * 0.8 * k
        if 0 < rv < 1:
            for sg in (-1, 1):
                add_glow(canvas, 540 + sg * rv * 1.08 * LOGO_W / 2 * sc, cy, sc * L["ch"] * 0.7, (1.0, 0.95, 0.8), 0.9, sx=0.08, sy=1.0)
        for s0 in TL.END_SHINES + (TL.END_FINAL,):
            if s0 <= t < s0 + 0.75:
                pos = lerp(-0.35, 1.35, ease_in_out((t - s0) / 0.75))
                xx = np.arange(L["w"], dtype=np.float32)[None, :] / L["w"]
                yy = np.arange(L["h"], dtype=np.float32)[:, None] / L["h"]
                canvas += warp(np.exp(-((xx + 0.35 * yy - pos) / 0.05) ** 2) * L["a"] * 0.3)[..., None]
    # address (pin + two lines on a glass panel)
    ka = ease_out_cubic(lin(t, TL.END_ADDR, TL.END_ADDR + 0.42))
    if ka > 0:
        pin = TLOC.pin_sprite(70)
        s_st = fit_size(ADDR[0], "InterDisplay-SemiBold.otf", 560, 56)
        s_ct = fit_size(ADDR[1], "InterDisplay-Medium.otf", 560, 44)
        f_st, f_ct = fx.font("InterDisplay-SemiBold.otf", s_st), fx.font("InterDisplay-Medium.otf", s_ct)
        bw = max(f_st.getlength(ADDR[0]), f_ct.getlength(ADDR[1]))
        row_w = pin.shape[1] + 24 + bw
        cy = ADDR_CY + yo + 26 * (1 - ka)
        pw, ph = int(row_w + 70), 136
        pm = TLOC.rounded_panel(pw, ph, 22)
        px0, py0 = int(ROW_CX - pw / 2), int(cy - ph / 2)
        fill = pm * 0.07 * ka
        blend_over(canvas, np.ones((ph, pw, 3), np.float32) * fill[..., None], fill, px0, py0)
        border = np.clip(cv2.GaussianBlur(pm, (0, 0), 0.8) - cv2.erode(pm, np.ones((5, 5), np.uint8)), 0, 1)
        blend_add(canvas, (border * 0.22 * ka)[..., None] * np.array([1.0, 0.92, 0.75], np.float32), px0, py0)
        x_pin = ROW_CX - row_w / 2
        kp = ease_out_back(lin(t, TL.END_ADDR + 0.08, TL.END_ADDR + 0.45), 2.4)
        if kp > 0:
            pa = clamp(lin(t, TL.END_ADDR + 0.08, TL.END_ADDR + 0.2))
            add_glow(canvas, x_pin + pin.shape[1] / 2, cy + 4, 46, (1.0, 0.3, 0.3), 0.25 * pa)
            blend_over(canvas, pin[..., :3] * pa, pin[..., 3] * pa, int(x_pin), int(cy - pin.shape[0] / 2 - 46 * (1 - kp)))
        x_txt = x_pin + pin.shape[1] + 24
        TLOC.text_left(canvas, ADDR[0], "InterDisplay-SemiBold.otf", s_st, x_txt, cy - s_ct * 0.62, alpha=ka, blur=(1 - ka) * 5, shadow=0.45)
        TLOC.text_left(canvas, ADDR[1], "InterDisplay-Medium.otf", s_ct, x_txt, cy + s_st * 0.52, alpha=ka * 0.9,
                       color=(0.92, 0.9, 0.95), blur=(1 - ka) * 5, shadow=0.45)
    # offer (kept right of the price label column)
    ko = ease_out_back(lin(t, TL.END_OFFER, TL.END_OFFER + 0.4), 1.8)
    koa = clamp(lin(t, TL.END_OFFER, TL.END_OFFER + 0.15))
    if koa > 0:
        offer = "2 FOR $5.99"
        so = fit_size(offer, "InterDisplay-Black.otf", 600, 120)
        draw_text(canvas, offer, "InterDisplay-Black.otf", so, ROW_CX, OFFER_CY + yo, alpha=koa, scale=max(0.01, ko),
                  grad=((1.0, 0.97, 0.7), (1.0, 0.72, 0.2)), glow=(1.0, 0.55, 0.2), glow_r=24, glow_s=0.8 + 0.5 * decay(t, TL.END_OFFER, 0.2),
                  shadow=0.6)
        draw_text(canvas, "BEFORE TAXES", "InterDisplay-Bold.otf", 38, ROW_CX, SUB_CY + yo, alpha=koa * clamp(lin(t, TL.END_OFFER + 0.1, TL.END_OFFER + 0.35)),
                  tracking=0.18, shadow=0.5)
    # CTA pill
    kc = ease_out_back(lin(t, TL.END_CTA, TL.END_CTA + 0.38), 2.2)
    kca = lin(t, TL.END_CTA, TL.END_CTA + 0.15)
    if kca > 0:
        cs = min(fit_size(CTA1, "InterDisplay-Black.otf", 560, 66), fit_size(CTA2, "InterDisplay-Black.otf", 560, 66))
        f2 = fx.font("InterDisplay-Black.otf", cs)
        tw = max(f2.getlength(CTA1), f2.getlength(CTA2))
        pw, ph = tw + 100, cs * 2.55
        cy = CTA_CY + yo
        pulse = 1 + 0.018 * np.sin(max(0, t - TL.END_CTA - 0.4) * 5.5) * (t > TL.END_CTA + 0.4)
        sc = max(0.01, kc) * pulse
        pad = 30
        lw, lh = int(pw * sc) + 2 * pad, int(ph * sc) + 2 * pad
        pm = np.zeros((lh, lw), np.float32)
        rr_ = int(ph * sc / 2)
        cv2.rectangle(pm, (pad + rr_, pad), (lw - pad - rr_, lh - pad), 1.0, -1, cv2.LINE_AA)
        cv2.circle(pm, (pad + rr_, lh // 2), rr_, 1.0, -1, cv2.LINE_AA)
        cv2.circle(pm, (lw - pad - rr_, lh // 2), rr_, 1.0, -1, cv2.LINE_AA)
        gy = np.linspace(0, 1, lh, dtype=np.float32)[:, None, None]
        col = np.array([1.0, 0.86, 0.45], np.float32) * (1 - gy) + np.array([0.98, 0.60, 0.16], np.float32) * gy
        a = pm * kca
        blend_add(canvas, (cv2.GaussianBlur(pm, (0, 0), 16) * 0.55 * kca)[..., None] * np.array(R.GOLD, np.float32), int(540 - lw / 2), int(cy - lh / 2))
        blend_over(canvas, col * a[..., None], a, int(540 - lw / 2), int(cy - lh / 2))
        for ln, dy_ in ((CTA1, -cs * 0.58), (CTA2, cs * 0.58)):
            draw_text(canvas, ln, "InterDisplay-Black.otf", cs, 540, cy + dy_ * sc, color=(0.07, 0.05, 0.10), alpha=kca, scale=sc)
    if shift > 1:
        canvas = vblur(canvas, shift * 0.12)
    zs = 1 + 0.03 * ease_in_out(lin(t, TL.END_START + 0.3, TL.DURATION))
    M = np.float32([[zs, 0, 540 - zs * 540], [0, zs, 960 - zs * 960]])
    canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    post["flash"] = 0.85 * (1 - ease_out_cubic(lin(t, TL.END_START, TL.END_START + 0.2)))
    post["flash"] += 0.18 * decay(t, TL.END_LOGO, 0.06) + (0.12 * decay(t, TL.END_FINAL, 0.08) if t >= TL.END_FINAL else 0.0)
    post["flash"] += 0.12 * decay(t, TL.END_OFFER, 0.06) if t >= TL.END_OFFER else 0.0
    post["flash_color"] = (1.0, 0.95, 0.85)
    post["ca"] = 6 * decay(t, TL.END_LOGO, 0.12)
    post["fade"] = 1 - smooth(lin(t, TL.END_FADE, TL.DURATION - 1.0 / FPS))
    return canvas, post


# ------------------------------------------------------------- dispatcher
SCENES = [scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9]


def scene_index(t):
    if t < TL.DRINK_STARTS[1]:
        return 0
    if t >= TL.END_START:
        return None
    return min(8, int((t - TL.DROP) / TL.SCENE))


def render_frame(fi):
    t = fi / FPS
    k = scene_index(t)
    canvas, post = (end_card(t, fi) if k is None else SCENES[k](t, fi))
    img = finish(canvas, fi, flash=min(1.0, post.get("flash", 0)), flash_color=post.get("flash_color", (1, 1, 1)),
                 ca=post.get("ca", 0), grain=post.get("grain", 0.014))
    if "fade" in post:
        img *= post["fade"]
    img = draw_badge(img, t)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


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
            if (k + 1) % 60 == 0:
                print(f"{k + 1}/{n} frames  {time.time() - t0:.0f}s", flush=True)
    print("render done", time.time() - t0)


if __name__ == "__main__":
    main()
