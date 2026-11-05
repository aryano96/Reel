"""Renders the Ezy Pzy Convenience product reel frame-by-frame.

    python render.py --test 1.0,3.4,8.0     # preview stills -> output/build/test/
    python render.py --all                   # full render + MP4 encode

Scene plan (all times in seconds, locked to the 120 BPM soundtrack):
    0.0 - 3.0   Intro: dark stage, light sweeps reveal Cherry Limeade, text slams, bass drop
    3.0 - 7.0   Cherry Limeade : explosive lime/cherry liquid splash, slow-mo, close-up, can crack
    7.0 - 11.0  Rainbow Candy  : rainbow liquid wave wipes in/out, orbiting candy orbs
   11.0 - 15.0  Green Apple    : whip-pan in, tumbling ice cubes, cold mist, vapour burst
   15.0 - 19.0  Banana Pineapple: lightning strike reveal, electric arcs, sparks, punch-ins
   19.0 - 24.5  Ezy Pzy Convenience end card with full address + call to action
Only ONE can is ever drawn in any frame (transitions fully clear one product
before the next appears).
"""
import os
import sys
import time
import argparse
import subprocess
from multiprocessing import Pool

import numpy as np
import cv2

import timeline as TL
from timeline import W, H, FPS
import fx
from fx import (clamp, lin, smooth, keys, decay, ease_out_cubic, ease_in_cubic, ease_in_out,
                ease_out_expo, ease_in_expo, ease_out_back, lerp, Cam, add_glow, render_liquid,
                draw_text, fit_size, radial_bg, hblur, vblur, zoom_blur, finish, draw_sprite,
                sphere_sprite, draw_ice, fractal_noise, scroll, smoothstep_arr, blend_over, blend_add)

ROOT = fx.ROOT
BUILD = os.path.join(ROOT, "output", "build")
FRAMES = os.path.join(BUILD, "frames")

FLOOR = 1575.0      # world y of the floor (can bottom)
CAN_Y = 1000.0      # world y of can centre
_cans = {}


def can(i):
    if i not in _cans:
        _cans[i] = fx.Can(TL.PRODUCTS[i]["key"])
    return _cans[i]


def shake(t, amp, seed=0.0):
    if amp <= 0.01:
        return 0.0, 0.0
    return (amp * (np.sin(t * 91 + seed) * 0.6 + np.sin(t * 143 + 2.1 + seed) * 0.4),
            amp * (np.sin(t * 77 + 1.3 + seed) * 0.6 + np.sin(t * 131 + 0.4 + seed) * 0.4))


def cam_discs(cam, discs, rscale=1.0):
    return [(cam.p(x, y)[0], cam.p(x, y)[1], r * cam.z * rscale) for (x, y, r) in discs]


def bokeh(canvas, t, seed, n, colors, cam=None, parallax=0.3, inten=0.12, rmin=20, rmax=70, speed=20):
    r = np.random.default_rng(seed)
    for i in range(n):
        x0, y0 = r.uniform(-100, W + 100), r.uniform(-100, H + 100)
        vx, vy = r.uniform(-1, 1) * speed, r.uniform(-1.5, 0.3) * speed
        rad = r.uniform(rmin, rmax)
        c = colors[i % len(colors)]
        ph = r.uniform(0, 6.28)
        x = (x0 + vx * t) % (W + 200) - 100
        y = (y0 + vy * t) % (H + 200) - 100
        if cam is not None:
            zz = 1 + (cam.z - 1) * parallax
            x = (x - 540) * zz + 540 + (cam.ax - 540) * parallax
            y = (y - 1000) * zz + 1000 + (cam.ay - 1000) * parallax
            rad *= zz
        k = inten * (0.6 + 0.4 * np.sin(t * 1.3 + ph))
        add_glow(canvas, x, y, rad, c, k)


def dust(canvas, t, seed, n, color, inten=0.5, mask_fn=None, rmin=2, rmax=6, vy=-15):
    r = np.random.default_rng(seed)
    for i in range(n):
        x0, y0 = r.uniform(0, W), r.uniform(0, H)
        ph = r.uniform(0, 6.28)
        x = (x0 + 18 * np.sin(t * 0.7 + ph)) % W
        y = (y0 + vy * t * r.uniform(0.5, 1.5)) % H
        k = inten * (0.5 + 0.5 * np.sin(t * 2.3 + ph * 3)) * (mask_fn(x, y) if mask_fn else 1.0)
        if k > 0.01:
            add_glow(canvas, x, y, r.uniform(rmin, rmax), color, k)


@fx.lru_cache(maxsize=4)
def ray_field(cx, cy, scale=4):
    h, w = H // scale, W // scale
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx * scale - cx, yy * scale - cy
    return np.arctan2(dy, dx), np.sqrt(dx * dx + dy * dy)


def god_rays(canvas, t, cx, cy, color, inten, n=14, speed=0.15, sharp=10):
    if inten <= 0.005:
        return
    ang, rad = ray_field(int(cx), int(cy))
    rays = (0.5 + 0.5 * np.cos(n * ang + t * speed * 6.28)) ** sharp
    rays *= (0.5 + 0.5 * np.cos(n * 0.5 * ang - t * speed * 4)) ** 2
    rays *= np.clip(1 - rad / 1500, 0, 1) ** 1.5
    rays = cv2.GaussianBlur(rays.astype(np.float32), (0, 0), 2)
    up = cv2.resize(rays, (W, H), interpolation=cv2.INTER_LINEAR)
    canvas += up[..., None] * np.array(color, np.float32) * inten


def shock_ring(canvas, cx, cy, r, thick, color, inten, ry_ratio=1.0):
    if inten <= 0.01 or r < 1:
        return
    s = 0.5
    lay = np.zeros((H // 2, W // 2), np.float32)
    cv2.ellipse(lay, (int(cx * s), int(cy * s)), (int(r * s), int(r * s * ry_ratio)), 0, 0, 360, 1.0,
                max(1, int(thick * s)), cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), max(1.5, thick * s * 0.6))
    canvas += cv2.resize(lay, (W, H))[..., None] * np.array(color, np.float32) * inten


def flavor_tag(canvas, t, t_in, t_out, name, glow):
    if t < t_in or t > t_out + 0.3:
        return
    a_in = ease_out_cubic(lin(t, t_in, t_in + 0.35))
    a_out = 1 - ease_in_cubic(lin(t, t_out, t_out + 0.25))
    a = a_in * a_out
    size = fit_size(name, "InterDisplay-BlackItalic.otf", 860, 108)
    yoff = (1 - a_in) * 30 - (1 - a_out) * 40
    draw_text(canvas, name, "InterDisplay-BlackItalic.otf", size, 540 + (1 - a_in) * -60, 262 + yoff,
              alpha=a, reveal=ease_out_cubic(lin(t, t_in, t_in + 0.4)), glow=glow, glow_r=22, glow_s=0.9,
              shadow=0.5, blur=(1 - a_out) * 8)
    chip = "25G PROTEIN  •  LIGHTLY CARBONATED"
    draw_text(canvas, chip, "InterDisplay-SemiBold.otf", 31, 540, 262 + size * 0.62 + 18 + yoff,
              alpha=a * ease_out_cubic(lin(t, t_in + 0.12, t_in + 0.45)), tracking=0.14, shadow=0.45,
              glow=glow, glow_r=10, glow_s=0.35)


# =========================================================== SCENE 1 (intro + cherry limeade)
def gen_splash(seed):
    r = np.random.default_rng(seed)
    streams = []
    for i in range(16):
        side = -1 if i % 2 == 0 else 1
        if i < 12:
            phi = side * (0.2 + 1.25 * (i // 2) / 5 + r.uniform(-0.08, 0.08))
        else:
            phi = side * r.uniform(1.7, 2.3)
        streams.append((phi, r.uniform(2700, 4600), r.uniform(26, 44), r.uniform(-2600, 2600) * 0.8,
                        r.uniform(0, 0.02), r.uniform(0, 6.28), r.uniform(18, 40)))
    drops = []
    for i in range(300):
        phi = r.normal(0, 0.95) if r.random() < 0.7 else r.uniform(-2.7, 2.7)
        drops.append((phi, r.uniform(1500, 6500), r.uniform(3, 15), r.uniform(0, 0.04)))
    red = []
    for i in range(70):
        phi = r.normal(0, 1.1)
        red.append((phi, r.uniform(1400, 5200), r.uniform(6, 17), r.uniform(0.0, 0.05)))
    front = []
    for i in range(12):
        side = -1 if i % 2 == 0 else 1
        x0 = 540 + side * r.uniform(260, 480)
        front.append((x0, r.uniform(250, 1700), r.uniform(18, 44), side * r.uniform(60, 180), r.uniform(-180, 60)))
    return streams, drops, red, front


SPLASH = gen_splash(21)


def splash_prims(s, ox, oy, streams, drops, drag=2.2, g=2600.0):
    discs = []
    for (phi, V, R0, curl, dl, wph, wamp) in streams:
        d = np.array([np.sin(phi), -np.cos(phi)])
        nrm = np.array([-d[1], d[0]])
        K = 34
        k = np.arange(K)
        tau = dl + k * 0.004
        dt = np.clip(s - tau, 0, None)
        v = V * (1 - 0.62 * k / K)
        dist = v * (1 - np.exp(-drag * dt)) / drag
        wob = wamp * np.sin(k * 0.35 + wph + s * 6) * np.clip(dt * 3, 0, 1)
        px = ox + d[0] * dist + nrm[0] * (curl * dt ** 2 + wob)
        py = oy + d[1] * dist + nrm[1] * (curl * dt ** 2 + wob) + 0.5 * g * dt ** 2
        f = k / (K - 1)
        rad = R0 * (0.32 + 0.68 * f ** 0.7)                     # thin tip, thick sheet near origin
        tip = f < 0.3
        rad = np.where(tip, rad * (0.75 + 0.45 * np.cos(k * 2.3 + phi * 5)), rad)  # tip beads into droplets
        rad *= np.clip(dt / 0.012, 0, 1) * np.clip(1.2 - dt * 0.5, 0.45, 1)
        al = dt > 0
        discs += list(zip(px[al], py[al], rad[al]))
    for (phi, V, R0, dl) in drops:
        dt = s - dl
        if dt <= 0:
            continue
        dist = V * (1 - np.exp(-drag * 1.3 * dt)) / (drag * 1.3)
        discs.append((ox + np.sin(phi) * dist, oy - np.cos(phi) * dist + 0.5 * g * dt * dt, R0 * clamp(dt / 0.01)))
    return discs


def s1_sim(u):
    if u <= 0:
        return 0.0
    return 0.17 * u + 0.83 * 0.11 * (1 - np.exp(-u / 0.11))


@fx.lru_cache(maxsize=1)
def cone_mask():
    xx, yy = fx.grid()
    half = (yy + 120) * 0.36 + 40
    m = np.clip(1 - np.abs(xx - 540) / half, 0, 1) ** 1.7
    m *= np.clip(1 - yy / 2100, 0, 1) ** 0.6
    return m.astype(np.float32)


def lime_hue(h, w):
    return None


def scene1(t, fi):
    c = can(0)
    post = dict(flash=0.0, ca=0.0, grain=0.016)
    u = t - TL.DROP
    pre = t < TL.DROP

    # ---------------- camera
    if pre:
        z = keys(t, [(0, 0.80), (2.95, 0.90)], ease_in_out)
        cam = Cam(z=z, fy=CAN_Y, ay=1190 - 50 * smooth(t / 3))
        rot = keys(t, [(0, -0.18), (3, -0.08)])
    elif t < TL.CUT_S1[0]:
        z = 1.0 + 0.08 * decay(t, TL.DROP, 0.16) + 0.04 * ease_in_out(lin(t, TL.DROP, TL.CUT_S1[0]))
        cam = Cam(z=z, fx=540 - 18 * np.sin(u * 0.9), fy=CAN_Y)
        rot = keys(t, [(TL.DROP, -0.14), (TL.CUT_S1[0], 0.1)])
    elif t < TL.CUT_S1[1]:
        x = lin(t, TL.CUT_S1[0], TL.CUT_S1[1])
        cam = Cam(z=2.05 + 0.1 * x + 0.05 * decay(t, TL.CUT_S1[0], 0.12), fy=lerp(640, 930, ease_in_out(x)))
        rot = lerp(-0.07, 0.07, x)
    else:
        z = keys(t, [(TL.CUT_S1[1], 1.5), (TL.CUT_S1[1] + 0.32, 1.03), (7.0, 0.97)], ease_out_expo)
        cam = Cam(z=z, fy=CAN_Y)
        rot = keys(t, [(TL.CUT_S1[1], 0.13), (7.0, -0.05)])
    amp = 26 * decay(t, TL.DROP, 0.13) + 9 * decay(t, TL.CUT_S1[0], 0.1) + 12 * decay(t, TL.CAN_OPEN_S1, 0.1)
    for t0, _ in TL.INTRO_WORDS:
        amp += 7 * decay(t, t0, 0.08)
    dx, dy = shake(t, amp)
    cam.ax += dx
    cam.ay += dy

    # ---------------- background
    if pre:
        rev = smooth(lin(t, 0.2, 2.8))
        canvas = radial_bg((0.03, 0.07, 0.03), (0.0, 0.0, 0.0), 540, 950, 900, 1300).copy() * (0.6 + 0.6 * rev)
        cone = cone_mask()
        canvas += cone[..., None] * np.array([0.55, 0.9, 0.35], np.float32) * (0.06 + 0.14 * rev)
        dust(canvas, t, 5, 80, (0.8, 1.0, 0.7), inten=0.35 + 0.35 * rev,
             mask_fn=lambda x, y: float(cone[int(min(H - 1, max(0, y))), int(min(W - 1, max(0, x)))]) + 0.08)
        # energy gathering before the drop
        g = ease_in_expo(lin(t, 2.2, 3.0))
        if g > 0:
            sx, sy = cam.p(540, 960)
            add_glow(canvas, sx, sy, 200 + 900 * g, (0.55, 1.0, 0.25), 0.9 * g)
            add_glow(canvas, sx, sy, 80 + 300 * g, (1.0, 1.0, 0.85), 0.8 * g)
            r = np.random.default_rng(9)
            lay = np.zeros((H, W), np.float32)
            for i in range(70):
                a = r.uniform(0, 6.28)
                rr = r.uniform(500, 1300) * (1 - g * r.uniform(0.75, 1.0))
                ln = 40 + 160 * g
                p0 = (sx + np.cos(a) * rr, sy + np.sin(a) * rr)
                p1 = (sx + np.cos(a) * (rr + ln), sy + np.sin(a) * (rr + ln))
                cv2.line(lay, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, 2, cv2.LINE_AA)
            lay = cv2.GaussianBlur(lay, (0, 0), 1.5)
            canvas += lay[..., None] * np.array([0.8, 1.0, 0.6], np.float32) * g * 0.9
    else:
        canvas = radial_bg((0.50, 0.86, 0.18), (0.015, 0.10, 0.02), 540, 880, 1000, 1350, 1.35).copy()
        god_rays(canvas, t, *cam.p(540, 900), (0.75, 1.0, 0.45), 0.22, n=11, speed=0.05)
        bokeh(canvas, t, 3, 22, [(0.8, 1.0, 0.3), (1.0, 0.95, 0.5), (0.5, 1.0, 0.4)], cam, inten=0.10, rmin=25, rmax=90)
        # floor
        fy = cam.p(0, FLOOR)[1]
        if fy < H:
            yy = np.arange(H, dtype=np.float32)
            fl = np.clip((yy - fy) / 300, 0, 1)[:, None, None]
            canvas *= 1 - 0.45 * fl

    # ---------------- splash behind the can
    sim = s1_sim(u)
    streams, drops, red, front = SPLASH
    if not pre:
        ox, oy = 540, 990
        discs = splash_prims(sim, ox, oy, streams, drops)
        core = 260 * ease_out_cubic(lin(u, 0, 0.12)) * (1 - 0.6 * lin(u, 0.1, 1.2))
        discs.append((ox, oy, core))
        sd = cam_discs(cam, discs)
        render_liquid(canvas, discs=[d for d in sd if d[2] > 4], color=(0.58, 0.95, 0.12), dark=(0.10, 0.42, 0.02),
                      sigma=4.0, thr=(0.30, 0.52), bump=40)
        for (x, y, r) in sd:
            if 1.0 < r <= 4:
                draw_sprite(canvas, sphere_sprite((0.6, 0.98, 0.2)), x, y, r * 2.2)
        rd = splash_prims(sim, ox, oy + 20, [], red, drag=2.0)
        for (x, y, r) in cam_discs(cam, rd):
            draw_sprite(canvas, sphere_sprite((0.85, 0.04, 0.09)), x, y, r * 2.0)
        # shockwave
        sx, sy = cam.p(540, 960)
        shock_ring(canvas, sx, sy, 1700 * ease_out_expo(lin(u, 0, 0.7)), 46 * (1 - lin(u, 0, 0.7)) + 6,
                   (0.8, 1.0, 0.6), 1.2 * (1 - lin(u, 0, 0.7)))
        add_glow(canvas, sx, sy, 650, (0.7, 1.0, 0.35), 0.5 * decay(t, TL.DROP, 0.5) + 0.12)

    # ---------------- the can
    if pre:
        rev = lin(t, 0.15, 2.9)
        sweep = [(keys(t, [(0.35, -1.9), (1.75, 1.9)], ease_in_out), 0.42, 1.25 * smooth(lin(t, 0.3, 0.6)) * (1 - smooth(lin(t, 1.6, 1.85)))),
                 (keys(t, [(1.75, 1.9), (2.75, -1.9)], ease_in_out), 0.5, 1.15 * smooth(lin(t, 1.7, 1.9)))]
        rim = (0.45, 1.0, 0.3)
        rs = 0.15 + 1.1 * smooth(rev)
        c.draw(canvas, cam, 540, CAN_Y, rot=rot, ambient=0.02 + 0.30 * ease_in_cubic(rev) + 0.5 * ease_in_expo(lin(t, 2.6, 3.0)),
               sweep=sweep, rim_l=tuple(np.array(rim) * rs * 0.7), rim_r=tuple(np.array(rim) * rs),
               reflect_floor=FLOOR, refl_strength=0.18 * rev)
    else:
        c.draw(canvas, cam, 540, CAN_Y, rot=rot, spec=lerp(-0.5, 0.6, lin(t, 3.0, 7.0)) - rot, spec_s=0.18,
               rim_l=(0.35, 0.40, 0.30), rim_r=(0.55, 0.9, 0.3), reflect_floor=FLOOR, refl_strength=0.3, shadow=0.55)

    # ---------------- foreground droplets (depth of field)
    if not pre:
        fd = []
        for (x0, y0, r0, vx, vy) in front:
            k = ease_out_cubic(lin(u, 0, 0.4))
            x = 540 + (x0 - 540) * (0.3 + 0.7 * k) + vx * sim * 3
            y = 960 + (y0 - 960) * (0.3 + 0.7 * k) + vy * sim * 3 + 400 * sim * sim
            fd.append((x, y, r0 * (0.4 + 0.6 * k) * (1 + 0.6 * sim)))
        sd = cam_discs(cam, fd)
        render_liquid(canvas, discs=sd, color=(0.62, 0.98, 0.18), dark=(0.12, 0.45, 0.03), sigma=3.0,
                      thr=(0.3, 0.52), bump=30, blur_out=5.0 if t < TL.CUT_S1[0] else 9.0, opacity=0.85)

    # ---------------- can crack: fizz + mist burst from the top
    if t >= TL.CAN_OPEN_S1:
        v = t - TL.CAN_OPEN_S1
        r = np.random.default_rng(44)
        tx, ty = cam.p(540, CAN_Y - 575)
        add_glow(canvas, tx, ty - cam.s(60) * ease_out_cubic(v / 0.6), cam.s(260) * ease_out_cubic(lin(v, 0, 0.5)),
                 (0.85, 1.0, 0.85), 0.55 * decay(t, TL.CAN_OPEN_S1, 0.35), sy=0.6)
        for i in range(46):
            st = r.uniform(0, 0.5)
            if v < st:
                continue
            vv = v - st
            bx = 540 + r.uniform(-150, 150) + 25 * np.sin(vv * 9 + i)
            by = CAN_Y - 575 - r.uniform(160, 420) * vv - 10
            x, y = cam.p(bx, by)
            draw_sprite(canvas, sphere_sprite((0.85, 1.0, 0.85)), x, y, cam.s(r.uniform(5, 13)), alpha=0.6 * clamp(1.4 - vv))

    # ---------------- text
    if pre:
        lines = [("THE ULTIMATE", 760, 290), ("REFRESHMENT", 960, 418), ("IS HERE.", 560, 540)]
        fonts = ["InterDisplay-Black.otf", "InterDisplay-BlackItalic.otf", "InterDisplay-Black.otf"]
        explode = ease_in_cubic(lin(t, 2.8, 3.0))
        for i, ((t0, _), (txt, mw, y)) in enumerate(zip(TL.INTRO_WORDS, lines)):
            if t < t0:
                continue
            k = ease_out_expo(lin(t, t0, t0 + 0.22))
            size = fit_size(txt, fonts[i], mw, 190)
            grad = ((1, 1, 1), (0.75, 1.0, 0.45)) if i == 1 else None
            draw_text(canvas, txt, fonts[i], size, 540, y, alpha=k * (1 - explode),
                      scale=lerp(1.45, 1.0, k) * (1 + 0.5 * explode), blur=(1 - k) * 14 + explode * 12,
                      glow=(0.45, 1.0, 0.25), glow_r=26, glow_s=0.7 + 0.6 * decay(t, t0, 0.15), grad=grad, shadow=0.4)
    else:
        flavor_tag(canvas, t, 3.35, 4.3, TL.PRODUCTS[0]["name"], (0.45, 1.0, 0.2))

    # ---------------- transition: rainbow wave rises over everything
    if t >= TL.WAVE_START:
        rainbow_wave(canvas, t)

    # flashes / post
    post["flash"] = 0.9 * decay(t, TL.DROP, 0.06) if not pre else 0.4 * ease_in_expo(lin(t, 2.75, 3.0))
    post["flash"] += 0.25 * decay(t, TL.CUT_S1[0], 0.05) + 0.3 * decay(t, TL.CUT_S1[1], 0.05) + 0.2 * decay(t, TL.CAN_OPEN_S1, 0.06)
    post["ca"] = 14 * decay(t, TL.DROP, 0.18) + 5 * decay(t, TL.CUT_S1[0], 0.1)
    for t0, _ in TL.INTRO_WORDS:
        post["ca"] += 6 * decay(t, t0, 0.1)
        post["flash"] += 0.10 * decay(t, t0, 0.05)
    return canvas, post


# =========================================================== rainbow wave (1 -> 2 transition)
def rainbow_hue(t):
    def fn(h, w):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        hue = xx / w * 0.85 + yy / h * 0.35 + t * 0.4
        return fx.hsv2rgb_arr(hue, 0.82, 1.0)
    return fn


@fx.lru_cache(maxsize=1)
def smooth_ripple():
    return fx.lowfreq_noise(31, 540, 960, 6) * 2.2


def rainbow_wave(canvas, t):
    """Liquid band moving upward through frame. Covers everything at t=7.0."""
    if t < TL.WAVE_START or t > TL.WAVE_CLEAR + 0.6:
        return
    xs = np.linspace(-60, W + 60, 48)
    if t <= 7.0:
        k = ease_in_cubic(lin(t, TL.WAVE_START, 7.0)) * 0.75 + 0.25 * lin(t, TL.WAVE_START, 7.0)
        ub = lerp(2050, -420, k)
        lb = 2600
    else:
        ub = -420 - 900 * (t - 7.0)
        k = ease_in_out(lin(t, 7.0, TL.WAVE_CLEAR))
        lb = lerp(2250, -520, k)
    up = ub + 95 * np.sin(xs * 0.006 + t * 7) + 60 * np.sin(xs * 0.017 - t * 9) + (xs - 540) * 0.22
    lo = lb + 80 * np.sin(xs * 0.008 - t * 6) + 50 * np.sin(xs * 0.02 + t * 8) - (xs - 540) * 0.18
    poly = np.concatenate([np.stack([xs, up], 1), np.stack([xs[::-1], lo[::-1]], 1)])
    discs = []
    r = np.random.default_rng(int(t * 1000) % 7 + 70)
    rr = np.random.default_rng(71)
    # crest spray
    for i in range(70):
        x = rr.uniform(-40, W + 40)
        ph = rr.uniform(0, 1)
        life = (t * 2.2 + ph) % 1.0
        yc = np.interp(x, xs, up)
        discs.append((x + 30 * np.sin(i), yc - 30 - 320 * life, rr.uniform(6, 20) * (1 - life)))
    # drips hanging from the trailing edge
    if t > 7.0:
        for i in range(26):
            x = rr.uniform(0, W)
            yl = np.interp(x, xs, lo)
            L = rr.uniform(80, 360) * smooth(lin(t, 7.0, 7.25))
            for j in range(9):
                f = j / 8
                discs.append((x + 6 * np.sin(j + i), yl + L * f, rr.uniform(10, 22) * (1 - 0.55 * f)))
            # detached falling drops
            ph = rr.uniform(0, 1)
            dt_ = max(0.0, t - 7.05 - ph * 0.3)
            discs.append((x, yl + L + 60 + 1800 * dt_ * dt_, rr.uniform(8, 16)))
    rip = smooth_ripple()
    rip = scroll(rip, t * 160, -t * 420)
    render_liquid(canvas, discs=discs, polys=[poly], hue_fn=rainbow_hue(t), sigma=3.4, thr=(0.3, 0.52),
                  bump=30, ripple=rip, opacity=1.0, spec_power=50)


# =========================================================== SCENE 2 (rainbow candy)
ORBS = []
_r = np.random.default_rng(12)
_cols = [(0.95, 0.12, 0.16), (1.0, 0.28, 0.66), (0.22, 0.85, 0.28), (0.15, 0.55, 1.0), (1.0, 0.48, 0.08),
         (1.0, 0.86, 0.12), (0.62, 0.28, 1.0)]
for i in range(18):
    ORBS.append(dict(rx=_r.uniform(330, 560), y=_r.uniform(380, 1620), om=_r.uniform(0.45, 0.9) * (1 if i % 2 else -1),
                     ph=_r.uniform(0, 6.28), size=_r.uniform(55, 125), col=_cols[i % len(_cols)], bob=_r.uniform(20, 60)))


FRONT_ORBS = [dict(cx=150, y=1450, size=170, col=_cols[4], ph=0.3, sp=-150), dict(cx=950, y=600, size=150, col=_cols[2], ph=2.0, sp=-120),
              dict(cx=980, y=1700, size=190, col=_cols[0], ph=4.0, sp=-180)]


def orb_pos(o, t):
    th = o["ph"] + o["om"] * t
    x = 540 + o["rx"] * np.cos(th)
    depth = -abs(np.sin(th)) - 0.05
    y = o["y"] + o["bob"] * np.sin(t * 1.3 + o["ph"]) - 40 * t
    return x, y, depth


def scene2(t, fi):
    c = can(1)
    post = dict(flash=0.0, ca=0.0, grain=0.016)
    t_in = 7.0
    shift = 0.0
    mblur = 0.0
    if t < TL.WAVE_CLEAR:
        z = keys(t, [(7.0, 1.24), (TL.WAVE_CLEAR, 1.0)], ease_out_cubic)
        cam = Cam(z=z, fy=CAN_Y, ay=1000 + 70 * (1 - ease_out_cubic(lin(t, 7.0, TL.WAVE_CLEAR))))
        rot = keys(t, [(7.0, 0.22), (TL.CUT_S2[0], -0.08)])
    elif t < TL.CUT_S2[0]:
        z = lerp(1.0, 1.07, ease_in_out(lin(t, TL.WAVE_CLEAR, TL.CUT_S2[0])))
        cam = Cam(z=z, fy=CAN_Y)
        rot = keys(t, [(7.0, 0.22), (TL.CUT_S2[0], -0.08)])
    elif t < TL.CUT_S2[1]:
        x = lin(t, TL.CUT_S2[0], TL.CUT_S2[1])
        cam = Cam(z=2.0 + 0.08 * x + 0.05 * decay(t, TL.CUT_S2[0], 0.12), fx=lerp(575, 540, x), fy=lerp(1330, 1030, ease_in_out(x)))
        rot = lerp(-0.05, 0.06, x)
    else:
        z = keys(t, [(TL.CUT_S2[1], 1.2), (TL.CUT_S2[1] + 0.3, 1.06), (11.0, 1.02)], ease_out_expo)
        cam = Cam(z=z, fy=CAN_Y)
        rot = keys(t, [(TL.CUT_S2[1], 0.12), (11.0, -0.08)])
        if t >= TL.WHIP_OUT_S2:
            k = ease_in_cubic(lin(t, TL.WHIP_OUT_S2, 10.96))
            shift = -1500 * k
            mblur = 10 + 210 * k
    dx, dy = shake(t, 10 * decay(t, 7.0, 0.12) + 8 * decay(t, TL.CUT_S2[0], 0.1))
    cam.ax += dx + shift
    cam.ay += dy

    canvas = radial_bg((0.36, 0.26, 0.98), (0.04, 0.02, 0.18), 540, 900, 1000, 1350, 1.3).copy()
    if shift != 0:
        canvas = np.roll(canvas, int(shift * 0.5), axis=1)
    god_rays(canvas, t, *cam.p(540, 900), (0.7, 0.6, 1.0), 0.14, n=9, speed=-0.04)
    bokeh(canvas, t, 8, 26, [(1.0, 0.3, 0.7), (0.3, 0.8, 1.0), (1.0, 0.85, 0.3), (0.6, 1.0, 0.4)], cam, inten=0.13, rmin=25, rmax=95)
    fy = cam.p(0, FLOOR)[1]
    if fy < H:
        yy = np.arange(H, dtype=np.float32)
        canvas *= 1 - 0.5 * np.clip((yy - fy) / 300, 0, 1)[:, None, None]

    tt = t - t_in
    # orbs behind
    front = []
    for o in ORBS:
        x, y, d = orb_pos(o, tt)
        if d < 0:
            sc = 1 + 0.3 * d
            sx, sy = cam.p(x + 0, y)
            draw_sprite(canvas, sphere_sprite(o["col"]), sx, sy, cam.s(o["size"]) * sc, blur=2.5 + 3 * -d, alpha=0.75 + 0.25 * (1 + d))
        else:
            front.append((o, x, y, d))
    # rainbow droplet remnants (slow-mo)
    rr = np.random.default_rng(5)
    drops = []
    for i in range(60):
        x0, y0 = rr.uniform(0, W), rr.uniform(-200, H)
        vy = rr.uniform(60, 200)
        drops.append((x0 + 20 * np.sin(tt * 2 + i), (y0 + vy * tt) % (H + 300) - 150, rr.uniform(5, 14)))
    render_liquid(canvas, discs=cam_discs(cam, drops), hue_fn=rainbow_hue(t), sigma=2.6, thr=(0.3, 0.52), bump=30,
                  opacity=0.9 * smooth(lin(t, 7.1, 7.6)))

    c.draw(canvas, cam, 540, CAN_Y, rot=rot, spec=lerp(0.6, -0.5, lin(t, 7.0, 11.0)) - rot, spec_s=0.2,
           rim_l=(0.9, 0.4, 0.9), rim_r=(0.3, 0.7, 1.0), reflect_floor=FLOOR, refl_strength=0.3, shadow=0.55)

    for o in FRONT_ORBS:
        x = o["cx"] + 60 * np.sin(tt * 0.9 + o["ph"])
        y = o["y"] + o["sp"] * tt
        zz = 1 + (cam.z - 1) * 1.3
        sx, sy = (x - 540) * zz + cam.ax, (y - 1000) * zz + cam.ay
        draw_sprite(canvas, sphere_sprite(o["col"]), sx, sy, o["size"] * zz, blur=9 * zz, alpha=0.8)

    flavor_tag(canvas, t, 7.6, 8.8, TL.PRODUCTS[1]["name"], (0.7, 0.45, 1.0))
    rainbow_wave(canvas, t)

    if mblur > 1:
        canvas = hblur(canvas, mblur)
    post["flash"] = 0.35 * decay(t, 7.0, 0.08) + 0.25 * decay(t, TL.CUT_S2[0], 0.05) + 0.25 * decay(t, TL.CUT_S2[1], 0.05)
    post["flash"] += 0.5 * ease_in_expo(lin(t, 10.85, 11.0))
    post["ca"] = 6 * decay(t, TL.CUT_S2[0], 0.1) + 10 * lin(t, TL.WHIP_OUT_S2, 11.0)
    return canvas, post


# =========================================================== SCENE 3 (green apple, ice)
ICE = []
_r = np.random.default_rng(33)
_land_x = [215, 870, 135, 948, 330, 760]
_behind = [False, False, False, False, True, True]
for i, th in enumerate(TL.ICE_HITS):
    size = _r.uniform(125, 175) * (0.75 if _behind[i] else 1.0)
    ICE.append(dict(kind="land", x=_land_x[i], th=th, fall=_r.uniform(0.45, 0.6), size=size, behind=_behind[i],
                    rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3.5, 3.5, 3)))
for i in range(3):
    ICE.append(dict(kind="fore", x=[110, 980, 820][i], t0=11.1 + i * 0.55, size=_r.uniform(280, 380),
                    rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-2, 2, 3), vy=_r.uniform(380, 560)))
for i in range(7):
    ICE.append(dict(kind="back", x=_r.uniform(60, 1020), t0=11.0 + _r.uniform(0, 3), size=_r.uniform(60, 95),
                    rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3, 3, 3), vy=_r.uniform(500, 800)))


def ice_state(o, t):
    if o["kind"] == "land":
        ts = o["th"] - o["fall"]
        land_y = FLOOR - o["size"] * 0.3 - (40 if o["behind"] else -30)
        if t < ts:
            return None
        if t < o["th"]:
            f = (t - ts) / o["fall"]
            y = lerp(-250, land_y, f * f)
            ang = o["rot0"] + o["om"] * (t - ts)
        else:
            d = t - o["th"]
            y = land_y - o["size"] * 0.45 * abs(np.sin(d * 9)) * np.exp(-d / 0.18)
            ang = o["rot0"] + o["om"] * (o["fall"] + 0.25 * (1 - np.exp(-d / 0.25)))
        return o["x"], y, ang
    if t < o["t0"]:
        return None
    d = t - o["t0"]
    y = -400 + o["vy"] * d + 120 * d * d
    if y > H + 400:
        return None
    return o["x"], y, o["rot0"] + o["om"] * d


def mist_layer(canvas, t, alpha_map, color=(0.82, 0.95, 0.97), seed=41, sx=60, sy=-25):
    n1 = scroll(fractal_noise(seed, 540, 960, 6), t * sx, t * sy)
    n2 = scroll(fractal_noise(seed + 1, 540, 960, 6), -t * sx * 0.6, t * sy * 1.6)
    n = np.clip((n1 * 0.6 + n2 * 0.4 - 0.32) * 2.2, 0, 1)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    a = n * alpha_map
    canvas *= (1 - a[..., None] * 0.6)
    canvas += a[..., None] * np.array(color, np.float32) * 0.75


def scene3(t, fi):
    c = can(2)
    post = dict(flash=0.0, ca=0.0, grain=0.016)
    shift, mblur, zb = 0.0, 0.0, 0.0
    if t < 11.25:
        k = ease_out_cubic(lin(t, 11.0, 11.27))
        shift = 1500 * (1 - k)
        mblur = 220 * (1 - k)
        cam = Cam(z=1.0, fy=CAN_Y)
        rot = -0.14
    elif t < TL.CUT_S3[0]:
        cam = Cam(z=lerp(1.0, 1.07, ease_in_out(lin(t, 11.25, TL.CUT_S3[0]))), fy=CAN_Y)
        rot = keys(t, [(11.25, -0.14), (TL.CUT_S3[0], 0.1)])
    elif t < TL.CUT_S3[1]:
        x = lin(t, TL.CUT_S3[0], TL.CUT_S3[1])
        cam = Cam(z=2.1 + 0.08 * x + 0.05 * decay(t, TL.CUT_S3[0], 0.12), fy=lerp(600, 820, ease_in_out(x)))
        rot = lerp(0.06, -0.06, x)
    elif t < TL.ZOOM_OUT_S3:
        cam = Cam(z=keys(t, [(TL.CUT_S3[1], 1.32), (TL.CUT_S3[1] + 0.3, 1.14), (TL.ZOOM_OUT_S3, 1.1)], ease_out_expo), fy=CAN_Y)
        rot = keys(t, [(TL.CUT_S3[1], 0.1), (TL.ZOOM_OUT_S3, -0.04)])
    else:
        x = ease_in_cubic(lin(t, TL.ZOOM_OUT_S3, 15.0))
        cam = Cam(z=1.1 * np.exp(np.log(7.5 / 1.1) * x), fx=540, fy=lerp(CAN_Y, 930, x))
        rot = -0.04
        zb = 0.28 * x
    dx, dy = shake(t, 8 * decay(t, TL.CUT_S3[0], 0.1) + 12 * decay(t, TL.CAN_OPEN_S3, 0.1) +
                   sum(5 * decay(t, h, 0.08) for h in TL.ICE_HITS))
    cam.ax += dx + shift
    cam.ay += dy

    canvas = radial_bg((0.20, 0.36, 0.32), (0.01, 0.04, 0.045), 540, 900, 950, 1350, 1.4).copy()
    if shift:
        canvas = np.roll(canvas, int(shift * 0.5), axis=1)
    god_rays(canvas, t, *cam.p(540, 300), (0.6, 0.9, 1.0), 0.10, n=8, speed=0.03, sharp=6)
    add_glow(canvas, *cam.p(540, 850), cam.s(700), (0.4, 0.75, 0.8), 0.18)
    fy = cam.p(0, FLOOR)[1]
    if fy < H:
        yy = np.arange(H, dtype=np.float32)
        canvas *= 1 - 0.5 * np.clip((yy - fy) / 300, 0, 1)[:, None, None]
    # background haze
    xx, yy = fx.grid()
    mist_layer(canvas, t, np.full((H, W), 0.18, np.float32), seed=51, sx=25, sy=-8)

    # ice behind
    for o in ICE:
        st = ice_state(o, t)
        if st is None:
            continue
        if o["kind"] == "back" or (o["kind"] == "land" and o["behind"]):
            x, y, ang = st
            sx_, sy_ = cam.p(x, y)
            draw_ice(canvas, sx_, sy_, cam.s(o["size"]), *ang, blur=2.5 if o["kind"] == "back" else 0.8, alpha=0.85)

    cold = 0.75 + 0.25 * np.sin(t * 2)
    c.draw(canvas, cam, 540, CAN_Y, rot=rot, spec=lerp(-0.6, 0.5, lin(t, 11.0, 15.0)) - rot, spec_s=0.22,
           rim_l=(0.45 * cold, 0.8 * cold, 0.9 * cold), rim_r=(0.55, 0.95, 1.0), reflect_floor=FLOOR, refl_strength=0.3, shadow=0.55)

    # vapour rising off the can (cold)
    top_x, top_y = cam.p(540, CAN_Y - 575)
    vap = np.zeros((H, W), np.float32)
    hw = cam.s(330)
    vap_mask = np.clip(1 - np.abs(xx - top_x) / hw, 0, 1) ** 1.5 * np.clip(1 - np.abs(yy - (top_y - cam.s(140))) / cam.s(330), 0, 1)
    burst = ease_out_cubic(lin(t, TL.CAN_OPEN_S3, TL.CAN_OPEN_S3 + 0.7)) * decay(t, TL.CAN_OPEN_S3 + 0.3, 0.8) if t >= TL.CAN_OPEN_S3 else 0
    vap = vap_mask * (0.38 + 1.1 * burst)
    # floor fog
    fog = np.clip(1 - np.abs(yy - fy) / cam.s(260), 0, 1) ** 1.2 * 0.55
    mist_layer(canvas, t, vap + fog, seed=61, sx=40, sy=-70)
    if burst > 0:
        add_glow(canvas, top_x, top_y - cam.s(80), cam.s(420) * (0.5 + burst), (0.75, 0.95, 1.0), 0.75 * burst, sy=0.7)

    # landing + foreground ice
    for o in ICE:
        st = ice_state(o, t)
        if st is None:
            continue
        x, y, ang = st
        sx_, sy_ = cam.p(x, y)
        if o["kind"] == "land" and not o["behind"]:
            draw_ice(canvas, sx_, sy_, cam.s(o["size"]), *ang, blur=0.6)
            dd = t - o["th"]
            if 0 <= dd < 0.25:  # impact sparkle
                add_glow(canvas, sx_, sy_ + cam.s(o["size"] * 0.3), cam.s(140), (0.8, 0.95, 1.0), 0.6 * (1 - dd / 0.25), sy=0.35)
        elif o["kind"] == "fore":
            draw_ice(canvas, sx_, sy_, cam.s(o["size"]), *ang, blur=7.0, alpha=0.9)

    # frost sparkles
    r = np.random.default_rng(77)
    for i in range(110):
        x0, y0 = r.uniform(0, W), r.uniform(0, H)
        ph = r.uniform(0, 6.28)
        x = (x0 + 12 * np.sin(t + ph)) % W
        y = (y0 + 35 * (t - 11) * r.uniform(0.5, 1.5)) % H
        k = max(0.0, np.sin(t * 3.1 + ph * 5)) ** 6
        if k > 0.03:
            add_glow(canvas, x, y, r.uniform(4, 10), (0.85, 0.97, 1.0), 0.9 * k)
            add_glow(canvas, x, y, r.uniform(14, 26), (0.85, 0.97, 1.0), 0.18 * k, sx=1.6, sy=0.12)

    flavor_tag(canvas, t, 11.45, 12.75, TL.PRODUCTS[2]["name"], (0.4, 0.9, 0.85))

    if mblur > 1:
        canvas = hblur(canvas, mblur)
    if zb > 0:
        canvas = zoom_blur(canvas, zb, *cam.p(540, 930))
    post["flash"] = 0.4 * decay(t, 11.0, 0.08) + 0.25 * decay(t, TL.CUT_S3[0], 0.05) + 0.3 * decay(t, TL.CUT_S3[1], 0.05)
    post["flash"] += 0.25 * decay(t, TL.CAN_OPEN_S3, 0.08) + ease_in_expo(lin(t, TL.ZOOM_OUT_S3 + 0.05, 15.0)) * 1.0
    post["flash_color"] = (0.9, 0.97, 1.0)
    post["ca"] = 8 * (1 - lin(t, 11.0, 11.3)) + 5 * decay(t, TL.CUT_S3[0], 0.1) + 14 * lin(t, TL.ZOOM_OUT_S3, 15.0)
    return canvas, post


# =========================================================== SCENE 4 (banana pineapple, electric)
def strike_env(t):
    """(age, index) of most recent strike."""
    best = None
    for i, s in enumerate(TL.STRIKES_S4):
        if t >= s:
            best = (t - s, i)
    return best


def scene4(t, fi):
    c = can(3)
    post = dict(flash=0.0, ca=0.0, grain=0.016)
    shift_y, vb, zb = 0.0, 0.0, 0.0
    se = strike_env(t)
    punch = sum(0.035 * decay(t, s, 0.12) for s in TL.STRIKES_S4 if t >= s)
    if t < 15.42:
        x = lin(t, 15.0, 15.42)
        cam = Cam(z=lerp(1.85, 1.0, ease_out_expo(x)), fy=CAN_Y)
        zb = 0.3 * (1 - ease_out_cubic(x))
        rot = 0.16
    elif t < TL.CUT_S4[0]:
        cam = Cam(z=lerp(1.0, 1.13, ease_in_out(lin(t, 15.42, TL.CUT_S4[0]))) + punch, fy=CAN_Y)
        rot = keys(t, [(15.0, 0.16), (TL.CUT_S4[0], -0.08)])
    elif t < TL.CUT_S4[1]:
        cam = Cam(z=1.95 + 0.1 * lin(t, 17.0, 17.5) + punch, fy=880)
        rot = lerp(-0.04, 0.04, lin(t, 17.0, 17.5))
    elif t < TL.CUT_S4[2]:
        cam = Cam(z=2.2 + 0.1 * lin(t, 17.5, 18.0) + punch, fx=590, fy=1320)
        rot = lerp(0.05, -0.03, lin(t, 17.5, 18.0))
    else:
        cam = Cam(z=lerp(0.93, 1.05, ease_out_cubic(lin(t, 18.0, 18.75))) + punch, fy=CAN_Y)
        rot = keys(t, [(18.0, 0.1), (19.0, -0.1)])
        if t >= TL.WHIP_OUT_S4:
            k = ease_in_cubic(lin(t, TL.WHIP_OUT_S4, 18.97))
            shift_y = -2200 * k
            vb = 10 + 240 * k
    amp = sum((22 if s in (15.0, 18.0) else 11) * decay(t, s, 0.1) for s in TL.STRIKES_S4 if t >= s)
    dx, dy = shake(t, amp, 3.0)
    cam.ax += dx
    cam.ay += dy + shift_y

    flash_bg = 0.0
    if se:
        flash_bg = np.exp(-se[0] / 0.07) * (1.0 if TL.STRIKES_S4[se[1]] in (15.0, 18.0) else 0.6)
    canvas = radial_bg((0.20, 0.58, 0.98), (0.0, 0.025, 0.10), 540, 880, 1000, 1350, 1.3).copy() * (1 + 0.8 * flash_bg)
    if shift_y:
        canvas = np.roll(canvas, int(shift_y * 0.5), axis=0)
    god_rays(canvas, t, *cam.p(540, 900), (0.55, 0.85, 1.0), 0.16 + 0.3 * flash_bg, n=16, speed=0.12, sharp=14)
    fy = cam.p(0, FLOOR)[1]
    if fy < H:
        yy = np.arange(H, dtype=np.float32)
        canvas *= 1 - 0.5 * np.clip((yy - fy) / 300, 0, 1)[:, None, None]

    # light streaks
    r = np.random.default_rng(90)
    lay = np.zeros((H // 2, W // 2), np.float32)
    for i in range(9):
        y = r.uniform(150, 1800)
        sp = r.uniform(2500, 5000) * (1 if i % 2 else -1)
        ph = r.uniform(0, 1)
        x = ((t * sp / 3000 + ph) % 1.6 - 0.3) * W
        L = r.uniform(250, 600)
        cv2.line(lay, (int(x / 2), int(y / 2)), (int((x - np.sign(sp) * L) / 2), int(y / 2)), 1.0, 2, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.2) + cv2.GaussianBlur(lay, (0, 0), 6) * 0.8
    canvas += cv2.resize(lay, (W, H))[..., None] * np.array([0.6, 0.9, 1.0], np.float32) * 0.45

    # lightning bolts behind the can
    core = np.zeros((H, W), np.float32)
    glow = np.zeros((H // 2, W // 2), np.float32)
    if se:
        age, idx = se
        fr = int(age * FPS)
        big = TL.STRIKES_S4[idx] in (15.0, 18.0)
        if fr in (0, 1, 2, 4, 5) or (big and fr < 9 and fr != 6):
            rb = np.random.default_rng(1000 + idx * 31 + fr // 2)
            nb = 4 if big else 2
            for b in range(nb):
                side = -1 if (b + idx) % 2 == 0 else 1
                x0 = 540 + side * rb.uniform(150, 520)
                p0 = cam.p(x0, -200)
                p1 = cam.p(540 + side * rb.uniform(200, 330), FLOOR - rb.uniform(0, 300))
                fx.draw_bolt(core, glow, p0, p1, rb, width=3 if big else 2, branches=4 if big else 2)
    fx.composite_electric(canvas, core, glow, (0.4, 0.75, 1.0), 1.0)

    # energy rings on the floor
    for s in TL.STRIKES_S4:
        if t >= s:
            a = t - s
            if a < 0.7:
                sx, sy = cam.p(540, FLOOR)
                shock_ring(canvas, sx, sy, cam.s(80 + 950 * ease_out_expo(a / 0.7)), 10, (0.5, 0.85, 1.0), 1.1 * (1 - a / 0.7), ry_ratio=0.17)

    c.draw(canvas, cam, 540, CAN_Y, rot=rot, spec=lerp(0.6, -0.6, lin(t, 15.0, 19.0)) - rot, spec_s=0.2,
           rim_l=tuple(np.array([0.4, 0.75, 1.0]) * (0.6 + 1.2 * flash_bg)), rim_r=tuple(np.array([1.0, 0.85, 0.35]) * (0.5 + 0.8 * flash_bg)),
           reflect_floor=FLOOR, refl_strength=0.3, shadow=0.55, bright=1.0 + 0.12 * flash_bg)

    # electric arcs crawling along the can edges
    core2 = np.zeros((H, W), np.float32)
    glow2 = np.zeros((H // 2, W // 2), np.float32)
    ra = np.random.default_rng(int(t * FPS) // 2 + 500)
    x0w, y0w, x1w, y1w = c.world_bounds(540, CAN_Y)
    for k in range(4 if t < 18 else 7):
        side = ra.choice([-1, 1])
        xe = (x1w if side > 0 else x0w) + side * ra.uniform(-6, 14)
        ya = ra.uniform(y0w + 120, y1w - 60)
        yb = ya + ra.uniform(90, 260) * ra.choice([-1, 1])
        p0 = cam.p(xe, ya)
        p1 = cam.p(xe + side * ra.uniform(0, 30), yb)
        fx.draw_bolt(core2, glow2, p0, p1, ra, width=2, branches=1)
    fx.composite_electric(canvas, core2, glow2, (0.45, 0.8, 1.0), 0.75 + 0.4 * flash_bg)

    # sparks (banana-yellow) on each strike
    lay = np.zeros((H, W), np.float32)
    for i, s in enumerate(TL.STRIKES_S4):
        a = t - s
        if a < 0 or a > 0.75:
            continue
        rs = np.random.default_rng(300 + i)
        for j in range(60 if s in (15.0, 18.0) else 32):
            ox = 540 + rs.uniform(-260, 260)
            oy = FLOOR - rs.uniform(0, 60)
            ang = rs.uniform(-np.pi * 0.95, -np.pi * 0.05)
            v = rs.uniform(700, 2300)
            px = ox + np.cos(ang) * v * a
            py = oy + np.sin(ang) * v * a + 1600 * a * a
            qx = ox + np.cos(ang) * v * (a - 0.035)
            qy = oy + np.sin(ang) * v * (a - 0.035) + 1600 * (a - 0.035) ** 2
            p, q = cam.p(px, py), cam.p(qx, qy)
            cv2.line(lay, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), float(1 - a / 0.75), 2, cv2.LINE_AA)
    if lay.any():
        g = cv2.GaussianBlur(lay, (0, 0), 1.0) + cv2.GaussianBlur(lay, (0, 0), 5) * 1.2
        canvas += g[..., None] * np.array([1.0, 0.85, 0.35], np.float32)
    dust(canvas, t, 15, 50, (1.0, 0.9, 0.4), inten=0.5, vy=-60)

    flavor_tag(canvas, t, 15.55, 16.75, TL.PRODUCTS[3]["name"], (0.3, 0.7, 1.0))

    if zb > 0:
        canvas = zoom_blur(canvas, zb, *cam.p(540, 930))
    if vb > 1:
        canvas = vblur(canvas, vb)
    post["flash"] = 0.0
    if se:
        post["flash"] = 0.55 * np.exp(-se[0] / 0.05) * (1.0 if TL.STRIKES_S4[se[1]] in (15.0, 18.0) else 0.5)
    post["flash"] += (1 - ease_out_cubic(lin(t, 15.0, 15.25))) * 0.9
    post["flash"] += 0.25 * decay(t, 17.0, 0.05) + 0.25 * decay(t, 17.5, 0.05)
    post["flash"] += 0.8 * ease_in_expo(lin(t, 18.85, 19.0))
    post["flash_color"] = (0.85, 0.95, 1.0)
    post["ca"] = 8 * (np.exp(-se[0] / 0.1) if se else 0) + 10 * lin(t, TL.WHIP_OUT_S4, 19.0)
    return canvas, post


# =========================================================== END CARD
NAME = "EZY PZY"
SUB = "CONVENIENCE"
ADDR1 = "153 Great Northern Rd,"
ADDR2 = "Sault Ste. Marie, ON P6B 4Y9"
CTA1 = "STOP BY & GRAB"
CTA2 = "YOUR FAVOURITES!"
GOLD = (1.0, 0.80, 0.38)


def end_layout():
    L = {}
    L["name_size"] = fit_size(NAME, "InterDisplay-Black.otf", 920, 280)
    L["sub_size"] = fit_size(SUB, "InterDisplay-SemiBold.otf", 820, 76, tracking=0.42)
    s1 = fit_size(ADDR1, "InterDisplay-Medium.otf", 840, 62)
    s2 = fit_size(ADDR2, "InterDisplay-Medium.otf", 840, 62)
    L["addr_size"] = min(s1, s2)
    c1 = fit_size(CTA1, "InterDisplay-Black.otf", 640, 76)
    c2 = fit_size(CTA2, "InterDisplay-Black.otf", 640, 76)
    L["cta_size"] = min(c1, c2)
    return L


EL = end_layout()


def end_card(t, fi):
    post = dict(flash=0.0, ca=0.0, grain=0.014)
    e = t - TL.END_START
    shift = 1800 * (1 - ease_out_cubic(lin(t, TL.END_START, TL.END_START + 0.28)))
    canvas = radial_bg((0.07, 0.06, 0.13), (0.004, 0.004, 0.012), 540, 860, 1000, 1300, 1.2).copy()
    # drifting light leaks in the four flavour colours
    leaks = [((0.45, 1.0, 0.2), 120, 380), ((0.45, 0.3, 1.0), 980, 520), ((0.3, 0.85, 0.8), 160, 1500), ((0.3, 0.7, 1.0), 950, 1650)]
    for i, (col, x, y) in enumerate(leaks):
        add_glow(canvas, x + 90 * np.sin(e * 0.5 + i), y + 70 * np.cos(e * 0.4 + i * 2) + shift, 620, col, 0.16)
    god_rays(canvas, t, 540, 700 + shift, GOLD, 0.11 + 0.10 * decay(t, TL.END_NAME, 0.5), n=18, speed=0.03, sharp=8)
    add_glow(canvas, 540, 760 + shift, 760, GOLD, 0.18 * smooth(lin(t, TL.END_NAME, TL.END_NAME + 0.6)))
    bokeh(canvas, t, 31, 18, [(0.5, 1.0, 0.3), (0.55, 0.4, 1.0), (0.35, 0.9, 0.85), (0.35, 0.75, 1.0), GOLD], inten=0.09, rmin=30, rmax=90, speed=14)
    dust(canvas, t, 21, 150, (1.0, 0.85, 0.5), inten=0.7, rmin=2, rmax=6, vy=-40)
    # anamorphic gold flare as the name lands
    fl = decay(t, TL.END_NAME + 0.1, 0.45) if t >= TL.END_NAME + 0.1 else 0.0
    if fl > 0.01:
        add_glow(canvas, 540, 700 + shift, 900, (1.0, 0.85, 0.55), 0.55 * fl, sx=1.0, sy=0.035)
        add_glow(canvas, 540, 700 + shift, 300, (1.0, 0.95, 0.85), 0.5 * fl, sx=1.0, sy=0.1)
    # reflective floor glow under the CTA
    add_glow(canvas, 540, 1560 + shift, 620, GOLD, 0.05, sy=0.25)

    yo = shift
    ny = 700 + yo
    nsize = EL["name_size"]
    f = fx.font("InterDisplay-Black.otf", nsize)
    adv = [f.getlength(ch) for ch in NAME]
    total = sum(adv)
    x = 540 - total / 2
    fin = TL.END_NAME + 0.06 * len(NAME) + 0.35
    final_punch = 1 + 0.035 * decay(t, 23.0, 0.25) if t >= 23.0 else 1.0
    if t < fin:
        for i, (ch, a) in enumerate(zip(NAME, adv)):
            ti = TL.END_NAME + i * 0.06
            k = ease_out_expo(lin(t, ti, ti + 0.32))
            if ch != " " and k > 0:
                draw_text(canvas, ch, "InterDisplay-Black.otf", nsize, x + a / 2, ny - 60 * (1 - k), alpha=k,
                          scale=lerp(1.9, 1.0, k), blur=(1 - k) * 16, grad=((1, 1, 1), (1.0, 0.88, 0.62)),
                          glow=GOLD, glow_r=30, glow_s=0.55)
            x += a
    else:
        info = draw_text(canvas, NAME, "InterDisplay-Black.otf", nsize, 540, ny, scale=final_punch,
                         grad=((1, 1, 1), (1.0, 0.88, 0.62)), glow=GOLD, glow_r=30, glow_s=0.55 + 0.25 * decay(t, 23.0, 0.3) if t >= 23 else 0.55)
        # shine sweeps
        for s in TL.END_SHINES:
            if s <= t < s + 0.7:
                x0, y0, w, h, a = info
                pos = lerp(-0.3, 1.3, ease_in_out((t - s) / 0.7))
                xs = np.arange(w, dtype=np.float32)[None, :] / w
                ys = np.arange(h, dtype=np.float32)[:, None] / h
                band = np.exp(-((xs + 0.25 * ys - pos) / 0.06) ** 2)
                blend_add(canvas, (a * band * 0.9)[..., None] * np.array([1, 1, 1], np.float32), x0, y0)
    # sub-title with tracking reveal
    k = ease_out_cubic(lin(t, TL.END_SUB, TL.END_SUB + 0.45))
    if k > 0:
        tr = lerp(0.9, 0.42, k)
        size = EL["sub_size"]
        draw_text(canvas, SUB, "InterDisplay-SemiBold.otf", size, 540, 878 + yo, alpha=k, tracking=round(tr, 2) if k < 1 else 0.42,
                  scale=min(1.0, lerp(0.8, 1.0, k)), glow=GOLD, glow_r=14, glow_s=0.3, blur=(1 - k) * 6)
    # gold rule
    k = ease_in_out(lin(t, TL.END_RULE, TL.END_RULE + 0.4))
    if k > 0:
        hw = 230 * k
        y = int(965 + yo)
        lay = np.zeros((H, W), np.float32)
        cv2.line(lay, (int(540 - hw), y), (int(540 + hw), y), 1.0, 3, cv2.LINE_AA)
        g = cv2.GaussianBlur(lay, (0, 0), 6)
        canvas += (lay * 0.9 + g * 1.2)[..., None] * np.array(GOLD, np.float32)
        add_glow(canvas, 540 + hw, y, 40, (1, 0.95, 0.8), 0.6 * (1 - k))
        add_glow(canvas, 540 - hw, y, 40, (1, 0.95, 0.8), 0.6 * (1 - k))
    # address
    for j, (txt, yy) in enumerate([(ADDR1, 1052), (ADDR2, 1128)]):
        tj = TL.END_ADDR + j * 0.1
        k = ease_out_cubic(lin(t, tj, tj + 0.4))
        if k > 0:
            draw_text(canvas, txt, "InterDisplay-Medium.otf", EL["addr_size"], 540, yy + yo + 26 * (1 - k), alpha=k * 0.95,
                      blur=(1 - k) * 5, shadow=0.4)
    # map-pin dot accents
    # CTA pill
    k = ease_out_back(lin(t, TL.END_CTA, TL.END_CTA + 0.38), 2.2)
    ka = lin(t, TL.END_CTA, TL.END_CTA + 0.15)
    if ka > 0:
        cs = EL["cta_size"]
        f2 = fx.font("InterDisplay-Black.otf", cs)
        tw = max(f2.getlength(CTA1), f2.getlength(CTA2))
        pw, ph = tw + 110, cs * 2.55
        cy = 1318 + yo
        pulse = 1 + 0.02 * np.sin(max(0, t - TL.END_CTA - 0.4) * 5.5) * (t > TL.END_CTA + 0.4)
        sc = max(0.01, k) * pulse
        pad = 30
        lw, lh = int(pw * sc) + 2 * pad, int(ph * sc) + 2 * pad
        pm = np.zeros((lh, lw), np.float32)
        rr = int(ph * sc / 2)
        x0p, y0p, x1p, y1p = pad, pad, lw - pad, lh - pad
        cv2.rectangle(pm, (x0p + rr, y0p), (x1p - rr, y1p), 1.0, -1, cv2.LINE_AA)
        cv2.circle(pm, (x0p + rr, (y0p + y1p) // 2), rr, 1.0, -1, cv2.LINE_AA)
        cv2.circle(pm, (x1p - rr, (y0p + y1p) // 2), rr, 1.0, -1, cv2.LINE_AA)
        gy = np.linspace(0, 1, lh, dtype=np.float32)[:, None, None]
        col = np.array([1.0, 0.86, 0.45], np.float32) * (1 - gy) + np.array([0.98, 0.60, 0.16], np.float32) * gy
        a = pm * ka
        glow = cv2.GaussianBlur(pm, (0, 0), 16) * 0.55 * ka
        blend_add(canvas, glow[..., None] * np.array(GOLD, np.float32), int(540 - lw / 2), int(cy - lh / 2))
        blend_over(canvas, col * a[..., None], a, int(540 - lw / 2), int(cy - lh / 2))
        # shine across pill
        sp = lin(t, TL.END_CTA + 0.5, TL.END_CTA + 1.2)
        if 0 < sp < 1:
            xs = np.arange(lw, dtype=np.float32)[None, :] / lw
            band = np.exp(-((xs - lerp(-0.2, 1.2, sp)) / 0.05) ** 2) * pm * 0.6
            blend_add(canvas, band[..., None] * np.ones(3, np.float32), int(540 - lw / 2), int(cy - lh / 2))
        for line, dy_ in ((CTA1, -cs * 0.58), (CTA2, cs * 0.58)):
            draw_text(canvas, line, "InterDisplay-Black.otf", cs, 540, cy + dy_ * sc, color=(0.07, 0.05, 0.10),
                      alpha=ka, scale=sc)

    if shift > 1:
        canvas = vblur(canvas, shift * 0.12)
    # slow cinematic push-in on the whole card
    zs = 1 + 0.04 * ease_in_out(lin(t, TL.END_START + 0.3, TL.DURATION))
    M = np.float32([[zs, 0, 540 - zs * 540], [0, zs, 960 - zs * 960]])
    canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    post["flash"] = 0.85 * (1 - ease_out_cubic(lin(t, TL.END_START, TL.END_START + 0.2)))
    post["flash"] += 0.18 * decay(t, TL.END_NAME, 0.06) + 0.12 * decay(t, 23.0, 0.08) if t >= 23.0 else 0.18 * decay(t, TL.END_NAME, 0.06)
    post["flash_color"] = (1.0, 0.95, 0.85)
    post["ca"] = 6 * decay(t, TL.END_NAME, 0.12)
    fade = 1 - smooth(lin(t, TL.END_FADE, TL.DURATION - 1.0 / FPS))
    post["fade"] = fade
    return canvas, post


# =========================================================== frame dispatcher
def render_frame(fi):
    t = fi / FPS
    if t < 7.0:
        canvas, post = scene1(t, fi)
    elif t < 11.0:
        canvas, post = scene2(t, fi)
    elif t < 15.0:
        canvas, post = scene3(t, fi)
    elif t < TL.END_START:
        canvas, post = scene4(t, fi)
    else:
        canvas, post = end_card(t, fi)
    img = finish(canvas, fi, flash=min(1.0, post.get("flash", 0)), flash_color=post.get("flash_color", (1, 1, 1)),
                 ca=post.get("ca", 0), grain=post.get("grain", 0.016))
    if "fade" in post:
        img *= post["fade"]
    return (img * 255 + 0.5).astype(np.uint8)


def _work(fi):
    path = os.path.join(FRAMES, f"{fi:05d}.png")
    if os.path.exists(path):
        return fi, 0.0
    t0 = time.time()
    img = render_frame(fi)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--range", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.test:
        out = os.path.join(BUILD, "test")
        os.makedirs(out, exist_ok=True)
        for ts in args.test.split(","):
            t = float(ts)
            fi = int(round(t * FPS))
            t0 = time.time()
            img = render_frame(fi)
            cv2.imwrite(os.path.join(out, f"t{t:06.2f}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"t={t:.2f} frame {fi} {time.time() - t0:.2f}s")
        return
    os.makedirs(FRAMES, exist_ok=True)
    n = int(round(TL.DURATION * FPS))
    frames = range(n)
    if args.range:
        a, b = map(int, args.range.split(":"))
        frames = range(a, b)
    t0 = time.time()
    done = 0
    with Pool(args.jobs) as p:
        for fi, dt in p.imap_unordered(_work, frames, chunksize=2):
            done += 1
            if done % 30 == 0:
                print(f"{done}/{len(frames)} frames  {time.time() - t0:.0f}s elapsed", flush=True)
    print("render done", time.time() - t0)


if __name__ == "__main__":
    main()
