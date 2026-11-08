"""Reel 5 - "Red Bull Energy Universe" for Ezy Pzy Convenience (13 editions, 2 FOR $5.99 BEFORE TAXES).

    python reel5_render.py --test 1.5,4.0     # stills -> output/build5/test/
    python reel5_render.py --all              # all frames (60 fps) -> output/build5/frames/
    python reel5_render.py --cover            # output/reel5/Ezy_Pzy_Red_Bull_Energy_Cover.jpg
    python reel5_render.py --stills           # output/reel5/product_visuals/*.jpg

A separate project from Reel 4: each can is cut out of its photo (packaging untouched) and placed
in a new environment. One can per frame; every transition passes through a covered state
(flash, blackout, iris, CRT dot, full defocus or full smear) so two cans never share a frame.
The motorsport price badge is composited on every frame, end card included.

  bar  edition                     environment            entrance              exit
   -   intro (Summer)              dark chamber           energy field forms    (EM shockwave)
   1   Summer  - Sudachi Lime      brushed-metal chamber  EM shockwave          light-speed jump
   2   Ice     - Iced Vanilla Berry speed tunnel          tunnel deceleration   iris shutter
   3   Yellow  - Tropical          carbon-fibre wall      metallic shockwave    CRT collapse
   4   Spring  - Cherry Sakura     hologram hex grid      hologram scan         particle dissolve
   5   Peach   - White Peach       nebula                 particle assembly     glitch
   6   Red     - Watermelon        racing circuit         headlight sweep       light-trail rush
   7   Sea Blue - Juneberry        EM ring chamber        blue pulse            energy surge
   8   Amber   - Strawberry Apricot plasma vortex         vortex collapse       defocus
   9   Pink    - Wild Berries      prism beams            refocus + RGB split   prism flash
  10   Lilac   - Grapefruit & Blossom tesla chamber       arc strike            arc whiteout
  11   Winter  - Pistachio & Berries wind tunnel          laser scan            overtake smear
  12   Coconut - Coconut Berry     zero-gravity void      drop-in + hover pad   power-down
  13   Apple   - Fuji Apple & Ginger turbine              ignition              launch
   -   end card: logo, both locations, 2 FOR $5.99 BEFORE TAXES, POWER UP YOUR DAY. VISIT US TODAY!
"""
import os
import time
import argparse
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import cv2
from PIL import Image

import timeline5 as TL
from timeline5 import W, H, FPS
import fx
from fx import (clamp, lin, smooth, decay, ease_out_cubic, ease_in_cubic, ease_in_out, ease_out_expo, ease_in_expo,
                lerp, add_glow, draw_text, fit_size, hblur, finish, blend_over)
import reel5_fx as X
import reel2_two_locations as TLOC

ROOT = fx.ROOT
BUILD = os.path.join(ROOT, "output", "build5")
FRAMES = os.path.join(BUILD, "frames")
OUT = os.path.join(ROOT, "output", "reel5")

ACC = [(1.0, 0.88, 0.2), (0.55, 0.85, 1.0), (1.0, 0.78, 0.15), (1.0, 0.2, 0.25), (1.0, 0.36, 0.52), (1.0, 0.16, 0.16),
       (0.25, 0.5, 1.0), (1.0, 0.5, 0.1), (1.0, 0.32, 0.75), (0.68, 0.42, 1.0), (0.72, 1.0, 0.25), (0.85, 0.93, 1.0), (1.0, 0.38, 0.12)]
SEC = [(0.3, 0.65, 1.0), (0.88, 0.94, 1.0), (0.85, 0.9, 1.0), (1.0, 0.95, 0.95), (1.0, 0.75, 0.65), (0.35, 1.0, 0.5),
       (0.6, 0.88, 1.0), (1.0, 0.22, 0.15), (0.6, 0.4, 1.0), (0.4, 0.9, 1.0), (1.0, 0.28, 0.58), (0.35, 0.65, 1.0), (1.0, 0.78, 0.28)]
BLUE = (0.3, 0.6, 1.0)

_cans = {}


def get_can(i):
    if i not in _cans:
        for k in [k for k in _cans if abs(k - i) > 1]:
            del _cans[k]
        _cans[i] = X.Can5(TL.PRODUCTS[i]["key"])
    return _cans[i]


# ---------------------------------------------------------------- scene-specific layers
def vortex(canvas, t, cx, cy, col_a, col_b, speed=1.0, inten=1.0, seed=88, n=320):
    r = np.random.default_rng(seed)
    lay = X.half()
    for j in range(n):
        ph, a0 = r.uniform(0, 1), r.uniform(0, 2 * np.pi)
        rr = 1100 * (((ph - t * 0.55 * speed) % 1.0) ** 1.4) + 20
        rt = rr * 1.10
        ang = a0 + 2600 / (rr + 260) + t * 1.5 * speed
        angt = a0 + 2600 / (rt + 260) + t * 1.5 * speed
        b = float(np.clip(1.3 - rr / 1100, 0, 1)) * inten
        col = col_a if j % 3 else col_b
        p0 = (cx + np.cos(angt) * rt, cy + np.sin(angt) * rt * 0.8)
        p1 = (cx + np.cos(ang) * rr, cy + np.sin(ang) * rr * 0.8)
        cv2.line(lay, (int(p0[0] / 2 * 16), int(p0[1] / 2 * 16)), (int(p1[0] / 2 * 16), int(p1[1] / 2 * 16)), tuple(float(c) * b for c in col), 1, cv2.LINE_AA, 4)
    X.glow_add(canvas, lay, core=1.2, g1=2, g2=9, k1=0.8, k2=0.9)
    add_glow(canvas, cx, cy, 380, col_a, 0.25 * inten)


def hover_pad(canvas, t, cx, y, inten, col):
    lay = np.zeros((H, W), np.float32)
    for k in range(4):
        ph = ((t * 1.6 + k / 4) % 1.0)
        rx, ry = 220 + 260 * ph, 26 + 30 * ph
        cv2.ellipse(lay, (int(cx * 16), int(y * 16)), (int(rx * 16), int(ry * 16)), 0, 0, 360, float(1 - ph), 2, cv2.LINE_AA, 4)
    canvas += (lay * 0.7 + cv2.GaussianBlur(lay, (0, 0), 6))[..., None] * np.array(col, np.float32) * inten
    add_glow(canvas, cx, y, 300, col, 0.35 * inten, sx=1.0, sy=0.18)


def make_env(i, t, u, M, can, floor_y, par, acc, sec, extra):
    env = TL.PRODUCTS[i]["env"]
    cx, cy = X.apply(M, can.cx, (can.top + can.bot) / 2)
    if env == "chamber":
        c = X.chamber(t, sec, floor_y, par)
    elif env == "tunnel":
        c = X.env_base((0.05, 0.08, 0.12), (0.005, 0.008, 0.015))
        c = X.speed_tunnel(c, t * 0.18, 0.25, [(0.85, 0.92, 1.0), (0.45, 0.75, 1.0)], inten=0.55, cx=540, cy=820, seed=21)
        c = X.floor(c, floor_y, acc)
    elif env == "carbon":
        c = X.carbon_wall(t, acc, floor_y, par)
    elif env == "hexgrid":
        c = X.hexgrid(t, acc, floor_y)
    elif env == "nebula":
        c = X.nebula(t, [acc, sec], floor_y)
        X.float_motes(c, t, 51, 50, (1.0, 0.75, 0.8), inten=0.5)
    elif env == "circuit":
        c = X.circuit(t, acc, floor_y)
        c = X.light_trails(c, t, [acc, (1.0, 0.95, 0.9)], seed=61, n=8, y_range=(1050, 1500), inten=0.9)
    elif env == "rings":
        c = X.chamber(t, acc, floor_y, par) * np.array([0.7, 0.8, 1.1], np.float32)
    elif env == "vortex":
        c = X.env_base((0.12, 0.05, 0.02), (0.01, 0.005, 0.0))
        vortex(c, t, cx, cy, acc, sec, speed=extra.get("vspeed", 1.0), inten=extra.get("vint", 1.0))
        c = X.floor(c, floor_y, acc, 0.4)
    elif env == "prism":
        c = X.env_base((0.09, 0.03, 0.08), (0.008, 0.004, 0.01))
        c = X.prism_beams(c, t, inten=extra.get("pint", 1.0), offset=extra.get("poff", 0.0))
        c = X.floor(c, floor_y, acc)
    elif env == "tesla":
        c = X.env_base((0.07, 0.04, 0.12), (0.006, 0.004, 0.012))
        add_glow(c, 540, cy - 100, 700, acc, 0.22)
        tg = list(extra.get("targets", ())) + [(0, 260), (W, 300), (0, 1500), (W, 1460)]
        X.tesla_arcs(c, t, int(t * 60), [(120, cy - 120), (960, cy - 120)], acc, inten=extra.get("aint", 1.2),
                     n_arcs=extra.get("narcs", 3), seed=i, targets=tg)
        c = X.floor(c, floor_y, acc)
    elif env == "wind":
        c = X.env_base((0.04, 0.08, 0.03), (0.004, 0.008, 0.003))
        x0, y0, x1, y1 = can.can_box(M)
        c = X.streamlines(c, t, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + 30, (y1 - y0) / 2, acc, inten=1.0)
        c = X.floor(c, floor_y, acc)
    elif env == "void":
        c = X.env_base((0.04, 0.05, 0.08), (0.003, 0.004, 0.008))
        X.float_motes(c, t, 121, 70, (0.75, 0.85, 1.0), inten=0.45, rmin=2, rmax=6)
        X.float_motes(c, t * 0.6, 122, 18, (0.6, 0.75, 1.0), inten=0.18, rmin=14, rmax=30, drift=30)
        hover_pad(c, t, cx, extra.get("pad_y", floor_y), extra.get("pad", 1.0), sec)
    elif env == "turbine":
        c = X.env_base((0.12, 0.03, 0.02), (0.01, 0.003, 0.0))
        X.turbine(c, cx, cy - 40, 640, extra.get("tang", t * 3), extra.get("spin", 1.0), extra.get("heat", 1.0), acc)
        c = X.heat_shimmer(c, t, extra.get("heat", 1.0))
        c = X.floor(c, floor_y, acc, 0.4)
    return c


# ---------------------------------------------------------------- intro
def intro(t, fi):
    can = get_can(0)
    post = dict(flash=0.0, flash_col=(0.8, 0.9, 1.0), ca=0.0)
    k = t / TL.DROP
    zmul = lerp(0.86, 1.0, ease_in_cubic(k))
    M = can.cam(zmul, lift=(1 - zmul) * 300)
    floor_y = X.apply(M, can.cx, can.bot)[1]
    cx, cy = X.apply(M, can.cx, (can.top + can.bot) / 2)
    canvas = X.chamber(t, BLUE, floor_y) * (0.25 + 0.35 * k)
    canvas = X.horizontal_streaks(canvas, t, 0.3 + 2.6 * k ** 2, [BLUE, (0.8, 0.9, 1.0)], inten=0.3 + 1.2 * k, n=50)
    # energy field forming: rings contract and spin up
    rf = lerp(2.6, 1.0, ease_out_cubic(lin(t, 0.6, TL.DROP)))
    specs = [dict(cx=cx, cy=cy - 140, rx=360 * rf, tilt=0.22, rot=0.08, col=BLUE, inten=lin(t, 0.4, 2.0), speed=1 + 4 * k, n=12),
             dict(cx=cx, cy=cy + 260, rx=420 * rf, tilt=0.2, rot=-0.06, col=(0.6, 0.85, 1.0), inten=lin(t, 0.9, 2.4), speed=-(1 + 5 * k), n=14)]
    back, front = X.rings(specs, t)
    canvas += back
    # converging sparks
    r = np.random.default_rng(12)
    lay = X.half()
    for j in range(90):
        a = r.uniform(0, 2 * np.pi)
        ph = r.uniform(0, 1)
        d = 1100 * (1 - ((t * (0.6 + 1.2 * k) + ph) % 1.0))
        p0 = (cx + np.cos(a) * d, cy + np.sin(a) * d)
        p1 = (cx + np.cos(a) * (d + 60 + 200 * k), cy + np.sin(a) * (d + 60 + 200 * k))
        cv2.line(lay, (int(p0[0] / 2 * 16), int(p0[1] / 2 * 16)), (int(p1[0] / 2 * 16), int(p1[1] / 2 * 16)), (0.4 * k, 0.65 * k, 1.0 * k), 1, cv2.LINE_AA, 4)
    canvas = X.glow_add(canvas, lay, core=1.0, g1=2, g2=7)
    g = ease_in_expo(lin(t, 2.1, TL.DROP))
    add_glow(canvas, cx, cy, 200 + 800 * g, BLUE, 0.9 * g)
    L = can.layer(M, expo=0.05 + 0.35 * ease_in_cubic(lin(t, 1.0, TL.DROP)) + 0.5 * ease_in_expo(lin(t, 2.3, TL.DROP)),
                  sweep=[(lerp(-1.6, 1.6, ease_in_out(lin(t, 1.4, 2.6))), 0.3, 0.9 * lin(t, 1.3, 1.6), (0.75, 0.88, 1.0))],
                  rim_l=tuple(np.array(BLUE) * (0.4 + 1.2 * k)), rim_r=tuple(np.array(BLUE) * (0.3 + 1.0 * k)))
    X.reflection(canvas, L, floor_y, 0.12)
    X.blit(canvas, L)
    canvas += front
    # headline decode -> burst at the drop
    txt = "UNLEASH THE ENERGY."
    size = fit_size(txt, "InterDisplay-Black.otf", 900, 92)
    tw = X.text_width(txt, "InterDisplay-Black.otf", size)
    kd = lin(t, TL.TEXT_IN, TL.TEXT_DONE)
    burst = ease_in_cubic(lin(t, TL.DROP - 0.16, TL.DROP))
    if kd > 0:
        shown = X.decode(txt, kd, fi, seed=7)
        if burst > 0:
            draw_text(canvas, txt, "InterDisplay-Black.otf", size, 540, 280, alpha=1 - burst, scale=1 + 0.6 * burst, blur=10 * burst,
                      grad=X.CHROME, glow=BLUE, glow_r=24, glow_s=1.0)
        else:
            X.text_at(canvas, shown, "InterDisplay-Black.otf", size, 540 - tw / 2, 280, grad=X.CHROME, glow=BLUE, glow_r=20,
                      glow_s=0.6 + 0.6 * kd, shadow=0.5)
    post["flash"] = 0.45 * ease_in_expo(lin(t, TL.DROP - 0.3, TL.DROP))
    post["ca"] = 6 * lin(t, 2.2, TL.DROP)
    return canvas, post


# ---------------------------------------------------------------- one edition (one bar)
def drink(i, t, fi, clean=False):
    P = TL.PRODUCTS[i]
    can = get_can(i)
    s = TL.S[i]
    u = t - s
    acc, sec = ACC[i], SEC[i]
    ent, ex, shot2 = P["enter"], P["exit"], P["shot2"]
    post = dict(flash=0.0, flash_col=(1.0, 1.0, 1.0), ca=0.0)
    ke = clamp((u - TL.EXIT) / (TL.SCENE - TL.EXIT))
    hero = u >= TL.HERO2
    zmul, lift, roll, ax, par = 1.0, 0.0, 0.0, 540.0, 0.0
    if not hero:
        zmul = 1 + 0.04 * ease_in_out(u / TL.HERO2)
    else:
        k = (u - TL.HERO2) / (TL.SCENE - TL.HERO2)
        if shot2 == "push":
            zmul = lerp(1.2, 1.32, ease_out_cubic(k))
        elif shot2 == "float":
            zmul = 1.12
            lift = 26 * ease_out_cubic(min(1, k * 2.5)) + 7 * np.sin(u * 4.2)
            roll = 2.0 * np.sin(u * 2.6 + i)
        else:
            zmul = 1.16
            ax = 540 + lerp(-28, 28, ease_in_out(k))
            par = lerp(-160, 160, ease_in_out(k))
    expo, alpha = 0.95, 1.0
    extra = {}
    sweep_pos = lerp(-1.3, 1.3, u / TL.SCENE) * (1 if i % 2 == 0 else -1)
    sweep = [(sweep_pos, 0.26, 0.32, (1.0, 1.0, 1.0))]
    rim_l, rim_r = tuple(np.array(sec) * 0.7), tuple(np.array(acc) * 0.55)
    flash_layer = 0.0
    # ---------------- entrance
    if ent == "awaken":
        post["flash"] += 0.95 * decay(t, s, 0.07)
        post["ca"] += 14 * decay(t, s, 0.2)
        lift += 12 * decay(t, s, 0.12) * np.sin(u * 60)
    elif ent == "tunnel":
        kt = lin(u, 0, 0.5)
        sc = lerp(0.22, 1.0, ease_out_expo(kt))
        zmul *= sc
        lift += (1 - sc) * 780
        post["flash"] += 0.95 * (1 - ease_out_cubic(lin(u, 0, 0.2)))
        post["flash_col"] = (0.85, 0.92, 1.0)
    elif ent == "shockwave":
        post["flash"] += 0.4 * decay(t, s, 0.06)
    elif ent == "assemble":
        alpha = smooth(lin(u, 0.4, 0.62))
    elif ent == "headlights":
        expo = lerp(0.2, 0.95, smooth(lin(u, 0.05, 0.5)))
        sweep = [(lerp(-1.7, 1.7, ease_in_out(lin(u, 0, 0.55))), 0.33, 1.3 * (1 - lin(u, 0.45, 0.7)), (1.0, 0.97, 0.92))] + sweep
    elif ent == "em_pulse":
        post["flash"] += 0.9 * decay(t, s, 0.07)
        post["flash_col"] = (0.55, 0.75, 1.0)
        flash_layer = 0.6 * decay(t, s, 0.15)
    elif ent == "vortex":
        kv = lin(u, 0, 0.55)
        extra["vspeed"] = 1 + 3 * (1 - kv)
        alpha = smooth(lin(u, 0.12, 0.42))
        sc = lerp(0.55, 1.0, ease_out_cubic(lin(u, 0.08, 0.5)))
        zmul *= sc
        lift += (1 - sc) * 780
        post["flash"] += 0.9 * (1 - ease_out_cubic(lin(u, 0, 0.2)))
    elif ent == "arc":
        post["flash"] += 0.9 * (1 - ease_out_cubic(lin(u, 0, 0.12))) + 0.5 * decay(t, s + 0.12, 0.06) * (u > 0.12)
        post["flash_col"] = (0.85, 0.75, 1.0)
        flick = 0.55 + 0.45 * (np.sin(u * 140) > 0) if u < 0.3 else 1.0
        expo *= flick
        extra["aint"] = 1.2 + 2.0 * decay(t, s + 0.12, 0.15)
        extra["narcs"] = 4 if u < 0.35 else 3
    elif ent == "scan":
        post["flash"] += 0.95 * (1 - ease_out_cubic(lin(u, 0, 0.18)))
    elif ent == "dropin":
        kd = lin(u, 0, 0.5)
        e = ease_out_cubic(kd)
        lift += 1750 * (1 - e)
        extra["pad"] = 0.4 + 1.4 * decay(t, s + 0.42, 0.25) * (u > 0.42) + 0.6 * smooth(lin(u, 0.3, 0.5))
    elif ent == "ignition":
        heat = smooth(lin(u, 0, 0.5))
        extra["heat"] = 0.25 + 0.75 * heat
        extra["spin"] = 0.3 + 1.2 * heat
        expo = lerp(0.12, 0.95, smooth(lin(u, 0.12, 0.5)))
        rim_l, rim_r = tuple(np.array(acc) * (1.6 - heat)), tuple(np.array(sec) * (1.4 - heat * 0.8))
    # ---------------- exit (camera part)
    if ex == "lightspeed":
        zmul *= 1 + 0.28 * ke ** 2
    elif ex == "launch":
        zmul *= 1 + 0.3 * ke ** 2
        extra["spin"] = extra.get("spin", 1.0) + 5 * ke
    if ex == "whiteout":
        extra["aint"] = extra.get("aint", 1.2) + 3 * ke
        extra["narcs"] = 5 if ke > 0 else extra.get("narcs", 3)
    if ex == "prism":
        extra["pint"] = 1 + 6 * ke
        extra["poff"] = 0.3 * ke
    if TL.PRODUCTS[i]["env"] == "turbine":
        spin = extra.get("spin", 1.0)
        extra["tang"] = (u * 2.0 + u * u * 2.0 * spin) * spin
    M = X.affine(can.wide_z() * zmul, can.cx, can.pbot, ax, X.SAFE_BOTTOM - lift, roll)
    floor_y = X.apply(M, can.cx, can.bot)[1]
    box = can.can_box(M)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if TL.PRODUCTS[i]["env"] == "void":
        extra["pad_y"] = X.SAFE_BOTTOM - 6
    if TL.PRODUCTS[i]["env"] == "tesla" and ent == "arc" and u < 0.35:
        extra["targets"] = [(box[0] + 10, cy - 200), (box[2] - 10, cy + 100)]
    # ---------------- environment
    canvas = make_env(i, t, u, M, can, floor_y, par, acc, sec, extra)
    if ent == "assemble":
        canvas *= smooth(lin(u, 0, 0.3))
    if ent == "headlights":
        canvas *= lerp(0.15, 1.0, smooth(lin(u, 0, 0.5)))
    if ent == "arc":
        canvas *= lerp(0.25, 1.0, smooth(lin(u, 0, 0.35)))
    if ent == "ignition":
        canvas *= lerp(0.3, 1.0, smooth(lin(u, 0, 0.4)))
    if ent == "dropin" and u < 0.22:
        canvas = hblur(canvas, 300 * (1 - ease_out_cubic(u / 0.22)))
    if hero:
        canvas = X.soft_blur(canvas, 5.0 * smooth(lin(u, TL.HERO2, TL.HERO2 + 0.15)))     # focus pull onto the hero
    if ex == "dissolve":
        canvas *= 1 - 0.85 * ke
    # rings (energy field) for Summer and Sea Blue
    front = None
    if i in (0, 6):
        rf = 1.0
        spd = 3.0
        inten = 1.0
        if ent == "em_pulse":
            rf = lerp(3.2, 1.0, ease_out_expo(lin(u, 0, 0.5)))
            spd = 3 + 6 * (1 - lin(u, 0, 0.6))
        if ex == "surge":
            rf *= lerp(1.0, 0.04, ease_in_cubic(ke))
            inten = 1 + 3 * ke
        col_a = BLUE if i == 0 else acc
        hh = box[3] - box[1]
        specs = [dict(cx=cx, cy=box[1] + hh * 0.28, rx=(box[2] - box[0]) * 0.95 * rf, tilt=0.22, rot=0.07, col=col_a, inten=inten, speed=spd, n=12),
                 dict(cx=cx, cy=box[1] + hh * 0.62, rx=(box[2] - box[0]) * 1.12 * rf, tilt=0.2, rot=-0.05, col=sec, inten=inten, speed=-spd * 1.2, n=14),
                 dict(cx=cx, cy=box[1] + hh * 0.9, rx=(box[2] - box[0]) * 1.3 * rf, tilt=0.18, rot=0.03, col=col_a, inten=0.7 * inten, speed=spd * 0.8, n=10)]
        back, front = X.rings(specs, t)
        canvas += back
    # shockwaves (behind the can)
    if ent in ("awaken", "shockwave") and u < 0.75:
        kw = ease_out_expo(lin(u, 0, 0.7))
        canvas = X.shockwave(canvas, cx, cy, 1800 * kw, 90, 46 * (1 - kw), BLUE if i == 0 else acc, inten=1.2 * (1 - lin(u, 0, 0.7)))
    # ---------------- product layer
    L = can.layer(M, expo=expo, sweep=sweep, rim_l=rim_l, rim_r=rim_r, flash=flash_layer, flash_col=acc, alpha=alpha,
                  vgrad=0.12)
    ghost = None
    if ent == "hologram" and u < 0.68:
        sy = lerp(box[1] - 30, box[3] + 30, ease_in_out(lin(u, 0.14, 0.64)))
        ghost, L = X.hologram(L, t, (1.0, 0.35, 0.35), solid_y=sy)
    elif ent == "scan" and u < 0.55:
        sx_ = lerp(box[0] - 30, box[2] + 30, ease_in_out(lin(u, 0.06, 0.5)))
        gh, _ = X.hologram(L, t, (0.6, 1.0, 0.35))
        xs = np.arange(L["a"].shape[1], dtype=np.float32) + L["x0"]
        m = np.clip((sx_ - xs) / 10, 0, 1)[None, :]
        ghost = dict(gh, rgb=gh["rgb"] * (1 - m)[..., None])
        L = dict(L, rgb=L["rgb"] * m[..., None], a=L["a"] * m)
    if ent == "dropin" and u < 0.45:
        sp = (1 - ease_out_cubic(lin(u, 0, 0.5)))
        kb = int(1 + 60 * sp)
        if kb > 2:
            L = dict(L, rgb=cv2.blur(L["rgb"], (1, kb)), a=cv2.blur(L["a"], (1, kb)))
    if ex == "dissolve" and ke > 0:
        m, edge = X.dissolve_mask(L, ke)
        L = dict(L, rgb=L["rgb"] * m[..., None] + (edge * L["a"])[..., None] * np.array(acc, np.float32) * 1.5, a=L["a"] * m)
    refl = 0.2
    if ent == "dropin":
        refl *= smooth(lin(u, 0.3, 0.5))
    X.reflection(canvas, L, floor_y, refl * alpha)
    X.blit(canvas, L)
    if ghost is not None:
        X.blit(canvas, ghost, additive=True)
        if ent == "hologram":
            lay = np.zeros((H, W), np.float32)
            yy_ = int(np.clip(sy, 0, H - 1))
            cv2.line(lay, (int(box[0] - 60), yy_), (int(box[2] + 60), yy_), 1.0, 2, cv2.LINE_AA)
            canvas += (lay + cv2.GaussianBlur(lay, (0, 0), 8) * 2)[..., None] * np.array((1.0, 0.45, 0.45), np.float32)
        else:
            lay = np.zeros((H, W), np.float32)
            xx_ = int(np.clip(sx_, 0, W - 1))
            cv2.line(lay, (xx_, int(box[1] - 60)), (xx_, int(box[3] + 30)), 1.0, 2, cv2.LINE_AA)
            canvas += (lay + cv2.GaussianBlur(lay, (0, 0), 8) * 2)[..., None] * np.array((0.6, 1.0, 0.35), np.float32)
    if front is not None:
        if L is not None:
            cm = np.zeros((H, W), np.float32)
            h_, w_ = L["a"].shape
            cm[L["y0"]:L["y0"] + h_, L["x0"]:L["x0"] + w_] = L["a"]
            front = front * (1 - 0.78 * cm)[..., None]
        canvas += front
    # ---------------- entrance overlays
    if ent == "assemble" and u < 0.65:
        px, py, on = X.can_particles(can, M, lin(u, 0, 0.6), "assemble", swirl=1.0)
        X.particles_draw(canvas, px[on], py[on], can.pcol[on], 1.0 - smooth(lin(u, 0.5, 0.65)))
    if ent in ("awaken", "shockwave", "ignition"):
        X.metal_debris(canvas, cx, cy, u - (0.08 if ent == "ignition" else 0.0), seed=i + 3,
                       col=(1.0, 0.8, 0.4) if ent == "ignition" else (0.85, 0.92, 1.0))
    if ent == "headlights" and u < 0.7:
        hx = lerp(-200, W + 200, ease_in_out(lin(u, 0, 0.55)))
        add_glow(canvas, hx, cy + 120, 900, (1.0, 0.97, 0.92), 0.7 * (1 - lin(u, 0.4, 0.7)), sx=1.0, sy=0.035)
        add_glow(canvas, hx, cy + 120, 160, (1.0, 1.0, 1.0), 0.6 * (1 - lin(u, 0.4, 0.7)))
    if ent == "shockwave" and u < 0.25:
        canvas = X.iris(canvas, ease_out_cubic(lin(u, 0, 0.22)), rot=0.5 * (1 - lin(u, 0, 0.22)))
    if ent == "hologram" and u < 0.2:
        canvas = X.crt(canvas, 1 - ease_out_cubic(lin(u, 0, 0.2)))
    if ent == "tunnel" and u < 0.55:
        kt = lin(u, 0, 0.5)
        canvas = X.speed_tunnel(canvas, 2.2 * (1 - (1 - kt) ** 3) + t * 0.1, 3.0 * (1 - kt) ** 2 + 0.2,
                                [(0.9, 0.95, 1.0), (0.45, 0.75, 1.0), (0.75, 0.82, 0.9)], inten=1.4 * (1 - kt) + 0.1, cx=540, cy=820, seed=33)
    if ent == "refocus" and u < 0.45:
        kr = 1 - ease_out_cubic(lin(u, 0, 0.42))
        canvas = X.soft_blur(canvas, 34 * kr)
        sh = int(36 * kr)
        if sh:
            canvas[..., 0] = np.roll(canvas[..., 0], sh, axis=1)
            canvas[..., 2] = np.roll(canvas[..., 2], -sh, axis=1)
    # ---------------- HUD (wide shot only)
    if not clean:
        ha = 1 - smooth(lin(u, TL.HERO2 - 0.12, TL.HERO2))
        if ha > 0 and u > 0.08:
            X.brackets(canvas, box, lin(u, 0.1, 0.45), acc, alpha=ha)
            X.hud_title(canvas, P["edition"], P["flavour"], lin(u, 0.12, 0.62), fi, acc, alpha=ha, idx=i)
    # ---------------- exit overlays
    if ex == "lightspeed" and ke > 0:
        canvas = X.speed_tunnel(canvas, t * 0.8 + ke * 2, 3 * ke, [(0.9, 0.95, 1.0), BLUE, (0.8, 0.85, 0.95)], inten=1.5 * ke, cx=540, cy=820, seed=33)
        post["flash"] += 0.95 * ease_in_expo(ke)
        post["flash_col"] = (0.85, 0.92, 1.0)
        post["ca"] += 14 * ke
    elif ex == "shutter" and ke > 0:
        canvas = X.iris(canvas, 1 - ease_in_cubic(ke), rot=0.5 * ke)
    elif ex == "crt" and ke > 0:
        canvas = X.crt(canvas, ke)
    elif ex == "dissolve" and ke > 0:
        px, py, on = X.can_particles(can, M, ke, "dissolve")
        X.particles_draw(canvas, px[on], py[on], can.pcol[on], 1.0)
    elif ex == "glitch" and ke > 0:
        canvas = X.glitch(canvas, ke, fi, seed=i) * (1 - smooth(lin(ke, 0.55, 1.0)))
        post["ca"] += 10 * ke
    elif ex == "trails" and ke > 0:
        canvas = X.light_trails(canvas, t, [acc, (1.0, 0.95, 0.9)], seed=66, n=24, y_range=(80, 1880), speed=2.4, density=1 + 2 * ke, inten=0.6 + 1.8 * ke)
        post["flash"] += 0.95 * ease_in_expo(ke)
        post["flash_col"] = (1.0, 0.85, 0.85)
    elif ex == "surge" and ke > 0:
        add_glow(canvas, cx, cy, 200 + 600 * ke, (0.6, 0.8, 1.0), 1.4 * ke)
        post["flash"] += 0.95 * ease_in_expo(ke)
        post["flash_col"] = (0.7, 0.85, 1.0)
    elif ex == "defocus" and ke > 0:
        canvas = X.soft_blur(canvas, 40 * ke ** 1.3) * (1 - 0.25 * ke)
    elif ex == "prism" and ke > 0:
        post["flash"] += 0.95 * ease_in_expo(ke)
    elif ex == "whiteout" and ke > 0:
        post["flash"] += ease_in_expo(ke)
        post["flash_col"] = (0.9, 0.85, 1.0)
    elif ex == "overtake" and ke > 0:
        canvas = X.horizontal_streaks(canvas, t, 1 + 5 * ke, [acc, (1.0, 1.0, 1.0), sec], inten=1 + 2 * ke, n=110, length=400)
        canvas = hblur(canvas, 300 * ke ** 2)
        post["ca"] += 10 * ke
    elif ex == "blackout" and ke > 0:
        flick = 1.0 if ke < 0.3 else (0.25 if int(ke * 22) % 2 else 1.0)
        canvas *= flick * (1 - smooth(lin(ke, 0.45, 0.95)))
    elif ex == "launch" and ke > 0:
        canvas = X.speed_tunnel(canvas, t * 0.9 + ke * 2.5, 3.2 * ke, [(1.0, 0.75, 0.3), (1.0, 0.35, 0.12), (1.0, 0.95, 0.85)], inten=1.6 * ke, cx=540, cy=860, seed=44)
        post["flash"] += 0.95 * ease_in_expo(ke)
        post["flash_col"] = (1.0, 0.92, 0.8)
    if ent == "vortex":
        add_glow(canvas, cx, cy, 300, acc, 0.8 * (1 - lin(u, 0.1, 0.5)))
    post["flash"] += 0.18 * decay(t, s + TL.HERO2, 0.05)
    post["ca"] += 4 * decay(t, s + TL.HERO2, 0.08)
    return canvas, post


# ---------------------------------------------------------------- end card
END_LEAK = [(0.25, 0.5, 1.0), (1.0, 0.2, 0.25)]
ADDRS = [("153 Great Northern Rd,", "Sault Ste. Marie, ON P6B 4Y9"),
         ("216 Second Line West,", "Sault Ste. Marie, ON P6C 2J3")]
LOGO_W, LOGO_CY = 800, 460
ADDR_Y = (752, 858)
OFFER_Y, SUB_Y = 1022, 1104
CTA_Y = 1290


@lru_cache(maxsize=1)
def logo_card():
    img = np.asarray(Image.open(TLOC.LOGO).convert("RGB")).astype(np.float32) / 255
    h0, w0 = img.shape[:2]
    cw, ch = LOGO_W, int(round(h0 * LOGO_W / w0))
    pic = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
    pad, c = 60, 22
    m = np.zeros((ch + 2 * pad, cw + 2 * pad), np.float32)
    poly = np.array([[pad + c, pad], [pad + cw, pad], [pad + cw, pad + ch - c], [pad + cw - c, pad + ch], [pad, pad + ch], [pad, pad + c]], np.float32)
    cv2.fillPoly(m, [(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    rgb = np.zeros((m.shape[0], m.shape[1], 3), np.float32)
    rgb[pad:pad + ch, pad:pad + cw] = pic
    edge = np.zeros_like(m)
    cv2.polylines(edge, [(poly * 16).astype(np.int32)], True, 1.0, 3, cv2.LINE_AA, 4)
    return dict(rgb=rgb, a=m, edge=edge, shadow=cv2.GaussianBlur(m, (0, 0), 22), w=m.shape[1], h=m.shape[0], ch=ch, pad=pad)


def chamfer_panel(w, h, c):
    m = np.zeros((h, w), np.float32)
    poly = np.array([[c, 0], [w - 1, 0], [w - 1, h - 1 - c], [w - 1 - c, h - 1], [0, h - 1], [0, c]], np.float32)
    cv2.fillPoly(m, [(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    e = np.zeros_like(m)
    cv2.polylines(e, [(poly * 16).astype(np.int32)], True, 1.0, 2, cv2.LINE_AA, 4)
    return m, e


def diamond(canvas, x, y, r, col, alpha):
    lay = np.zeros((H, W), np.float32)
    pts = np.array([[x, y - r], [x + r, y], [x, y + r], [x - r, y]], np.float32)
    cv2.polylines(lay, [(pts * 16).astype(np.int32)], True, 1.0, 2, cv2.LINE_AA, 4)
    cv2.circle(lay, (int(x * 16), int(y * 16)), int(r * 0.32 * 16), 1.0, -1, cv2.LINE_AA, 4)
    canvas += (lay + cv2.GaussianBlur(lay, (0, 0), 5) * 1.2)[..., None] * np.array(col, np.float32) * alpha


def end_card(t, fi):
    post = dict(flash=0.0, flash_col=(1.0, 0.92, 0.8), ca=0.0)
    e = t - TL.END_START
    canvas = X.carbon_wall(t, BLUE, 1560) * 0.7
    canvas = X.light_trails(canvas, t, [BLUE, (1.0, 0.22, 0.25), (0.9, 0.95, 1.0)], seed=91, n=8, y_range=(1480, 1620), speed=0.7, inten=0.7)
    for k, col in enumerate(END_LEAK):
        add_glow(canvas, 120 + 840 * k + 60 * np.sin(e + k), 380 + 900 * k, 650, col, 0.12)
    # energy rings around the logo
    L = logo_card()
    k_logo = ease_out_expo(lin(t, TL.END_LOGO, TL.END_LOGO + 0.5))
    rb = 1 + 0.5 * decay(t, TL.END_FINAL, 0.3) if t >= TL.END_FINAL else 1.0
    specs = [dict(cx=540, cy=LOGO_CY + 120, rx=560 * rb, tilt=0.1, rot=0.0, col=BLUE, inten=0.9 * k_logo, speed=2.0, n=12),
             dict(cx=540, cy=LOGO_CY + 130, rx=640 * rb, tilt=0.09, rot=0.0, col=(1.0, 0.25, 0.3), inten=0.6 * k_logo, speed=-1.6, n=10)]
    back, front = X.rings(specs, t)
    canvas += back
    if k_logo > 0:
        punch = 1 + 0.03 * decay(t, TL.END_FINAL, 0.25) if t >= TL.END_FINAL else 1.0
        sc = (1 + 0.012 * np.sin(e * 1.1)) * punch
        M = np.float32([[sc, 0, 540 - sc * L["w"] / 2], [0, sc, LOGO_CY - sc * L["h"] / 2]])
        warp = lambda a: cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        ys = (np.arange(L["h"], dtype=np.float32) - L["pad"]) / L["ch"]
        scan = lerp(-0.05, 1.05, ease_in_out(lin(t, TL.END_LOGO, TL.END_LOGO + 0.38)))
        rev = np.clip((scan - ys) / 0.03, 0, 1)[:, None]
        canvas *= 1 - np.roll(warp(L["shadow"] * rev), 18, axis=0)[..., None] * 0.7
        lay = np.clip(warp(np.dstack([L["rgb"] * (L["a"] * rev)[..., None], L["a"] * rev])), 0, None)
        blend_over(canvas, lay[..., :3], np.clip(lay[..., 3], 0, 1))
        canvas += warp(L["edge"] * rev)[..., None] * np.array([0.85, 0.92, 1.0], np.float32) * 0.9
        if 0 < scan < 1.02:
            sy = LOGO_CY - sc * L["ch"] / 2 + scan * sc * L["ch"]
            add_glow(canvas, 540, sy, 520, (0.55, 0.8, 1.0), 1.0, sx=1.0, sy=0.02)
        for s0 in (TL.END_SHINE, TL.END_FINAL):
            if s0 <= t < s0 + 0.6:
                pos = lerp(-0.35, 1.35, ease_in_out((t - s0) / 0.6))
                xx = np.arange(L["w"], dtype=np.float32)[None, :] / L["w"]
                yy = np.arange(L["h"], dtype=np.float32)[:, None] / L["h"]
                canvas += warp(np.exp(-((xx + 0.35 * yy - pos) / 0.05) ** 2) * L["a"] * 0.28)[..., None]
    canvas += front
    # addresses (HUD markers, decoding text)
    s_st = min(fit_size(a, "InterDisplay-SemiBold.otf", 640, 52) for a, _ in ADDRS)
    s_ct = min(fit_size(b, "InterDisplay-Medium.otf", 640, 40) for _, b in ADDRS)
    bw = max(max(X.text_width(a, "InterDisplay-SemiBold.otf", s_st), X.text_width(b, "InterDisplay-Medium.otf", s_ct)) for a, b in ADDRS)
    x_txt = 540 - (bw + 60) / 2 + 60
    for j, ((st, ct), y) in enumerate(zip(ADDRS, ADDR_Y)):
        ka = lin(t, TL.END_ADDR[j], TL.END_ADDR[j] + 0.4)
        if ka <= 0:
            continue
        a = ease_out_cubic(ka)
        diamond(canvas, x_txt - 36, y, 16, BLUE, a)
        X.text_at(canvas, X.decode(st, ka * 1.3, fi, seed=200 + j), "InterDisplay-SemiBold.otf", s_st, x_txt, y - s_ct * 0.55, alpha=a, shadow=0.5)
        X.text_at(canvas, X.decode(ct, ka * 1.3 - 0.15, fi, seed=210 + j), "InterDisplay-Medium.otf", s_ct, x_txt, y + s_st * 0.5, alpha=a * 0.9,
                  color=(0.78, 0.85, 0.95), shadow=0.5)
    # offer (chrome)
    ko = ease_out_expo(lin(t, TL.END_OFFER, TL.END_OFFER + 0.35))
    if ko > 0:
        so = fit_size("2 FOR $5.99", "InterDisplay-Black.otf", 760, 130)
        draw_text(canvas, "2 FOR $5.99", "InterDisplay-Black.otf", so, 540, OFFER_Y, alpha=ko, scale=lerp(1.25, 1.0, ko), blur=(1 - ko) * 8,
                  grad=X.CHROME, glow=BLUE, glow_r=22, glow_s=0.6 + 0.6 * decay(t, TL.END_OFFER, 0.2), shadow=0.6)
        draw_text(canvas, "BEFORE TAXES", "InterDisplay-Bold.otf", 36, 540, SUB_Y, alpha=ko * clamp(lin(t, TL.END_OFFER + 0.1, TL.END_OFFER + 0.35)),
                  tracking=0.34, color=(0.55, 0.78, 1.0), shadow=0.5)
    # CTA panel
    kc = ease_out_expo(lin(t, TL.END_CTA, TL.END_CTA + 0.4))
    if kc > 0:
        pw, ph = 860, 168
        m, ed = chamfer_panel(pw, ph, 28)
        x0, y0 = int(540 - pw / 2), int(CTA_Y - ph / 2)
        reveal = np.clip((kc * 1.1 - np.abs(np.arange(pw, dtype=np.float32) - pw / 2) / (pw / 2)) / 0.08, 0, 1)[None, :]
        gy = np.linspace(0, 1, ph, dtype=np.float32)[:, None, None]
        fill = (np.array([0.16, 0.18, 0.23], np.float32) * (1 - gy) + np.array([0.05, 0.06, 0.08], np.float32) * gy) * (m * reveal)[..., None]
        blend_over(canvas, fill, m * reveal * 0.92, x0, y0)
        X.blit(canvas, dict(x0=x0, y0=y0, rgb=(ed * reveal)[..., None] * np.array([0.85, 0.9, 1.0], np.float32), a=np.zeros_like(m)), additive=True)
        canvas[y0 + 34:y0 + ph - 34, x0 + 18:x0 + 24] += np.array([1.0, 0.2, 0.25], np.float32) * kc
        canvas[y0 + 34:y0 + ph - 34, x0 + pw - 24:x0 + pw - 18] += np.array([0.3, 0.6, 1.0], np.float32) * kc
        l1, l2 = "POWER UP YOUR DAY.", "VISIT US TODAY!"
        s1 = fit_size(l1, "InterDisplay-Black.otf", 720, 54)
        s2 = fit_size(l2, "InterDisplay-Black.otf", 720, 62)
        draw_text(canvas, l1, "InterDisplay-Black.otf", s1, 540, CTA_Y - 34, alpha=kc, color=(0.95, 0.97, 1.0))
        draw_text(canvas, l2, "InterDisplay-Black.otf", s2, 540, CTA_Y + 38, alpha=kc, grad=((0.75, 0.9, 1.0), (0.35, 0.62, 1.0)), glow=BLUE, glow_r=14, glow_s=0.5)
    if t >= TL.END_FINAL and t < TL.END_FINAL + 0.8:
        kw = ease_out_expo(lin(t, TL.END_FINAL, TL.END_FINAL + 0.7))
        canvas = X.shockwave(canvas, 540, LOGO_CY, 1700 * kw, 90, 30 * (1 - kw), BLUE, inten=0.9 * (1 - kw))
    zs = 1 + 0.03 * ease_in_out(lin(t, TL.END_START, TL.DURATION))
    Mz = np.float32([[zs, 0, 540 - zs * 540], [0, zs, 960 - zs * 960]])
    canvas = cv2.warpAffine(canvas, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    post["flash"] = 0.95 * (1 - ease_out_cubic(lin(t, TL.END_START, TL.END_START + 0.25)))
    post["flash"] += 0.2 * decay(t, TL.END_LOGO, 0.06) + 0.15 * decay(t, TL.END_OFFER, 0.06) * (t >= TL.END_OFFER)
    post["flash"] += 0.18 * decay(t, TL.END_FINAL, 0.08) * (t >= TL.END_FINAL)
    post["fade"] = 1 - smooth(lin(t, TL.END_FADE, TL.DURATION - 1.0 / FPS)) * 0.97
    return canvas, post


# ---------------------------------------------------------------- dispatch
def scene_index(t):
    if t < TL.DROP:
        return -1
    if t >= TL.END_START:
        return None
    return min(TL.N_DRINKS - 1, int((t - TL.DROP) / TL.SCENE))


def render_frame(fi, clean=False, t=None):
    t = fi / FPS if t is None else t
    k = scene_index(t)
    if k == -1:
        canvas, post = intro(t, fi)
    elif k is None:
        canvas, post = end_card(t, fi)
    else:
        canvas, post = drink(k, t, fi, clean)
    img = finish(canvas, fi, flash=min(1.0, post.get("flash", 0)), flash_color=post.get("flash_col", (1, 1, 1)),
                 ca=post.get("ca", 0), grain=0.012, bloom_s=0.36)
    img *= post.get("fade", 1.0)
    if not clean:
        img = X.draw_badge(img, t, TL.BEAT)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _work(fi):
    path = os.path.join(FRAMES, f"{fi:05d}.png")
    if os.path.exists(path):
        return fi
    cv2.imwrite(path, cv2.cvtColor(render_frame(fi), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi


def cover():
    """Vertical Reel cover: Sea Blue edition in its EM-ring chamber + title + store + badge."""
    i = 6
    t = TL.S[i] + 0.7
    canvas, post = drink(i, t, int(t * FPS), clean=True)
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    canvas *= 0.55 + 0.45 * np.clip((yy - 120) / 420, 0, 1)
    txt = "UNLEASH THE ENERGY."
    draw_text(canvas, txt, "InterDisplay-Black.otf", fit_size(txt, "InterDisplay-Black.otf", 920, 96), 540, 190, grad=X.CHROME,
              glow=BLUE, glow_r=22, glow_s=0.9, shadow=0.6)
    sub = "NOW AT EZY PZY CONVENIENCE"
    draw_text(canvas, sub, "InterDisplay-SemiBold.otf", fit_size(sub, "InterDisplay-SemiBold.otf", 860, 38, tracking=0.25), 540, 282,
              tracking=0.25, color=(0.75, 0.88, 1.0), shadow=0.6)
    img = finish(canvas, 0, grain=0.01, bloom_s=0.36)
    img = X.draw_badge(img, 0.3, TL.BEAT)
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "Ezy_Pzy_Red_Bull_Energy_Cover.jpg")
    cv2.imwrite(p, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
    print("wrote", p)


def stills():
    """One clean hero visual per can (no HUD, no badge), taken from its scene."""
    d = os.path.join(OUT, "product_visuals")
    os.makedirs(d, exist_ok=True)
    for i, P in enumerate(TL.PRODUCTS):
        t = TL.S[i] + 0.8
        img = render_frame(int(t * FPS), clean=True, t=t)
        p = os.path.join(d, f"{P['key']}.jpg")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.cover:
        cover()
    if args.stills:
        stills()
    if args.test:
        out = os.path.join(BUILD, "test")
        os.makedirs(out, exist_ok=True)
        for ts in args.test.split(","):
            t = float(ts)
            t0 = time.time()
            img = render_frame(int(round(t * FPS)), t=t)
            cv2.imwrite(os.path.join(out, f"t{t:06.2f}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"t={t:.2f} {time.time() - t0:.2f}s", flush=True)
    if args.all:
        os.makedirs(FRAMES, exist_ok=True)
        n = int(round(TL.DURATION * FPS))
        t0 = time.time()
        with Pool(args.jobs) as pool:
            for k, _ in enumerate(pool.imap(_work, range(n), chunksize=4)):
                if (k + 1) % 60 == 0:
                    print(f"{k + 1}/{n} frames  {time.time() - t0:.0f}s", flush=True)
        print("render done", time.time() - t0)


if __name__ == "__main__":
    main()
