"""Reel 4 renderer - Red Bull Editions x13, "2 FOR $5.99 BEFORE TAXES" offer, for Ezy Pzy Convenience.

    python reel4_render.py --test 1.5,4.0     # preview stills -> output/build4/test/
    python reel4_render.py --all              # render every frame (60 fps) -> output/build4/frames/

Same 2.5D plate approach as Reels 2/3: each product photo is used untouched, a matte
lets effects pass BEHIND the can, and exactly one product is visible in any frame.
The price banner (bottom centre) is composited on every frame until the end card and
slides out before the store logo appears; every shot keeps the can clear of it.

   time    drink                         look                                   transition out
   0.0-3.0  intro (Summer Edition)       dark stage, light sweeps, headline      bass drop
   3.0      Summer - Sudachi Lime        sun rays, lime/yellow explosion         lime->ice liquid wave
   4.9      Ice - Iced Vanilla Berry     ice cubes land on the beat, cold mist   whip left
   6.8      Yellow - Tropical            golden sunburst, gold splash, bubbles   sun-flare burst
   8.6      Spring - Cherry Sakura       falling sakura petals, cherry splash    petal storm
  10.5      Peach - White Peach          zoom-in reveal, bubbles, soft glow      whip right
  12.4      Red - Watermelon             red/green splash                        red->blue liquid wave
  14.3      Sea Blue - Juneberry         underwater bubbles, caustic rays        hyperspace warp
  16.1      Amber - Strawberry Apricot   rising embers, orange/red splash        whip left
  18.0      Pink - Wild Berries          berry confetti, pink splash             magenta smoke wipe
  19.9      Lilac - Grapefruit & Blossom purple smoke, falling blossoms          zoom-through flash
  21.8      Winter - Pistachio & Berries snowfall, green/magenta splash          green->ice liquid wave
  23.6      Coconut - Coconut Berry      blue lightning on the beat, sparks      whip left
  25.5      Apple - Fuji Apple & Ginger  flames, embers, red/gold splash         flash + whip up
  27.4-31.9 end card: logo, both locations, 2 FOR $5.99 BEFORE TAXES, call to action
"""
import os
import json
import time
import argparse
from types import SimpleNamespace
from multiprocessing import Pool

import numpy as np
import cv2

import timeline4 as TL
from timeline4 import W, H, FPS
import fx
from fx import (lin, smooth, keys, decay, ease_out_cubic, ease_in_cubic, ease_in_out, ease_out_expo, ease_in_expo,
                lerp, Cam, add_glow, draw_text, fit_size, hblur, vblur, zoom_blur, finish, draw_ice, fractal_noise,
                scroll, blend_over)
import render as R
import reel2_render as R2
import reel3_render as R3

# Reuse Reel 3's price banner, whips, keep-clear camera guard and end card with Reel 4's timing.
R3.TL = TL

ROOT = fx.ROOT
A4 = os.path.join(ROOT, "assets", "reel4")
GEO4 = json.load(open(os.path.join(A4, "geometry.json")))
BUILD = os.path.join(ROOT, "output", "build4")
FRAMES = os.path.join(BUILD, "frames")

_plates = {}


def plate(i):
    """Load product plate i; keep only neighbouring plates in memory (13 plates x 2 resolutions is a lot)."""
    if i not in _plates:
        for k in [k for k in _plates if abs(k - i) > 1]:
            del _plates[k]
        _plates[i] = R2.Plate(TL.PRODUCTS[i]["key"], base=A4, geo=GEO4, backdrop_gain=0.8, feather=260)
    return _plates[i]


def origin(p, f=0.42):
    return (p.cx, p.top + (p.bot - p.top) * f)


# ------------------------------------------------------------- per-drink design table
def D(a, ad, b, bd, glow, rim_l, rim_r, move, enter, exit, **kw):
    return dict(a=a, ad=ad, b=b, bd=bd, glow=glow, rim_l=rim_l, rim_r=rim_r, move=move, enter=enter, exit=exit, **kw)


LIME, LIME_D = (0.62, 1.0, 0.12), (0.15, 0.45, 0.0)
YEL, YEL_D = (1.0, 0.92, 0.18), (0.55, 0.42, 0.0)
ICE, ICE_D = (0.45, 0.85, 1.0), (0.05, 0.3, 0.55)
ICEW, ICEW_D = (0.85, 0.95, 1.0), (0.35, 0.5, 0.7)
GOLD, GOLD_D = (1.0, 0.76, 0.12), (0.6, 0.33, 0.0)
CHERRY, CHERRY_D = (0.85, 0.05, 0.15), (0.35, 0.0, 0.05)
SAKURA, SAKURA_D = (1.0, 0.62, 0.76), (0.6, 0.2, 0.35)
PEACH, PEACH_D = (1.0, 0.64, 0.56), (0.6, 0.25, 0.2)
ROSE, ROSE_D = (1.0, 0.38, 0.6), (0.55, 0.08, 0.25)
WRED, WRED_D = (0.95, 0.1, 0.15), (0.4, 0.0, 0.03)
WGRN, WGRN_D = (0.2, 0.85, 0.3), (0.02, 0.35, 0.08)
SBLUE, SBLUE_D = (0.15, 0.4, 1.0), (0.02, 0.1, 0.45)
VIOL, VIOL_D = (0.6, 0.35, 1.0), (0.22, 0.08, 0.5)
ORNG, ORNG_D = (1.0, 0.5, 0.08), (0.55, 0.18, 0.0)
SRED, SRED_D = (0.95, 0.15, 0.2), (0.45, 0.02, 0.06)
PINK, PINK_D = (1.0, 0.25, 0.65), (0.5, 0.03, 0.28)
BERRY, BERRY_D = (0.6, 0.1, 0.35), (0.25, 0.0, 0.12)
LILAC, LILAC_D = (0.65, 0.4, 1.0), (0.25, 0.1, 0.5)
GFRT, GFRT_D = (1.0, 0.4, 0.45), (0.5, 0.08, 0.12)
PIST, PIST_D = (0.7, 0.95, 0.25), (0.25, 0.42, 0.02)
MAG, MAG_D = (0.85, 0.1, 0.4), (0.35, 0.0, 0.14)
COCO, COCO_D = (0.8, 0.92, 1.0), (0.3, 0.42, 0.6)
CBLUE, CBLUE_D = (0.2, 0.55, 1.0), (0.03, 0.18, 0.5)
ARED, ARED_D = (0.95, 0.15, 0.1), (0.42, 0.02, 0.0)
AGOLD, AGOLD_D = (1.0, 0.7, 0.15), (0.55, 0.28, 0.0)

DRINKS = [
    D(LIME, LIME_D, YEL, YEL_D, (0.7, 1.0, 0.2), (0.5, 0.9, 0.2), (1.0, 0.9, 0.3), "push", "drop", "wave",
      wave_pal=[LIME, YEL, ICE, ICEW]),
    D(ICE, ICE_D, ICEW, ICEW_D, (0.4, 0.8, 1.0), (0.5, 0.85, 1.0), (0.9, 0.95, 1.0), "pull", "wave", "hwhip_l"),
    D(GOLD, GOLD_D, YEL, YEL_D, (1.0, 0.8, 0.2), (1.0, 0.8, 0.3), (1.0, 0.9, 0.5), "drift_r", "hwhip_r", "flare"),
    D(CHERRY, CHERRY_D, SAKURA, SAKURA_D, (1.0, 0.45, 0.6), (1.0, 0.6, 0.75), (0.95, 0.25, 0.35), "rise", "zoom_out", "petals"),
    D(PEACH, PEACH_D, ROSE, ROSE_D, (1.0, 0.5, 0.6), (1.0, 0.7, 0.65), (1.0, 0.45, 0.6), "push", "zoom_in", "hwhip_r"),
    D(WRED, WRED_D, WGRN, WGRN_D, (1.0, 0.25, 0.25), (0.3, 0.9, 0.4), (1.0, 0.3, 0.3), "drift_l", "hwhip_l", "wave",
      wave_pal=[WRED, WGRN, SBLUE, VIOL]),
    D(SBLUE, SBLUE_D, VIOL, VIOL_D, (0.35, 0.5, 1.0), (0.6, 0.4, 1.0), (0.3, 0.7, 1.0), "rise", "wave", "warp",
      warp_col=(0.7, 0.75, 1.0)),
    D(ORNG, ORNG_D, SRED, SRED_D, (1.0, 0.55, 0.15), (1.0, 0.6, 0.2), (1.0, 0.3, 0.3), "pull", "warp", "hwhip_l"),
    D(PINK, PINK_D, BERRY, BERRY_D, (1.0, 0.3, 0.7), (1.0, 0.45, 0.75), (0.75, 0.25, 0.55), "drift_r", "hwhip_r", "smoke",
      smoke_col=(0.42, 0.04, 0.24)),
    D(LILAC, LILAC_D, GFRT, GFRT_D, (0.7, 0.45, 1.0), (0.75, 0.5, 1.0), (1.0, 0.5, 0.55), "push", "smoke_clear", "zoom_through"),
    D(PIST, PIST_D, MAG, MAG_D, (0.75, 1.0, 0.3), (0.75, 1.0, 0.35), (0.95, 0.25, 0.5), "rise", "zoom_out", "wave",
      wave_pal=[PIST, MAG, COCO, CBLUE]),
    D(COCO, COCO_D, CBLUE, CBLUE_D, (0.45, 0.8, 1.0), (0.4, 0.75, 1.0), (0.85, 0.95, 1.0), "pull", "wave", "hwhip_l"),
    D(ARED, ARED_D, AGOLD, AGOLD_D, (1.0, 0.45, 0.15), (1.0, 0.35, 0.1), (1.0, 0.75, 0.25), "drift_l", "hwhip_r", "end"),
]
for _i, _d in enumerate(DRINKS):
    _d["name"] = TL.PRODUCTS[_i]["name"]
    _d["sub"] = "RED BULL  •  " + TL.PRODUCTS[_i]["edition"]

ICE_LAND = []
_r = np.random.default_rng(404)
for _k, (_x, _th) in enumerate(zip([110, 1010, 190, 950], TL.ICE_HITS)):
    ICE_LAND.append(dict(x=_x, th=_th, fall=_r.uniform(0.32, 0.42), size=_r.uniform(115, 150),
                         rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3.5, 3.5, 3)))
ICE_BACK = [dict(x=_r.uniform(840, 1100) if _k % 2 else _r.uniform(20, 300), t0=TL.DRINK_STARTS[1] + _r.uniform(-0.2, 1.5),
                 size=_r.uniform(60, 90), rot0=_r.uniform(0, 6.28, 3), om=_r.uniform(-3, 3, 3), vy=_r.uniform(600, 900))
            for _k in range(7)]

SAKURA_COLS = [(1.0, 0.72, 0.82), (1.0, 0.82, 0.9), (0.98, 0.6, 0.75), (1.0, 0.92, 0.95)]
BLOSSOM_COLS = [(1.0, 0.97, 0.98), (0.98, 0.9, 0.98), (1.0, 0.94, 0.85)]
BERRY_COLS = [(1.0, 0.3, 0.7), (0.65, 0.12, 0.4), (1.0, 0.75, 0.9), (0.35, 0.2, 0.7), (0.9, 0.1, 0.3)]
SNOW_COLS = [(1.0, 1.0, 1.0), (0.92, 0.97, 1.0)]


# ------------------------------------------------------------- particles: petals / confetti / snow
def flakes(canvas, u, seed, n, cols, kind="petal", smin=10, smax=22, fall=(150, 320), sway=50, blur=0.0, alpha=0.95):
    if n <= 0 or alpha <= 0.01:
        return
    r = np.random.default_rng(seed)
    lay = np.zeros((H, W, 3), np.float32)
    A = np.zeros((H, W), np.float32)
    for j in range(n):
        x0, y0 = r.uniform(-60, W + 60), r.uniform(-200, H)
        vy, ph, spin = r.uniform(*fall), r.uniform(0, 6.28), r.uniform(1.5, 5)
        sw, fl = r.uniform(0.8, 2.0), r.uniform(2, 5)
        sz = r.uniform(smin, smax)
        col = tuple(float(v) for v in cols[j % len(cols)])
        x = x0 + sway * np.sin(u * sw + ph)
        y = (y0 + u * vy) % (H + 300) - 150
        c = (int(x * 16), int(y * 16))
        flip = 0.35 + 0.65 * abs(np.cos(u * fl + ph))
        if kind == "snow":
            rr = max(1, int(sz * 16))
            cv2.circle(lay, c, rr, col, -1, cv2.LINE_AA, 4)
            cv2.circle(A, c, rr, 1.0, -1, cv2.LINE_AA, 4)
        elif kind == "confetti":
            ang = u * spin + ph
            L, Wd = sz, sz * 0.45 * flip
            pts = np.array([[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]], np.float32)
            Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
            P = ((pts @ Rm.T + [x, y]) * 16).astype(np.int32)
            cv2.fillPoly(lay, [P], col, cv2.LINE_AA, 4)
            cv2.fillPoly(A, [P], 1.0, cv2.LINE_AA, 4)
        else:   # petal
            ang = float(np.degrees(u * spin + ph))
            ax = (max(1, int(sz * 16)), max(1, int(sz * 0.55 * flip * 16)))
            cv2.ellipse(lay, c, ax, ang, 0, 360, col, -1, cv2.LINE_AA, 4)
            cv2.ellipse(A, c, ax, ang, 0, 360, 1.0, -1, cv2.LINE_AA, 4)
            light = tuple(min(1.0, v * 0.5 + 0.5) for v in col)
            cv2.ellipse(lay, c, (max(1, ax[0] // 2), max(1, ax[1] // 2)), ang, 0, 360, light, -1, cv2.LINE_AA, 4)
    if blur > 0.3:
        lay = cv2.GaussianBlur(lay, (0, 0), blur)
        A = cv2.GaussianBlur(A, (0, 0), blur)
    blend_over(canvas, lay * alpha, A * alpha)


# ------------------------------------------------------------- camera
def scene_cam(i, t):
    p = plate(i)
    s = TL.DRINK_STARTS[i]
    u = t - s
    mv = DRINKS[i]["move"]
    if u < TL.CLOSE:
        x = ease_in_out(u / TL.CLOSE)
        if mv == "push":
            return R3.wide_cam(p, 1 + 0.07 * x, dx=8 * np.sin(u * 1.9 + i))
        if mv == "pull":
            return R3.wide_cam(p, 1.1 - 0.08 * x)
        if mv in ("drift_l", "drift_r"):
            sg = -1 if mv == "drift_l" else 1
            return R3.wide_cam(p, 1.02 + 0.03 * x, dx=sg * lerp(-32, 32, x))
        if mv == "rise":
            return R3.wide_cam(p, 1.0 + 0.04 * x, ay=R3.CAN_CY + lerp(15, -45, x))
    if u < TL.MEDIUM:   # hero close-up: slow push-in anchored at the can's base, drifting sideways
        x = lin(u, TL.CLOSE, TL.MEDIUM)
        zc = R3.CLOSE_H / (p.bot - p.top) * (1 + 0.05 * x + 0.025 * decay(u, TL.CLOSE, 0.1))
        drift = (1 if i % 2 == 0 else -1) * lerp(-16, 16, ease_in_out(x))
        return Cam(z=zc, fx=p.cx + drift, fy=p.bot, ax=540, ay=R3.CAN_BOTTOM_LIMIT - 4)
    zm = keys(u, [(TL.MEDIUM, 1.22), (TL.MEDIUM + 0.2, 1.04), (TL.SCENE, 1.02)], ease_out_expo)
    return R3.wide_cam(p, zm, dx=-6 * np.sin(u * 2 + i))


def tag(canvas, t, t_in, t_out, name, sub, glow):
    if t < t_in or t > t_out + 0.25:
        return
    a_in = ease_out_cubic(lin(t, t_in, t_in + 0.25))
    a_out = 1 - ease_in_cubic(lin(t, t_out, t_out + 0.2))
    a = a_in * a_out
    size = fit_size(name, "InterDisplay-BlackItalic.otf", 900, 92)
    yoff = (1 - a_in) * 26 - (1 - a_out) * 34
    draw_text(canvas, name, "InterDisplay-BlackItalic.otf", size, 540 - (1 - a_in) * 50, 222 + yoff, alpha=a,
              reveal=ease_out_cubic(lin(t, t_in, t_in + 0.3)), glow=glow, glow_r=22, glow_s=0.9, shadow=0.55,
              blur=(1 - a_out) * 8)
    ss = fit_size(sub, "InterDisplay-SemiBold.otf", 860, 30, tracking=0.2)
    draw_text(canvas, sub, "InterDisplay-SemiBold.otf", ss, 540, 222 + size * 0.6 + 20 + yoff,
              alpha=a * ease_out_cubic(lin(t, t_in + 0.08, t_in + 0.33)), tracking=0.2, shadow=0.5,
              glow=glow, glow_r=10, glow_s=0.35)


# ------------------------------------------------------------- per-drink effects
def strike_flash(t):
    last = max([st for st in TL.STRIKES if st <= t], default=None)
    if last is None:
        return 0.0
    return float(np.exp(-(t - last) / 0.07) * (1.0 if last == TL.STRIKES[0] else 0.6))


def bg(i, c, cv):
    cam, p, t, u, d, o = c.cam, c.p, c.t, c.u, c.d, c.o
    if i == 0:      # Summer - Sudachi Lime
        R.god_rays(cv, t, *cam.p(p.cx - 300, p.top - 80), (1.0, 0.95, 0.5), 0.26, n=13, speed=0.04, sharp=8)
        R.bokeh(cv, t, 3, 20, [LIME, YEL, (0.9, 1.0, 0.6)], cam, inten=0.10)
        R3.splash_two_tone(cv, cam, u, o, LIME, YEL, LIME_D, YEL_D, core=True)
    elif i == 1:    # Ice - Iced Vanilla Berry
        R.god_rays(cv, t, *cam.p(p.cx - 250, p.top - 60), (0.7, 0.9, 1.0), 0.16, n=10, speed=0.03, sharp=6)
        R.bokeh(cv, t, 14, 16, [(0.5, 0.8, 1.0), (0.85, 0.95, 1.0), (0.85, 0.25, 0.45)], cam, inten=0.10)
        R.mist_layer(cv, t, np.full((H, W), 0.2, np.float32), color=(0.85, 0.95, 1.0), seed=51, sx=25, sy=-8)
        for ob in ICE_BACK:
            st = R3.ice_state(ob, t, p)
            if st:
                x, y = cam.p(st[0], st[1])
                draw_ice(cv, x, y, cam.s(ob["size"]), *st[2], blur=2.5, alpha=0.85)
    elif i == 2:    # Yellow - Tropical
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (1.0, 0.85, 0.35), 0.3, n=18, speed=0.06, sharp=10)
        add_glow(cv, *cam.p(p.cx, p.top + 400), cam.s(800), (1.0, 0.75, 0.2), 0.25)
        R3.splash_two_tone(cv, cam, max(0.0, u - 0.05), o, GOLD, YEL, GOLD_D, YEL_D, core=False)
        R3.bubbles(cv, cam, u, p, 31, 26, tint=(1.0, 0.9, 0.6), smin=10, smax=26, spd=(240, 460))
    elif i == 3:    # Spring - Cherry Sakura
        R.god_rays(cv, t, *cam.p(p.cx - 300, p.top - 60), (1.0, 0.85, 0.9), 0.2, n=12, speed=0.04)
        R.bokeh(cv, t, 41, 18, [(1.0, 0.7, 0.8), (1.0, 0.9, 0.95), (0.9, 0.2, 0.3)], cam, inten=0.10)
        R3.splash_two_tone(cv, cam, max(0.0, u - 0.05), o, CHERRY, SAKURA, CHERRY_D, SAKURA_D, core=False)
        flakes(cv, u, 431, 40, SAKURA_COLS, "petal", smin=9, smax=18, fall=(120, 260), blur=1.2, alpha=0.9)
    elif i == 4:    # Peach - White Peach
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 250), (1.0, 0.8, 0.8), 0.16, n=10, speed=-0.03)
        R.bokeh(cv, t, 51, 20, [(1.0, 0.7, 0.65), (1.0, 0.55, 0.7), (1.0, 0.9, 0.85)], cam, inten=0.12)
        add_glow(cv, *cam.p(p.cx, p.bot - 100), cam.s(700), (1.0, 0.6, 0.65), 0.22, sy=0.5)
        R3.bubbles(cv, cam, u, p, 52, 34, tint=(1.0, 0.88, 0.88), smin=10, smax=30, spd=(200, 420))
    elif i == 5:    # Red - Watermelon
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (1.0, 0.4, 0.4), 0.16, n=12, speed=0.05)
        add_glow(cv, *cam.p(p.cx + 450, p.top + 200), cam.s(600), (0.2, 0.9, 0.3), 0.18)
        R3.splash_two_tone(cv, cam, max(0.0, u - 0.05), o, WRED, WGRN, WRED_D, WGRN_D, core=False)
    elif i == 6:    # Sea Blue - Juneberry (underwater)
        R.god_rays(cv, t, *cam.p(p.cx, -400), (0.5, 0.7, 1.0), 0.22, n=16, speed=0.08, sharp=6)
        R.bokeh(cv, t, 62, 16, [(0.3, 0.5, 1.0), (0.6, 0.4, 1.0), (0.6, 0.9, 1.0)], cam, inten=0.12)
        R.mist_layer(cv, t, np.full((H, W), 0.12, np.float32), color=(0.5, 0.6, 1.0), seed=71, sx=20, sy=-10)
        R3.bubbles(cv, cam, u, p, 61, 44, tint=(0.8, 0.9, 1.0), smin=8, smax=28, spd=(260, 560))
    elif i == 7:    # Amber - Strawberry Apricot
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (1.0, 0.6, 0.25), 0.2, n=14, speed=0.05)
        R3.splash_two_tone(cv, cam, max(0.0, u - 0.05), o, ORNG, SRED, ORNG_D, SRED_D, core=False)
        R2.embers(cv, u, 71, 40, cam, (p.cx, p.top + 600), color=(1.0, 0.45, 0.1), mode="rise", inten=0.9)
    elif i == 8:    # Pink - Wild Berries
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (1.0, 0.5, 0.8), 0.16, n=12, speed=-0.04)
        R.bokeh(cv, t, 81, 20, [(1.0, 0.3, 0.7), (0.75, 0.2, 0.5), (1.0, 0.75, 0.9)], cam, inten=0.12)
        R3.splash_two_tone(cv, cam, max(0.0, u - 0.05), o, PINK, BERRY, PINK_D, BERRY_D, core=False)
        flakes(cv, u, 881, 34, BERRY_COLS, "confetti", smin=10, smax=20, fall=(200, 380), blur=1.4, alpha=0.9)
    elif i == 9:    # Lilac - Grapefruit & Blossom
        R2.smoke(cv, t, 0.42, (0.35, 0.18, 0.55), seed=91, sx=30, sy=-40)
        add_glow(cv, *cam.p(p.cx, p.top + 450), cam.s(800), (0.7, 0.45, 1.0), 0.28)
        R.bokeh(cv, t, 91, 16, [(0.75, 0.5, 1.0), (1.0, 0.5, 0.55), (1.0, 0.95, 1.0)], cam, inten=0.10)
        flakes(cv, u, 991, 34, BLOSSOM_COLS, "petal", smin=9, smax=17, fall=(110, 230), blur=1.0, alpha=0.9)
    elif i == 10:   # Winter - Pistachio & Berries
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 250), (0.85, 1.0, 0.75), 0.12, n=10, speed=0.03)
        R3.splash_two_tone(cv, cam, max(0.0, u - 0.05), o, PIST, MAG, PIST_D, MAG_D, core=False)
        flakes(cv, u, 1101, 130, SNOW_COLS, "snow", smin=2, smax=5, fall=(90, 200), sway=30, blur=0.8, alpha=0.9)
    elif i == 11:   # Coconut - Coconut Berry (electric ice)
        fl = c.fl
        cv *= 0.8 + 0.6 * fl
        R.god_rays(cv, t, *cam.p(p.cx, p.top + 300), (0.6, 0.85, 1.0), 0.12 + 0.3 * fl, n=16, speed=0.12, sharp=14)
        R.mist_layer(cv, t, np.full((H, W), 0.16, np.float32), color=(0.8, 0.92, 1.0), seed=72, sx=30, sy=-12)
        R2.bolts(cv, cam, t, TL.STRIKES, p, color=(0.5, 0.85, 1.0), big=(TL.STRIKES[0],))
    elif i == 12:   # Apple - Fuji Apple & Ginger
        cv *= 0.62
        R2.fire(cv, t, 1.3, cam.p(0, p.bot - 280)[1], cam.s(1350), seed=91)
        R3.splash_two_tone(cv, cam, max(0.0, u - 0.05), o, ARED, AGOLD, ARED_D, AGOLD_D, core=False)
        R2.embers(cv, u, 131, 30, cam, (p.cx, p.top + 600), color=(1.0, 0.4, 0.05), mode="rise", inten=0.8)


def fg(i, c, canvas):
    cam, p, t, u, o = c.cam, c.p, c.t, c.u, c.o
    blur = 6.0 if u < TL.CLOSE else 10.0
    if i in (0, 2, 5, 7, 12):
        col, dark = {0: (LIME, LIME_D), 2: (GOLD, GOLD_D), 5: (WRED, WRED_D), 7: (ORNG, ORNG_D), 12: (ARED, ARED_D)}[i]
        R3.fg_drops(canvas, cam, u if i == 0 else max(0.0, u - 0.05), o, col, dark, blur=blur)
    if i == 1:
        xx, yy = fx.grid()
        fy_ = cam.p(0, p.bot)[1]
        fog = np.clip(1 - np.abs(yy - fy_) / cam.s(240), 0, 1) ** 1.2 * 0.42
        R.mist_layer(canvas, t, fog, color=(0.88, 0.96, 1.0), seed=61, sx=40, sy=-70)
        for ob in ICE_LAND:
            st = R3.ice_state(ob, t, p)
            if st:
                x, y = cam.p(st[0], st[1])
                draw_ice(canvas, x, y, cam.s(ob["size"]), *st[2], blur=0.8)
        R3.frost(canvas, t, u)
    elif i == 2:
        R.dust(canvas, t, 32, 40, (1.0, 0.9, 0.5), inten=0.5, vy=-50)
    elif i == 3:
        flakes(canvas, u, 432, 7, SAKURA_COLS, "petal", smin=26, smax=44, fall=(160, 300), blur=5.0, alpha=0.85)
    elif i == 4:
        R3.bubbles(canvas, cam, u, p, 53, 10, tint=(1.0, 0.92, 0.9), smin=22, smax=50, spd=(300, 520), blur=0.5, alpha=0.95)
    elif i == 6:
        R3.bubbles(canvas, cam, u, p, 63, 12, tint=(0.85, 0.92, 1.0), smin=24, smax=56, spd=(320, 600), blur=0.5, alpha=0.95)
        R3.bubbles(canvas, cam, u, p, 64, 5, tint=(0.85, 0.92, 1.0), smin=100, smax=160, spd=(400, 700), avoid_can=False, blur=7, alpha=0.7)
    elif i == 7:
        R.dust(canvas, t, 72, 34, (1.0, 0.65, 0.3), inten=0.55, vy=-70)
    elif i == 8:
        flakes(canvas, u, 882, 8, BERRY_COLS, "confetti", smin=22, smax=34, fall=(260, 420), blur=4.0, alpha=0.9)
    elif i == 9:
        flakes(canvas, u, 992, 6, BLOSSOM_COLS, "petal", smin=26, smax=42, fall=(140, 260), blur=5.0, alpha=0.85)
    elif i == 10:
        flakes(canvas, u, 1102, 22, SNOW_COLS, "snow", smin=6, smax=12, fall=(160, 300), sway=40, blur=3.0, alpha=0.85)
        R3.frost(canvas, t, u, seed=79, n=60, col=(0.92, 1.0, 0.95))
    elif i == 11:
        R2.sparks(canvas, cam, t, TL.STRIKES, (p.cx, p.bot - 60), color=(0.75, 0.92, 1.0), n=34, seed=1200)
        R3.frost(canvas, t, u, seed=81, n=70, col=(0.85, 0.95, 1.0))
    elif i == 12:
        R2.embers(canvas, u, 132, 20, cam, (p.cx, p.top + 600), color=(1.0, 0.5, 0.1), mode="rise", inten=1.0)


# ------------------------------------------------------------- intro (first drink, before the drop)
def intro(t, fi):
    d = DRINKS[0]
    p = plate(0)
    post = dict(flash=0.0, ca=0.0)
    zm = keys(t, [(0, 0.77), (2.95, 0.835)])
    cam = R3.wide_cam(p, zm, ay=1610 - 0.5 * R3.CAN_H * zm)
    R3.shaken(cam, t, sum(7 * decay(t, w, 0.08) for w, _ in TL.INTRO_WORDS))
    o = origin(p)

    def behind(cv):
        g = ease_in_expo(lin(t, 2.2, 3.0))
        if g > 0:
            sx, sy = cam.p(*o)
            add_glow(cv, sx, sy, 200 + 900 * g, LIME, 0.85 * g)
            add_glow(cv, sx, sy, 80 + 300 * g, (1.0, 1.0, 0.92), 0.8 * g)
        R.dust(cv, t, 5, 70, (1.0, 1.0, 0.85), inten=0.5)
    rev = lin(t, 0.15, 2.9)
    sweep = [(keys(t, [(0.4, -1.9), (1.8, 1.9)]), 0.42, 1.2 * smooth(lin(t, 0.35, 0.6)) * (1 - smooth(lin(t, 1.65, 1.9)))),
             (keys(t, [(1.8, 1.9), (2.8, -1.9)]), 0.5, 1.1 * smooth(lin(t, 1.75, 1.95)))]
    rs = 0.15 + 1.0 * smooth(rev)
    canvas = R3.render_plate(p, cam, behind, amb_bg=0.05 + 0.18 * ease_in_cubic(rev),
                             amb_can=0.03 + 0.3 * ease_in_cubic(rev) + 0.5 * ease_in_expo(lin(t, 2.6, 3.0)),
                             amb_props=0.04 + 0.2 * ease_in_cubic(rev), sweep=sweep,
                             rim_l=tuple(np.array(YEL) * rs * 0.8), rim_r=tuple(np.array(LIME) * rs), protect=False)
    lines = [("ICE COLD.", 720, 290), ("UNBEATABLE", 960, 412), ("DEALS.", 620, 534)]
    fonts = ["InterDisplay-Black.otf", "InterDisplay-BlackItalic.otf", "InterDisplay-Black.otf"]
    grads = [((0.88, 1.0, 1.0), (0.5, 0.88, 1.0)), ((1, 1, 1), (1.0, 0.9, 0.35)), None]
    glows = [(0.4, 0.8, 1.0), (1.0, 0.85, 0.2), (0.7, 1.0, 0.2)]
    explode = ease_in_cubic(lin(t, 2.8, 3.0))
    for k, ((t0, _), (txt, mw, y)) in enumerate(zip(TL.INTRO_WORDS, lines)):
        if t < t0:
            continue
        kk = ease_out_expo(lin(t, t0, t0 + 0.22))
        size = fit_size(txt, fonts[k], mw, 190)
        draw_text(canvas, txt, fonts[k], size, 540, y, alpha=kk * (1 - explode), scale=lerp(1.45, 1.0, kk) * (1 + 0.5 * explode),
                  blur=(1 - kk) * 14 + explode * 12, glow=glows[k], glow_r=26, glow_s=0.7 + 0.6 * decay(t, t0, 0.15),
                  grad=grads[k], shadow=0.4)
    post["flash"] = 0.4 * ease_in_expo(lin(t, 2.75, 3.0)) + sum(0.1 * decay(t, w, 0.05) for w, _ in TL.INTRO_WORDS)
    post["ca"] = sum(6 * decay(t, w, 0.1) for w, _ in TL.INTRO_WORDS)
    return canvas, post


# ------------------------------------------------------------- one drink scene
def smoke_wipe(canvas, u, color):
    wipe = ease_in_cubic(lin(u, TL.SCENE - 0.42, TL.SCENE - 0.02))
    if wipe <= 0:
        return canvas
    n1 = scroll(fractal_noise(87, 540, 960, 6), u * 50, -u * 170)
    n = cv2.resize(np.clip((n1 - 0.25) * 2.2, 0, 1), (W, H))
    a = np.clip(n * wipe * 3.2 + wipe ** 1.5 * 1.3, 0, 1)[..., None]
    return canvas * (1 - a) + a * np.array(color, np.float32)


def scene(i, t, fi):
    if i == 0 and t < TL.DROP:
        return intro(t, fi)
    d = DRINKS[i]
    p = plate(i)
    s = TL.DRINK_STARTS[i]
    u = t - s
    post = dict(flash=0.0, ca=0.0, flash_color=(1.0, 1.0, 1.0))
    cam = scene_cam(i, t)
    ent, ex = d["enter"], d["exit"]
    prev = DRINKS[i - 1] if i > 0 else None
    sx = sy = hb = vb = zb = 0.0
    amp = 10 * decay(t, s, 0.12) + 8 * decay(t, s + TL.CLOSE, 0.1)
    # ---- entrance
    if ent == "drop":
        if u < TL.CLOSE:
            cam.z *= 1 + 0.07 * decay(t, s, 0.16)
        post["flash"] += 0.9 * decay(t, s, 0.06)
        post["ca"] += 14 * decay(t, s, 0.18)
        amp += 16 * decay(t, s, 0.13)
    elif ent in ("hwhip_r", "hwhip_l"):
        dx, hb = R3.hwhip_in(u, +1 if ent == "hwhip_r" else -1)
        sx += dx
        post["ca"] += 8 * (1 - lin(u, 0, 0.3))
    elif ent == "zoom_out":
        x = ease_out_expo(lin(u, 0, 0.35))
        cam.z *= lerp(1.6, 1.0, x)
        zb = 0.25 * (1 - x)
        post["flash"] += (1 - ease_out_cubic(lin(u, 0, 0.25))) * 0.95
    elif ent == "zoom_in":
        cam.z *= lerp(0.8, 1.0, ease_out_expo(lin(u, 0, 0.5)))
        post["flash"] += (1 - ease_out_cubic(lin(u, 0, 0.25))) * 0.95
        post["flash_color"] = (1.0, 0.92, 0.95)
    elif ent == "warp":
        post["flash"] += (1 - ease_out_cubic(lin(u, 0, 0.24))) * 0.95
        post["ca"] += 12 * (1 - ease_out_cubic(lin(u, 0, 0.45)))
        post["flash_color"] = (0.9, 0.9, 1.0)
    # ---- exit
    if ex in ("hwhip_l", "hwhip_r"):
        dx, hb2 = R3.hwhip_out(u, -1 if ex == "hwhip_l" else +1)
        sx += dx
        hb = max(hb, hb2)
        post["ca"] += 8 * lin(u, TL.EXIT, TL.SCENE)
    elif ex == "zoom_through":
        x = ease_in_cubic(lin(u, TL.EXIT, TL.SCENE))
        cam.z *= float(np.exp(np.log(3.0) * x))
        zb = max(zb, 0.26 * x)
        post["flash"] += ease_in_expo(lin(u, TL.EXIT + 0.06, TL.SCENE))
        post["ca"] += 12 * x
    elif ex == "end":
        dy, vb = R3.vwhip_out(u, -1)
        sy += dy
        post["flash"] += 0.85 * ease_in_expo(lin(u, TL.SCENE - 0.15, TL.SCENE))
    fl = strike_flash(t) if i == 11 else 0.0
    if i == 11:
        amp += sum((16 if st == TL.STRIKES[0] else 9) * decay(t, st, 0.1) for st in TL.STRIKES if t >= st)
    cam.ax += sx
    cam.ay += sy
    R3.shaken(cam, t, amp, seed=float(i))
    c = SimpleNamespace(cam=cam, p=p, t=t, u=u, s=s, d=d, o=origin(p), fl=fl)
    canvas = R3.render_plate(p, cam, lambda cv: bg(i, c, cv),
                             amb_bg=0.85 if i == 11 else 1.0,
                             spec=lerp(-0.6, 0.6, lin(t, s, s + TL.SCENE)) * (1 if i % 2 == 0 else -1), spec_s=0.16,
                             rim_l=tuple(np.array(d["rim_l"]) * (0.6 + 1.2 * fl)) if i == 11 else d["rim_l"],
                             rim_r=d["rim_r"], hero_gain=1.0 + 0.1 * fl,
                             backdrop_shift=(sx * 0.5, sy * 0.5))
    fg(i, c, canvas)
    for t0 in TL.CAN_OPENS:
        if s <= t0 < s + TL.SCENE:
            R2.can_open_fx(canvas, cam, p, t, t0, (1.0, 0.97, 0.9))
    # ---- transition overlays (exit)
    if ex == "wave":
        R3.liquid_wave(canvas, t, s + TL.SCENE - 0.4, s + TL.SCENE, s + TL.SCENE + 0.45, d["wave_pal"])
    elif ex == "flare":
        k = ease_in_cubic(lin(u, TL.SCENE - 0.35, TL.SCENE))
        if k > 0:
            fx_, fy_ = cam.p(p.cx - 280, p.top - 40)
            add_glow(canvas, fx_, fy_, 300 + 1700 * k, (1.0, 0.85, 0.45), 1.6 * k)
            add_glow(canvas, fx_, fy_, 900 + 1200 * k, (1.0, 0.95, 0.8), 1.2 * k, sx=1.0, sy=0.06)
            post["flash"] += ease_in_expo(k) * 0.95
            post["flash_color"] = (1.0, 0.95, 0.8)
    elif ex == "petals":
        k = ease_in_cubic(lin(u, TL.SCENE - 0.42, TL.SCENE))
        if k > 0:
            flakes(canvas, u, 401, int(30 + 230 * k), SAKURA_COLS, "petal", smin=18 + 22 * k, smax=40 + 50 * k,
                   fall=(220, 520), sway=90, blur=1.5, alpha=1.0)
            post["flash"] += ease_in_expo(lin(u, TL.SCENE - 0.15, TL.SCENE)) * 0.9
            post["flash_color"] = (1.0, 0.92, 0.95)
    elif ex == "warp":
        k = ease_in_cubic(lin(u, TL.SCENE - 0.4, TL.SCENE))
        R3.warp_streaks(canvas, k, col=d["warp_col"])
        post["flash"] += ease_in_expo(k) * 0.95
        post["flash_color"] = (0.9, 0.9, 1.0)
        post["ca"] += 12 * k
    elif ex == "smoke":
        canvas = smoke_wipe(canvas, u, d["smoke_col"])
    # ---- transition overlays (entrance)
    if ent == "wave":
        R3.liquid_wave(canvas, t, s - 0.4, s, s + 0.45, prev["wave_pal"])
    elif ent == "smoke_clear" and u < 0.42:
        R2.smoke(canvas, t, 3.0 * (1 - ease_out_cubic(u / 0.42)), prev["smoke_col"], seed=87)
    elif ent == "warp":
        R3.warp_streaks(canvas, 1 - ease_out_cubic(lin(u, 0, 0.45)), col=prev["warp_col"])
    elif ent == "zoom_in" and u < 0.5:
        k = ease_out_cubic(u / 0.5)
        flakes(canvas, u + TL.SCENE, 401, int(260 * (1 - k)), SAKURA_COLS, "petal", smin=40, smax=90,
               fall=(220, 520), sway=90, blur=1.5 + 4 * k, alpha=1 - k)
    # ---- motion blur
    if hb > 1:
        canvas = hblur(canvas, hb)
    if vb > 1:
        canvas = vblur(canvas, vb)
    if zb > 0.004:
        canvas = zoom_blur(canvas, zb, 540, 900)
    tag(canvas, t, s + 0.12, s + 0.6, d["name"], d["sub"], d["glow"])
    post["flash"] += 0.25 * decay(t, s + TL.CLOSE, 0.05) + 0.2 * decay(t, s + TL.MEDIUM, 0.05) + 0.45 * fl
    post["ca"] += 5 * decay(t, s + TL.CLOSE, 0.1) + 8 * fl
    return canvas, post


# ------------------------------------------------------------- dispatcher
def scene_index(t):
    if t < TL.DRINK_STARTS[1]:
        return 0
    if t >= TL.END_START:
        return None
    return min(TL.N_DRINKS - 1, int((t - TL.DROP) / TL.SCENE))


def render_frame(fi):
    t = fi / FPS
    k = scene_index(t)
    canvas, post = (R3.end_card(t, fi) if k is None else scene(k, t, fi))
    img = finish(canvas, fi, flash=min(1.0, post.get("flash", 0)), flash_color=post.get("flash_color", (1, 1, 1)),
                 ca=post.get("ca", 0), grain=post.get("grain", 0.014))
    if "fade" in post:
        img *= post["fade"]
    img = R3.draw_banner(img, t)
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
        for k, _ in enumerate(pool.imap(_work, range(n), chunksize=4)):
            if (k + 1) % 60 == 0:
                print(f"{k + 1}/{n} frames  {time.time() - t0:.0f}s", flush=True)
    print("render done", time.time() - t0)


if __name__ == "__main__":
    main()
