"""Reel 2 - "Two Locations" edition.

Only the closing store-promotion scene changes. Everything before it is reused
untouched:
  * video: the original Reel 2 export's compressed video is copied bit-for-bit
    up to its last keyframe before the end card (frame 699 = 23.3 s); frames
    699-719 are re-encoded from the identical original source frames
    (output/build2/frames, hard-linked, not re-rendered); only 720+ is new;
  * audio: the original Reel 2 mix, bit-identical up to 26.99 s. To give the
    two addresses more reading time the ending is extended by one bar (1.5 s)
    by repeating the end card's last groove bar (25.5-27.0 s) before the final
    hit - a standard music edit, so the soundtrack stays the original.

New closing scene (24.0-30.0 s, synced to the original end-card hits):
  24.00  fire burst -> white flash, card slides in (as before)
  24.15  store logo picture revealed (light-wipe + settle)       <- low hit
  24.30  "VISIT US AT TWO LOCATIONS!"                           <- whoosh
  24.75  location 1, then location 2 (pins drop in)             <- whoosh
  25.20  "STOP BY & GRAB YOUR FAVOURITES!" pill                 <- bell
  26.00 / 27.50  shine sweeps                                   <- sparkles
  28.50  final hit: punch + glow                                <- final hit
  29.60  fade to black, ends 30.0 s

    python reel2_two_locations.py            # build everything
    python reel2_two_locations.py --test 25.6,28.6
"""
import os
import argparse
import shutil
import subprocess
import time
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

import fx
from fx import (clamp, lin, smooth, decay, ease_out_cubic, ease_in_cubic, ease_in_out, ease_out_expo, ease_out_back,
                lerp, add_glow, draw_text, fit_size, vblur, finish, blend_over, blend_add, radial_bg)
import render as R
from timeline2 import W, H, FPS

ROOT = fx.ROOT
SRC_FRAMES = os.path.join(ROOT, "output", "build2", "frames")
SRC_AUDIO = os.path.join(ROOT, "output", "build2", "audio_mix.wav")
BUILD = os.path.join(ROOT, "output", "build2_two_locations")
FRAMES = os.path.join(BUILD, "frames")
OUT_MP4 = os.path.join(ROOT, "output", "reel2", "Ezy_Pzy_Convenience_Two_Locations_Reel.mp4")
LOGO = os.path.join(ROOT, "assets", "brand", "ezy_pzy_logo.jpg")
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


class T:
    END_START = 24.0
    LOGO = 24.15
    HEAD = 24.30
    RULE = 24.60
    ADDR = (24.75, 24.93)
    CTA = 25.20
    SHINES = (26.0, 27.5)
    FINAL = 28.5
    FADE = 29.6
    DURATION = 30.0
    LOOP_FROM, LOOP_TO = 25.5, 27.0     # audio bar that is repeated


FIRST_NEW_FRAME = int(round(T.END_START * FPS))     # 720
N_FRAMES = int(round(T.DURATION * FPS))             # 900

HEADLINE = "VISIT US AT TWO LOCATIONS!"
LOCATIONS = [("153 Great Northern Rd,", "Sault Ste. Marie, ON P6B 4Y9"),
             ("216 Second Line West,", "Sault Ste. Marie, ON")]
CTA1, CTA2 = "STOP BY & GRAB", "YOUR FAVOURITES!"
GOLD = R.GOLD
BRAND_LEAKS = [((0.85, 0.12, 0.18), 110, 360), ((0.22, 0.28, 0.85), 990, 520),
               ((1.0, 0.55, 0.15), 150, 1520), ((0.22, 0.28, 0.85), 960, 1660)]

# layout (screen px)
CARD_W = 940
CARD_CY = 598
HEAD_Y = 905
RULE_Y = 962
ROW_Y = (1080, 1248)
ROW_H = 142
CTA_Y = 1436


# ----------------------------------------------------------------- assets
@lru_cache(maxsize=1)
def logo_card():
    """The uploaded picture, unaltered, on a rounded card (premultiplied RGBA) + its shadow."""
    img = np.asarray(Image.open(LOGO).convert("RGB")).astype(np.float32) / 255
    h0, w0 = img.shape[:2]
    cw = CARD_W
    ch = int(round(h0 * cw / w0))
    pic = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA if cw < w0 else cv2.INTER_CUBIC)
    pad = 90
    m = np.zeros((ch + 2 * pad, cw + 2 * pad), np.float32)
    r = 26
    S = 16
    x0, y0, x1, y1 = pad, pad, pad + cw - 1, pad + ch - 1
    cv2.rectangle(m, ((x0 + r) * S, y0 * S), ((x1 - r) * S, y1 * S), 1.0, -1, cv2.LINE_AA, 4)
    cv2.rectangle(m, (x0 * S, (y0 + r) * S), (x1 * S, (y1 - r) * S), 1.0, -1, cv2.LINE_AA, 4)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx * S, cy * S), r * S, 1.0, -1, cv2.LINE_AA, 4)
    rgb = np.zeros((m.shape[0], m.shape[1], 3), np.float32)
    rgb[pad:pad + ch, pad:pad + cw] = pic
    # thin gold rim around the card
    edge = cv2.GaussianBlur(m, (0, 0), 1.2) - cv2.GaussianBlur(cv2.erode(m, np.ones((5, 5), np.uint8)), (0, 0), 1.2)
    edge = np.clip(edge * 2, 0, 1)
    shadow = cv2.GaussianBlur(m, (0, 0), 28)
    return dict(rgb=rgb, a=m, edge=edge, shadow=shadow, w=m.shape[1], h=m.shape[0], ch=ch, pad=pad)


@lru_cache(maxsize=2)
def pin_sprite(px=60):
    """The real map-pin emoji, rendered with Noto Color Emoji."""
    try:
        f = ImageFont.truetype(EMOJI_FONT, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), "\U0001F4CD", font=f, embedded_color=True)
        bb = im.getbbox()
        im = im.crop(bb)
        s = px / max(im.size)
        im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 255
        return np.dstack([a[..., :3] * a[..., 3:], a[..., 3]])
    except Exception:
        # fallback: drawn red pin
        n = px
        m = np.zeros((n, n), np.float32)
        cv2.circle(m, (n // 2, int(n * 0.36)), int(n * 0.3), 1.0, -1, cv2.LINE_AA)
        pts = np.array([[n * 0.24, n * 0.5], [n * 0.76, n * 0.5], [n * 0.5, n * 0.98]], np.int32)
        cv2.fillPoly(m, [pts], 1.0, cv2.LINE_AA)
        cv2.circle(m, (n // 2, int(n * 0.36)), int(n * 0.11), 0.0, -1, cv2.LINE_AA)
        col = np.array([0.9, 0.15, 0.2], np.float32)
        return np.dstack([col * m[..., None], m])


@lru_cache(maxsize=4)
def sizes(locations=tuple(LOCATIONS)):
    s1 = min(fit_size(a, "InterDisplay-SemiBold.otf", 740, 64) for a, _ in locations)
    s2 = min(fit_size(b, "InterDisplay-Medium.otf", 740, 48) for _, b in locations)
    hs = fit_size(HEADLINE, "InterDisplay-Black.otf", 900, 74)
    cs = min(fit_size(CTA1, "InterDisplay-Black.otf", 620, 72), fit_size(CTA2, "InterDisplay-Black.otf", 620, 72))
    return s1, s2, hs, cs


S_STREET, S_CITY, S_HEAD, S_CTA = sizes()


def text_left(canvas, text, font, size, x_left, cy, alpha=1.0, color=(1, 1, 1), blur=0.0, shadow=0.0):
    w = fx.font(font, size).getlength(text)
    draw_text(canvas, text, font, size, x_left + w / 2, cy, alpha=alpha, color=color, blur=blur, shadow=shadow)


def rounded_panel(w, h, r):
    m = np.zeros((h, w), np.float32)
    cv2.rectangle(m, (r, 0), (w - r, h), 1.0, -1, cv2.LINE_AA)
    cv2.rectangle(m, (0, r), (w, h - r), 1.0, -1, cv2.LINE_AA)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(m, (cx, cy), r, 1.0, -1, cv2.LINE_AA)
    return m


# ----------------------------------------------------------------- the scene
def end_card_two_locations(t, fi, T_=None, locations=None, leaks=None):
    """Closing scene. Defaults = Reel 2; Reel 1 passes its own timing, addresses and light leaks."""
    T_ = T if T_ is None else T_
    locs = tuple(LOCATIONS if locations is None else locations)
    S_STREET, S_CITY = sizes(locs)[:2]
    leaks = BRAND_LEAKS if leaks is None else leaks
    post = dict(flash=0.0, ca=0.0, grain=0.014)
    e = t - T_.END_START
    shift = 1800 * (1 - ease_out_cubic(lin(t, T_.END_START, T_.END_START + 0.28)))
    canvas = radial_bg((0.07, 0.06, 0.13), (0.004, 0.004, 0.012), 540, 860, 1000, 1300, 1.2).copy()
    for i, (col, x, y) in enumerate(leaks):
        add_glow(canvas, x + 90 * np.sin(e * 0.5 + i), y + 70 * np.cos(e * 0.4 + i * 2) + shift, 620, col, 0.15)
    R.god_rays(canvas, t, 540, CARD_CY + shift, GOLD, 0.10 + 0.10 * decay(t, T_.LOGO, 0.5), n=18, speed=0.03, sharp=8)
    add_glow(canvas, 540, CARD_CY + shift, 820, GOLD, 0.16 * smooth(lin(t, T_.LOGO, T_.LOGO + 0.6)))
    R.bokeh(canvas, t, 31, 16, [(0.9, 0.2, 0.25), (0.3, 0.35, 0.95), (1.0, 0.6, 0.2), GOLD], inten=0.08, rmin=30, rmax=90, speed=14)
    R.dust(canvas, t, 21, 150, (1.0, 0.85, 0.5), inten=0.7, rmin=2, rmax=6, vy=-40)
    yo = shift

    # ---- logo picture card
    L = logo_card()
    k = ease_out_expo(lin(t, T_.LOGO, T_.LOGO + 0.55))
    if k > 0:
        punch = 1 + 0.03 * decay(t, T_.FINAL, 0.25) if t >= T_.FINAL else 1.0
        sc = lerp(1.14, 1.0, k) * (1 + 0.012 * np.sin(e * 0.9)) * punch
        float_y = 5 * np.sin(e * 1.1)
        cy = CARD_CY + yo + float_y
        # light-wipe reveal from the centre outward
        rv = ease_in_out(lin(t, T_.LOGO, T_.LOGO + 0.42))
        xs = (np.arange(L["w"], dtype=np.float32) - L["w"] / 2) / (CARD_W / 2)
        reveal = np.clip((rv * 1.08 - np.abs(xs)) / 0.06, 0, 1)[None, :]
        M = np.float32([[sc, 0, 540 - sc * L["w"] / 2], [0, sc, cy - sc * L["h"] / 2]])
        warp = lambda a, flags=cv2.INTER_LINEAR: cv2.warpAffine(a, M, (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT)
        a_card = L["a"] * reveal
        # soft shadow + warm glow behind the card
        sh = warp(L["shadow"] * reveal) * 0.75 * k
        sh = np.roll(sh, 22, axis=0)
        canvas *= (1 - sh[..., None])
        glow = warp(cv2.GaussianBlur(L["a"], (0, 0), 40) * reveal) * 0.35 * k
        canvas += glow[..., None] * np.array([1.0, 0.92, 0.8], np.float32)
        # faint floor reflection under the card
        ref_a = cv2.flip(a_card, 0) * np.clip(1 - np.arange(L["h"], dtype=np.float32) / (L["h"] * 0.35), 0, 1)[:, None] * 0.12
        ref_rgb = cv2.flip(L["rgb"], 0)
        Mr = M.copy()
        Mr[1, 2] += sc * (L["h"] - 2 * L["pad"]) + 6
        rr = cv2.warpAffine(np.dstack([ref_rgb * ref_a[..., None], ref_a]), Mr, (W, H), flags=cv2.INTER_LINEAR)
        rr = cv2.GaussianBlur(rr, (0, 0), 3)
        blend_over(canvas, rr[..., :3] * k, rr[..., 3] * k)
        # the picture itself (pixels untouched), with a slight blur only while it settles
        lay = warp(np.dstack([L["rgb"] * a_card[..., None], a_card]), cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        lay = np.clip(lay, 0, None)
        if k < 0.98:
            lay = cv2.GaussianBlur(lay, (0, 0), 0.3 + 8 * (1 - k))
        blend_over(canvas, lay[..., :3] * k, lay[..., 3] * k)
        # gold rim + bright wipe edge
        edge = warp(L["edge"] * reveal) * k
        canvas += edge[..., None] * np.array(GOLD, np.float32) * 0.8
        if 0 < rv < 1:
            for sgn in (-1, 1):
                ex = 540 + sgn * rv * 1.08 * CARD_W / 2 * sc
                add_glow(canvas, ex, cy, sc * L["ch"] * 0.7, (1.0, 0.95, 0.8), 0.9, sx=0.08, sy=1.0)
        # shine sweeps across the card (transient light, picture unchanged)
        for s0 in T_.SHINES + (T_.FINAL,):
            if s0 <= t < s0 + 0.75:
                pos = lerp(-0.35, 1.35, ease_in_out((t - s0) / 0.75))
                xx = np.arange(L["w"], dtype=np.float32)[None, :] / L["w"]
                yy = np.arange(L["h"], dtype=np.float32)[:, None] / L["h"]
                band = np.exp(-((xx + 0.35 * yy - pos) / 0.05) ** 2) * L["a"] * 0.32
                canvas += warp(band)[..., None]

    # ---- headline
    kh = ease_out_cubic(lin(t, T_.HEAD, T_.HEAD + 0.45))
    if kh > 0:
        tr = lerp(0.35, 0.04, kh)
        draw_text(canvas, HEADLINE, "InterDisplay-Black.otf", S_HEAD, 540, HEAD_Y + yo, alpha=kh,
                  tracking=round(tr, 3) if kh < 1 else 0.04, scale=min(1.0, lerp(0.85, 1.0, kh)), blur=(1 - kh) * 6,
                  grad=((1, 1, 1), (1.0, 0.86, 0.52)), glow=GOLD, glow_r=18, glow_s=0.5, shadow=0.5)
    # ---- gold rule
    kr = ease_in_out(lin(t, T_.RULE, T_.RULE + 0.38))
    if kr > 0:
        hw = 250 * kr
        y = int(RULE_Y + yo)
        lay = np.zeros((H, W), np.float32)
        cv2.line(lay, (int(540 - hw), y), (int(540 + hw), y), 1.0, 3, cv2.LINE_AA)
        canvas += (lay * 0.9 + cv2.GaussianBlur(lay, (0, 0), 6) * 1.2)[..., None] * np.array(GOLD, np.float32)
        for sgn in (-1, 1):
            add_glow(canvas, 540 + sgn * hw, y, 40, (1, 0.95, 0.8), 0.6 * (1 - kr))

    # ---- two locations
    pin = pin_sprite(72)
    f_st = fx.font("InterDisplay-SemiBold.otf", S_STREET)
    f_ct = fx.font("InterDisplay-Medium.otf", S_CITY)
    block_w = max(max(f_st.getlength(a), f_ct.getlength(b)) for a, b in locs)
    pin_gap = 26
    row_w = pin.shape[1] + pin_gap + block_w
    x_pin = 540 - row_w / 2
    x_txt = x_pin + pin.shape[1] + pin_gap
    for j, ((street, city), ry) in enumerate(zip(locs, ROW_Y)):
        ka = ease_out_cubic(lin(t, T_.ADDR[j], T_.ADDR[j] + 0.42))
        if ka <= 0:
            continue
        cy = ry + yo + 28 * (1 - ka)
        # glass panel
        pw, ph = int(row_w + 90), ROW_H
        pm = rounded_panel(pw, ph, 22)
        px0, py0 = int(540 - pw / 2), int(cy - ph / 2)
        fill = pm * 0.07 * ka
        blend_over(canvas, np.ones((ph, pw, 3), np.float32) * fill[..., None], fill, px0, py0)
        border = np.clip(cv2.GaussianBlur(pm, (0, 0), 0.8) - cv2.erode(pm, np.ones((5, 5), np.uint8)), 0, 1)
        blend_add(canvas, (border * 0.22 * ka)[..., None] * np.array([1.0, 0.92, 0.75], np.float32), px0, py0)
        # pin drops in with a little bounce
        kp = ease_out_back(lin(t, T_.ADDR[j] + 0.08, T_.ADDR[j] + 0.45), 2.4)
        if kp > 0:
            pa = clamp(lin(t, T_.ADDR[j] + 0.08, T_.ADDR[j] + 0.2))
            psy = cy - pin.shape[0] / 2 - 46 * (1 - kp)
            add_glow(canvas, x_pin + pin.shape[1] / 2, cy + 4, 46, (1.0, 0.3, 0.3), 0.25 * pa)
            blend_over(canvas, pin[..., :3] * pa, pin[..., 3] * pa, int(x_pin), int(psy))
        text_left(canvas, street, "InterDisplay-SemiBold.otf", S_STREET, x_txt, cy - S_CITY * 0.62, alpha=ka, blur=(1 - ka) * 5, shadow=0.45)
        text_left(canvas, city, "InterDisplay-Medium.otf", S_CITY, x_txt, cy + S_STREET * 0.52, alpha=ka * 0.88,
                  color=(0.92, 0.9, 0.95), blur=(1 - ka) * 5, shadow=0.45)

    # ---- CTA pill
    kc = ease_out_back(lin(t, T_.CTA, T_.CTA + 0.38), 2.2)
    kca = lin(t, T_.CTA, T_.CTA + 0.15)
    if kca > 0:
        cs = S_CTA
        f2 = fx.font("InterDisplay-Black.otf", cs)
        tw = max(f2.getlength(CTA1), f2.getlength(CTA2))
        pw, ph = tw + 110, cs * 2.55
        cy = CTA_Y + yo
        pulse = 1 + 0.018 * np.sin(max(0, t - T_.CTA - 0.4) * 5.5) * (t > T_.CTA + 0.4)
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
        blend_add(canvas, (cv2.GaussianBlur(pm, (0, 0), 16) * 0.55 * kca)[..., None] * np.array(GOLD, np.float32), int(540 - lw / 2), int(cy - lh / 2))
        blend_over(canvas, col * a[..., None], a, int(540 - lw / 2), int(cy - lh / 2))
        for s0 in (T_.SHINES[1],):
            sp = lin(t, s0, s0 + 0.7)
            if 0 < sp < 1:
                xs = np.arange(lw, dtype=np.float32)[None, :] / lw
                band = np.exp(-((xs - lerp(-0.2, 1.2, sp)) / 0.05) ** 2) * pm * 0.6
                blend_add(canvas, band[..., None] * np.ones(3, np.float32), int(540 - lw / 2), int(cy - lh / 2))
        for line, dy_ in ((CTA1, -cs * 0.58), (CTA2, cs * 0.58)):
            draw_text(canvas, line, "InterDisplay-Black.otf", cs, 540, cy + dy_ * sc, color=(0.07, 0.05, 0.10), alpha=kca, scale=sc)

    if shift > 1:
        canvas = vblur(canvas, shift * 0.12)
    zs = 1 + 0.035 * ease_in_out(lin(t, T_.END_START + 0.3, T_.DURATION))
    M = np.float32([[zs, 0, 540 - zs * 540], [0, zs, 960 - zs * 960]])
    canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    post["flash"] = 0.85 * (1 - ease_out_cubic(lin(t, T_.END_START, T_.END_START + 0.2)))
    post["flash"] += 0.18 * decay(t, T_.LOGO, 0.06) + (0.12 * decay(t, T_.FINAL, 0.08) if t >= T_.FINAL else 0.0)
    post["flash_color"] = (1.0, 0.95, 0.85)
    post["ca"] = 6 * decay(t, T_.LOGO, 0.12)
    post["fade"] = 1 - smooth(lin(t, T_.FADE, T_.DURATION - 1.0 / FPS))
    return canvas, post


def render_frame(fi):
    t = fi / FPS
    canvas, post = end_card_two_locations(t, fi)
    img = finish(canvas, fi, flash=min(1.0, post["flash"]), flash_color=post["flash_color"], ca=post["ca"], grain=post["grain"])
    img *= post["fade"]
    return (img * 255 + 0.5).astype(np.uint8)


def _work(fi):
    img = render_frame(fi)
    cv2.imwrite(os.path.join(FRAMES, f"{fi:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi


# ----------------------------------------------------------------- audio splice
def build_audio(path):
    sr, a = wavfile.read(SRC_AUDIO)
    xf = int(0.012 * sr)                      # 12 ms equal-power crossfade just before the downbeat
    j = int(round(T.LOOP_TO * sr))            # 27.0 s
    k = int(round(T.LOOP_FROM * sr))          # 25.5 s
    af = a.astype(np.float64)
    tt = np.linspace(0, np.pi / 2, xf)[:, None]
    seam = af[j - xf:j] * np.cos(tt) + af[k - xf:k] * np.sin(tt)
    out = np.concatenate([af[:j - xf], seam, af[k:]])
    out = np.clip(np.round(out), -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, out)
    return sr, a, out, j - xf


ORIG_MP4 = os.path.join(ROOT, "output", "reel2", "ezy_pzy_reel2_1080x1920.mp4")
X264 = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M", "-tune", "film",
        "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-g", str(FPS)]


def keyframes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=key_frame",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()
    return [i for i, k in enumerate(out) if k.strip().startswith("1")]


def encode_smart(wav):
    """Splice at the container level: the original Reel 2 video packets up to its last
    keyframe before the end card are copied untouched; only the rest is encoded.
    (SPS/PPS are identical for both pieces, so they join seamlessly.)"""
    ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    K = max(k for k in keyframes(ORIG_MP4) if k <= FIRST_NEW_FRAME)
    head = os.path.join(BUILD, "head.mp4")
    tail = os.path.join(BUILD, "tail.mp4")
    lst = os.path.join(BUILD, "list.txt")
    subprocess.run(ff + ["-i", ORIG_MP4, "-map", "0:v:0", "-c:v", "copy", "-frames:v", str(K), head], check=True)
    subprocess.run(ff + ["-framerate", str(FPS), "-start_number", str(K), "-i", os.path.join(FRAMES, "%05d.png"),
                         "-frames:v", str(N_FRAMES - K)] + X264 + ["-video_track_timescale", "15360", tail], check=True)
    with open(lst, "w") as f:
        f.write(f"file '{head}'\nfile '{tail}'\n")
    subprocess.run(ff + ["-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-map", "0:v", "-map", "1:a",
                         "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", "-shortest",
                         OUT_MP4], check=True)
    print(f"video: frames 0-{K - 1} stream-copied from the original Reel 2 export, frames {K}-{N_FRAMES - 1} encoded")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.test:
        out = os.path.join(BUILD, "test")
        os.makedirs(out, exist_ok=True)
        for ts in args.test.split(","):
            t0 = time.time()
            img = render_frame(int(round(float(ts) * FPS)))
            cv2.imwrite(os.path.join(out, f"t{float(ts):06.2f}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"t={ts} {time.time() - t0:.2f}s", flush=True)
        return
    assert len(os.listdir(SRC_FRAMES)) >= FIRST_NEW_FRAME, "render Reel 2 first (build_reel2.sh)"
    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)
    # 1) reuse the original frames 0..719 untouched
    for fi in range(FIRST_NEW_FRAME):
        src = os.path.join(SRC_FRAMES, f"{fi:05d}.png")
        dst = os.path.join(FRAMES, f"{fi:05d}.png")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    # 2) render the new closing scene
    t0 = time.time()
    with Pool(args.jobs) as p:
        for n, _ in enumerate(p.imap_unordered(_work, range(FIRST_NEW_FRAME, N_FRAMES), chunksize=2)):
            if (n + 1) % 30 == 0:
                print(f"end card {n + 1}/{N_FRAMES - FIRST_NEW_FRAME} frames {time.time() - t0:.0f}s", flush=True)
    # 3) audio: original mix + one repeated bar before the final hit
    wav = os.path.join(BUILD, "audio_mix.wav")
    sr, orig, new, seam = build_audio(wav)
    same = np.array_equal(orig[:seam], new[:seam])
    print(f"audio: {len(new) / sr:.3f}s, first {seam / sr:.3f}s bit-identical to original: {same}")
    # 4) encode: copy the ORIGINAL Reel 2 video stream bit-for-bit up to its last
    #    keyframe before the end card, encode only from there (same x264 settings)
    encode_smart(wav)
    print("wrote", OUT_MP4)


if __name__ == "__main__":
    main()
