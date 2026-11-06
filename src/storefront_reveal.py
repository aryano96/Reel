"""Ezy Pzy Convenience - cinematic storefront reveal from the real store video.

HONEST SCOPE: the supplied clip (assets/storefront/store_video_original.MOV) is a
ground-level, night-time phone ZOOM toward the store (360x480, 5.5 s). It has no
aerial footage and no perspective change, so a genuine drone descent cannot be
made from it. This script makes the best honest result from the real footage:

  * camera tracking (SIFT + RANSAC) puts every frame in one shared "world" space,
    so the move can be re-timed into one ultra-smooth eased push with clean
    slow motion (frames are re-projected and blended, no AI interpolation);
  * the iPhone lens-switch exposure/colour jump (frame 45) is corrected;
  * cleanup in world space: the filmer's shadow is relit (texture kept), litter
    specks on the asphalt are removed, the parked car is cloned out;
  * night cinematic grade: deep-navy sky, warm store light, bloom, vignette,
    grain, a foreground grad; 9:16 1080x1920 framing; gentle hero push-in + hold;
  * original synthesized sound design (pad, build, whoosh, impact, ending).

    python storefront_reveal.py --test 0.5,3.5,7.5
    python storefront_reveal.py              # full render + MP4
"""
import os
import glob
import argparse
import subprocess
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import cv2

import fx

ROOT = fx.ROOT
SRC = os.path.join(ROOT, "assets", "storefront", "store_video_original.MOV")
BUILD = os.path.join(ROOT, "output", "build_storefront")
FR_SRC = os.path.join(BUILD, "src")
FR_OUT = os.path.join(BUILD, "frames")
OUT_DIR = os.path.join(ROOT, "output", "storefront")
OUT_MP4 = os.path.join(OUT_DIR, "Ezy_Pzy_Cinematic_Storefront_Reveal.mp4")
W, H, FPS = 1080, 1920, 30
DUR = 9.5
T_ZOOM0, T_ZOOM1 = 1.6, 5.6          # eased approach between these times
Z0, Z_A, Z_B, Z_END = 1.31, 1.37, 4.0, 4.38
CENTER = (184.0, 236.0)              # world point kept centred (the storefront)
WIDE = range(0, 22)                  # static wide frames (0-based)
HERO0 = 84                           # first frame of the static front-on hold
SWITCH = 44                          # lens switch (first frame with new lens)

EXPOSURE = 0.82

cv2.setNumThreads(1)


# ---------------------------------------------------------------- data prep
def extract():
    os.makedirs(FR_SRC, exist_ok=True)
    if len(glob.glob(os.path.join(FR_SRC, "*.png"))) < 100:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-vf", "fps=24", os.path.join(FR_SRC, "f_%04d.png")], check=True)


@lru_cache(maxsize=1)
def frames():
    fs = sorted(glob.glob(os.path.join(FR_SRC, "f_*.png")))
    out = []
    for f in fs:
        im = cv2.imread(f)
        im = cv2.fastNlMeansDenoisingColored(im, None, 4, 6, 5, 15)
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    return out


@lru_cache(maxsize=1)
def tracks():
    """W[i]: 3x3 similarity mapping frame i pixels -> world (= last frame) pixels."""
    path = os.path.join(BUILD, "W.npy")
    if os.path.exists(path):
        return np.load(path)
    fs = sorted(glob.glob(os.path.join(FR_SRC, "f_*.png")))
    gray = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in fs]
    sift = cv2.SIFT_create(3000)
    feats = [sift.detectAndCompute(g, None) for g in gray]
    bf = cv2.BFMatcher()
    n = len(gray)
    Wm = [None] * n
    Wm[-1] = np.eye(3)
    for i in range(n - 2, -1, -1):
        (k1, d1), (k2, d2) = feats[i], feats[i + 1]
        m = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
        p1 = np.float32([k1[x.queryIdx].pt for x in m])
        p2 = np.float32([k2[x.trainIdx].pt for x in m])
        M, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        Wm[i] = Wm[i + 1] @ np.vstack([M, [0, 0, 1]])
    Wm = np.array(Wm)
    np.save(path, Wm)
    return Wm


def scale_of(M):
    return float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))


def to_world(img, M, size=(360, 480), flags=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, M[:2], size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@lru_cache(maxsize=1)
def gains():
    """Per-frame RGB gain matching each frame's exposure to the final hold (removes the
    auto-exposure ramp and the lens-switch jump)."""
    F, Wm = frames(), tracks()
    ref = np.median(np.stack([F[i] for i in range(100, len(F))]), 0)
    g = np.ones((len(F), 3), np.float32)
    for i in range(len(F)):
        wi = to_world(F[i], Wm[i])
        valid = to_world(np.ones((480, 360), np.float32), Wm[i]) > 0.99
        lum = ref.mean(2)
        sel = valid & (lum > 0.08) & (lum < 0.9)
        sel[:150] = False                       # storefront band + lot only
        if sel.sum() > 500:
            g[i] = np.median(ref[sel], 0) / np.maximum(np.median(wi[sel], 0), 1e-3)
    # smooth on each side of the lens switch, keep the jump itself
    out = g.copy()
    for a, b in ((0, SWITCH), (SWITCH, len(F))):
        seg = g[a:b]
        k = np.exp(-0.5 * (np.arange(-6, 7) / 3.0) ** 2)
        k /= k.sum()
        for c in range(3):
            pad = np.pad(seg[:, c], 6, mode="edge")
            out[a:b, c] = np.convolve(pad, k, mode="valid")
    return np.clip(out, 0.5, 4.0)


def frame_g(i):
    return np.clip(frames()[i] * gains()[i][None, None, :], 0, 1.5)


# ---------------------------------------------------------------- cleanup
def remove_specks(img, ymin, max_area=30, k=9):
    """Remove small bright litter specks on the asphalt (rows >= ymin)."""
    out = img.copy()
    lum = img.mean(2)
    bg = cv2.medianBlur((np.clip(img, 0, 1) * 255).astype(np.uint8), k).astype(np.float32) / 255
    d = lum - bg.mean(2)
    m = (d > 0.06).astype(np.uint8)
    m[:int(ymin)] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for kk in range(1, n):
        k_ = kk
        if st[k_, cv2.CC_STAT_AREA] <= max_area:
            keep[lab == k_] = 1
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8)).astype(np.float32)
    keep = cv2.GaussianBlur(keep, (0, 0), 0.8)[..., None]
    return out * (1 - keep) + bg * keep


def relight_shadow(img, box):
    """Lift a cast shadow inside box=(x0,y0,x1,y1) by estimating the surrounding
    illumination (texture, cracks and paint lines are kept)."""
    x0, y0, x1, y1 = box
    out = img.copy()
    lum = cv2.GaussianBlur(img.mean(2), (0, 0), 3)
    pad = 30
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    L = lum[Y0:Y1, X0:X1]
    est0 = cv2.GaussianBlur(L, (0, 0), 14)
    shadow = (L < est0 * 0.78).astype(np.float32)
    inbox = np.zeros_like(L)
    inbox[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = 1
    shadow *= inbox
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lit = 1 - shadow
    num = cv2.GaussianBlur(L * lit, (0, 0), 16)
    den = cv2.GaussianBlur(lit, (0, 0), 16) + 1e-4
    illum = num / den
    gain = np.clip(illum / np.maximum(cv2.GaussianBlur(L, (0, 0), 6), 1e-3), 1.0, 3.0)
    msk = cv2.GaussianBlur(shadow, (0, 0), 4)
    gmap = 1 + (gain - 1) * msk
    out[Y0:Y1, X0:X1] *= gmap[..., None]
    return out


def clone_out(img, box, src_dx):
    """Replace box with the patch src_dx pixels to the side (feathered)."""
    x0, y0, x1, y1 = box
    out = img.copy()
    patch = img[y0:y1, x0 + src_dx:x1 + src_dx]
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    m[2:-2, 2:-2] = 1
    m = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None]
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m) + patch * m
    return out


CAR_BOX = (258, 230, 314, 258)       # parked car in the wide frames (frame px)
SHADOW_BOX = (222, 326, 322, 480)    # filmer's shadow in the wide frames


@lru_cache(maxsize=1)
def wide_plate():
    """Clean, denoised plate of the static wide shot, in its own frame coords (frame 11)."""
    F, Wm = frames(), tracks()
    ref = 10
    al = []
    for i in WIDE:
        M = np.linalg.inv(Wm[ref]) @ Wm[i]
        al.append(to_world(frame_g(i), M))
    plate = np.median(np.stack(al), 0)
    plate = relight_shadow(plate, SHADOW_BOX)
    plate = clone_out(plate, CAR_BOX, -56)
    plate = remove_specks(plate, 300, max_area=90, k=15)
    return plate


SR = 2          # hero plates are built at 2x by multi-frame shift-and-add


@lru_cache(maxsize=1)
def hero_plates():
    """Multi-frame super-resolution of the near-static hero hold: each output frame is the
    average of 9 neighbouring frames, each warped (sub-pixel accurate) onto a 2x grid of
    frame i. Real detail from the real frames - nothing is invented."""
    F, Wm = frames(), tracks()
    out = {}
    n = len(F)
    up = np.diag([SR, SR, 1.0])
    for i in range(HERO0, n):
        win = [j for j in range(i - 4, i + 5) if HERO0 <= j < n]
        acc = np.mean([cv2.warpAffine(frame_g(j), (up @ np.linalg.inv(Wm[i]) @ Wm[j])[:2], (360 * SR, 480 * SR),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) for j in win], 0)
        out[i] = remove_specks(acc, 290 * SR, max_area=30 * SR * SR, k=15)
    return out


# ---------------------------------------------------------------- camera path
def smoother(x):
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (x * 6 - 15) + 10)


def zoom_at(t):
    if t < T_ZOOM0:
        return Z0 + (Z_A - Z0) * fx.ease_in_out(t / T_ZOOM0)
    if t < T_ZOOM1:
        k = smoother((t - T_ZOOM0) / (T_ZOOM1 - T_ZOOM0))
        return float(np.exp(np.log(Z_A) + (np.log(Z_B) - np.log(Z_A)) * k))
    return Z_B + (Z_END - Z_B) * fx.ease_out_cubic((t - T_ZOOM1) / (DUR - T_ZOOM1))


def source_index(t, z):
    """Fractional source frame to use: during the approach, the most detailed frame that
    still covers the view; during the hold, real time (slowed) through the hero frames."""
    Wm = tracks()
    n = len(Wm)
    if t < T_ZOOM0:
        return None
    if t >= T_ZOOM1:
        return HERO0 + (n - 1 - HERO0) * min(1.0, (t - T_ZOOM1) / (DUR - T_ZOOM1))
    need = (H / z) / 480.0 * 1.04            # world-per-frame-px needed to cover the view height
    s = np.array([scale_of(Wm[i]) for i in range(n)])
    idx = np.arange(n, dtype=np.float32)
    # s decreases with i over the zoom: find u with s(u) == need, restricted to the zoom section
    lo, hi = WIDE[-1], HERO0
    seg_s, seg_i = s[lo:hi + 1], idx[lo:hi + 1]
    if need >= seg_s[0]:
        return float(lo)
    if need <= seg_s[-1]:
        return float(hi)
    return float(np.interp(-need, -seg_s, seg_i))


def screen_from_world(z, c=CENTER):
    return np.array([[z, 0, W / 2 - z * c[0]], [0, z, H / 2 - z * c[1]], [0, 0, 1]], np.float64)


def warp_to_screen(img, M):
    flags = cv2.INTER_CUBIC
    return cv2.warpAffine(img, M[:2], (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@lru_cache(maxsize=4)
def feather(h, w, r=6):
    m = np.zeros((h, w), np.float32)
    m[r:-r, r:-r] = 1
    return cv2.GaussianBlur(m, (0, 0), r / 2)


# ---------------------------------------------------------------- render
def sky_and_grade(img, S):
    """Night cinematic grade. S: screen<-world (for world-anchored sky mask)."""
    yy = np.arange(H, dtype=np.float32)
    roof_y = S[1, 1] * 92 + S[1, 2]               # world y=92 is just above the roof line
    sky = (1 - fx.smoothstep_arr(roof_y - 40 * S[1, 1] / 4, roof_y, yy))[:, None]
    lum = img.mean(2, keepdims=True)
    dark = np.clip(1 - lum / 0.16, 0, 1)
    grad_t = np.clip(yy / max(1.0, roof_y), 0, 1)[:, None, None]
    navy = np.array([0.020, 0.040, 0.105], np.float32) * (1 - grad_t) + np.array([0.055, 0.075, 0.150], np.float32) * grad_t
    img = img + (navy * (sky[..., None] * dark)) * 0.55
    # split tone: cool shadows, warm highlights
    lum = img.mean(2, keepdims=True)
    sh = np.clip(1 - lum / 0.35, 0, 1)
    hi = np.clip((lum - 0.45) / 0.5, 0, 1)
    img = img * (1 - 0.18 * sh) + sh * np.array([0.0, 0.012, 0.03], np.float32) * 1.0
    img = img * (1 + hi * (np.array([1.06, 1.0, 0.9], np.float32) - 1))
    # exposure (moodier night) + gentle S-curve that keeps highlights intact
    img = np.clip(img * EXPOSURE, 0, None)
    x = np.clip(img, 0, 1)
    img = img + 0.22 * (x * x * (3 - 2 * x) - x)
    # foreground grad (darkens the bottom of frame)
    fg = 1 - 0.38 * fx.smoothstep_arr(H * 0.70, H * 1.0, yy)
    return img * fg[:, None, None]


def render(t, fi):
    Wm = tracks()
    z = zoom_at(t)
    S = screen_from_world(z)
    # base: clean wide plate (world via frame 11's track) - always covers the view
    plate = wide_plate()
    base_M = S @ Wm[10]
    canvas = warp_to_screen(plate, base_M)
    u = source_index(t, z)
    if u is not None:
        hero = t >= T_ZOOM1
        i0 = int(np.floor(u))
        f = u - i0
        i1 = min(i0 + 1, len(Wm) - 1)
        acc = np.zeros((H, W, 3), np.float32)
        aacc = np.zeros((H, W), np.float32)
        for i, wgt in ((i0, 1 - f), (i1, f)):
            if wgt <= 1e-3:
                continue
            down = np.eye(3)
            if hero:
                src = hero_plates()[i]
                down = np.diag([1.0 / SR, 1.0 / SR, 1.0])
            else:
                src = frame_g(i)
                src = remove_specks(src, 300 if scale_of(Wm[i]) > 2 else 290, max_area=90, k=15)
                if scale_of(Wm[i]) > 1.9:            # car still in view: clone it out in world space
                    Mi = np.linalg.inv(Wm[i]) @ Wm[10]
                    cleaned = cv2.warpAffine(plate, Mi[:2], (360, 480), flags=cv2.INTER_LINEAR)
                    car = np.zeros((480, 360), np.float32)
                    x0, y0, x1, y1 = CAR_BOX
                    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                    pts = cv2.transform(pts[None], Mi[:2])[0]
                    cv2.fillConvexPoly(car, pts.astype(np.int32), 1.0)
                    car = cv2.GaussianBlur(cv2.dilate(car, np.ones((7, 7), np.uint8)), (0, 0), 2)[..., None]
                    src = src * (1 - car) + cleaned * car
            M = S @ Wm[i] @ down
            lay = warp_to_screen(src, M)
            a = warp_to_screen(feather(src.shape[0], src.shape[1], 6 * (SR if hero else 1)), M)
            acc += lay * (a * wgt)[..., None]
            aacc += a * wgt
        k = np.clip(aacc, 0, 1)[..., None]
        canvas = canvas * (1 - k) + acc / np.maximum(aacc, 1e-4)[..., None] * k
    # sharpen the upscale a touch
    bl = cv2.GaussianBlur(canvas, (0, 0), 1.6)
    canvas = np.clip(canvas + 0.45 * (canvas - bl), 0, None)
    # subtle zoom motion blur proportional to zoom speed
    dz = np.log(zoom_at(min(DUR, t + 1 / FPS))) - np.log(z)
    if dz > 0.004:
        cx, cy = S[0, 0] * CENTER[0] + S[0, 2], S[1, 1] * CENTER[1] + S[1, 2]
        canvas = fx.zoom_blur(canvas, min(0.05, dz * 1.6), cx, cy, n=8)
    canvas = sky_and_grade(canvas, S)
    flash = 0.18 * fx.decay(t, T_ZOOM1, 0.12) if t >= T_ZOOM1 else 0.0
    img = fx.finish(canvas, fi, flash=flash, flash_color=(1.0, 0.95, 0.85), grain=0.012, bloom_s=0.42)
    fade_in = fx.smooth(t / 0.6)
    fade_out = 1 - fx.smooth((t - (DUR - 0.5)) / 0.5)
    img *= fade_in * fade_out
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _work(fi):
    img = render(fi / FPS, fi)
    cv2.imwrite(os.path.join(FR_OUT, f"{fi:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi


def _warm():
    frames(); tracks(); gains(); wide_plate(); hero_plates()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default=None)
    args = ap.parse_args()
    extract()
    tracks()
    if args.test:
        out = os.path.join(BUILD, "test")
        os.makedirs(out, exist_ok=True)
        for ts in args.test.split(","):
            t = float(ts)
            img = render(t, int(t * FPS))
            cv2.imwrite(os.path.join(out, f"t{t:05.2f}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("t", t, "z", round(zoom_at(t), 3), "u", source_index(t, zoom_at(t)), flush=True)
        return
    os.makedirs(FR_OUT, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = int(round(DUR * FPS))
    with Pool(os.cpu_count(), initializer=_warm) as p:
        for k, _ in enumerate(p.imap_unordered(_work, range(n), chunksize=3)):
            if (k + 1) % 45 == 0:
                print(f"{k + 1}/{n} frames", flush=True)
    import storefront_audio
    wav = storefront_audio.build(os.path.join(BUILD, "audio.wav"), DUR, T_ZOOM0, T_ZOOM1)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(FR_OUT, "%05d.png"),
                    "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-maxrate", "14M", "-bufsize", "28M",
                    "-tune", "film", "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-g", str(FPS),
                    "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest", "-movflags", "+faststart", OUT_MP4], check=True)
    print("wrote", OUT_MP4)


if __name__ == "__main__":
    main()
