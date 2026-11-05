"""Visual effects toolkit for the reel: compositing, can rendering, liquid,
ice, lightning, particles, text. Everything is a pure function of time so
frames can be rendered in parallel and in any order."""
import os
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from timeline import W, H

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FONT_DIR = os.path.join(ROOT, "assets", "fonts")
CUT_DIR = os.path.join(ROOT, "assets", "cutouts")

cv2.setNumThreads(1)


# =========================================================== easing / timing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lin(t, t0, t1):
    if t1 == t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_cubic(x):
    x = clamp(x)
    return x ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_expo(x):
    x = clamp(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_in_expo(x):
    x = clamp(x)
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)


def ease_out_back(x, s=1.70158):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def lerp(a, b, x):
    return a + (b - a) * x


def keys(t, kf, ease=ease_in_out):
    """kf: list of (time, value). Piecewise eased interpolation."""
    if t <= kf[0][0]:
        return kf[0][1]
    for (t0, v0), (t1, v1) in zip(kf, kf[1:]):
        if t <= t1:
            return lerp(v0, v1, ease(lin(t, t0, t1)))
    return kf[-1][1]


def decay(t, t0, tau):
    return 0.0 if t < t0 else float(np.exp(-(t - t0) / tau))


def smoothstep_arr(a, b, x):
    y = np.clip((x - a) / (b - a), 0, 1)
    return y * y * (3 - 2 * y)


# =========================================================== image basics
_grid_cache = {}


def grid():
    if "g" not in _grid_cache:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _grid_cache["g"] = (xx, yy)
    return _grid_cache["g"]


@lru_cache(maxsize=32)
def radial_bg(c_center, c_edge, cx, cy, rx, ry, power=1.6):
    xx, yy = grid()
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    m = np.clip(1 - d, 0, 1) ** power
    c0 = np.array(c_center, np.float32)
    c1 = np.array(c_edge, np.float32)
    return (c1 + (c0 - c1) * m[..., None]).astype(np.float32)


@lru_cache(maxsize=4)
def gauss_sprite(n=256):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    r2 = ((x - c) ** 2 + (y - c) ** 2) / (c * c)
    return np.exp(-r2 * 4.5).astype(np.float32)


def add_glow(canvas, x, y, radius, color, intensity=1.0, sx=1.0, sy=1.0):
    """Additive soft gaussian light at (x,y)."""
    rw, rh = int(radius * sx), int(radius * sy)
    if rw < 1 or rh < 1 or intensity <= 0:
        return
    x0, y0 = int(x - rw), int(y - rh)
    x1, y1 = x0 + 2 * rw, y0 + 2 * rh
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return
    spr = cv2.resize(gauss_sprite(), (2 * rw, 2 * rh), interpolation=cv2.INTER_LINEAR)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    s = spr[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    canvas[cy0:cy1, cx0:cx1] += s[..., None] * (np.array(color, np.float32) * intensity)


def blend_over(canvas, rgb_pm, a, x0=0, y0=0):
    """Premultiplied 'over' of a layer whose top-left is at (x0,y0)."""
    h, w = a.shape
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sa = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, None]
    sr = rgb_pm[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    roi = canvas[cy0:cy1, cx0:cx1]
    roi *= 1 - sa
    roi += sr


def blend_add(canvas, rgb, x0=0, y0=0):
    h, w = rgb.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas[cy0:cy1, cx0:cx1] += rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]


def screen_white(canvas, amount):
    if amount > 0:
        canvas += (1 - np.clip(canvas, 0, 1)) * amount


def hblur(img, k):
    k = int(k)
    if k < 2:
        return img
    return cv2.blur(img, (k, 1), borderType=cv2.BORDER_REPLICATE)


def vblur(img, k):
    k = int(k)
    if k < 2:
        return img
    return cv2.blur(img, (1, k), borderType=cv2.BORDER_REPLICATE)


def zoom_blur(img, amount, cx=W / 2, cy=H / 2, n=10):
    """Radial blur: average of progressively scaled copies (amount ~ 0..0.4)."""
    if amount < 0.004:
        return img
    acc = np.zeros_like(img)
    for i in range(n):
        s = 1 + amount * i / (n - 1)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


def chroma_ab(img, px):
    if px < 0.5:
        return img
    out = img.copy()
    s = 1 + px / 600
    for ch, sc in ((0, s), (2, 1 / s)):
        M = np.float32([[sc, 0, W / 2 - sc * W / 2], [0, sc, H / 2 - sc * H / 2]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


@lru_cache(maxsize=1)
def vignette_mask():
    xx, yy = grid()
    d = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    return np.clip(1 - 0.55 * smoothstep_arr(0.45, 1.25, d), 0, 1).astype(np.float32)


@lru_cache(maxsize=1)
def grain_bank():
    r = np.random.default_rng(3)
    bank = []
    for _ in range(6):
        g = r.standard_normal((H // 2, W // 2)).astype(np.float32)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        bank.append(g)
    return bank


def bloom(img, thr=0.72, strength=0.32):
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = np.clip(small - thr, 0, None)
    g = cv2.GaussianBlur(b, (0, 0), 6) * 0.6 + cv2.GaussianBlur(b, (0, 0), 22) * 0.6
    return img + cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) * strength


def finish(img, frame_idx, flash=0.0, flash_color=(1, 1, 1), ca=0.0, grain=0.016, bloom_s=0.3):
    img = bloom(img, strength=bloom_s)
    if ca > 0:
        img = chroma_ab(img, ca)
    img *= vignette_mask()[..., None]
    if flash > 0:
        fc = np.array(flash_color, np.float32)
        img = img + (fc - np.clip(img, 0, 1) * fc) * flash
    # soft highlight roll-off (filmic shoulder)
    img = np.where(img > 0.8, 0.8 + 0.2 * np.tanh((img - 0.8) / 0.2), img)
    img += grain_bank()[frame_idx % 6][..., None] * grain
    return np.clip(img, 0, 1)


# =========================================================== noise textures
@lru_cache(maxsize=4)
def fractal_noise(seed, w=540, h=960, octaves=6, persistence=0.55):
    r = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cw, ch = max(2, w >> (octaves - o)), max(2, h >> (octaves - o))
        base = r.random((ch, cw)).astype(np.float32)
        # tileable upscale: wrap-pad then crop
        big = np.pad(base, 1, mode="wrap")
        up = cv2.resize(big, ((cw + 2) * w // cw, (ch + 2) * h // ch), interpolation=cv2.INTER_CUBIC)
        oy, ox = h // ch, w // cw
        up = up[oy:oy + h, ox:ox + w]
        acc += up * amp
        tot += amp
        amp *= persistence
    acc /= tot
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return acc


@lru_cache(maxsize=4)
def lowfreq_noise(seed, w=540, h=960, cells=7):
    """Smooth, tileable large-scale undulation noise."""
    r = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for o, amp in ((cells, 1.0), (cells * 2, 0.45)):
        cw, ch = o, int(o * h / w)
        base = r.random((ch, cw)).astype(np.float32)
        big = np.pad(base, 2, mode="wrap")
        up = cv2.resize(big, ((cw + 4) * w // cw, (ch + 4) * h // ch), interpolation=cv2.INTER_CUBIC)
        oy, ox = 2 * h // ch, 2 * w // cw
        acc += up[oy:oy + h, ox:ox + w] * amp
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return acc


def scroll(tex, dx, dy):
    return np.roll(np.roll(tex, int(dy) % tex.shape[0], 0), int(dx) % tex.shape[1], 1)


# =========================================================== camera
class Cam:
    """world -> screen: s = (p - f) * z + anchor (+ shake)."""

    def __init__(self, z=1.0, fx=540.0, fy=1000.0, ax=540.0, ay=1000.0, roll=0.0):
        self.z, self.fx, self.fy, self.ax, self.ay, self.roll = z, fx, fy, ax, ay, roll

    def p(self, x, y):
        return (x - self.fx) * self.z + self.ax, (y - self.fy) * self.z + self.ay

    def s(self, r):
        return r * self.z


# =========================================================== the can
class _Level:
    pass


class Can:
    REF_H = 1150.0  # world height of every can

    def __init__(self, key):
        im = np.asarray(Image.open(os.path.join(CUT_DIR, key + ".png")).convert("RGBA")).astype(np.float32) / 255.0
        self.key = key
        rgb = np.ascontiguousarray(im[..., :3])
        a = np.ascontiguousarray(im[..., 3])
        self.levels = {1: self._build(rgb, a, 1)}
        L = self.levels[1]
        self.h, self.w = L.a.shape
        self.top, self.bot = L.top, L.bot
        self.axis, self.cy_native = L.axis, L.cy

    def _build(self, rgb, a, k):
        L = _Level()
        if k != 1:
            pm = np.dstack([rgb * a[..., None], a])
            up = cv2.resize(pm, (pm.shape[1] * k, pm.shape[0] * k), interpolation=cv2.INTER_LANCZOS4)
            bl = cv2.GaussianBlur(up, (0, 0), 1.4)
            up = np.clip(up + 0.7 * (up - bl), 0, None)
            a = np.clip(up[..., 3], 0, 1)
            rgb = np.clip(up[..., :3] / np.maximum(a[..., None], 1e-4), 0, 1)
            rgb[a < 1e-3] = 0
        L.rgb, L.a = rgb.astype(np.float32), a.astype(np.float32)
        h, w = a.shape
        m = a > 0.5
        rows = np.where(m.any(1))[0]
        anyr = m.any(1)
        left = np.where(anyr, np.argmax(m, 1), 0).astype(np.float32)
        right = np.where(anyr, w - 1 - np.argmax(m[:, ::-1], 1), 0).astype(np.float32)
        body = rows[int(len(rows) * 0.2):int(len(rows) * 0.8)]
        L.axis = float(np.median((left[body] + right[body]) / 2))
        R = np.maximum(right - L.axis, L.axis - left) + 1.0
        R[~anyr] = R[body].max()
        L.R = R.astype(np.float32)
        L.top, L.bot = int(rows[0]), int(rows[-1])
        L.cy = (L.top + L.bot) / 2.0
        xs = np.arange(w, dtype=np.float32)
        u = np.clip((xs[None, :] - L.axis) / L.R[:, None], -1, 1)
        L.theta = np.arcsin(u).astype(np.float32)
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        L.rim = (np.exp(-dist / (7.0 * k)) * a).astype(np.float32)
        L.side_l = (np.clip(-u, 0, 1) ** 3).astype(np.float32)
        L.side_r = (np.clip(u, 0, 1) ** 3).astype(np.float32)
        L.map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1)
        L.k = k
        return L

    def level(self, k):
        if k not in self.levels:
            L1 = self.levels[1]
            self.levels[k] = self._build(L1.rgb, L1.a, k)
        return self.levels[k]

    def shaded(self, L, rot=0.0, bright=1.0, spec=None, spec_w=0.32, spec_s=0.0,
               rim_l=(0, 0, 0), rim_r=(0, 0, 0), sweep=None, ambient=None, tint=None):
        """Premultiplied RGBA at the level's res with cylinder rotation + lighting."""
        if abs(rot) > 1e-4:
            th = np.clip(L.theta + rot, -np.pi / 2, np.pi / 2)
            mx = (L.axis + L.R[:, None] * np.sin(th)).astype(np.float32)
            rgb = cv2.remap(L.rgb, mx, L.map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rgb = L.rgb.copy()
        if ambient is not None:
            light = np.full(L.a.shape, ambient, np.float32)
            if sweep is not None:
                for (pos, width, power) in sweep:
                    light += power * np.exp(-((L.theta - pos) / width) ** 2)
            rgb = rgb * light[..., None]
        elif bright != 1.0:
            rgb = rgb * bright
        if tint is not None:
            rgb = rgb * np.array(tint, np.float32)
        if spec is not None and spec_s > 0:
            band = np.exp(-((L.theta - spec) / spec_w) ** 2) * spec_s
            rgb = 1 - (1 - np.clip(rgb, 0, 1)) * (1 - band[..., None])
        rl = np.array(rim_l, np.float32)
        rr = np.array(rim_r, np.float32)
        if rl.any() or rr.any():
            rgb = rgb + L.rim[..., None] * (L.side_l[..., None] * rl + L.side_r[..., None] * rr)
            rgb = rgb + L.rim[..., None] * 0.12 * (rl + rr)
        return np.dstack([rgb * L.a[..., None], L.a])

    def draw(self, canvas, cam, wx=540.0, wy=1000.0, scale=1.0, angle=0.0,
             reflect_floor=None, refl_strength=0.28, shadow=0.0, alpha=1.0, **shade_kw):
        """Draw can centred at world (wx,wy) (world height REF_H*scale). Returns screen bbox."""
        s_screen = cam.z * scale * self.REF_H / (self.bot - self.top)
        k = 2 if s_screen > 1.2 else 1
        L = self.level(k)
        src = self.shaded(L, **shade_kw)
        sx, sy = cam.p(wx, wy)
        ss = s_screen / k
        ang = np.deg2rad(angle)
        ca, sa = np.cos(ang) * ss, np.sin(ang) * ss
        axn, ayn = L.axis, L.cy
        M = np.float32([[ca, -sa, sx - (ca * axn - sa * ayn)],
                        [sa, ca, sy - (sa * axn + ca * ayn)]])
        hh, ww = src.shape[:2]
        corners = np.array([[0, 0, 1], [ww, 0, 1], [0, hh, 1], [ww, hh, 1]], np.float32) @ M.T
        x0 = int(max(0, np.floor(corners[:, 0].min()) - 2))
        y0 = int(max(0, np.floor(corners[:, 1].min()) - 2))
        x1 = int(min(W, np.ceil(corners[:, 0].max()) + 2))
        y1 = int(min(H, np.ceil(corners[:, 1].max()) + 2))
        if reflect_floor is not None and refl_strength > 0:
            fy_s = cam.p(0, reflect_floor)[1]
            ext = (corners[:, 1].max() - corners[:, 1].min()) * 0.5
            Mf = M.copy()
            Mf[1] = -M[1]
            Mf[1, 2] = 2 * fy_s - M[1, 2]
            rx0 = int(max(0, np.floor(corners[:, 0].min()) - 2))
            rx1 = int(min(W, np.ceil(corners[:, 0].max()) + 2))
            ry0 = int(max(0, fy_s - 6))
            ry1 = int(min(H, fy_s + ext))
            if ry1 > ry0 and rx1 > rx0:
                Mfr = Mf.copy()
                Mfr[0, 2] -= rx0
                Mfr[1, 2] -= ry0
                ref = cv2.warpAffine(src, Mfr, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                yy = np.arange(ry0, ry1, dtype=np.float32)
                fade = np.clip(1 - (yy - fy_s) / max(1.0, ext), 0, 1) ** 2.4
                ref *= (fade * refl_strength * alpha)[:, None, None]
                ref = cv2.GaussianBlur(ref, (0, 0), 2.0 * k)
                blend_over(canvas, ref[..., :3], ref[..., 3], rx0, ry0)
        if shadow > 0 and reflect_floor is not None:
            fy_s = cam.p(0, reflect_floor)[1]
            add_shadow(canvas, sx, fy_s, (corners[:, 0].max() - corners[:, 0].min()) * 0.5, cam.s(22 * scale), shadow)
        if x1 <= x0 or y1 <= y0:
            return None
        Mr = M.copy()
        Mr[0, 2] -= x0
        Mr[1, 2] -= y0
        lay = cv2.warpAffine(src, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        lay = np.clip(lay, 0, None)
        lay[..., 3] = np.clip(lay[..., 3], 0, 1)
        if alpha < 1:
            lay *= alpha
        blend_over(canvas, lay[..., :3], lay[..., 3], x0, y0)
        return (x0, y0, x1, y1)

    def world_bounds(self, wx=540.0, wy=1000.0, scale=1.0):
        h = self.REF_H * scale
        w = h * (2 * self.levels[1].R.max()) / (self.bot - self.top)
        return wx - w / 2, wy - h / 2, wx + w / 2, wy + h / 2


def add_shadow(canvas, cx, cy, rx, ry, strength):
    rx, ry = int(max(2, rx)), int(max(2, ry))
    x0, y0 = int(cx - rx * 1.5), int(cy - ry * 1.5)
    w, h = rx * 3, ry * 3
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (rx, ry), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), max(2, ry * 0.6)) * strength
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas[cy0:cy1, cx0:cx1] *= (1 - m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0])[..., None]


# =========================================================== liquid (metaball) renderer
def render_liquid(canvas, discs=(), lines=(), polys=(), color=(0.5, 0.9, 0.2), dark=None,
                  scale=0.5, sigma=5.0, thr=(0.38, 0.55), light=(-0.55, -0.75, 0.65),
                  bump=26.0, hue_fn=None, opacity=0.96, ripple=None, blur_out=0.0, spec_power=70.0):
    """Metaball liquid: splat primitives into a field, threshold, shade with
    normals from the field gradient -> glossy 3D liquid with highlights."""
    h, w = int(H * scale), int(W * scale)
    F = np.zeros((h, w), np.float32)
    S = 16
    for (x, y, r) in discs:
        rr = r * scale
        if rr < 0.35:
            continue
        cx, cy = x * scale, y * scale
        if cx < -rr - 40 or cy < -rr - 40 or cx > w + rr + 40 or cy > h + rr + 40:
            continue
        cv2.circle(F, (int(cx * S), int(cy * S)), int(rr * S), 1.0, -1, cv2.LINE_AA, 4)
    for (x0, y0, x1, y1, th) in lines:
        cv2.line(F, (int(x0 * scale * S), int(y0 * scale * S)), (int(x1 * scale * S), int(y1 * scale * S)),
                 1.0, max(1, int(th * scale)), cv2.LINE_AA, 4)
    for pts in polys:
        p = (np.asarray(pts, np.float32) * scale * S).astype(np.int32)
        cv2.fillPoly(F, [p], 1.0, cv2.LINE_AA, 4)
    if not F.any():
        return
    F = cv2.GaussianBlur(F, (0, 0), sigma)
    A = smoothstep_arr(thr[0], thr[1], F)
    if A.max() <= 0.001:
        return
    Hh = F.copy()
    if ripple is not None:
        Hh = Hh + ripple * A * 0.18
    gx = cv2.Sobel(Hh, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Hh, cv2.CV_32F, 0, 1, ksize=3)
    nx, ny = -gx * bump, -gy * bump
    nz = np.ones_like(nx)
    inv = 1 / np.sqrt(nx * nx + ny * ny + 1)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    Hv = L + np.array([0, 0, 1], np.float32)
    Hv /= np.linalg.norm(Hv)
    spec = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1) ** spec_power
    # second, softer light from lower right for liquid "glow"
    L2 = np.array([0.6, 0.5, 0.6], np.float32)
    L2 /= np.linalg.norm(L2)
    back = np.clip(nx * L2[0] + ny * L2[1] + nz * L2[2], 0, 1) ** 6
    thick = smoothstep_arr(thr[0], 1.0, F)
    if hue_fn is not None:
        base = hue_fn(h, w)
    else:
        base = np.broadcast_to(np.array(color, np.float32), (h, w, 3))
    dk = base * 0.35 if dark is None else np.broadcast_to(np.array(dark, np.float32), (h, w, 3))
    col = dk + (base - dk) * (0.45 + 0.55 * diff)[..., None]
    col = col * (1.15 - 0.35 * thick[..., None])          # thin edges are brighter (translucent)
    edge = (1 - nz)
    col += base * edge[..., None] * 0.25
    col += back[..., None] * base * 0.5
    col += spec[..., None] * 1.25
    a = A * opacity * (0.82 + 0.18 * thick)
    rgb_pm = col * a[..., None]
    lay = np.dstack([rgb_pm, a])
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
    if blur_out > 0:
        lay = cv2.GaussianBlur(lay, (0, 0), blur_out)
    blend_over(canvas, lay[..., :3], lay[..., 3])


def hsv2rgb_arr(hh, s, v):
    hh = (hh % 1.0) * 6
    i = np.floor(hh).astype(np.int32) % 6
    f = hh - np.floor(hh)
    p, q, tt = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, tt, v])
    g = np.choose(i, [tt, v, v, q, p, p])
    b = np.choose(i, [p, p, tt, v, v, q])
    return np.stack([r, g, b], -1).astype(np.float32)


# =========================================================== spheres (candy orbs)
@lru_cache(maxsize=16)
def sphere_sprite(color, n=256):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    X, Y = (x - c) / c, (y - c) / c
    r2 = X * X + Y * Y
    inside = r2 < 1
    Z = np.sqrt(np.clip(1 - r2, 0, 1))
    L = np.array([-0.45, -0.6, 0.66])
    L /= np.linalg.norm(L)
    diff = np.clip(X * L[0] + Y * L[1] + Z * L[2], 0, 1)
    col = np.array(color, np.float32)
    rgb = col * (0.28 + 0.85 * diff[..., None])
    # subsurface glow at the bottom
    rgb += col * (np.clip(Y, 0, 1) ** 2 * 0.45)[..., None]
    Hv = L + np.array([0, 0, 1.0])
    Hv /= np.linalg.norm(Hv)
    spec = np.clip(X * Hv[0] + Y * Hv[1] + Z * Hv[2], 0, 1) ** 90
    rgb += spec[..., None] * 1.4
    # rim reflection
    rim = (1 - Z) ** 3
    rgb += rim[..., None] * 0.35
    a = np.clip((1 - np.sqrt(r2)) * c / 1.5, 0, 1) * inside
    return np.dstack([np.clip(rgb, 0, 2) * a[..., None], a]).astype(np.float32)


def draw_sprite(canvas, spr, x, y, size, blur=0.0, alpha=1.0):
    n = int(size)
    if n < 2:
        return
    s = cv2.resize(spr, (n, n), interpolation=cv2.INTER_AREA if n < spr.shape[0] else cv2.INTER_LINEAR)
    if blur > 0.5:
        pad = int(blur * 3)
        s = np.pad(s, ((pad, pad), (pad, pad), (0, 0)))
        s = cv2.GaussianBlur(s, (0, 0), blur)
    else:
        pad = 0
    s = s * alpha
    blend_over(canvas, s[..., :3], s[..., 3], int(x - n / 2 - pad), int(y - n / 2 - pad))


# =========================================================== ice cubes
_ROT = lambda ax, ay, az: (
    np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]) @
    np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]) @
    np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]]))

_CUBE_V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
_CUBE_F = [(0, 1, 3, 2, (-1, 0, 0)), (4, 6, 7, 5, (1, 0, 0)), (0, 4, 5, 1, (0, -1, 0)),
           (2, 3, 7, 6, (0, 1, 0)), (0, 2, 6, 4, (0, 0, -1)), (1, 5, 7, 3, (0, 0, 1))]


@lru_cache(maxsize=2)
def ice_texture(n=256):
    t = fractal_noise(11, n, n, octaves=5)
    cr = fractal_noise(12, n, n, octaves=6)
    cracks = np.clip(1 - np.abs(cr - 0.5) * 22, 0, 1)
    return (t * 0.6 + cracks * 0.8).astype(np.float32)


def draw_ice(canvas, x, y, size, ax, ay, az, blur=0.0, alpha=1.0, tint=(0.80, 0.93, 1.0)):
    R = _ROT(ax, ay, az)
    v = _CUBE_V @ R.T
    half = size / 2
    pts2 = v[:, :2] * half * 0.62
    pad = int(size * 0.75 + blur * 3 + 6)
    n = 2 * pad
    ox, oy = int(x) - pad, int(y) - pad
    if ox > W or oy > H or ox + n < 0 or oy + n < 0:
        return
    loc = pts2 + pad
    # refraction: blur + brighten what is behind the cube silhouette
    hull = cv2.convexHull(loc.astype(np.float32)).reshape(-1, 2)
    mask = np.zeros((n, n), np.float32)
    cv2.fillPoly(mask, [(hull * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, size * 0.03))
    cx0, cy0 = max(0, ox), max(0, oy)
    cx1, cy1 = min(W, ox + n), min(H, oy + n)
    roi = canvas[cy0:cy1, cx0:cx1]
    msub = mask[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
    if roi.size == 0:
        return
    behind = cv2.GaussianBlur(roi, (0, 0), max(2.0, size * 0.05))
    behind = behind * 0.85 + 0.12 * np.array(tint, np.float32)
    k = (msub * 0.92 * alpha)[..., None]
    if blur > 0.5:
        k = cv2.GaussianBlur(k[..., 0], (0, 0), blur)[..., None]
    roi[:] = roi * (1 - k) + behind * k
    # faces
    lay = np.zeros((n, n, 4), np.float32)
    Lr = np.array([-0.4, -0.75, -0.5])
    Lr /= np.linalg.norm(Lr)
    tex = ice_texture()
    faces = []
    for (a, b, c, d, nrm) in _CUBE_F:
        nn = R @ np.array(nrm, np.float32)
        if nn[2] < 0:  # facing camera (camera looks along +z)
            faces.append((nn, [a, b, c, d]))
    for nn, idx in faces:
        poly = loc[idx]
        fm = np.zeros((n, n), np.float32)
        cv2.fillPoly(fm, [(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
        lit = 0.5 + 0.5 * max(0.0, float(-(nn @ Lr)))
        face_a = 0.26 + 0.24 * (-nn[2])
        tx = cv2.resize(tex, (n, n))
        col = np.array(tint, np.float32) * lit
        a = fm * face_a
        rgb = col[None, None, :] * a[..., None] + (tx * fm * 0.32)[..., None]
        lay[..., :3] = lay[..., :3] * (1 - a[..., None]) + rgb
        lay[..., 3] = lay[..., 3] * (1 - a) + a
    # bright bevel edges
    edges = np.zeros((n, n), np.float32)
    for nn, idx in faces:
        poly = loc[idx]
        cv2.polylines(edges, [(poly * 16).astype(np.int32)], True, 1.0, max(1, int(size * 0.025)), cv2.LINE_AA, 4)
    edges = cv2.GaussianBlur(edges, (0, 0), max(0.8, size * 0.012))
    lay[..., :3] += edges[..., None] * 0.55
    lay[..., 3] = np.clip(lay[..., 3] + edges * 0.35, 0, 1)
    # specular hot spot
    top = loc.mean(0) + np.array([-size * 0.12, -size * 0.14])
    g = cv2.resize(gauss_sprite(), (max(2, int(size * 0.5)), max(2, int(size * 0.32))))
    gx0, gy0 = int(top[0] - g.shape[1] / 2), int(top[1] - g.shape[0] / 2)
    gx1, gy1 = gx0 + g.shape[1], gy0 + g.shape[0]
    if gx0 >= 0 and gy0 >= 0 and gx1 <= n and gy1 <= n:
        lay[gy0:gy1, gx0:gx1, :3] += (g * 0.55 * mask[gy0:gy1, gx0:gx1])[..., None]
    if blur > 0.5:
        lay = cv2.GaussianBlur(lay, (0, 0), blur)
    lay *= alpha
    blend_over(canvas, lay[..., :3], lay[..., 3], ox, oy)


# =========================================================== lightning
def bolt_points(p0, p1, rng, disp=0.22, depth=7):
    pts = [np.array(p0, np.float32), np.array(p1, np.float32)]
    d = disp * np.linalg.norm(pts[1] - pts[0])
    for _ in range(depth):
        new = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            m = (a + b) / 2
            v = b - a
            nrm = np.array([-v[1], v[0]], np.float32)
            nrm /= (np.linalg.norm(nrm) + 1e-6)
            m = m + nrm * rng.uniform(-d, d)
            new += [m, b]
        pts = new
        d *= 0.55
    return np.array(pts, np.float32)


def draw_bolt(layer_core, layer_glow, p0, p1, rng, width=3, branches=3, glow_scale=0.5):
    pts = bolt_points(p0, p1, rng)
    segs = [(pts, width)]
    for _ in range(branches):
        i = rng.integers(len(pts) // 5, len(pts) * 4 // 5)
        a = pts[i]
        dirv = pts[min(len(pts) - 1, i + 8)] - a
        ang = np.arctan2(dirv[1], dirv[0]) + rng.uniform(-0.9, 0.9)
        L = np.linalg.norm(p1 - np.array(p0)) * rng.uniform(0.15, 0.35)
        b = a + np.array([np.cos(ang), np.sin(ang)]) * L
        segs.append((bolt_points(a, b, rng, 0.25, 5), max(1, width - 1)))
    for p, wd in segs:
        P = (p * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer_core, [P], False, 1.0, wd, cv2.LINE_AA, 4)
        Pg = (p * glow_scale * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer_glow, [Pg], False, 1.0, max(1, int(wd * 2 * glow_scale)), cv2.LINE_AA, 4)


def composite_electric(canvas, core, glow_half, color=(0.35, 0.75, 1.0), intensity=1.0):
    g1 = cv2.GaussianBlur(glow_half, (0, 0), 4)
    g2 = cv2.GaussianBlur(glow_half, (0, 0), 16)
    g = cv2.resize(g1 * 1.2 + g2 * 1.6, (W, H), interpolation=cv2.INTER_LINEAR)
    c = cv2.GaussianBlur(core, (0, 0), 0.8)
    col = np.array(color, np.float32)
    canvas += (g[..., None] * col + c[..., None] * (0.6 + 0.4 * col)) * intensity


# =========================================================== text
def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), int(size))


@lru_cache(maxsize=64)
def text_layer(text, fontname, size, tracking=0.0, pad=40):
    """Return premultiplied RGBA (white) float32 for the text, tightly padded."""
    f = font(fontname, size)
    if tracking == 0:
        bb = f.getbbox(text)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), text, font=f, fill=255)
    else:
        adv = [f.getlength(ch) for ch in text]
        tr = tracking * size
        total = sum(adv) + tr * (len(text) - 1)
        bb = f.getbbox(text)
        h = bb[3] - bb[1]
        im = Image.new("L", (int(total) + 2 * pad, h + 2 * pad), 0)
        d = ImageDraw.Draw(im)
        x = pad
        for ch, a in zip(text, adv):
            d.text((x, pad - bb[1]), ch, font=f, fill=255)
            x += a + tr
    a = np.asarray(im).astype(np.float32) / 255.0
    return a


def fit_size(text, fontname, max_w, max_size, tracking=0.0):
    size = max_size
    while size > 10:
        f = font(fontname, size)
        w = sum(f.getlength(c) for c in text) + tracking * size * (len(text) - 1) if tracking else f.getlength(text)
        if w <= max_w:
            return size
        size -= 2
    return size


def draw_text(canvas, text, fontname, size, cx, cy, color=(1, 1, 1), alpha=1.0, scale=1.0,
              blur=0.0, tracking=0.0, glow=None, glow_r=18, glow_s=0.8, shadow=0.0, grad=None,
              reveal=None, skew=0.0):
    """Draw text centred at (cx,cy). grad: (top_color, bottom_color). reveal: 0..1 left->right wipe."""
    if alpha <= 0.002:
        return
    a = text_layer(text, fontname, int(size), tracking)
    h, w = a.shape
    if scale != 1.0 or skew != 0.0:
        M = np.float32([[scale, -skew * scale, 0], [0, scale, 0]])
        nw, nh = int(w * scale + abs(skew) * h * scale) + 2, int(h * scale) + 2
        M[0, 2] = (nw - w * scale) / 2 + skew * scale * h / 2
        a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR)
        h, w = a.shape
    if reveal is not None:
        xs = np.arange(w, dtype=np.float32) / w
        edge = reveal * 1.15 - 0.075
        a = a * np.clip((edge - xs) / 0.075 + 0.5, 0, 1)[None, :]
    if blur > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    if shadow > 0:
        sh = cv2.GaussianBlur(a, (0, 0), 10) * shadow * alpha
        cx0, cy0 = max(0, x0 + 4), max(0, y0 + 8)
        cx1, cy1 = min(W, x0 + 4 + w), min(H, y0 + 8 + h)
        if cx1 > cx0 and cy1 > cy0:
            canvas[cy0:cy1, cx0:cx1] *= (1 - sh[cy0 - y0 - 8:cy1 - y0 - 8, cx0 - x0 - 4:cx1 - x0 - 4])[..., None]
    if glow is not None:
        g = cv2.GaussianBlur(a, (0, 0), glow_r) * glow_s * alpha
        blend_add(canvas, g[..., None] * np.array(glow, np.float32), x0, y0)
    if grad is not None:
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        col = np.array(grad[0], np.float32) * (1 - t) + np.array(grad[1], np.float32) * t
        col = np.broadcast_to(col, (h, w, 3))
    else:
        col = np.broadcast_to(np.array(color, np.float32), (h, w, 3))
    aa = a * alpha
    blend_over(canvas, col * aa[..., None], aa, x0, y0)
    return (x0, y0, w, h, a)
