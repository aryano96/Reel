"""Synthesis toolkit shared by every reel's soundtrack.

Every drum, synth, splash, ice clink, fizz, can-crack, zap and whoosh is
generated from scratch with numpy/scipy - nothing is sampled or licensed, so
the audio is 100% original and free to use commercially."""
import os
import numpy as np
from scipy import signal

SR = 48000
rng = np.random.default_rng(7)


def reseed(seed):
    rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state


# ----------------------------------------------------------------- utilities
def ta(n):
    return np.arange(n) / SR


def ns(n):
    return rng.standard_normal(n)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def sos_bp(lo, hi, order=2):
    return signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")


def sos_lp(f, order=2):
    return signal.butter(order, f, btype="low", fs=SR, output="sos")


def sos_hp(f, order=2):
    return signal.butter(order, f, btype="high", fs=SR, output="sos")


def filt(sos, x):
    return signal.sosfilt(sos, x, axis=0)


def stereo(x, pan=0.0):
    if x.ndim == 2:
        return x
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1) * np.sqrt(2)


class Bus:
    def __init__(self, n):
        self.n = n
        self.b = np.zeros((n, 2))

    def add(self, x, t0, gain=1.0, pan=0.0):
        x = stereo(np.asarray(x, dtype=np.float64), pan) * gain
        i0 = int(round(t0 * SR))
        if i0 < 0:
            x = x[-i0:]
            i0 = 0
        i1 = min(self.n, i0 + len(x))
        if i1 > i0:
            self.b[i0:i1] += x[: i1 - i0]


def env_ar(n, a=0.003, r=0.05, dur=None):
    t = ta(n)
    dur = n / SR if dur is None else dur
    e = np.clip(t / max(a, 1e-4), 0, 1)
    e *= np.clip((dur - t) / max(r, 1e-4), 0, 1)
    return e


def saw(freq, n, phase0=None):
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    dt = f / SR
    p0 = rng.random() if phase0 is None else phase0
    ph = (p0 + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1
    m = ph > 1 - dt
    x = (ph[m] - 1) / dt[m]
    y[m] -= x * x + x + x + 1
    return y


def sweep_noise(n, f_curve, width=0.35, bands=18, lo=150, hi=14000, src=None):
    """Noise whose spectral centre follows f_curve (Hz per sample)."""
    src = ns(n) if src is None else src
    centers = np.geomspace(lo, hi, bands)
    out = np.zeros(n)
    lf = np.log(np.maximum(f_curve, 20))
    for c in centers:
        b = filt(sos_bp(c / 1.25, min(c * 1.25, SR / 2 - 100)), src)
        w = np.exp(-((lf - np.log(c)) / width) ** 2)
        out += b * w
    return out


# ------------------------------------------------------------------ reverb/delay
def make_ir(rt=1.8, length=2.6, pre=0.018):
    n = int(length * SR)
    t = ta(n)
    tau = rt / 6.9
    ir = np.zeros((n, 2))
    for c in range(2):
        x = ns(n) * np.exp(-t / tau)
        # darker tail
        x = 0.6 * filt(sos_lp(7000), x) + 0.4 * filt(sos_lp(2500), x)
        ir[:, c] = x
    ir[: int(pre * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
    return ir


def reverb(x, ir):
    y = np.zeros_like(x)
    for c in range(2):
        y[:, c] = signal.fftconvolve(x[:, c], ir[:, c])[: len(x)]
    return y


def pingpong(x, d=0.375, fb=0.42, taps=5):
    y = np.zeros_like(x)
    k = int(d * SR)
    g = 1.0
    for i in range(1, taps + 1):
        g *= fb
        sh = k * i
        if sh >= len(x):
            break
        src = x[:-sh].mean(1)
        ch = i % 2
        y[sh:, ch] += src * g
    return y


# ------------------------------------------------------------------ drums
def kick(punch=1.0):
    n = int(0.5 * SR)
    t = ta(n)
    f = 46 + 120 * np.exp(-t / 0.032) + 60 * np.exp(-t / 0.005)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.30)
    click = filt(sos_hp(1800), ns(n)) * np.exp(-t / 0.0025) * 0.35 * punch
    x = body + click
    return np.tanh(1.8 * x) / np.tanh(1.8)


def clap():
    n = int(0.45 * SR)
    t = ta(n)
    e = np.zeros(n)
    for k, d in enumerate([0.0, 0.010, 0.021]):
        tt = np.clip(t - d, 0, None)
        e += (t >= d) * np.exp(-tt / 0.0045) * (0.8 + 0.1 * k)
    tt = np.clip(t - 0.03, 0, None)
    e += (t >= 0.03) * np.exp(-tt / 0.11)
    x = filt(sos_bp(850, 3200), ns(n)) * e
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05) * 0.35
    return (x + body) * 0.9


def snare(bright=1.0):
    n = int(0.3 * SR)
    t = ta(n)
    body = np.sin(2 * np.pi * (200 + 40 * np.exp(-t / 0.01)) * t) * np.exp(-t / 0.05)
    nz = filt(sos_hp(1500), ns(n)) * np.exp(-t / 0.09) * bright
    return 0.55 * body + 0.7 * nz


def hat(open_=False):
    n = int((0.4 if open_ else 0.08) * SR)
    t = ta(n)
    x = filt(sos_hp(7500, 4), ns(n))
    x += 0.5 * filt(sos_bp(9000, 13000), ns(n))
    return x * np.exp(-t / (0.22 if open_ else 0.022)) * 0.6


def crash(dur=2.2):
    n = int(dur * SR)
    t = ta(n)
    x = filt(sos_hp(3200), ns(n)) * np.exp(-t / 0.7)
    x += 0.4 * filt(sos_bp(5000, 9000), ns(n)) * np.exp(-t / 0.3)
    att = np.clip(t / 0.002, 0, 1)
    return x * att * 0.55


# ------------------------------------------------------------------ synths
def sub808(freq, dur, glide_to=None):
    n = int((dur + 0.06) * SR)
    t = ta(n)
    f = np.full(n, freq) * (1 + 0.25 * np.exp(-t / 0.02))
    if glide_to is not None:
        g = np.clip((t - dur * 0.6) / (dur * 0.4), 0, 1)
        f = f * (glide_to / freq) ** (g ** 2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph)
    x = np.tanh(2.4 * x) / np.tanh(2.4)
    e = np.clip(t / 0.004, 0, 1) * np.exp(-t / 1.6) * np.clip((dur + 0.06 - t) / 0.06, 0, 1)
    return x * e


def supersaw(freq, n, voices=7, spread_cents=22):
    det = np.linspace(-1, 1, voices)
    out = np.zeros((n, 2))
    for d in det:
        f = freq * 2 ** (d * spread_cents / 1200)
        s = saw(f, n)
        pan = d * 0.8
        out += stereo(s, pan)
    return out / voices


def chord_stab(midis, dur, bright=1.0):
    n = int((dur + 0.15) * SR)
    t = ta(n)
    raw = sum(supersaw(mtof(m), n) for m in midis) / len(midis)
    hi = filt(sos_lp(min(9000, 3500 + 5000 * bright)), raw)
    lo = filt(sos_lp(1400), raw)
    e_hi = np.exp(-t / 0.07)
    e_lo = np.exp(-t / 0.45)
    rel = np.clip((dur + 0.15 - t) / 0.15, 0, 1) * np.clip(t / 0.003, 0, 1)
    return (hi * e_hi[:, None] * 0.9 + lo * e_lo[:, None] * 0.6) * rel[:, None]


def pad(midis, dur, cutoff=1300, att=0.25, rel=0.5):
    n = int((dur + rel) * SR)
    t = ta(n)
    raw = sum(supersaw(mtof(m), n, voices=5, spread_cents=14) for m in midis) / len(midis)
    x = filt(sos_lp(cutoff, 2), raw)
    e = np.clip(t / att, 0, 1) * np.clip((dur + rel - t) / rel, 0, 1)
    return x * e[:, None]


def pluck(m, dur=0.2):
    n = int((dur + 0.1) * SR)
    t = ta(n)
    f = mtof(m)
    s = saw(f, n) * 0.6 + np.sign(np.sin(2 * np.pi * f * t + rng.random() * 6)) * 0.3
    hi = filt(sos_lp(6000), s) * np.exp(-t / 0.04)
    lo = filt(sos_lp(1800), s) * np.exp(-t / 0.18)
    return (hi + lo) * np.clip(t / 0.002, 0, 1) * np.clip((dur + 0.1 - t) / 0.1, 0, 1)


def bell(m, dur=1.4):
    n = int(dur * SR)
    t = ta(n)
    f = mtof(m)
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.5)
    x += 0.35 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / 0.25)
    x += 0.18 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t / 0.12)
    return x * np.clip(t / 0.002, 0, 1)


# ------------------------------------------------------------------ cinematic FX
def impact(size=1.0, dur=2.4):
    n = int(dur * SR)
    t = ta(n)
    f = 28 + 42 * np.exp(-t / 0.12)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.7 * size))
    boom = np.tanh(2.0 * boom)
    nz = filt(sos_lp(3500), ns(n)) * np.exp(-t / 0.12)
    crack = filt(sos_hp(2000), ns(n)) * np.exp(-t / 0.012)
    return (boom * 0.9 + nz * 0.45 * size + crack * 0.35) * np.clip(t / 0.001, 0, 1)


def low_hit(dur=1.2):
    n = int(dur * SR)
    t = ta(n)
    f = 38 + 60 * np.exp(-t / 0.05)
    x = np.tanh(2.5 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t / 0.35)
    x += filt(sos_lp(1800), ns(n)) * np.exp(-t / 0.05) * 0.4
    return x


def whoosh(dur=0.5, lo=300, hi=6000, peak=0.6, pan_from=-0.7, pan_to=0.7):
    n = int(dur * SR)
    t = ta(n) / dur
    shape = np.where(t < peak, (t / peak) ** 1.6, ((1 - t) / (1 - peak)) ** 1.3)
    fc = lo * (hi / lo) ** shape
    x = sweep_noise(n, fc, width=0.45)
    x *= shape ** 1.4
    pan = pan_from + (pan_to - pan_from) * t
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1) * np.sqrt(2) * 0.9


def riser(dur=2.0):
    n = int(dur * SR)
    t = ta(n) / dur
    fc = 250 * (11000 / 250) ** (t ** 1.4)
    x = sweep_noise(n, fc, width=0.4) * (t ** 1.8)
    fs = 180 * (1400 / 180) ** (t ** 1.5)
    tone = saw(fs, n) * 0.25 + saw(fs * 1.007, n) * 0.25
    tone = filt(sos_lp(5000), tone) * t ** 2.2
    return x * 0.8 + tone * 0.5


def reverse_cymbal(dur=1.0):
    c = crash(dur)
    c = c[::-1].copy()
    t = ta(len(c)) / dur
    return c * (t ** 2) * 1.4


def splash(size=1.0, dur=1.6):
    n = int(dur * SR)
    t = ta(n)
    burst = filt(sos_lp(6000), ns(n)) * np.clip(t / 0.004, 0, 1) * np.exp(-t / 0.22)
    body = filt(sos_bp(250, 1400), ns(n)) * np.exp(-t / 0.10)
    out = np.zeros((n, 2))
    out += stereo(burst * 0.8 + body * 0.9)
    # droplets: tiny rising-pitch bubbles scattered in time
    for _ in range(int(80 * size)):
        t0 = 0.03 + rng.exponential(0.35)
        if t0 > dur - 0.05:
            continue
        m = int(0.035 * SR)
        tt = ta(m)
        f0 = rng.uniform(650, 1700)
        f = f0 * (1 + 1.2 * tt / 0.035)
        d = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.009)
        a = rng.uniform(0.05, 0.28) * np.exp(-t0 / 0.6)
        i0 = int(t0 * SR)
        out[i0:i0 + m] += stereo(d * a, rng.uniform(-0.9, 0.9))[: n - i0]
    return out * size


def fizz(dur=1.5, rate=2600):
    n = int(dur * SR)
    t = ta(n)
    imp = np.zeros(n)
    k = int(rate * dur)
    idx = rng.integers(0, n, k)
    imp[idx] = rng.uniform(-1, 1, k) * rng.uniform(0.2, 1, k) ** 2
    pops = filt(sos_bp(3500, 11000), imp) * 2.0
    hiss = filt(sos_hp(5000), ns(n)) * 0.08
    e = np.clip(t / 0.05, 0, 1) * np.exp(-t / (dur * 0.6))
    L = (pops + hiss) * e
    imp2 = np.zeros(n)
    idx = rng.integers(0, n, k)
    imp2[idx] = rng.uniform(-1, 1, k) * rng.uniform(0.2, 1, k) ** 2
    R = (filt(sos_bp(3500, 11000), imp2) * 2.0 + hiss[::-1]) * e
    return np.stack([L, R], 1)


def can_open():
    n = int(0.9 * SR)
    t = ta(n)
    click = filt(sos_hp(1200), ns(n)) * np.exp(-t / 0.0015) * 1.2
    ping = (np.sin(2 * np.pi * 2350 * t) + 0.6 * np.sin(2 * np.pi * 4120 * t)) * np.exp(-t / 0.02) * 0.35
    snap_t = np.clip(t - 0.045, 0, None)
    snap = (t > 0.045) * filt(sos_hp(2500), ns(n)) * np.exp(-snap_t / 0.004) * 0.9
    hiss_t = np.clip(t - 0.05, 0, None)
    hiss = (t > 0.05) * filt(sos_bp(2500, 9500), ns(n)) * np.clip(hiss_t / 0.01, 0, 1) * np.exp(-hiss_t / 0.28)
    return click + ping + snap + hiss * 0.85


def ice_clink(f0=None, amp=1.0):
    f0 = rng.uniform(2300, 4300) if f0 is None else f0
    n = int(0.35 * SR)
    t = ta(n)
    x = np.zeros(n)
    for r, a, d in [(1, 1, 0.16), (2.76, 0.6, 0.08), (5.40, 0.38, 0.045), (8.93, 0.22, 0.025)]:
        x += a * np.sin(2 * np.pi * f0 * r * t + rng.random() * 6) * np.exp(-t / d)
    x += filt(sos_hp(3000), ns(n)) * np.exp(-t / 0.002) * 0.8
    knock = np.sin(2 * np.pi * 320 * t) * np.exp(-t / 0.025) * 0.4
    return (x * 0.5 + knock) * amp


def zap(dur=0.45, big=False):
    n = int(dur * SR)
    t = ta(n)
    f = 70 + np.cumsum(rng.standard_normal(n)) * 0.6
    f = np.clip(f, 40, 220)
    buzz = np.sign(saw(f, n)) * 0.5 + saw(f * 2.01, n) * 0.5
    flick = np.repeat(rng.random(n // 600 + 1) > 0.35, 600)[:n].astype(float)
    imp = np.zeros(n)
    k = int(3000 * dur)
    imp[rng.integers(0, n, k)] = rng.uniform(-1, 1, k)
    crack = filt(sos_hp(2200), imp) * 1.5
    pew_f = 2600 * np.exp(-t / 0.05) + 180
    pew = np.sin(2 * np.pi * np.cumsum(pew_f) / SR) * np.exp(-t / 0.08)
    e = np.clip(t / 0.002, 0, 1) * np.exp(-t / (dur * 0.45))
    x = (filt(sos_bp(150, 6000), buzz) * flick * 0.6 + crack * 0.7 + pew * 0.5) * e
    if big:
        rumble = filt(sos_lp(200), ns(n)) * np.exp(-t / 0.5) * 3.0
        x = x * 1.3 + rumble
    return np.tanh(1.6 * x)


def thunder(dur=2.0):
    n = int(dur * SR)
    t = ta(n)
    crack = filt(sos_hp(800), ns(n)) * np.exp(-t / 0.04) * 1.2
    rumble = filt(sos_lp(240), ns(n)) * np.exp(-t / 0.7) * 2.5
    rumble *= 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3.3 * t))
    return np.tanh(crack + rumble)


def bloop(f0=None):
    f0 = rng.uniform(320, 620) if f0 is None else f0
    n = int(0.12 * SR)
    t = ta(n)
    f = f0 * (1 + 1.6 * (t / 0.12) ** 0.7)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.035) * np.clip(t / 0.003, 0, 1)


def wind(dur):
    n = int(dur * SR)
    t = ta(n)
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t + 1.0)
    x = filt(sos_bp(300, 1800), ns(n)) * lfo
    x += filt(sos_bp(2000, 5000), ns(n)) * 0.25 * (1 - lfo + 0.4)
    e = np.clip(t / 0.4, 0, 1) * np.clip((dur - t) / 0.5, 0, 1)
    return x * e


def gurgle(dur=0.6):
    out = np.zeros(int(dur * SR))
    for _ in range(26):
        t0 = rng.uniform(0, dur - 0.13)
        b = bloop(rng.uniform(180, 520)) * rng.uniform(0.2, 0.6)
        i0 = int(t0 * SR)
        out[i0:i0 + len(b)] += b[: len(out) - i0]
    return out


def sparkle(dur=1.2, count=14, base=84):
    out = np.zeros((int((dur + 1.2) * SR), 2))
    scale = [0, 3, 5, 7, 10, 12, 15]
    for i in range(count):
        t0 = i * dur / count + rng.uniform(0, 0.03)
        m = base + scale[rng.integers(0, len(scale))]
        b = bell(m, 1.0) * rng.uniform(0.15, 0.35)
        i0 = int(t0 * SR)
        out[i0:i0 + len(b)] += stereo(b, rng.uniform(-0.8, 0.8))[: len(out) - i0]
    return out


