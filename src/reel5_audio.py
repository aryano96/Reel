"""Reel 5 soundtrack - a new sonic identity: 128 BPM breakbeat techno in E minor.

Every voice here is new for this reel (only generic helpers - noise, filters, reverb, the Bus
mixer - come from synth.py): an analogue-style techno kick, a syncopated breakbeat, a detuned
Reese bass, arpeggiated pulse leads, engine-like acceleration sweeps, metallic resonant hits,
electrical crackle, mechanical charge-ups, Doppler fly-bys and low-frequency shockwaves.
All synthesized from scratch - no samples, no licensed audio.

Output: output/build5/audio_mix.wav
"""
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d

import timeline5 as TL
import synth
from synth import SR, Bus, ta, ns, mtof, filt, sos_bp, sos_lp, sos_hp, stereo, saw, make_ir, reverb

synth.reseed(5150)
rng = synth.rng
N = int(SR * TL.DURATION)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "build5")
os.makedirs(OUT, exist_ok=True)
B, S16 = TL.BEAT, TL.S16


# ================================================================ voices
def kick909(drive=1.0):
    n = int(0.42 * SR)
    t = ta(n)
    f = 42 + 210 * np.exp(-t / 0.018) + 40 * np.exp(-t / 0.11)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.24)
    click = filt(sos_bp(2500, 9000), ns(n)) * np.exp(-t / 0.0018) * 0.45
    x = body + click
    return np.tanh(2.2 * drive * x) / np.tanh(2.2 * drive)


def snare_break(tone=1.0):
    n = int(0.28 * SR)
    t = ta(n)
    body = np.sin(2 * np.pi * (180 + 60 * np.exp(-t / 0.012)) * t) * np.exp(-t / 0.06) * 0.7
    nz = filt(sos_bp(1800, 9000), ns(n)) * np.exp(-t / 0.075) * tone
    ring = np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.03) * 0.2
    return np.tanh((body + nz + ring) * 1.4)


def ghost():
    n = int(0.09 * SR)
    t = ta(n)
    return filt(sos_bp(2500, 8000), ns(n)) * np.exp(-t / 0.018) * 0.5


def tick(open_=False):
    n = int((0.25 if open_ else 0.05) * SR)
    t = ta(n)
    x = filt(sos_hp(9000, 4), ns(n)) + 0.4 * np.sin(2 * np.pi * 7400 * t) * np.exp(-t / 0.004)
    return x * np.exp(-t / (0.12 if open_ else 0.012)) * 0.5


def ride():
    n = int(0.5 * SR)
    t = ta(n)
    x = sum(np.sin(2 * np.pi * f * t + rng.random() * 6) for f in (3120, 4710, 5930, 7480)) / 4
    x += filt(sos_hp(6000), ns(n)) * 0.3
    return x * np.exp(-t / 0.22) * 0.3


def reese(freq, dur, cutoff=420, drive=1.6):
    n = int((dur + 0.03) * SR)
    t = ta(n)
    x = (saw(freq * 0.997, n) + saw(freq * 1.006, n) + 0.5 * saw(freq / 2, n)) / 2.5
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.9 * t + rng.random() * 6)
    x = filt(sos_lp(cutoff * 0.8, 4), x) * (1 - lfo) + filt(sos_lp(cutoff * 1.4, 4), x) * lfo   # LFO-swept filter
    e = np.clip(t / 0.006, 0, 1) * np.clip((dur + 0.03 - t) / 0.03, 0, 1)
    return np.tanh(drive * x * e)


def subline(freq, dur):
    n = int((dur + 0.02) * SR)
    t = ta(n)
    return np.sin(2 * np.pi * freq * t) * np.clip(t / 0.004, 0, 1) * np.clip((dur + 0.02 - t) / 0.02, 0, 1)


def pulse_lead(m, dur=0.1, pw=0.3, cutoff=3500):
    n = int((dur + 0.06) * SR)
    t = ta(n)
    ph = (mtof(m) * t + rng.random()) % 1.0
    x = np.where(ph < pw, 1.0, -1.0) * 0.6 + saw(mtof(m) * 1.004, n) * 0.4
    x = filt(sos_lp(cutoff, 2), x)
    return x * np.clip(t / 0.002, 0, 1) * np.exp(-t / 0.07) * np.clip((dur + 0.06 - t) / 0.06, 0, 1)


def dark_pad(ms, dur, cutoff=900, att=0.6, rel=0.6):
    n = int((dur + rel) * SR)
    t = ta(n)
    out = np.zeros((n, 2))
    for m in ms:
        for d, pan in ((-9, -0.7), (0, 0), (9, 0.7)):
            out += stereo(saw(mtof(m) * 2 ** (d / 1200), n), pan)
    out = filt(sos_lp(cutoff, 2), out / (len(ms) * 3))
    e = np.clip(t / att, 0, 1) * np.clip((dur + rel - t) / rel, 0, 1)
    return out * e[:, None]


def engine_sweep(dur, f0=55, f1=420, gear_shifts=2):
    """Engine-like acceleration: harmonic buzz rising in pitch with gear shifts + intake noise."""
    n = int(dur * SR)
    t = ta(n) / dur
    seg = np.minimum((t * (gear_shifts + 1)).astype(int), gear_shifts)
    local = t * (gear_shifts + 1) - seg
    f = f0 * (f1 / f0) ** ((seg + local ** 0.8) / (gear_shifts + 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(ph * k) / k ** 0.8 for k in range(1, 9))
    x = np.tanh(1.8 * x) * (0.6 + 0.4 * np.sin(ph * 0.5) ** 2)
    x = filt(sos_bp(60, 4000), x)
    intake = filt(sos_bp(800, 5000), ns(n)) * (t ** 1.5) * 0.35
    e = np.clip(t / 0.05, 0, 1) * (0.3 + 0.7 * t) * np.clip((1 - t) / 0.03, 0, 1)
    return (x * 0.7 + intake) * e


def doppler(dur=0.7, f=210, pan_from=-1, pan_to=1):
    """Fly-by: pitch falls through the pass, level peaks at the centre, pans across."""
    n = int(dur * SR)
    t = ta(n) / dur
    c = 0.5
    ff = f * (1 + 0.35 * np.tanh((c - t) * 6))
    ph = 2 * np.pi * np.cumsum(ff) / SR
    x = np.tanh(2 * sum(np.sin(ph * k) / k for k in range(1, 6))) + filt(sos_bp(500, 6000), ns(n)) * 0.6
    amp = np.exp(-((t - c) / 0.18) ** 2)
    x = filt(sos_lp(9000), x) * amp
    pan = pan_from + (pan_to - pan_from) * t
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1) * np.sqrt(2)


def metal_hit(size=1.0, dur=1.8, base=180):
    """Resonant metallic impact: inharmonic partials + low thump + transient."""
    n = int(dur * SR)
    t = ta(n)
    parts = [(1.0, 1.0, 0.9), (2.32, 0.6, 0.6), (3.76, 0.45, 0.4), (5.41, 0.3, 0.25), (7.12, 0.2, 0.15)]
    x = sum(a * np.sin(2 * np.pi * base * r * t + rng.random() * 6) * np.exp(-t / (d * size)) for r, a, d in parts)
    thump = np.sin(2 * np.pi * (45 + 80 * np.exp(-t / 0.04)) * t) * np.exp(-t / 0.3) * 1.2
    trans = filt(sos_hp(2000), ns(n)) * np.exp(-t / 0.006) * 0.8
    return np.tanh((x * 0.5 + thump + trans) * 1.2)


def sub_drop(dur=1.6, f0=90, f1=28):
    n = int(dur * SR)
    t = ta(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.18)
    return np.tanh(2.4 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t / 0.7) * np.clip(t / 0.003, 0, 1)


def shock_lfo(dur=1.2):
    """Low-frequency shockwave: sub thump + air pressure rumble."""
    n = int(dur * SR)
    t = ta(n)
    x = np.sin(2 * np.pi * (34 + 30 * np.exp(-t / 0.08)) * t) * np.exp(-t / 0.4)
    air = filt(sos_lp(300), ns(n)) * np.exp(-t / 0.25) * 0.8
    return np.tanh(2 * (x + air))


def crackle(dur, rate=120, bright=1.0):
    n = int(dur * SR)
    out = np.zeros(n)
    for t0 in rng.uniform(0, dur - 0.01, int(rate * dur)):
        m = int(rng.uniform(0.001, 0.006) * SR)
        i0 = int(t0 * SR)
        out[i0:i0 + m] += (ns(m) * np.exp(-ta(m) / (m / SR / 4)) * rng.uniform(0.2, 1.0) ** 2)[: n - i0]
    return filt(sos_bp(1500, 12000 * bright), out) * 1.6


def zap_arc(dur=0.4):
    n = int(dur * SR)
    t = ta(n)
    buzz = np.sign(np.sin(2 * np.pi * (90 + 40 * np.sin(2 * np.pi * 13 * t)) * t))
    x = filt(sos_bp(200, 7000), buzz) * 0.5 + crackle(dur, 400) * 0.8
    return np.tanh(x * np.exp(-t / (dur * 0.5)) * 1.5)


def charge_up(dur=0.8, f0=180, f1=2400):
    """Mechanical charging whine: rising sine + servo ratchet clicks."""
    n = int(dur * SR)
    t = ta(n) / dur
    f = f0 * (f1 / f0) ** (t ** 1.6)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.4 + np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR) * 0.2
    clicks = np.zeros(n)
    k, pos = 0, 0.0
    while pos < 1.0:
        i = int(pos * n)
        m = min(n - i, int(0.004 * SR))
        clicks[i:i + m] += filt(sos_hp(2500), ns(m)) * np.exp(-ta(m) / 0.0008)
        k += 1
        pos += 0.12 * (0.82 ** k) + 0.01
    return (x * t ** 0.7 + clicks * 0.7) * np.clip((1 - t) / 0.02, 0, 1)


def power_down(dur=0.6):
    n = int(dur * SR)
    t = ta(n) / dur
    f = 900 * (40 / 900) ** (t ** 0.7)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.4 * np.sign(np.sin(2 * np.pi * np.cumsum(f * 0.5) / SR))
    return filt(sos_lp(4000), x) * (1 - t) ** 1.2 * 0.6


def warp_rise(dur=0.5):
    """Hyperspace rise: noise + saw sweeping up, ending abruptly."""
    n = int(dur * SR)
    t = ta(n) / dur
    fc = 200 * (12000 / 200) ** (t ** 1.3)
    x = synth.sweep_noise(n, fc, width=0.35)
    tone = saw(80 * (1600 / 80) ** (t ** 1.5), n) * 0.3
    return (x + filt(sos_lp(8000), tone)) * t ** 1.6


def glitch_burst(dur=0.25):
    n = int(dur * SR)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        m = int(rng.uniform(0.008, 0.03) * SR)
        f = rng.choice([180, 360, 720, 1440, 2880])
        tt = ta(min(m, n - pos))
        out[pos:pos + len(tt)] = np.sign(np.sin(2 * np.pi * f * tt)) * rng.uniform(0.3, 1.0)
        pos += m
    return filt(sos_bp(150, 9000), out) * 0.5


def holo_scan(dur=0.5, f0=1200, f1=300):
    n = int(dur * SR)
    t = ta(n) / dur
    f = f0 * (f1 / f0) ** t
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 40 * t * dur)))
    return filt(sos_bp(200, 6000), x) * np.sin(np.pi * t) * 0.5


def iris_mech(dur=0.3):
    n = int(dur * SR)
    t = ta(n) / dur
    x = filt(sos_bp(300, 3000), ns(n)) * 0.4 * np.sin(np.pi * t)
    for p in np.linspace(0.05, 0.95, 8):
        i = int(p * n)
        m = min(n - i, int(0.006 * SR))
        x[i:i + m] += filt(sos_hp(1500), ns(m)) * np.exp(-ta(m) / 0.001) * 1.2
    clunk = np.sin(2 * np.pi * 90 * ta(n)) * np.exp(-np.clip(ta(n) - dur * 0.92, 0, None) / 0.05) * (t > 0.92)
    return x + clunk * 0.9


def shimmer(dur=1.2, base=88):
    n = int(dur * SR)
    t = ta(n)
    out = np.zeros((n, 2))
    for k in range(9):
        m = base + [0, 3, 7, 10, 12, 15, 19][k % 7]
        t0 = k * dur / 12
        i0 = int(t0 * SR)
        tt = ta(n - i0)
        x = np.sin(2 * np.pi * mtof(m) * tt) * np.exp(-tt / 0.25) * 0.2
        out[i0:] += stereo(x, rng.uniform(-0.8, 0.8))
    return out


def ignition(dur=1.6):
    n = int(dur * SR)
    t = ta(n)
    crank = np.zeros(n)
    for k in range(5):
        i0 = int(k * 0.07 * SR)
        m = int(0.05 * SR)
        crank[i0:i0 + m] += filt(sos_bp(80, 800), ns(m)) * np.exp(-ta(m) / 0.02)
    roar = engine_sweep(dur - 0.3, 40, 160, 1)
    out = crank * 0.8
    out[int(0.3 * SR):int(0.3 * SR) + len(roar)] += roar * 1.2
    return out


# ================================================================ arrangement
drums, bass, music, sfx, send = Bus(N), Bus(N), Bus(N), Bus(N), Bus(N)
kick_times = []
ROOTS = [40, 40, 36, 38]          # E, E, C, D  (E minor) - one root per bar, 4-bar cycle
CHORDS = [[52, 55, 59, 64], [52, 55, 59, 62], [48, 52, 55, 60], [50, 54, 57, 62]]
ARP = [0, 2, 1, 3, 2, 1, 3, 1]


def at(bar, s16, base=TL.DROP):
    return base + bar * TL.BAR + s16 * S16


# ---------- INTRO (0 - DROP): "the energy awakens"
music.add(dark_pad([40, 47, 52], TL.DROP + 0.2, cutoff=500, att=1.2, rel=0.3), 0.0, 0.5)
for k in range(int(TL.DROP / B)):                      # faint heartbeat pulse that intensifies
    tk = k * B
    hb = subline(mtof(28), 0.12)
    music.add(hb * np.exp(-ta(len(hb)) / 0.05), tk, 0.25 + 0.12 * k)
    sfx.add(pulse_lead(76, 0.05, cutoff=1800 + 600 * k), tk + B / 2, 0.05 + 0.03 * k, pan=0.4 if k % 2 else -0.4)
sfx.add(engine_sweep(TL.DROP - 0.3, 45, 380, 2), 0.3, 0.28)            # acceleration build
sfx.add(warp_rise(1.0), TL.DROP - 1.0, 0.4)
sfx.add(charge_up(1.2, 150, 2600), TL.DROP - 1.2, 0.32)
for k in range(8):                                                     # snare/tick roll into the drop
    tk = TL.DROP - B * 2 + k * B / 4
    drums.add(snare_break(0.7), tk, 0.12 + 0.07 * k)
for k, w in enumerate(np.arange(TL.TEXT_IN, TL.TEXT_DONE, S16 / 2)):   # headline decode ticks
    sfx.add(tick(), w, 0.12, pan=rng.uniform(-0.5, 0.5))
sfx.add(crackle(0.6, 160), 1.8, 0.25)

# ---------- MAIN SECTION: 13 bars of breakbeat techno
KICK_PAT = [0, 6, 8, 10]                  # syncopated breakbeat kick
SNARE_PAT = [4, 12]
GHOSTS = [7, 9, 15]
for bar in range(TL.N_DRINKS):
    r = ROOTS[bar % 4]
    ch = CHORDS[bar % 4]
    build = bar >= 10                     # last three bars build toward the climax
    for s in KICK_PAT + ([14] if bar % 2 else []):
        drums.add(kick909(1.15 if build else 1.0), at(bar, s), 1.0)
        kick_times.append(at(bar, s))
    for s in SNARE_PAT:
        drums.add(snare_break(), at(bar, s), 0.62)
        send.add(snare_break(), at(bar, s), 0.22)
    for s in GHOSTS:
        drums.add(ghost(), at(bar, s), 0.35)
    for s in range(16):
        drums.add(tick(s % 4 == 2), at(bar, s), (0.22 if s % 2 else 0.12) * (1.3 if build else 1.0), pan=0.3 if s % 2 else -0.15)
    if bar % 2 == 0:
        for s in (2, 10):
            drums.add(ride(), at(bar, s), 0.18, pan=-0.35)
    # Reese bass: offbeat 16ths riding over a sub line
    bass.add(subline(mtof(r - 12), TL.BAR - 0.02), at(bar, 0), 0.38)
    for s in (2, 3, 6, 7, 10, 11, 14):
        bass.add(reese(mtof(r), S16 * 0.95, cutoff=380 + (260 if build else 0) + 140 * (s % 4 == 3)), at(bar, s), 0.36)
    # pulse arp (16ths), brighter in the build
    for s in range(16):
        m = ch[ARP[s % 8]] + 12
        p = pulse_lead(m, 0.09, pw=0.25 + 0.15 * (s % 2), cutoff=2600 + 2200 * (bar / 12))
        music.add(p, at(bar, s), 0.1 + (0.05 if build else 0), pan=-0.4 if s % 2 else 0.4)
        send.add(p, at(bar, s), 0.05)
    music.add(dark_pad(ch, TL.BAR, cutoff=800 + 70 * bar, att=0.05, rel=0.2), at(bar, 0), 0.16)

# every reveal: sub drop + metallic hit on the downbeat
for i, s in enumerate(TL.S):
    big = i == 0
    sfx.add(sub_drop(1.6 if big else 1.0), s, 0.9 if big else 0.55)
    sfx.add(metal_hit(1.2 if big else 0.7, base=150 + 15 * (i % 5)), s, 0.75 if big else 0.42)
    send.add(metal_hit(1.0, base=150 + 15 * (i % 5)), s, 0.25)
    sfx.add(holo_scan(0.3, 1600, 600), s + 0.1, 0.16)           # HUD lock-on chirp
    sfx.add(tick(), s + TL.HERO2, 0.3)
    sfx.add(shock_lfo(0.5), s + TL.HERO2, 0.18)                  # focus-pull thump into the hero shot

S = TL.S
EXIT = TL.EXIT
# ---------- per-edition sound design (entrance / exit matched to the picture)
# 1 Summer: EM shockwave
sfx.add(shock_lfo(1.4), S[0], 0.8)
sfx.add(crackle(1.0, 120), S[0] + 0.05, 0.35)
sfx.add(warp_rise(0.32), S[1] - 0.32, 0.6)                       # light-speed exit
# 2 Ice: tunnel deceleration (Doppler) ; iris close
sfx.add(doppler(0.6, 260, 1, -1), S[1] - 0.12, 0.5)
sfx.add(engine_sweep(0.5, 400, 90, 0), S[1], 0.3)
sfx.add(iris_mech(0.28), S[2] - 0.28, 0.55)
# 3 Yellow: metallic shockwave ; CRT collapse
sfx.add(metal_hit(1.4, 2.2, base=130), S[2], 0.55)
sfx.add(shock_lfo(1.0), S[2], 0.5)
sfx.add(power_down(0.28), S[3] - 0.28, 0.6)
# 4 Spring: hologram scan ; particle dissolve
sfx.add(holo_scan(0.55, 1800, 500), S[3] + 0.12, 0.45)
sfx.add(shimmer(0.5, 93), S[4] - 0.3, 0.35)
sfx.add(crackle(0.3, 300, 1.2), S[4] - 0.28, 0.3)
# 5 Peach: particle assembly (rising shimmer) ; glitch
sfx.add(shimmer(0.6, 81), S[4], 0.45)
sfx.add(charge_up(0.55, 300, 1800), S[4], 0.25)
sfx.add(glitch_burst(0.28), S[5] - 0.28, 0.6)
# 6 Red: headlight sweep (engine pass) ; light-trail rush
sfx.add(doppler(0.8, 180, -1, 1), S[5] - 0.1, 0.55)
sfx.add(engine_sweep(0.6, 70, 520, 1), S[6] - 0.6, 0.45)
# 7 Sea Blue: EM pulse ; energy surge
sfx.add(zap_arc(0.35), S[6], 0.4)
sfx.add(shock_lfo(0.9), S[6], 0.55)
sfx.add(charge_up(0.3, 400, 3200), S[7] - 0.3, 0.45)
# 8 Amber: vortex collapse ; defocus (filtered drop-out)
sfx.add(warp_rise(0.4)[::-1].copy(), S[7], 0.45)
sfx.add(engine_sweep(0.3, 300, 60, 0), S[8] - 0.3, 0.3)
# 9 Pink: refocus ; prism flash
sfx.add(holo_scan(0.4, 400, 2000), S[8], 0.3)
sfx.add(shimmer(0.4, 96), S[9] - 0.3, 0.35)
# 10 Lilac: arc strike + crackling chamber ; arc whiteout
sfx.add(zap_arc(0.5), S[9], 0.6)
sfx.add(zap_arc(0.3), S[9] + 0.12, 0.45)
sfx.add(crackle(TL.BAR, 70), S[9], 0.3)
sfx.add(zap_arc(0.3), S[10] - 0.3, 0.55)
# 11 Winter: laser scan ; overtake (Doppler)
sfx.add(holo_scan(0.5, 900, 2600), S[10] + 0.05, 0.4)
sfx.add(doppler(0.55, 240, 1, -1), S[11] - 0.4, 0.6)
# 12 Coconut: drop-in + hover hum ; power-down blackout
sfx.add(doppler(0.5, 300, 0, 0)[:, :], S[11] - 0.2, 0.35)
hum_n = int((TL.BAR - 0.5) * SR)
hum_t = ta(hum_n)
hum = filt(sos_bp(60, 600), saw(55, hum_n) * 0.5 + np.sin(2 * np.pi * 110 * hum_t) * 0.3) * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * hum_t))
sfx.add(hum * np.clip(hum_t / 0.2, 0, 1), S[11] + 0.4, 0.14)
sfx.add(power_down(0.5), S[12] - 0.5, 0.6)
# 13 Apple: ignition ; launch
sfx.add(ignition(1.4), S[12], 0.5)
sfx.add(engine_sweep(0.5, 120, 900, 1), TL.END_START - 0.5, 0.55)
sfx.add(warp_rise(0.35), TL.END_START - 0.35, 0.5)

# ---------- END CARD
E = TL.END_START
sfx.add(sub_drop(2.0, 100, 26), E, 0.95)
sfx.add(metal_hit(1.6, 2.8, base=110), E, 0.7)
send.add(metal_hit(1.4, 2.8, base=110), E, 0.35)
sfx.add(shock_lfo(1.6), E, 0.6)
music.add(dark_pad([40, 47, 52, 55, 59], TL.DURATION - E, cutoff=1400, att=0.1, rel=0.6), E, 0.3)
bass.add(subline(mtof(28), 2 * TL.BAR), E, 0.42)
for k in range(8):                                                    # half-time groove under the card
    tk = E + k * B
    if k % 2 == 0:
        drums.add(kick909(), tk, 0.85)
        kick_times.append(tk)
    else:
        drums.add(snare_break(0.8), tk, 0.4)
        send.add(snare_break(0.8), tk, 0.18)
    for j in range(2):
        drums.add(tick(), tk + j * B / 2, 0.14, pan=0.3)
    m = [64, 67, 71, 74][k % 4] + 12
    p = pulse_lead(m, 0.2, cutoff=4000)
    music.add(p, tk, 0.12, pan=-0.3 if k % 2 else 0.3)
    send.add(p, tk, 0.12)
sfx.add(holo_scan(0.4, 2000, 800), TL.END_LOGO, 0.35)                # logo scan-in
for j, ta_ in enumerate(TL.END_ADDR):                                 # address decode ticks
    for k in range(6):
        sfx.add(tick(), ta_ + k * S16 / 2, 0.12, pan=-0.3 + 0.6 * j)
sfx.add(metal_hit(0.8, 1.2, base=220), TL.END_OFFER, 0.45)           # offer slams in
sfx.add(shock_lfo(0.6), TL.END_OFFER, 0.35)
sfx.add(charge_up(0.4, 500, 3000), TL.END_CTA - 0.3, 0.3)            # CTA panel powers on
sfx.add(shimmer(0.8, 88), TL.END_SHINE, 0.3)
F = TL.END_FINAL
sfx.add(sub_drop(1.0, 80, 30), F, 0.9)
sfx.add(metal_hit(1.3, 1.0, base=140), F, 0.6)
send.add(metal_hit(1.3, 1.0, base=140), F, 0.4)
drums.add(kick909(1.2), F, 1.0)
kick_times.append(F)
sfx.add(crackle(0.5, 100), F, 0.25)

# ================================================================ mix
duck = np.ones(N)
for kt in kick_times:
    i0 = int(kt * SR)
    m = min(N - i0, int(0.25 * SR))
    duck[i0:i0 + m] = np.minimum(duck[i0:i0 + m], 1 - 0.6 * np.exp(-ta(m) / 0.07))
duck = signal.lfilter([0.03], [1, -0.97], duck)
music.b *= duck[:, None]
bass.b *= (0.35 + 0.65 * duck)[:, None]
ir = make_ir(1.4, 2.0)
wet = reverb(send.b, ir)
# short slap delay on the leads (dotted 16th) - a tighter space than earlier reels
d = int(S16 * 1.5 * SR)
slap = np.zeros_like(music.b)
slap[d:, 0] = music.b[:-d, 1] * 0.35
slap[d:, 1] = music.b[:-d, 0] * 0.35
mix = drums.b * 0.9 + bass.b * 0.6 + music.b * 4.2 + sfx.b * 1.2 + wet * 0.6 + slap * 1.5
mix = filt(sos_hp(25), mix)
mix += filt(sos_bp(900, 5000), mix) * 0.55          # presence lift (leads, crackle, metal)
mix += filt(sos_hp(6000), mix) * 0.6                # air shelf
t = ta(N)
mix *= np.clip((TL.DURATION - t) / 0.55, 0, 1)[:, None]
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 2.0
look = int(0.003 * SR)
peak = maximum_filter1d(np.abs(mix).max(1), size=2 * look + 1)
g = np.minimum(1.0, 0.89 / (peak + 1e-9))
alpha = np.exp(-1 / (0.07 * SR))
g_s = np.minimum(signal.lfilter([1 - alpha], [1, -alpha], g), g)
mix *= g_s[:, None]
mix = np.clip(np.tanh(mix * 1.05) / np.tanh(1.05) * 0.68, -0.74, 0.74)
wavfile.write(os.path.join(OUT, "audio_mix.wav"), SR, (mix * 32767).astype(np.int16))
for name, bus, gn in [("drums", drums, 0.9), ("bass", bass, 0.6), ("music", music, 4.2), ("sfx", sfx, 1.2)]:
    print(f"{name:6s} rms={20 * np.log10(np.sqrt(((bus.b * gn) ** 2).mean()) + 1e-12):6.1f} dB")
print("wrote", os.path.join(OUT, "audio_mix.wav"), f"{TL.DURATION}s")
