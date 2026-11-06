"""Reel 2 soundtrack: original 160 BPM half-time trap in G minor + sound design.

Everything is synthesized from scratch (synth.py + the few extra voices below):
no samples, no licensed audio. Output: output/build2/audio_mix.wav
"""
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d

import timeline2 as TL
import synth
from synth import (SR, Bus, ta, ns, mtof, filt, sos_bp, sos_lp, sos_hp, stereo, saw, kick, clap, snare, hat, crash,
                   sub808, chord_stab, pad, pluck, bell, impact, low_hit, whoosh, riser, reverse_cymbal, splash, fizz,
                   can_open, ice_clink, zap, thunder, bloop, wind, gurgle, sparkle, make_ir, reverb, pingpong)

synth.reseed(2025)
rng = synth.rng
N = int(SR * TL.DURATION)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "build2")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ extra voices
def scratch(dur=0.55, rate=7.0):
    """Vinyl scratch: a formant-y source read back and forth at a wobbling speed."""
    n_src = int(1.2 * SR)
    t = ta(n_src)
    src = saw(220 * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)), n_src) * 0.6 + ns(n_src) * 0.25
    src = filt(sos_bp(500, 3200), src)
    n = int(dur * SR)
    tt = ta(n)
    pos = 0.18 * (1 - np.cos(2 * np.pi * rate * tt)) + 0.05 * np.sin(2 * np.pi * rate * 2.1 * tt)
    out = np.interp(pos * SR + 0.2 * SR, np.arange(n_src), src)
    chop = (np.sin(2 * np.pi * rate * 2 * tt) > -0.3).astype(float)
    chop = signal.lfilter([0.15], [1, -0.85], chop)
    return out * chop * np.clip(tt / 0.005, 0, 1) * np.clip((dur - tt) / 0.03, 0, 1)


def crackle(dur, rate=60):
    n = int(dur * SR)
    out = np.zeros(n)
    k = int(rate * dur)
    for t0 in rng.uniform(0, dur - 0.02, k):
        m = int(rng.uniform(0.002, 0.012) * SR)
        i0 = int(t0 * SR)
        burst = ns(m) * np.exp(-ta(m) / (m / SR / 3)) * rng.uniform(0.2, 1.0) ** 2
        out[i0:i0 + m] += burst[: n - i0]
    return filt(sos_bp(900, 7000), out)


def fire_roar(dur=1.6):
    n = int(dur * SR)
    t = ta(n)
    body = filt(sos_bp(120, 1400), ns(n)) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.3 * t + 1))
    rumble = filt(sos_lp(160), ns(n)) * 2.2
    e = np.clip(t / 0.08, 0, 1) * np.exp(-t / (dur * 0.5))
    return np.tanh((body + rumble) * e * 1.4) + crackle(dur, 90) * e * 1.4


# ================================================================== arrangement
drums, bass, music, sfx, send = Bus(N), Bus(N), Bus(N), Bus(N), Bus(N)
kick_times = []
CH = {
    "Gm": [55, 58, 62, 67, 70],
    "Eb": [51, 55, 58, 63, 67],
    "Bb": [53, 58, 62, 65, 70],
    "F": [53, 57, 60, 65, 69],
}
ROOT = {"Gm": 31, "Eb": 39, "Bb": 34, "F": 29}
PROG = ["Gm", "Eb", "Bb", "F"]


def T(bar, s16, base=TL.DROP):
    return base + bar * TL.BAR + s16 * TL.S16


# ---------- INTRO (0 - 3.0, 2 bars)
music.add(pad(CH["Gm"], 2.88, cutoff=620, att=0.9, rel=0.1), 0.0, 0.55)
music.add(pad([m + 12 for m in CH["Gm"]], 1.5, cutoff=2400, att=1.2, rel=0.1), 1.4, 0.24)
bass.add(sub808(mtof(31), 2.88), 0.0, 0.33)
for t0, _ in TL.INTRO_WORDS:
    sfx.add(whoosh(0.3, 400, 5000, peak=0.85, pan_from=-0.5, pan_to=0.3), t0 - 0.25, 0.35)
    drums.add(low_hit(), t0, 0.95)
    send.add(low_hit(), t0, 0.35)
    drums.add(clap(), t0, 0.35)
    send.add(clap(), t0, 0.5)
sfx.add(riser(1.95), 1.0, 0.42)
roll = [2.25 + i * TL.S16 * 2 for i in range(2)] + [2.4375 + i * TL.S16 for i in range(4)] + [2.8125 + i * TL.S16 / 2 for i in range(2)]
for i, t0 in enumerate(roll):
    drums.add(snare(0.6 + 0.4 * i / len(roll)), t0, 0.2 + 0.45 * i / len(roll))
sfx.add(reverse_cymbal(1.1), TL.DROP - 1.1, 0.55)

# ---------- MAIN GROOVE (3.0 - 24.0) : 14 bars
KA = [0, 7, 10]
KB = [0, 3, 10, 14]
for bar in range(14):
    chord = PROG[bar % 4]
    kicks = KA if bar % 2 == 0 else KB
    for s in kicks:
        drums.add(kick(), T(bar, s), 1.0)
        kick_times.append(T(bar, s))
    for i, s in enumerate(kicks):
        nxt = kicks[i + 1] if i + 1 < len(kicks) else 16
        glide = mtof(ROOT[PROG[(bar + 1) % 4]]) if (bar % 2 == 1 and i == len(kicks) - 1) else None
        bass.add(sub808(mtof(ROOT[chord]), (nxt - s) * TL.S16, glide), T(bar, s), 0.85)
    drums.add(clap(), T(bar, 8), 0.7)                      # half-time backbeat
    send.add(clap(), T(bar, 8), 0.38)
    drums.add(snare(0.5), T(bar, 8), 0.25)
    for s in range(0, 16, 2):
        drums.add(hat(), T(bar, s), 0.30 if s % 4 == 0 else 0.2, pan=0.25)
    if bar % 2 == 1:   # trap hat roll into every scene change
        for k in range(6):
            drums.add(hat(), T(bar, 12 + k * 0.6667), 0.12 + 0.035 * k, pan=0.25)
        drums.add(hat(), T(bar, 15.5), 0.3, pan=0.25)
    else:
        for s in (5, 11, 13):
            drums.add(hat(), T(bar, s), 0.13, pan=-0.2)
    drums.add(hat(True), T(bar, 6), 0.18, pan=-0.15)
    hits = [0, 3, 6, 10, 12] if bar % 2 == 0 else [0, 3, 6, 8, 11, 14]
    for s in hits:
        st = chord_stab(CH[chord], 0.14 if s != 12 else 0.32)
        music.add(st, T(bar, s), 0.42)
        send.add(st, T(bar, s), 0.16)
    music.add(pad(CH[chord], TL.BAR, cutoff=1500, att=0.05, rel=0.1), T(bar, 0), 0.30)
    if bar in (2, 3, 6, 7, 10, 11):          # arp layer in scenes 2, 4, 6
        tones = sorted(CH[chord])[1:] + [CH[chord][1] + 12]
        pattern = [0, 1, 2, 3, 4, 3, 2, 1]
        for s in range(16):
            m = tones[pattern[s % 8] % len(tones)] + 12
            p = pluck(m, 0.08)
            music.add(p, T(bar, s), 0.15, pan=-0.35 if s % 2 else 0.35)
            send.add(p, T(bar, s), 0.07)

for i, t0 in enumerate(TL.DRINK_STARTS):
    big = i == 0
    sfx.add(impact(1.2 if big else 0.7), t0, 1.0 if big else 0.6)
    send.add(impact(1.0), t0, 0.35 if big else 0.2)
    drums.add(crash(), t0, 0.5)
    send.add(crash(), t0, 0.2)
    c = t0 + TL.CLOSE
    sfx.add(whoosh(0.3, 300, 5000, peak=0.8, pan_from=0.5 if i % 2 else -0.5, pan_to=-0.5 if i % 2 else 0.5), c - 0.27, 0.42)
    drums.add(low_hit(0.6), c, 0.42)

# ---------- S1 Crush Pineapple : golden splash
sfx.add(splash(1.4, 2.0), 3.0, 0.95)
send.add(splash(1.0, 2.0), 3.0, 0.25)
# ---------- can cracks (S1, S3, S6)
for t0 in TL.CAN_OPENS:
    sfx.add(can_open(), t0 - 0.04, 0.85)
    send.add(can_open(), t0 - 0.04, 0.2)
    sfx.add(fizz(1.1), t0 + 0.06, 0.45)
# transition 1 -> 2: neon wave
sfx.add(whoosh(0.4, 180, 3500, peak=0.9, pan_from=0, pan_to=0), 5.6, 0.6)
sfx.add(splash(1.1, 1.3), 5.92, 0.75)
# ---------- S2 Big Troppa Punch : neon wave, 90s scratch
sfx.add(gurgle(0.6), 6.05, 0.5)
sfx.add(scratch(0.5, 7.5), 7.5, 0.55, pan=-0.2)
send.add(scratch(0.5, 7.5), 7.5, 0.15)
sfx.add(scratch(0.3, 9.0), 8.25, 0.4, pan=0.2)
sfx.add(whoosh(0.3, 500, 9000, peak=0.75, pan_from=0.8, pan_to=-0.9), 8.64, 0.75)
# ---------- S3 Crush Peach : fizz bubbles
sfx.add(whoosh(0.3, 600, 8000, peak=0.2, pan_from=0.9, pan_to=0), 9.0, 0.5)
for k in range(16):
    t0 = 9.05 + k * 0.17 + rng.uniform(0, 0.05)
    sfx.add(bloop(rng.uniform(380, 820)), t0, 0.22, pan=rng.uniform(-0.7, 0.7))
sfx.add(fizz(2.4, 3200), 9.1, 0.32)
sfx.add(riser(0.4), 11.6, 0.5)
sfx.add(whoosh(0.32, 400, 10000, peak=0.95, pan_from=0, pan_to=0), 11.66, 0.55)
# ---------- S4 Fanta Strawberry : crimson smoke + embers
sp = sparkle(0.8, 10, base=86)
sfx.add(sp, 12.05, 0.5)
send.add(sp, 12.05, 0.3)
sfx.add(wind(3.0) * 0.8, 12.0, 0.12)
sfx.add(whoosh(0.5, 120, 1800, peak=0.85, pan_from=-0.6, pan_to=0.6), 14.52, 0.7)
drums.add(low_hit(1.0), 14.97, 0.4)
# ---------- S5 Fanta Pineapple : lightning
sfx.add(thunder(2.0), 15.0, 0.75)
send.add(thunder(2.0), 15.0, 0.2)
for t0 in TL.STRIKES_S5:
    big = t0 == 15.0
    z = zap(0.55 if big else 0.32, big)
    sfx.add(z, t0, 0.55 if big else 0.36, pan=rng.uniform(-0.5, 0.5))
    send.add(z, t0, 0.12)
hum_n = int(2.6 * SR)
hum_t = ta(hum_n)
hum = filt(sos_bp(80, 900), np.sign(np.sin(2 * np.pi * 60 * hum_t)) * 0.5 + saw(120, hum_n) * 0.3)
hum *= np.clip(hum_t / 0.3, 0, 1) * np.clip((2.6 - hum_t) / 0.3, 0, 1)
sfx.add(hum, 15.05, 0.08)
sfx.add(riser(0.36), 17.62, 0.5)
sfx.add(whoosh(0.3, 400, 10000, peak=0.95, pan_from=0, pan_to=0), 17.68, 0.55)
# ---------- S6 Fanta Mango : ice + mist
sfx.add(whoosh(0.3, 600, 8000, peak=0.2, pan_from=0, pan_to=0), 18.0, 0.45)
sfx.add(wind(3.0), 18.0, 0.10)
for t0 in TL.ICE_HITS_S6:
    sfx.add(ice_clink(amp=0.8), t0, 0.5, pan=rng.uniform(-0.6, 0.6))
    sfx.add(ice_clink(amp=0.5), t0 + rng.uniform(0.03, 0.07), 0.3, pan=rng.uniform(-0.6, 0.6))
    send.add(ice_clink(), t0, 0.15)
sfx.add(whoosh(0.32, 300, 9000, peak=0.9, pan_from=0, pan_to=0), 20.64, 0.75)
# ---------- S7 Fanta Crimson Sour Cherry : dark fire
sfx.add(fire_roar(2.2), 21.0, 0.55)
send.add(fire_roar(2.2), 21.0, 0.15)
sfx.add(crackle(2.8, 40), 21.1, 0.35)
sfx.add(fire_roar(0.9), 23.55, 0.6)
sfx.add(whoosh(0.4, 200, 7000, peak=0.9, pan_from=0, pan_to=0), 23.6, 0.6)

# ---------- END CARD (24.0 - 28.5)
E = TL.END_START
sfx.add(impact(1.1), E, 0.9)
send.add(impact(1.0), E, 0.3)
drums.add(crash(3.0), E, 0.5)
send.add(crash(3.0), E, 0.25)
st = chord_stab(CH["Gm"], 0.9)
music.add(st, E, 0.5)
send.add(st, E, 0.3)
music.add(pad(CH["Gm"], 1.5, cutoff=1800, att=0.05, rel=0.2), E, 0.32)
music.add(pad(CH["Eb"], 0.75, cutoff=1800, att=0.05, rel=0.2), E + 1.5, 0.32)
music.add(pad(CH["F"], 0.75, cutoff=1800, att=0.05, rel=0.2), E + 2.25, 0.32)
bass.add(sub808(mtof(31), 1.45), E, 0.8)
bass.add(sub808(mtof(39), 0.72), E + 1.5, 0.75)
bass.add(sub808(mtof(29), 0.72, glide_to=mtof(31)), E + 2.25, 0.75)
for bar in range(2):
    for s in (0, 10):
        drums.add(kick(), T(bar, s, E), 0.85)
        kick_times.append(T(bar, s, E))
    drums.add(clap(), T(bar, 8, E), 0.5)
    send.add(clap(), T(bar, 8, E), 0.3)
    for s in range(0, 16, 2):
        drums.add(hat(), T(bar, s, E), 0.17, pan=0.25)
    seq = [CH["Gm"]] * 16 if bar == 0 else [CH["Eb"]] * 8 + [CH["F"]] * 8
    for s in range(16):
        tones = sorted(seq[s])[1:] + [seq[s][1] + 12]
        m = tones[[0, 1, 2, 3, 4, 3, 2, 1][s % 8] % len(tones)] + 12
        p = pluck(m, 0.08)
        music.add(p, T(bar, s, E), 0.13, pan=-0.35 if s % 2 else 0.35)
        send.add(p, T(bar, s, E), 0.07)
drums.add(low_hit(), TL.END_NAME, 0.8)
sfx.add(whoosh(0.35, 500, 7000, peak=0.6), TL.END_SUB - 0.15, 0.35)
sfx.add(whoosh(0.3, 800, 9000, peak=0.6, pan_from=0.5, pan_to=-0.5), TL.END_ADDR - 0.1, 0.25)
music.add(bell(91, 1.5), TL.END_CTA, 0.22)
music.add(bell(98, 1.5), TL.END_CTA + 0.06, 0.15)
send.add(bell(91, 1.5), TL.END_CTA, 0.2)
for t0 in TL.END_SHINES:
    sp = sparkle(0.6, 9, base=86)
    sfx.add(sp, t0, 0.4)
    send.add(sp, t0, 0.25)
FIN = TL.END_FINAL
drums.add(kick(), FIN, 0.9)
kick_times.append(FIN)
sfx.add(impact(1.0, 1.5), FIN, 0.55)
st = chord_stab(CH["Gm"] + [74], 1.3)
music.add(st, FIN, 0.55)
send.add(st, FIN, 0.45)
music.add(pad(CH["Gm"], 1.3, cutoff=2200, att=0.02, rel=0.2), FIN, 0.35)
bass.add(sub808(mtof(31), 1.3), FIN, 0.85)
drums.add(crash(1.5), FIN, 0.45)

# ================================================================== mix (same chain as reel 1)
duck = np.ones(N)
for kt in kick_times:
    i0 = int(kt * SR)
    m = min(N - i0, int(0.3 * SR))
    duck[i0:i0 + m] = np.minimum(duck[i0:i0 + m], 1 - 0.55 * np.exp(-ta(m) / 0.08))
duck = signal.lfilter([0.02], [1, -0.98], duck)
music.b *= duck[:, None]
send_ducked = send.b * (0.5 + 0.5 * duck[:, None])
wet = reverb(send_ducked, make_ir(1.8))
wet += pingpong(music.b * 1.2, d=0.28125)     # dotted-8th at 160 BPM
mix = drums.b * 0.9 + bass.b * 0.7 + music.b * 3.4 + sfx.b * 1.0 + wet * 0.6
mix = filt(sos_hp(28), mix)
mix += filt(sos_hp(8000), mix) * 0.12
t = ta(N)
mix *= np.clip((TL.DURATION - t) / 0.75, 0, 1)[:, None]
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 2.0
look = int(0.003 * SR)
peak = maximum_filter1d(np.abs(mix).max(1), size=2 * look + 1)
g = np.minimum(1.0, 0.89 / (peak + 1e-9))
alpha = np.exp(-1 / (0.08 * SR))
g_s = np.minimum(signal.lfilter([1 - alpha], [1, -alpha], g), g)
mix *= g_s[:, None]
mix = np.clip(np.tanh(mix * 1.05) / np.tanh(1.05) * 0.78, -0.84, 0.84)
wavfile.write(os.path.join(OUT, "audio_mix.wav"), SR, (mix * 32767).astype(np.int16))
for name, bus, gn in [("drums", drums, 0.9), ("bass", bass, 0.7), ("music", music, 3.4), ("sfx", sfx, 1.0)]:
    r = np.sqrt(((bus.b * gn) ** 2).mean())
    print(f"{name:6s} rms={20 * np.log10(r + 1e-12):6.1f} dB")
print("wrote", os.path.join(OUT, "audio_mix.wav"), f"{TL.DURATION}s")
