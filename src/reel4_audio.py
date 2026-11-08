"""Reel 4 soundtrack: original 160 BPM half-time trap in D minor + scene-matched sound design.
Everything is synthesized from scratch (synth.py + two small voices below) - no samples, no licensed audio.

Each drink lasts 5 beats; every reveal gets an impact + crash on its beat, every close-up cut a whoosh.
Output: output/build4/audio_mix.wav
"""
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d

import timeline4 as TL
import synth
from synth import (SR, Bus, ta, ns, mtof, filt, sos_bp, sos_lp, sos_hp, kick, clap, snare, hat, crash, sub808,
                   chord_stab, pad, pluck, bell, impact, low_hit, whoosh, riser, reverse_cymbal, splash, fizz, can_open,
                   ice_clink, zap, thunder, bloop, wind, gurgle, sparkle, make_ir, reverb, pingpong)

synth.reseed(4044)
rng = synth.rng
N = int(SR * TL.DURATION)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "build4")
os.makedirs(OUT, exist_ok=True)


def crackle(dur, rate=60):
    n = int(dur * SR)
    out = np.zeros(n)
    for t0 in rng.uniform(0, dur - 0.02, int(rate * dur)):
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


drums, bass, music, sfx, send = Bus(N), Bus(N), Bus(N), Bus(N), Bus(N)
kick_times = []
CH = {"Dm": [50, 53, 57, 62, 65], "Bb": [46, 50, 53, 58, 62], "F": [53, 57, 60, 65, 69], "C": [48, 52, 55, 60, 64]}
ROOT = {"Dm": 26, "Bb": 34, "F": 29, "C": 36}
PROG = ["Dm", "Bb", "F", "C"]


def T(bar, s16, base=TL.DROP):
    return base + bar * TL.BAR + s16 * TL.S16


# ---------- INTRO (0 - 3.0)
music.add(pad(CH["Dm"], 2.88, cutoff=620, att=0.9, rel=0.1), 0.0, 0.55)
music.add(pad([m + 12 for m in CH["Dm"]], 1.5, cutoff=2400, att=1.2, rel=0.1), 1.4, 0.24)
bass.add(sub808(mtof(26), 2.88), 0.0, 0.33)
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

# ---------- MAIN GROOVE (3.0 - END_START)
n_bars = int(np.ceil((TL.END_START - TL.DROP) / TL.BAR - 1e-6))
KA, KB = [0, 7, 10], [0, 3, 10, 14]
for bar in range(n_bars):
    chord = PROG[bar % 4]
    bar_t = T(bar, 0)
    steps = int(min(16, round((TL.END_START - bar_t) / TL.S16)))
    kicks = [s for s in (KA if bar % 2 == 0 else KB) if s < steps]
    for s in kicks:
        drums.add(kick(), T(bar, s), 1.0)
        kick_times.append(T(bar, s))
    for i, s in enumerate(kicks):
        nxt = kicks[i + 1] if i + 1 < len(kicks) else steps
        bass.add(sub808(mtof(ROOT[chord]), (nxt - s) * TL.S16), T(bar, s), 0.85)
    if steps > 8:
        drums.add(clap(), T(bar, 8), 0.7)
        send.add(clap(), T(bar, 8), 0.38)
        drums.add(snare(0.5), T(bar, 8), 0.25)
    for s in range(0, steps, 2):
        drums.add(hat(), T(bar, s), 0.30 if s % 4 == 0 else 0.2, pan=0.25)
    for s in (5, 11, 13):
        if s < steps:
            drums.add(hat(), T(bar, s), 0.13, pan=-0.2)
    for s in ([0, 3, 6, 10, 12] if bar % 2 == 0 else [0, 3, 6, 8, 11, 14]):
        if s < steps:
            st = chord_stab(CH[chord], 0.14 if s != 12 else 0.32)
            music.add(st, T(bar, s), 0.42)
            send.add(st, T(bar, s), 0.16)
    music.add(pad(CH[chord], TL.BAR * steps / 16, cutoff=1500, att=0.05, rel=0.1), bar_t, 0.30)
    if bar % 4 in (2, 3):
        tones = sorted(CH[chord])[1:] + [CH[chord][1] + 12]
        for s in range(steps):
            m = tones[[0, 1, 2, 3, 4, 3, 2, 1][s % 8] % len(tones)] + 12
            p = pluck(m, 0.08)
            music.add(p, T(bar, s), 0.15, pan=-0.35 if s % 2 else 0.35)
            send.add(p, T(bar, s), 0.07)

S = TL.DRINK_STARTS
for i, s in enumerate(S):
    big = i == 0
    sfx.add(impact(1.2 if big else 0.7), s, 1.0 if big else 0.6)
    send.add(impact(1.0), s, 0.35 if big else 0.2)
    drums.add(crash(), s, 0.5)
    send.add(crash(), s, 0.2)
    for k in range(6):
        drums.add(hat(), s - TL.BEAT + k * TL.BEAT / 6, 0.12 + 0.035 * k, pan=0.25)
    c = s + TL.CLOSE
    sfx.add(whoosh(0.3, 300, 5000, peak=0.8, pan_from=0.5 if i % 2 else -0.5, pan_to=-0.5 if i % 2 else 0.5), c - 0.27, 0.42)
    drums.add(low_hit(0.6), c, 0.42)

# ---------- transitions
for w in (S[1], S[6], S[11]):                      # liquid waves
    sfx.add(whoosh(0.42, 180, 3500, peak=0.9, pan_from=0, pan_to=0), w - 0.42, 0.6)
    sfx.add(splash(1.1, 1.3), w - 0.08, 0.7)
    sfx.add(gurgle(0.6), w + 0.05, 0.45)
for w, pf, pt in ((S[2], 0.8, -0.9), (S[5], -0.8, 0.9), (S[8], 0.8, -0.9), (S[12], 0.8, -0.9)):   # whips
    sfx.add(whoosh(0.3, 500, 9000, peak=0.75, pan_from=pf, pan_to=pt), w - 0.28, 0.72)
sfx.add(riser(0.45), S[3] - 0.45, 0.5)                         # Yellow -> sun flare
sfx.add(reverse_cymbal(0.5), S[3] - 0.5, 0.45)
sp = sparkle(0.45, 8, base=88)                                 # Spring -> petal storm
sfx.add(sp, S[4] - 0.45, 0.45)
sfx.add(whoosh(0.45, 250, 6000, peak=0.9, pan_from=-0.4, pan_to=0.4), S[4] - 0.43, 0.5)
sfx.add(riser(0.45), S[7] - 0.45, 0.55)                        # Sea Blue -> warp
sfx.add(whoosh(0.45, 300, 11000, peak=0.95, pan_from=0, pan_to=0), S[7] - 0.42, 0.6)
sfx.add(whoosh(0.5, 120, 1800, peak=0.85, pan_from=-0.6, pan_to=0.6), S[9] - 0.48, 0.7)   # Pink -> smoke
sfx.add(riser(0.32), S[10] - 0.32, 0.5)                        # Lilac -> zoom-through
sfx.add(whoosh(0.3, 400, 10000, peak=0.95, pan_from=0, pan_to=0), S[10] - 0.28, 0.6)
sfx.add(whoosh(0.35, 400, 9000, peak=0.9, pan_from=0, pan_to=0), TL.END_START - 0.33, 0.7)   # Apple -> end card

# ---------- per-drink sound design
sfx.add(splash(1.4, 2.0), S[0], 0.95)                          # 1 Summer: explosion
send.add(splash(1.0, 2.0), S[0], 0.25)
for t0 in TL.CAN_OPENS:                                        # Summer + Apple can cracks
    sfx.add(can_open(), t0 - 0.04, 0.85)
    send.add(can_open(), t0 - 0.04, 0.2)
    sfx.add(fizz(1.1), t0 + 0.06, 0.45)
sfx.add(wind(1.9), S[1], 0.10)                                 # 2 Ice: clinks on each landing
for t0 in TL.ICE_HITS:
    sfx.add(ice_clink(amp=0.8), t0, 0.5, pan=rng.uniform(-0.6, 0.6))
    sfx.add(ice_clink(amp=0.5), t0 + rng.uniform(0.03, 0.07), 0.3, pan=rng.uniform(-0.6, 0.6))
    send.add(ice_clink(), t0, 0.15)
sfx.add(splash(1.1, 1.6), S[2] + 0.05, 0.8)                    # 3 Yellow
sfx.add(fizz(1.6, 2400), S[2] + 0.2, 0.28)
sfx.add(splash(1.0, 1.5), S[3] + 0.05, 0.7)                    # 4 Spring: chimes
for k, m in enumerate([84, 88, 91, 96]):
    b = bell(m, 0.9)
    music.add(b, S[3] + 0.3 + k * TL.BEAT / 2, 0.08)
    send.add(b, S[3] + 0.3 + k * TL.BEAT / 2, 0.1)
for k in range(10):                                            # 5 Peach: bubbles
    sfx.add(bloop(rng.uniform(380, 820)), S[4] + 0.2 + k * 0.15, 0.2, pan=rng.uniform(-0.7, 0.7))
sfx.add(fizz(1.6, 2000), S[4] + 0.1, 0.25)
sfx.add(splash(1.3, 1.8), S[5] + 0.05, 0.9)                    # 6 Red
sfx.add(gurgle(1.0), S[6] + 0.3, 0.3)                          # 7 Sea Blue: underwater
for k in range(12):
    sfx.add(bloop(rng.uniform(200, 520)), S[6] + 0.3 + k * 0.12, 0.22, pan=rng.uniform(-0.7, 0.7))
sfx.add(splash(1.0, 1.5), S[7] + 0.05, 0.7)                    # 8 Amber: embers
sfx.add(crackle(1.8, 50), S[7] + 0.1, 0.35)
sfx.add(splash(1.1, 1.6), S[8] + 0.05, 0.8)                    # 9 Pink: confetti shimmer
sp = sparkle(0.7, 10, base=86)
sfx.add(sp, S[8] + 0.1, 0.4)
send.add(sp, S[8] + 0.1, 0.25)
sfx.add(wind(1.9) * 0.8, S[9], 0.12)                           # 10 Lilac: blossom chimes
for k, m in enumerate([81, 86, 89]):
    b = bell(m, 1.0)
    music.add(b, S[9] + 0.4 + k * TL.BEAT, 0.08)
    send.add(b, S[9] + 0.4 + k * TL.BEAT, 0.1)
sfx.add(splash(1.1, 1.6), S[10] + 0.05, 0.75)                  # 11 Winter: wind + snow shimmer
sfx.add(wind(1.9), S[10], 0.12)
sp = sparkle(0.9, 12, base=93)
sfx.add(sp, S[10] + 0.15, 0.35)
send.add(sp, S[10] + 0.15, 0.25)
sfx.add(thunder(1.9), S[11], 0.72)                             # 12 Coconut: thunder + zaps
send.add(thunder(1.9), S[11], 0.2)
for t0 in TL.STRIKES:
    big = t0 == TL.STRIKES[0]
    z = zap(0.55 if big else 0.32, big)
    sfx.add(z, t0, 0.55 if big else 0.36, pan=rng.uniform(-0.5, 0.5))
    send.add(z, t0, 0.12)
sfx.add(fire_roar(1.9), S[12], 0.55)                           # 13 Apple: fire
send.add(fire_roar(1.9), S[12], 0.15)
sfx.add(splash(1.2, 1.6), S[12] + 0.05, 0.8)

# ---------- END CARD
E = TL.END_START
sfx.add(impact(1.1), E, 0.9)
send.add(impact(1.0), E, 0.3)
drums.add(crash(3.0), E, 0.5)
send.add(crash(3.0), E, 0.25)
st = chord_stab(CH["Dm"], 0.9)
music.add(st, E, 0.5)
send.add(st, E, 0.3)
music.add(pad(CH["Dm"], 1.5, cutoff=1800, att=0.05, rel=0.2), E, 0.32)
music.add(pad(CH["Bb"], 0.75, cutoff=1800, att=0.05, rel=0.2), E + 1.5, 0.32)
music.add(pad(CH["C"], 0.75, cutoff=1800, att=0.05, rel=0.2), E + 2.25, 0.32)
bass.add(sub808(mtof(26), 1.45), E, 0.8)
bass.add(sub808(mtof(34), 0.72), E + 1.5, 0.75)
bass.add(sub808(mtof(36), 0.72, glide_to=mtof(38)), E + 2.25, 0.75)
for bar in range(2):
    for s16 in (0, 10):
        drums.add(kick(), T(bar, s16, E), 0.85)
        kick_times.append(T(bar, s16, E))
    drums.add(clap(), T(bar, 8, E), 0.5)
    send.add(clap(), T(bar, 8, E), 0.3)
    for s16 in range(0, 16, 2):
        drums.add(hat(), T(bar, s16, E), 0.17, pan=0.25)
    seq = [CH["Dm"]] * 16 if bar == 0 else [CH["Bb"]] * 8 + [CH["C"]] * 8
    for s16 in range(16):
        tones = sorted(seq[s16])[1:] + [seq[s16][1] + 12]
        m = tones[[0, 1, 2, 3, 4, 3, 2, 1][s16 % 8] % len(tones)] + 12
        p = pluck(m, 0.08)
        music.add(p, T(bar, s16, E), 0.13, pan=-0.35 if s16 % 2 else 0.35)
        send.add(p, T(bar, s16, E), 0.07)
drums.add(low_hit(), TL.END_LOGO, 0.8)
sfx.add(whoosh(0.3, 800, 9000, peak=0.6, pan_from=0.5, pan_to=-0.5), TL.END_ADDR - 0.1, 0.25)
sfx.add(whoosh(0.3, 800, 9000, peak=0.6, pan_from=-0.5, pan_to=0.5), TL.END_ADDR + 0.08, 0.2)
drums.add(low_hit(0.8), TL.END_OFFER, 0.7)
send.add(low_hit(0.8), TL.END_OFFER, 0.25)
music.add(bell(86, 1.2), TL.END_OFFER, 0.14)
music.add(bell(81, 1.5), TL.END_CTA, 0.22)
music.add(bell(88, 1.5), TL.END_CTA + 0.06, 0.15)
send.add(bell(81, 1.5), TL.END_CTA, 0.2)
for t0 in TL.END_SHINES:
    sp = sparkle(0.6, 9, base=86)
    sfx.add(sp, t0, 0.4)
    send.add(sp, t0, 0.25)
FIN = TL.END_FINAL
drums.add(kick(), FIN, 0.9)
kick_times.append(FIN)
sfx.add(impact(1.0, 1.5), FIN, 0.55)
st = chord_stab(CH["Dm"] + [69], 1.3)
music.add(st, FIN, 0.55)
send.add(st, FIN, 0.45)
music.add(pad(CH["Dm"], 1.3, cutoff=2200, att=0.02, rel=0.2), FIN, 0.35)
bass.add(sub808(mtof(26), 1.3), FIN, 0.85)
drums.add(crash(1.5), FIN, 0.45)

# ---------- mix (same chain as Reels 1-3)
duck = np.ones(N)
for kt in kick_times:
    i0 = int(kt * SR)
    m = min(N - i0, int(0.3 * SR))
    duck[i0:i0 + m] = np.minimum(duck[i0:i0 + m], 1 - 0.55 * np.exp(-ta(m) / 0.08))
duck = signal.lfilter([0.02], [1, -0.98], duck)
music.b *= duck[:, None]
wet = reverb(send.b * (0.5 + 0.5 * duck[:, None]), make_ir(1.8))
wet += pingpong(music.b * 1.2, d=0.28125)
mix = drums.b * 0.9 + bass.b * 0.7 + music.b * 3.4 + sfx.b * 1.0 + wet * 0.6
mix = filt(sos_hp(28), mix)
mix += filt(sos_hp(8000), mix) * 0.12
t = ta(N)
mix *= np.clip((TL.DURATION - t) / 0.6, 0, 1)[:, None]
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 2.0
look = int(0.003 * SR)
peak = maximum_filter1d(np.abs(mix).max(1), size=2 * look + 1)
g = np.minimum(1.0, 0.89 / (peak + 1e-9))
alpha = np.exp(-1 / (0.08 * SR))
g_s = np.minimum(signal.lfilter([1 - alpha], [1, -alpha], g), g)
mix *= g_s[:, None]
mix = np.clip(np.tanh(mix * 1.05) / np.tanh(1.05) * 0.73, -0.79, 0.79)
wavfile.write(os.path.join(OUT, "audio_mix.wav"), SR, (mix * 32767).astype(np.int16))
print("wrote", os.path.join(OUT, "audio_mix.wav"), f"{TL.DURATION}s")
