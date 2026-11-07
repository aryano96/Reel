"""Reel 3 soundtrack: original 160 BPM half-time trap in A minor + sound design, all
synthesized from scratch (synth.py) - no samples, no licensed audio.

Scene changes land every 6 beats (one per drink); every reveal gets an impact.
Output: output/build3/audio_mix.wav
"""
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d

import timeline3 as TL
import synth
from synth import (SR, Bus, ta, ns, mtof, filt, sos_bp, sos_lp, sos_hp, saw, kick, clap, snare, hat, crash, sub808,
                   chord_stab, pad, pluck, bell, impact, low_hit, whoosh, riser, reverse_cymbal, splash, fizz, can_open,
                   ice_clink, zap, thunder, bloop, wind, gurgle, sparkle, make_ir, reverb, pingpong)

synth.reseed(3033)
rng = synth.rng
N = int(SR * TL.DURATION)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "build3")
os.makedirs(OUT, exist_ok=True)

drums, bass, music, sfx, send = Bus(N), Bus(N), Bus(N), Bus(N), Bus(N)
kick_times = []
CH = {"Am": [57, 60, 64, 69, 72], "F": [53, 57, 60, 65, 69], "C": [55, 60, 64, 67, 72], "G": [55, 59, 62, 67, 71]}
ROOT = {"Am": 33, "F": 29, "C": 36, "G": 31}
PROG = ["Am", "F", "C", "G"]


def T(bar, s16, base=TL.DROP):
    return base + bar * TL.BAR + s16 * TL.S16


# ---------- INTRO (0 - 3.0)
music.add(pad(CH["Am"], 2.88, cutoff=620, att=0.9, rel=0.1), 0.0, 0.55)
music.add(pad([m + 12 for m in CH["Am"]], 1.5, cutoff=2400, att=1.2, rel=0.1), 1.4, 0.24)
bass.add(sub808(mtof(33), 2.88), 0.0, 0.33)
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

# ---------- MAIN GROOVE (3.0 - 23.25): 13.5 bars
n_bars = int(np.ceil((TL.END_START - TL.DROP) / TL.BAR))
KA, KB = [0, 7, 10], [0, 3, 10, 14]
for bar in range(n_bars):
    chord = PROG[bar % 4]
    bar_t = T(bar, 0)
    last = bar_t + TL.BAR > TL.END_START + 1e-6       # half bar before the end card
    steps = 8 if last else 16
    kicks = [s for s in (KA if bar % 2 == 0 else KB) if s < steps]
    for s in kicks:
        drums.add(kick(), T(bar, s), 1.0)
        kick_times.append(T(bar, s))
    for i, s in enumerate(kicks):
        nxt = kicks[i + 1] if i + 1 < len(kicks) else steps
        bass.add(sub808(mtof(ROOT[chord]), (nxt - s) * TL.S16), T(bar, s), 0.85)
    if 8 < steps or not last:
        drums.add(clap(), T(bar, 8), 0.7)
        send.add(clap(), T(bar, 8), 0.38)
        drums.add(snare(0.5), T(bar, 8), 0.25)
    for s in range(0, steps, 2):
        drums.add(hat(), T(bar, s), 0.30 if s % 4 == 0 else 0.2, pan=0.25)
    for s in (5, 11, 13):
        if s < steps:
            drums.add(hat(), T(bar, s), 0.13, pan=-0.2)
    hits = [0, 3, 6, 10, 12] if bar % 2 == 0 else [0, 3, 6, 8, 11, 14]
    for s in hits:
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

# scene changes: impact + crash on each reveal, hat roll into it, whoosh at the close-up cut
for i, s in enumerate(TL.DRINK_STARTS):
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

S = TL.DRINK_STARTS
# 1 Cotton Candy: splash + can crack; exit liquid wave
sfx.add(splash(1.4, 2.0), S[0], 0.95)
send.add(splash(1.0, 2.0), S[0], 0.25)
for t0 in TL.CAN_OPENS:
    sfx.add(can_open(), t0 - 0.04, 0.85)
    send.add(can_open(), t0 - 0.04, 0.2)
    sfx.add(fizz(1.1), t0 + 0.06, 0.45)
for w in (S[1], S[6]):                   # liquid-wave transitions (into Purple Cotton Candy, Breezeberry)
    sfx.add(whoosh(0.42, 180, 3500, peak=0.9, pan_from=0, pan_to=0), w - 0.42, 0.6)
    sfx.add(splash(1.1, 1.3), w - 0.08, 0.7)
    sfx.add(gurgle(0.6), w + 0.05, 0.45)
# 2 Purple Cotton Candy: carnival chimes; exit whip up
sp = sparkle(0.9, 12, base=81)
sfx.add(sp, S[1] + 0.05, 0.5)
send.add(sp, S[1] + 0.05, 0.3)
for k, m in enumerate([76, 79, 83, 88]):
    b = bell(m, 0.9)
    music.add(b, S[1] + 0.4 + k * TL.BEAT / 2, 0.08)
for wt in (S[2], S[4], S[8]):            # vertical whips
    sfx.add(whoosh(0.3, 500, 9000, peak=0.75, pan_from=0, pan_to=0), wt - 0.28, 0.72)
sfx.add(whoosh(0.3, 400, 8000, peak=0.75, pan_from=0, pan_to=0), S[4] - 0.28, 0.4)
# 3 Witch's Brew: bubbling cauldron; exit smoke wipe
for k in range(16):
    sfx.add(bloop(rng.uniform(220, 560)), S[2] + 0.1 + k * 0.12 + rng.uniform(0, 0.04), 0.26, pan=rng.uniform(-0.7, 0.7))
sfx.add(gurgle(1.0), S[2] + 0.2, 0.35)
sfx.add(wind(2.2) * 0.8, S[2], 0.12)
sfx.add(whoosh(0.5, 120, 1800, peak=0.85, pan_from=-0.6, pan_to=0.6), S[3] - 0.48, 0.7)
# 4 Hawaiian Shaved Ice: ice clinks on each landing
sfx.add(wind(2.2), S[3], 0.10)
for t0 in TL.ICE_HITS_S4:
    sfx.add(ice_clink(amp=0.8), t0, 0.5, pan=rng.uniform(-0.6, 0.6))
    sfx.add(ice_clink(amp=0.5), t0 + rng.uniform(0.03, 0.07), 0.3, pan=rng.uniform(-0.6, 0.6))
    send.add(ice_clink(), t0, 0.15)
# 5 Orange Kiss: splash + fizz; exit sun-flare burst
sfx.add(splash(1.2, 1.8), S[4] + 0.05, 0.85)
sfx.add(fizz(1.8, 2600), S[4] + 0.2, 0.3)
sfx.add(riser(0.45), S[5] - 0.45, 0.5)
sfx.add(reverse_cymbal(0.5), S[5] - 0.5, 0.45)
# 6 Cherry Slush: thunder + zaps on every strike
sfx.add(thunder(2.0), S[5], 0.72)
send.add(thunder(2.0), S[5], 0.2)
for t0 in TL.STRIKES_S6:
    big = t0 == S[5]
    z = zap(0.55 if big else 0.32, big)
    sfx.add(z, t0, 0.55 if big else 0.36, pan=rng.uniform(-0.5, 0.5))
    send.add(z, t0, 0.12)
# 7 Breezeberry: droplets; exit hyperspace warp
for k in range(10):
    sfx.add(bloop(rng.uniform(600, 1100)), S[6] + 0.3 + k * 0.17, 0.16, pan=rng.uniform(-0.7, 0.7))
sfx.add(riser(0.45), S[7] - 0.45, 0.55)
sfx.add(whoosh(0.45, 300, 11000, peak=0.95, pan_from=0, pan_to=0), S[7] - 0.42, 0.6)
# 8 Cosmic Stardust: shimmer
for t0 in (S[7] + 0.05, S[7] + 1.2):
    sp = sparkle(0.8, 10, base=86)
    sfx.add(sp, t0, 0.45)
    send.add(sp, t0, 0.3)
music.add(pad([m + 12 for m in CH["C"]], 2.0, cutoff=3000, att=0.4, rel=0.3), S[7], 0.12)
# 9 Watermelon Wave: splash + can crack; exit flash + whip into the end card
sfx.add(splash(1.3, 1.8), S[8] + 0.05, 0.9)
sfx.add(whoosh(0.35, 400, 9000, peak=0.9, pan_from=0, pan_to=0), TL.END_START - 0.33, 0.7)

# ---------- END CARD (23.25 - 27.75)
E = TL.END_START
sfx.add(impact(1.1), E, 0.9)
send.add(impact(1.0), E, 0.3)
drums.add(crash(3.0), E, 0.5)
send.add(crash(3.0), E, 0.25)
st = chord_stab(CH["Am"], 0.9)
music.add(st, E, 0.5)
send.add(st, E, 0.3)
music.add(pad(CH["Am"], 1.5, cutoff=1800, att=0.05, rel=0.2), E, 0.32)
music.add(pad(CH["F"], 0.75, cutoff=1800, att=0.05, rel=0.2), E + 1.5, 0.32)
music.add(pad(CH["G"], 0.75, cutoff=1800, att=0.05, rel=0.2), E + 2.25, 0.32)
bass.add(sub808(mtof(33), 1.45), E, 0.8)
bass.add(sub808(mtof(29), 0.72), E + 1.5, 0.75)
bass.add(sub808(mtof(31), 0.72, glide_to=mtof(33)), E + 2.25, 0.75)
for bar in range(2):
    for s16 in (0, 10):
        drums.add(kick(), T(bar, s16, E), 0.85)
        kick_times.append(T(bar, s16, E))
    drums.add(clap(), T(bar, 8, E), 0.5)
    send.add(clap(), T(bar, 8, E), 0.3)
    for s16 in range(0, 16, 2):
        drums.add(hat(), T(bar, s16, E), 0.17, pan=0.25)
    seq = [CH["Am"]] * 16 if bar == 0 else [CH["F"]] * 8 + [CH["G"]] * 8
    for s16 in range(16):
        tones = sorted(seq[s16])[1:] + [seq[s16][1] + 12]
        m = tones[[0, 1, 2, 3, 4, 3, 2, 1][s16 % 8] % len(tones)] + 12
        p = pluck(m, 0.08)
        music.add(p, T(bar, s16, E), 0.13, pan=-0.35 if s16 % 2 else 0.35)
        send.add(p, T(bar, s16, E), 0.07)
drums.add(low_hit(), TL.END_LOGO, 0.8)
sfx.add(whoosh(0.3, 800, 9000, peak=0.6, pan_from=0.5, pan_to=-0.5), TL.END_ADDR - 0.1, 0.25)
drums.add(low_hit(0.8), TL.END_OFFER, 0.7)                 # the offer lands on a hit
send.add(low_hit(0.8), TL.END_OFFER, 0.25)
music.add(bell(93, 1.2), TL.END_OFFER, 0.14)
music.add(bell(88, 1.5), TL.END_CTA, 0.22)
music.add(bell(95, 1.5), TL.END_CTA + 0.06, 0.15)
send.add(bell(88, 1.5), TL.END_CTA, 0.2)
for t0 in TL.END_SHINES:
    sp = sparkle(0.6, 9, base=86)
    sfx.add(sp, t0, 0.4)
    send.add(sp, t0, 0.25)
FIN = TL.END_FINAL
drums.add(kick(), FIN, 0.9)
kick_times.append(FIN)
sfx.add(impact(1.0, 1.5), FIN, 0.55)
st = chord_stab(CH["Am"] + [76], 1.3)
music.add(st, FIN, 0.55)
send.add(st, FIN, 0.45)
music.add(pad(CH["Am"], 1.3, cutoff=2200, att=0.02, rel=0.2), FIN, 0.35)
bass.add(sub808(mtof(33), 1.3), FIN, 0.85)
drums.add(crash(1.5), FIN, 0.45)

# ---------- mix (same chain as Reels 1 and 2)
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
