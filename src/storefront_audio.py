"""Original sound design for the storefront reveal (all synthesized, no samples):
night ambience + atmospheric pad, a build-up during the approach, an aerial whoosh,
a tasteful impact on the storefront reveal and a warm musical ending."""
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d

import synth
from synth import (SR, Bus, ta, ns, mtof, filt, sos_bp, sos_lp, sos_hp, pad, bell, impact, low_hit, whoosh, riser,
                   reverse_cymbal, sub808, chord_stab, pluck, make_ir, reverb, crash, kick)


def build(path, dur, t0, t1):
    synth.reseed(4242)
    N = int(SR * dur)
    music, sfx, send = Bus(N), Bus(N), Bus(N)
    # night ambience: distant traffic rumble + soft air
    n = N
    t = ta(n)
    amb = filt(sos_lp(220), ns(n)) * 0.5 + filt(sos_bp(500, 2500), ns(n)) * 0.08 * (0.7 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
    sfx.add(amb * np.clip(t / 0.8, 0, 1), 0.0, 0.10)
    # atmospheric pad: D major-ish, opens up during the approach
    Dm = [50, 57, 62, 66, 69]
    music.add(pad(Dm, t1 + 0.2, cutoff=700, att=1.2, rel=0.4), 0.0, 0.42)
    music.add(pad([m + 12 for m in Dm], t1 - t0, cutoff=2600, att=1.8, rel=0.3), t0, 0.22)
    # slow plucked motif while hovering wide
    for k, m in enumerate([74, 81, 78, 76]):
        p = pluck(m, 0.5)
        music.add(p, 0.35 + k * 0.38, 0.09, pan=-0.3 if k % 2 else 0.3)
        send.add(p, 0.35 + k * 0.38, 0.12)
    # build-up across the approach
    sfx.add(riser(t1 - t0 + 0.05), t0, 0.32)
    sfx.add(reverse_cymbal(1.1), t1 - 1.1, 0.42)
    for k in range(8):                              # accelerating soft pulses
        tk = t0 + (t1 - t0) * (1 - (1 - k / 8) ** 1.6)
        music.add(low_hit(0.5), tk, 0.10 + 0.04 * k)
    # aerial whooshes on the descent
    sfx.add(whoosh(1.4, 200, 4500, peak=0.75, pan_from=-0.6, pan_to=0.5), t0 + 0.2, 0.30)
    sfx.add(whoosh(0.9, 300, 7000, peak=0.85, pan_from=0.5, pan_to=-0.2), t1 - 0.9, 0.38)
    # reveal impact
    sfx.add(impact(1.0, 2.6), t1, 0.70)
    send.add(impact(0.9, 2.6), t1, 0.30)
    sfx.add(crash(2.4), t1, 0.30)
    send.add(crash(2.4), t1, 0.18)
    Dmaj = [50, 57, 62, 66, 69, 74]
    st = chord_stab(Dmaj, 1.6, bright=0.6)
    music.add(st, t1, 0.40)
    send.add(st, t1, 0.40)
    music.add(pad(Dmaj, dur - t1, cutoff=2200, att=0.05, rel=0.6), t1, 0.32)
    music.add(sub808(mtof(38), 2.2), t1, 0.55)
    # warm ending: two bells and a final low note
    for tk, m, g in ((t1 + 1.2, 86, 0.16), (t1 + 1.25, 93, 0.10), (dur - 1.6, 81, 0.14), (dur - 1.55, 86, 0.10)):
        b = bell(m, 1.6)
        music.add(b, tk, g)
        send.add(b, tk, 0.2)
    music.add(sub808(mtof(38), 1.4), dur - 1.6, 0.35)
    # mix
    wet = reverb(send.b, make_ir(2.6, 3.2))
    mix = music.b * 1.6 + sfx.b * 1.0 + wet * 0.7
    mix = filt(sos_hp(28), mix)
    tt = ta(N)
    mix *= (np.clip(tt / 0.25, 0, 1) * np.clip((dur - tt) / 0.6, 0, 1))[:, None]
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 1.6
    look = int(0.003 * SR)
    peak = maximum_filter1d(np.abs(mix).max(1), size=2 * look + 1)
    g = np.minimum(1.0, 0.85 / (peak + 1e-9))
    a = np.exp(-1 / (0.1 * SR))
    g = np.minimum(signal.lfilter([1 - a], [1, -a], g), g)
    mix = np.clip(np.tanh(mix * g[:, None] * 1.05) / np.tanh(1.05) * 0.8, -0.84, 0.84)
    wavfile.write(path, SR, (mix * 32767).astype(np.int16))
    return path


if __name__ == "__main__":
    import os
    os.makedirs("../output/build_storefront", exist_ok=True)
    print(build("../output/build_storefront/audio.wav", 9.5, 1.6, 5.6))
