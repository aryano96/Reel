"""Reel 1 Final: the Updated Reel 1 with the opening headline changed to
"POWER EVERY REP WITH BUCKED UP!" (three slams on the same three bass hits:
"POWER EVERY REP" / "WITH" / "BUCKED UP!" - same fonts, colours, glow and
animation; the lime-gradient highlight now lands on BUCKED UP!).

Only the opening frames showing the headline are re-rendered (after checking the
renderer reproduces the current frames exactly); every other GOP of the Updated
Reel 1 MP4 is copied bit-for-bit; the audio is identical.

    python reel1_final.py
"""
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np
import cv2

import render as R
import reel1_two_locations as R1U
import patching

ROOT = R.ROOT
SRC_MP4 = R1U.OUT_MP4                        # Ezy_Pzy_Convenience_Reel_1_Updated.mp4
SRC_FRAMES = R1U.FRAMES
SRC_WAV = os.path.join(R1U.BUILD, "audio_mix.wav")
BUILD = os.path.join(ROOT, "output", "build_reel1_final")
FRAMES = os.path.join(BUILD, "frames")
OUT_MP4 = os.path.join(ROOT, "output", "Ezy_Pzy_Bucked_Up_Reel_1_Final.mp4")
N = R1U.N_FRAMES

NEW_LINES = [("POWER EVERY REP", 880, 282), ("WITH", 250, 392), ("BUCKED UP!", 960, 522)]
NEW_FONTS = ["InterDisplay-Black.otf", "InterDisplay-Black.otf", "InterDisplay-BlackItalic.otf"]
NEW_HERO = 2


def use_new_headline():
    R.INTRO_LINES, R.INTRO_FONTS, R.INTRO_HERO = NEW_LINES, NEW_FONTS, NEW_HERO


def _check(fi):
    use_new_headline()
    new = R.render_frame(fi)
    old = cv2.cvtColor(cv2.imread(os.path.join(SRC_FRAMES, f"{fi:05d}.png")), cv2.COLOR_BGR2RGB)
    if np.array_equal(new, old):
        return fi, False
    cv2.imwrite(os.path.join(BUILD, "new", f"{fi:05d}.png"), cv2.cvtColor(new, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi, True


def main():
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(os.path.join(BUILD, "new"))
    os.makedirs(FRAMES)
    for fi in (10, 50, 85):     # original headline must reproduce the current frames exactly
        a = R.render_frame(fi)
        b = cv2.cvtColor(cv2.imread(os.path.join(SRC_FRAMES, f"{fi:05d}.png")), cv2.COLOR_BGR2RGB)
        assert np.array_equal(a, b), f"renderer does not reproduce frame {fi}; aborting"
    print("determinism check passed")
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        res = p.map(_check, range(0, int(3.0 * 30) + 2), chunksize=2)     # the opening (headline is pre-drop only)
    changed = sorted(fi for fi, c in res if c)
    print(f"{len(changed)} opening frames updated (frames {changed[0]}-{changed[-1]}, {time.time() - t0:.0f}s)")
    chs = set(changed)
    for fi in range(N):
        src = os.path.join(BUILD, "new", f"{fi:05d}.png") if fi in chs else os.path.join(SRC_FRAMES, f"{fi:05d}.png")
        dst = os.path.join(FRAMES, f"{fi:05d}.png")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    patching.splice(SRC_MP4, FRAMES, SRC_WAV, BUILD, OUT_MP4, changed, N, R1U.X264_R1)
    print("wrote", OUT_MP4)


if __name__ == "__main__":
    main()
