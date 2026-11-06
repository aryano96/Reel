"""Logo Top Reel: the Logo Final Reel with the store logo moved ABOVE each product
name (logo takes the name's old spot, the name moves down into the logo's old spot).

Same patching method as reel2_final.py / reel2_logo_final.py:
  * determinism check: the "logo" style must reproduce the Logo Final frames exactly;
  * only frames where the name/logo block is visible are re-rendered;
  * GOPs without a changed frame are copied bit-for-bit from the Logo Final MP4,
    the others re-encoded to the original GOP bit budget (identical SPS/PPS);
  * the audio is the identical soundtrack.

    python reel2_logo_top.py
"""
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np
import cv2

import reel2_render as R2
import reel2_final as F
import reel2_logo_final as LF

ROOT = R2.ROOT
SRC_MP4 = LF.OUT_MP4                                      # Ezy_Pzy_Convenience_Logo_Final_Reel.mp4
SRC_FRAMES = os.path.join(LF.BUILD, "frames")             # its source frames
BUILD = os.path.join(ROOT, "output", "build2_logo_top")
OUT_MP4 = os.path.join(ROOT, "output", "reel2", "Ezy_Pzy_Convenience_Logo_Top_Final_Reel.mp4")


def _check(fi):
    R2.CHIP_STYLE = "logo_top"
    new = R2.render_frame(fi)
    old = cv2.cvtColor(cv2.imread(os.path.join(SRC_FRAMES, f"{fi:05d}.png")), cv2.COLOR_BGR2RGB)
    changed = not np.array_equal(new, old)
    if changed:
        cv2.imwrite(os.path.join(BUILD, "new", f"{fi:05d}.png"), cv2.cvtColor(new, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi, changed


def main():
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(os.path.join(BUILD, "new"))
    R2.CHIP_STYLE = "logo"
    for fi in (110, 400, 650):
        a = R2.render_frame(fi)
        b = cv2.cvtColor(cv2.imread(os.path.join(SRC_FRAMES, f"{fi:05d}.png")), cv2.COLOR_BGR2RGB)
        assert np.array_equal(a, b), f"renderer does not reproduce frame {fi}; aborting"
    print("determinism check passed (logo style reproduces the Logo Final frames exactly)")
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        res = p.map(_check, F.candidate_frames(), chunksize=2)
    changed = sorted(fi for fi, c in res if c)
    print(f"{len(changed)} frames show the name/logo block and were updated ({time.time() - t0:.0f}s)")
    F.SRC_MP4, F.SRC_FRAMES, F.BUILD, F.OUT_MP4 = SRC_MP4, SRC_FRAMES, BUILD, OUT_MP4
    F.FRAMES = os.path.join(BUILD, "frames")
    F.SEGS = os.path.join(BUILD, "segments")
    os.makedirs(F.FRAMES)
    os.makedirs(F.SEGS)
    chs = set(changed)
    for fi in range(F.N):
        dst = os.path.join(F.FRAMES, f"{fi:05d}.png")
        src = os.path.join(BUILD, "new", f"{fi:05d}.png") if fi in chs else os.path.join(SRC_FRAMES, f"{fi:05d}.png")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    F.splice(changed)


if __name__ == "__main__":
    main()
