"""Final Reel: the Two Locations edition with the store name corrected.

The only change: the small line under each product name in the seven drink
scenes reads "NOW AT EZY PZY CONVENIENCE" instead of "NOW AT EZY PZY". (The end
card already shows the full name via the store logo picture; no other on-screen
text contains the store name.)

How it is built, so nothing else changes:
  1. Frames where that line is visible are re-rendered (the renderer is
     deterministic: with the old text it reproduces the original frames
     pixel-for-pixel; see verify step).
  2. The Two Locations MP4 is split at its keyframes into GOPs. Every GOP with
     no changed frame is copied bit-for-bit; only GOPs containing a changed
     frame are re-encoded (same x264 settings) from the source frames.
  3. The audio is the identical Two Locations soundtrack.

    python reel2_final.py
"""
import os
import shutil
import subprocess
import time
from multiprocessing import Pool

import numpy as np
import cv2

import reel2_render as R2
import reel2_two_locations as TLOC
from timeline2 import FPS

ROOT = R2.ROOT
SRC_MP4 = TLOC.OUT_MP4                                   # Ezy_Pzy_Convenience_Two_Locations_Reel.mp4
SRC_FRAMES = TLOC.FRAMES                                  # its source frames (0-719 = original Reel 2 frames)
SRC_WAV = os.path.join(TLOC.BUILD, "audio_mix.wav")
BUILD = os.path.join(ROOT, "output", "build2_final")
FRAMES = os.path.join(BUILD, "frames")
SEGS = os.path.join(BUILD, "segments")
OUT_MP4 = os.path.join(ROOT, "output", "reel2", "Ezy_Pzy_Convenience_Final_Reel.mp4")
N = TLOC.N_FRAMES
OLD_TEXT = "NOW AT EZY PZY"
NEW_TEXT = "NOW AT EZY PZY CONVENIENCE"
# tag windows (t_in, t_out) used by the seven drink scenes in reel2_render.py
TAG_WINDOWS = [(3.25, 4.3), (6.35, 7.35), (9.3, 10.35), (12.3, 13.35), (15.3, 16.35), (18.3, 19.35), (21.3, 22.35)]


def candidate_frames():
    out = set()
    for t_in, t_out in TAG_WINDOWS:
        for fi in range(int(t_in * FPS) - 3, int((t_out + 0.3) * FPS) + 4):
            out.add(fi)
    return sorted(out)


def _check(fi):
    R2.CHIP_TEXT = NEW_TEXT
    new = R2.render_frame(fi)
    old = cv2.imread(os.path.join(SRC_FRAMES, f"{fi:05d}.png"))
    old = cv2.cvtColor(old, cv2.COLOR_BGR2RGB)
    changed = not np.array_equal(new, old)
    if changed:
        cv2.imwrite(os.path.join(BUILD, "new", f"{fi:05d}.png"), cv2.cvtColor(new, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi, changed


def keyframes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=key_frame",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()
    return [i for i, k in enumerate(out) if k.strip().startswith("1")]


def framemd5(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def encode_matched(a, b, target_bytes, out):
    """Re-encode frames [a, b) with the original x264 settings, choosing the VBV start
    level so the GOP gets the same bit budget as the original GOP (keeps compression
    consistent with the bit-for-bit copied neighbours). CRF/profile are unchanged, so
    SPS/PPS stay identical."""
    ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    lo, hi, best = 0.0, 0.9, None
    for _ in range(7):
        mid = (lo + hi) / 2 if best is not None else 0.9
        subprocess.run(ff + ["-framerate", str(FPS), "-start_number", str(a), "-i", os.path.join(FRAMES, "%05d.png"),
                             "-frames:v", str(b - a)] + TLOC.X264 + ["-x264-params", f"vbv-init={mid:.4f}",
                             "-video_track_timescale", "15360", out + ".try.mp4"], check=True)
        size = os.path.getsize(out + ".try.mp4")
        if best is None or abs(size - target_bytes) < abs(best[1] - target_bytes):
            best = (mid, size)
            shutil.copy2(out + ".try.mp4", out)
        if size > target_bytes:
            hi = mid
        else:
            lo = mid
        if abs(size - target_bytes) / target_bytes < 0.03:
            break
    os.remove(out + ".try.mp4")
    return best


def sps_pps(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-frames:v", "1",
                          "-f", "h264", "-"], capture_output=True).stdout
    out = {}
    for chunk in raw.split(b"\x00\x00\x01"):
        if chunk and (chunk[0] & 0x1F) in (7, 8):
            out.setdefault(chunk[0] & 0x1F, chunk.rstrip(b"\x00"))
    return out


def main(encode_only=False):
    ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if encode_only:
        changed = sorted(int(f[:5]) for f in os.listdir(os.path.join(BUILD, "new")))
        shutil.rmtree(SEGS, ignore_errors=True)
        os.makedirs(SEGS)
        return splice(changed)
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(os.path.join(BUILD, "new"))
    os.makedirs(FRAMES)
    os.makedirs(SEGS)

    # 0) determinism guard: old text must reproduce the original frames exactly
    R2.CHIP_TEXT = OLD_TEXT
    for fi in (110, 400, 650):
        a = R2.render_frame(fi)
        b = cv2.cvtColor(cv2.imread(os.path.join(SRC_FRAMES, f"{fi:05d}.png")), cv2.COLOR_BGR2RGB)
        assert np.array_equal(a, b), f"renderer is not reproducing frame {fi}; aborting"
    print("determinism check passed (old text reproduces original frames exactly)")

    # 1) re-render candidate frames with the corrected text; keep only those that differ
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        res = p.map(_check, candidate_frames(), chunksize=2)
    changed = sorted(fi for fi, c in res if c)
    print(f"{len(changed)} frames show the store line and were updated ({time.time() - t0:.0f}s); "
          f"ranges: {[(g[0], g[-1]) for g in np.split(np.array(changed), np.where(np.diff(changed) > 1)[0] + 1)]}")

    # 2) frame set: hard links to the Two Locations source frames, updated frames as new files
    for fi in range(N):
        dst = os.path.join(FRAMES, f"{fi:05d}.png")
        src = os.path.join(BUILD, "new", f"{fi:05d}.png") if fi in set(changed) else os.path.join(SRC_FRAMES, f"{fi:05d}.png")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    splice(changed)


def splice(changed):
    ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    # 3) GOP-level splice of the Two Locations MP4
    kf = keyframes(SRC_MP4)
    bounds = kf + [N]
    gops = list(zip(bounds[:-1], bounds[1:]))
    subprocess.run(ff + ["-i", SRC_MP4, "-map", "0:v:0", "-c", "copy", "-f", "segment",
                         "-segment_frames", ",".join(str(k) for k in kf[1:]), "-reset_timestamps", "1",
                         "-segment_format", "mp4", os.path.join(SEGS, "orig_%03d.mp4")], check=True)
    chs = set(changed)
    files, n_copy, n_reenc = [], 0, 0
    for gi, (a, b) in enumerate(gops):
        if any(a <= fi < b for fi in chs):
            seg = os.path.join(SEGS, f"new_{gi:03d}.mp4")
            target = os.path.getsize(os.path.join(SEGS, f"orig_{gi:03d}.mp4"))
            vi, size = encode_matched(a, b, target, seg)
            assert sps_pps(seg) == sps_pps(os.path.join(SEGS, f"orig_{gi:03d}.mp4")), f"SPS/PPS mismatch in GOP {gi}"
            print(f"  GOP {gi:2d} frames {a}-{b - 1}: original {target / 1e6:.2f} MB -> re-encoded {size / 1e6:.2f} MB (vbv-init {vi:.3f})", flush=True)
            n_reenc += b - a
        else:
            seg = os.path.join(SEGS, f"orig_{gi:03d}.mp4")
            n_copy += b - a
        files.append(seg)
    lst = os.path.join(BUILD, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in files)
    subprocess.run(ff + ["-f", "concat", "-safe", "0", "-i", lst, "-i", SRC_WAV, "-map", "0:v", "-map", "1:a",
                         "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", "-shortest",
                         OUT_MP4], check=True)
    print(f"GOPs: {len(gops)} total, frames copied bit-for-bit: {n_copy}, frames in re-encoded GOPs: {n_reenc}")

    # 4) verification
    a, b = framemd5(SRC_MP4), framemd5(OUT_MP4)
    assert len(b) == N, f"frame count {len(b)} != {N}"
    copied = [fi for g, (s, e) in enumerate(gops) for fi in range(s, e) if not any(s <= c < e for c in chs)]
    same = sum(a[fi] == b[fi] for fi in copied)
    print(f"verify: {len(b)} frames; {same}/{len(copied)} frames outside updated GOPs decode identical to the Two Locations reel")
    print("wrote", OUT_MP4)


if __name__ == "__main__":
    import sys
    main(encode_only="--encode-only" in sys.argv)
