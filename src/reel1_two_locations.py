"""Reel 1 (Bucked Up) - updated ending: store logo + both locations.

Only the closing store-promotion scene changes. Everything before it is reused:
  * video: the original Reel 1 export's compressed stream is copied bit-for-bit up
    to its last keyframe before the end card; the few frames from that keyframe to
    18.97 s are re-encoded from the identical original source frames
    (output/build/frames, hard-linked, not re-rendered); only 19.0 s+ is new;
  * audio: the original Reel 1 mix, bit-identical up to 22.99 s. The ending is
    extended by one bar (2.0 s at 120 BPM) by repeating the end card's last groove
    bar (21.0-23.0 s) before the final hit, so the soundtrack stays the original.

New closing scene (19.0-26.5 s), synced to the original end-card hits:
  19.00  vertical whip from Banana Pineapple -> flash, card slides in (as before)
  19.15  Ezy Pzy Convenience logo picture revealed               <- low hit
  19.30  "VISIT US AT TWO LOCATIONS!" + gold rule                <- whoosh
  19.75  location 1, then location 2 (pins drop in)              <- whoosh
  20.20  "STOP BY & GRAB YOUR FAVOURITES!"                       <- bell
  21.00 / 23.00  shine sweeps                                    <- sparkles
  25.00  final hit                                               <- final hit
  26.10  fade to black, ends 26.5 s

    python reel1_two_locations.py
    python reel1_two_locations.py --test 20.8,25.1
"""
import os
import argparse
import shutil
import subprocess
import time
from multiprocessing import Pool

import numpy as np
import cv2
from scipy.io import wavfile

import fx
from fx import finish
import reel2_two_locations as TLOC
from timeline import FPS

ROOT = fx.ROOT
SRC_FRAMES = os.path.join(ROOT, "output", "build", "frames")
SRC_AUDIO = os.path.join(ROOT, "output", "build", "audio_mix.wav")
ORIG_MP4 = os.path.join(ROOT, "output", "ezy_pzy_reel_1080x1920.mp4")
BUILD = os.path.join(ROOT, "output", "build_reel1_updated")
FRAMES = os.path.join(BUILD, "frames")
OUT_MP4 = os.path.join(ROOT, "output", "Ezy_Pzy_Convenience_Reel_1_Updated.mp4")


class T1:
    END_START = 19.0
    LOGO = 19.15
    HEAD = 19.30
    RULE = 19.60
    ADDR = (19.75, 19.93)
    CTA = 20.20
    SHINES = (21.0, 23.0)
    FINAL = 25.0
    FADE = 26.1
    DURATION = 26.5
    LOOP_FROM, LOOP_TO = 21.0, 23.0


LOCATIONS_R1 = (("153 Great Northern Rd,", "Sault Ste. Marie, ON P6B 4Y9"),
                ("216 Second Line W,", "Sault Ste. Marie, ON P6C 2J3"))
# Reel 1's own end-card light leaks (its four flavour colours)
LEAKS_R1 = [((0.45, 1.0, 0.2), 120, 380), ((0.45, 0.3, 1.0), 980, 520), ((0.3, 0.85, 0.8), 160, 1500), ((0.3, 0.7, 1.0), 950, 1650)]
FIRST_NEW = int(round(T1.END_START * FPS))     # 570
N_FRAMES = int(round(T1.DURATION * FPS))       # 795
# Reel 1's original encode settings (identical SPS/PPS => seamless splice)
X264_R1 = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "14M", "-bufsize", "28M", "-tune", "film",
           "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-g", str(FPS)]


def render_frame(fi):
    t = fi / FPS
    canvas, post = TLOC.end_card_two_locations(t, fi, T_=T1, locations=LOCATIONS_R1, leaks=LEAKS_R1)
    img = finish(canvas, fi, flash=min(1.0, post["flash"]), flash_color=post["flash_color"], ca=post["ca"], grain=post["grain"])
    img *= post["fade"]
    return (img * 255 + 0.5).astype(np.uint8)


def _work(fi):
    cv2.imwrite(os.path.join(FRAMES, f"{fi:05d}.png"), cv2.cvtColor(render_frame(fi), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return fi


def build_audio(path):
    sr, a = wavfile.read(SRC_AUDIO)
    xf = int(0.012 * sr)
    j, k = int(round(T1.LOOP_TO * sr)), int(round(T1.LOOP_FROM * sr))
    af = a.astype(np.float64)
    tt = np.linspace(0, np.pi / 2, xf)[:, None]
    seam = af[j - xf:j] * np.cos(tt) + af[k - xf:k] * np.sin(tt)
    out = np.clip(np.round(np.concatenate([af[:j - xf], seam, af[k:]])), -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, out)
    return sr, a, out, j - xf


def keyframes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=key_frame",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()
    return [i for i, k in enumerate(out) if k.strip().startswith("1")]


def sps_pps(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-frames:v", "1",
                          "-f", "h264", "-"], capture_output=True).stdout
    out = {}
    for chunk in raw.split(b"\x00\x00\x01"):
        if chunk and (chunk[0] & 0x1F) in (7, 8):
            out.setdefault(chunk[0] & 0x1F, chunk.rstrip(b"\x00"))
    return out


def framemd5(path, n=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v"] + (["-frames:v", str(n)] if n else []) + ["-f", "framemd5", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def encode_smart(wav):
    ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    K = max(k for k in keyframes(ORIG_MP4) if k <= FIRST_NEW)
    head, tail, lst = (os.path.join(BUILD, n) for n in ("head.mp4", "tail.mp4", "list.txt"))
    subprocess.run(ff + ["-i", ORIG_MP4, "-map", "0:v:0", "-c:v", "copy", "-frames:v", str(K), head], check=True)
    subprocess.run(ff + ["-framerate", str(FPS), "-start_number", str(K), "-i", os.path.join(FRAMES, "%05d.png"),
                         "-frames:v", str(N_FRAMES - K)] + X264_R1 + ["-video_track_timescale", "15360", tail], check=True)
    assert sps_pps(head) == sps_pps(tail), "SPS/PPS differ between original and new segment"
    with open(lst, "w") as f:
        f.write(f"file '{head}'\nfile '{tail}'\n")
    subprocess.run(ff + ["-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-map", "0:v", "-map", "1:a",
                         "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", "-shortest",
                         OUT_MP4], check=True)
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.test:
        out = os.path.join(BUILD, "test")
        os.makedirs(out, exist_ok=True)
        for ts in args.test.split(","):
            t0 = time.time()
            cv2.imwrite(os.path.join(out, f"t{float(ts):06.2f}.png"), cv2.cvtColor(render_frame(int(round(float(ts) * FPS))), cv2.COLOR_RGB2BGR))
            print(f"t={ts} {time.time() - t0:.2f}s", flush=True)
        return
    assert len(os.listdir(SRC_FRAMES)) >= FIRST_NEW, "render Reel 1 first (build.sh)"
    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)
    for fi in range(FIRST_NEW):
        src, dst = os.path.join(SRC_FRAMES, f"{fi:05d}.png"), os.path.join(FRAMES, f"{fi:05d}.png")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    t0 = time.time()
    with Pool(args.jobs) as p:
        for n, _ in enumerate(p.imap_unordered(_work, range(FIRST_NEW, N_FRAMES), chunksize=2)):
            if (n + 1) % 45 == 0:
                print(f"end card {n + 1}/{N_FRAMES - FIRST_NEW} frames {time.time() - t0:.0f}s", flush=True)
    wav = os.path.join(BUILD, "audio_mix.wav")
    sr, orig, new, seam = build_audio(wav)
    print(f"audio: {len(new) / sr:.3f}s, first {seam / sr:.3f}s bit-identical to original: {np.array_equal(orig[:seam], new[:seam])}")
    K = encode_smart(wav)
    same = framemd5(ORIG_MP4, K) == framemd5(OUT_MP4, K)
    print(f"video: frames 0-{K - 1} stream-copied from the original Reel 1 (decoded identical: {same}); frames {K}-{N_FRAMES - 1} encoded")
    print("wrote", OUT_MP4)


if __name__ == "__main__":
    main()
