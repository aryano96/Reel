"""Generic "text-only patch" helpers: re-encode only the GOPs that contain changed
frames, copy every other GOP of the source MP4 bit-for-bit, keep the audio."""
import os
import shutil
import subprocess

FPS = 30
FF = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]


def keyframes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=key_frame",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()
    return [i for i, k in enumerate(out) if k.strip().startswith("1")]


def framemd5(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def sps_pps(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-frames:v", "1",
                          "-f", "h264", "-"], capture_output=True).stdout
    out = {}
    for chunk in raw.split(b"\x00\x00\x01"):
        if chunk and (chunk[0] & 0x1F) in (7, 8):
            out.setdefault(chunk[0] & 0x1F, chunk.rstrip(b"\x00"))
    return out


def encode_matched(frames_dir, a, b, target_bytes, out, x264):
    """Encode frames [a, b) with the source's x264 settings, picking the VBV start level so the
    GOP gets the same bit budget as the original GOP (SPS/PPS unchanged)."""
    lo, hi, best = 0.0, 0.9, None
    for _ in range(7):
        mid = (lo + hi) / 2 if best is not None else 0.9
        subprocess.run(FF + ["-framerate", str(FPS), "-start_number", str(a), "-i", os.path.join(frames_dir, "%05d.png"),
                             "-frames:v", str(b - a)] + x264 + ["-x264-params", f"vbv-init={mid:.4f}",
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


def splice(src_mp4, frames_dir, wav, build, out_mp4, changed, n_frames, x264):
    segs = os.path.join(build, "segments")
    shutil.rmtree(segs, ignore_errors=True)
    os.makedirs(segs)
    kf = keyframes(src_mp4)
    bounds = kf + [n_frames]
    gops = list(zip(bounds[:-1], bounds[1:]))
    subprocess.run(FF + ["-i", src_mp4, "-map", "0:v:0", "-c", "copy", "-f", "segment",
                         "-segment_frames", ",".join(str(k) for k in kf[1:]), "-reset_timestamps", "1",
                         "-segment_format", "mp4", os.path.join(segs, "orig_%03d.mp4")], check=True)
    chs = set(changed)
    files, n_copy, n_reenc = [], 0, 0
    for gi, (a, b) in enumerate(gops):
        orig = os.path.join(segs, f"orig_{gi:03d}.mp4")
        if any(a <= fi < b for fi in chs):
            seg = os.path.join(segs, f"new_{gi:03d}.mp4")
            vi, size = encode_matched(frames_dir, a, b, os.path.getsize(orig), seg, x264)
            assert sps_pps(seg) == sps_pps(orig), f"SPS/PPS mismatch in GOP {gi}"
            print(f"  GOP {gi:2d} frames {a}-{b - 1}: original {os.path.getsize(orig) / 1e6:.2f} MB -> {size / 1e6:.2f} MB", flush=True)
            n_reenc += b - a
        else:
            seg = orig
            n_copy += b - a
        files.append(seg)
    lst = os.path.join(build, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in files)
    subprocess.run(FF + ["-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-map", "0:v", "-map", "1:a",
                         "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", "-shortest",
                         out_mp4], check=True)
    a, b = framemd5(src_mp4), framemd5(out_mp4)
    assert len(b) == n_frames, f"frame count {len(b)} != {n_frames}"
    copied = [fi for (s, e) in gops for fi in range(s, e) if not any(s <= c < e for c in chs)]
    same = sum(a[fi] == b[fi] for fi in copied)
    print(f"GOPs: {len(gops)}; frames copied bit-for-bit: {n_copy}, in re-encoded GOPs: {n_reenc}")
    print(f"verify: {len(b)} frames; {same}/{len(copied)} copied frames decode identical to the source reel")
