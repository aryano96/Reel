#!/usr/bin/env bash
# One-command rebuild of Reel 2 (Crush & Fanta line-up).
#   ./build_reel2.sh              full pipeline (mattes -> audio -> frames -> MP4 -> thumbnail)
#   SKIP_PREP=1 ./build_reel2.sh  reuse existing assets/reel2/mattes
set -euo pipefail
cd "$(dirname "$0")"
OUT=output/reel2
BUILD=output/build2
mkdir -p "$OUT" "$BUILD"

if [ -z "${SKIP_PREP:-}" ] || [ ! -f assets/reel2/geometry.json ]; then
  echo "== 1/5 building mattes + can geometry"
  python3 src/reel2_prep.py
fi
echo "== 2/5 synthesizing original soundtrack + sound design"
(cd src && python3 reel2_audio.py)
echo "== 3/5 rendering 1080x1920 frames @30fps"
rm -rf "$BUILD/frames"
(cd src && python3 reel2_render.py --all)
echo "== 4/5 encoding Instagram-ready MP4 (H.264 High, yuv420p, AAC 48k)"
ffmpeg -y -hide_banner -loglevel error -framerate 30 -i "$BUILD/frames/%05d.png" -i "$BUILD/audio_mix.wav" \
  -c:v libx264 -preset slow -crf 18 -maxrate 12M -bufsize 24M -tune film -profile:v high -level 4.2 -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 30 -g 30 \
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart \
  "$OUT/ezy_pzy_reel2_1080x1920.mp4"
echo "== 5/5 thumbnail"
(cd src && python3 reel2_thumbnail.py)
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate:format=duration,size,bit_rate \
  -of default=nw=1 "$OUT/ezy_pzy_reel2_1080x1920.mp4"
echo "done -> $OUT/ezy_pzy_reel2_1080x1920.mp4"

# Optional: "Two Locations" edition (new closing scene only; everything before it is reused untouched)
if [ -n "${TWO_LOCATIONS:-}" ]; then
  echo "== two-locations edition"
  (cd src && python3 reel2_two_locations.py)
fi
