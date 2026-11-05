#!/usr/bin/env bash
# One-command rebuild of the Ezy Pzy Convenience product reel.
#   ./build.sh            full pipeline (cutouts -> audio -> frames -> MP4 -> thumbnail)
#   SKIP_CUTOUTS=1 ./build.sh   reuse existing assets/cutouts
set -euo pipefail
cd "$(dirname "$0")"
OUT=output
BUILD=$OUT/build
mkdir -p "$BUILD"

if [ -z "${SKIP_CUTOUTS:-}" ] || [ ! -d assets/cutouts ]; then
  echo "== 1/5 cutting cans out of product photos"
  python3 src/extract_cans.py
fi

echo "== 2/5 synthesizing original soundtrack + sound design"
(cd src && python3 audio.py)

echo "== 3/5 rendering 1080x1920 frames @30fps"
rm -rf "$BUILD/frames"
(cd src && python3 render.py --all)

echo "== 4/5 encoding Instagram-ready MP4 (H.264 High, yuv420p, AAC 48k)"
ffmpeg -y -hide_banner -loglevel error -framerate 30 -i "$BUILD/frames/%05d.png" -i "$BUILD/audio_mix.wav" \
  -c:v libx264 -preset slow -crf 18 -maxrate 14M -bufsize 28M -tune film -profile:v high -level 4.2 -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 30 -g 30 \
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart \
  "$OUT/ezy_pzy_reel_1080x1920.mp4"

echo "== 5/5 thumbnail"
(cd src && python3 thumbnail.py)

ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate:format=duration,size,bit_rate \
  -of default=nw=1 "$OUT/ezy_pzy_reel_1080x1920.mp4"
echo "done -> $OUT/ezy_pzy_reel_1080x1920.mp4"
