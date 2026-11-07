#!/usr/bin/env bash
# One-command rebuild of Reel 3 (Alani Nu x9, 2 FOR $5.99 offer) - current version:
# bottom price banner (hidden on the end card) + both store locations on the end card.
# The first version (side price tag, one address) is output/reel3/Ezy_Pzy_Alani_Nu_2_for_5.99_Reel.mp4,
# built by this script at commit 0b91aa3.
#   ./build_reel3.sh              full pipeline (mattes -> audio -> frames -> MP4)
#   SKIP_PREP=1 ./build_reel3.sh  reuse existing assets/reel3/mattes
set -euo pipefail
cd "$(dirname "$0")"
OUT=output/reel3
BUILD=output/build3_v2
mkdir -p "$OUT" "$BUILD"
if [ -z "${SKIP_PREP:-}" ] || [ ! -f assets/reel3/geometry.json ]; then
  echo "== 1/4 building mattes + can geometry"
  python3 src/reel2_prep.py reel3 --robust
fi
echo "== 2/4 synthesizing original soundtrack + sound design"
(cd src && python3 reel3_audio.py) && mkdir -p "$BUILD" && cp output/build3/audio_mix.wav "$BUILD/audio_mix.wav"
echo "== 3/4 rendering 1080x1920 frames @60fps"
rm -rf "$BUILD/frames"
(cd src && python3 reel3_render.py --all)
echo "== 4/4 encoding Instagram-ready MP4 (H.264 High, 60 fps, yuv420p, AAC 48k)"
ffmpeg -y -hide_banner -loglevel error -framerate 60 -i "$BUILD/frames/%05d.png" -i "$BUILD/audio_mix.wav" \
  -c:v libx264 -preset slow -crf 18 -maxrate 13M -bufsize 26M -tune film -profile:v high -level 4.2 -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 60 -g 60 \
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart \
  "$OUT/Ezy_Pzy_Alani_Nu_2_for_5.99_Two_Locations_Reel.mp4"
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate:format=duration,size,bit_rate \
  -of default=nw=1 "$OUT/Ezy_Pzy_Alani_Nu_2_for_5.99_Two_Locations_Reel.mp4"
