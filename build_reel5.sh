#!/usr/bin/env bash
# One-command rebuild of Reel 5 - "Red Bull Energy Universe" (separate project from Reel 4).
#   ./build_reel5.sh              full pipeline (cutouts -> audio -> frames -> MP4 -> cover + product visuals)
set -euo pipefail
cd "$(dirname "$0")"
OUT=output/reel5
BUILD=output/build5
mkdir -p "$OUT" "$BUILD"
echo "== 1/5 product cutouts for the dark set"
python3 src/reel5_prep.py
echo "== 2/5 synthesizing the original soundtrack"
(cd src && python3 reel5_audio.py)
echo "== 3/5 rendering 1080x1920 frames @60fps"
rm -rf "$BUILD/frames"
(cd src && python3 reel5_render.py --all)
echo "== 4/5 encoding Instagram-ready MP4 (H.264 High, 60 fps, yuv420p, AAC 48k)"
ffmpeg -y -hide_banner -loglevel error -framerate 60 -i "$BUILD/frames/%05d.png" -i "$BUILD/audio_mix.wav" \
  -c:v libx264 -preset slow -crf 18 -maxrate 12M -bufsize 24M -tune film -profile:v high -level 4.2 -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -r 60 -g 60 \
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart \
  "$OUT/Ezy_Pzy_Red_Bull_Energy_Universe_Reel.mp4"
echo "== 5/5 cover image + per-can product visuals"
(cd src && python3 reel5_render.py --cover --stills)
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate:format=duration,size,bit_rate \
  -of default=nw=1 "$OUT/Ezy_Pzy_Red_Bull_Energy_Universe_Reel.mp4"
