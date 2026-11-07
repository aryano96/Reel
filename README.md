# Ezy Pzy Convenience — Cinematic Product Reels

Two 9:16 Instagram Reel ads for **Ezy Pzy Convenience**
(153 Great Northern Rd, Sault Ste. Marie, ON P6B 4Y9). In both, every drink gets
its own scene and only one product is ever on screen at a time.

| Reel | Products | Length | Files |
|---|---|---|---|
| **Reel 1** | Bucked Up Lightly Carbonated Protein: Cherry Limeade, Rainbow Candy, Green Apple, Banana Pineapple | 24.5 s | `output/ezy_pzy_reel_1080x1920.mp4`, `output/ezy_pzy_reel_thumbnail.jpg`, `output/storyboard.jpg` |
| **Reel 1, Updated** | Same as Reel 1, with a new ending showing the store logo and both locations | 26.5 s | `output/Ezy_Pzy_Convenience_Reel_1_Updated.mp4`, `output/Reel_1_Updated_ending_storyboard.jpg` |
| **Reel 1, Final** (latest) | Reel 1 Updated with the opening headline changed to "POWER EVERY REP WITH BUCKED UP!" | 26.5 s | `output/Ezy_Pzy_Bucked_Up_Reel_1_Final.mp4` |
| **Reel 2** | Crush Pineapple, Crush Big Troppa Punch, Crush Peach, Fanta Strawberry, Fanta Pineapple, Fanta Mango, Fanta Crimson Sour Cherry | 28.5 s | `output/reel2/ezy_pzy_reel2_1080x1920.mp4`, `output/reel2/ezy_pzy_reel2_thumbnail.jpg`, `output/reel2/storyboard.jpg` |
| **Reel 2, Two Locations edition** | Same as Reel 2, with a new closing scene showing the store logo picture and both locations | 30.0 s | `output/reel2/Ezy_Pzy_Convenience_Two_Locations_Reel.mp4`, `output/reel2/Two_Locations_ending_storyboard.jpg` |
| **Reel 2, Final** | The Two Locations edition with the store name corrected to "EZY PZY CONVENIENCE" in every drink scene | 30.0 s | `output/reel2/Ezy_Pzy_Convenience_Final_Reel.mp4` |
| **Reel 2, Logo Final** (latest) | The Final Reel with the store-name text line under each product name replaced by the store logo | 30.0 s | `output/reel2/Ezy_Pzy_Convenience_Logo_Final_Reel.mp4` |

| **Reel 3** | Alani Nu: Cotton Candy, Purple Cotton Candy, Witch's Brew, Hawaiian Shaved Ice, Orange Kiss, Cherry Slush, Breezeberry, Cosmic Stardust, Watermelon Wave, with a "2 FOR $5.99 BEFORE TAXES" price label on every frame | 27.75 s @ 60 fps | `output/reel3/Ezy_Pzy_Alani_Nu_2_for_5.99_Reel.mp4`, `output/reel3/storyboard.jpg` |

Reels 1 and 2 are 1080×1920, 30 fps, H.264 High at about 12–15 Mbps, yuv420p, with AAC 320 kbps / 48 kHz stereo audio and faststart. Each is under 50 MB.

---

## Reel 2: Crush & Fanta (160 BPM, 2 bars = 3 s per drink)

| Time | Scene | Visuals | Sound |
|---|---|---|---|
| 0.0–3.0 | **Intro** | Dark stage. Light sweeps and rim lights slowly reveal Crush Pineapple during a slow push-in. "THE ULTIMATE / REFRESHMENT / IS HERE." slams in word by word, then energy gathers behind the can | Dark pad, sub hits, riser, snare roll, reverse cymbal |
| 3.0–6.0 | **Crush Pineapple** | Bass drop, white flash, shockwave, and a golden liquid explosion in slow motion behind the can. Sun rays, out-of-focus foreground drops, a close-up pan to "Pineapple", and a can crack with rising bubbles | Impact, splash, can opening, fizz |
| 5.6–6.45 | *Transition* | A neon pink/orange/yellow/cyan liquid wave covers the frame, then drains to reveal the next can | Liquid surge, splash, gurgle |
| 6.0–9.0 | **Crush Big Troppa Punch** | Retro striped synthwave sun behind the can, floating 90s shapes, neon droplets, VHS glitch hits on the cuts, then a whip pan out | Vinyl scratches, arp, whoosh |
| 9.0–12.0 | **Crush Peach** | Whip pan in with a bubble burst. Fizzy carbonation bubbles rise in front of and behind the can, with a can crack and bubble stream. A bubble swarm rushes the camera into a white flash | Bubble "bloops", fizz, can opening, riser |
| 12.0–15.0 | **Fanta Strawberry** | Zoom out of the flash. Crimson smoke, glowing embers orbiting the can (passing behind and in front of it) and red light rays. Smoke then rises to fill the frame | Shimmer, low wind, smoke whoosh |
| 15.0–18.0 | **Fanta Pineapple** | The smoke clears as a lightning strike hits. Bolts flicker on the beat, arcs crawl along the can and sparks burst out, then a zoom-through rush into a flash | Thunder, zaps on every strike, electric hum |
| 18.0–21.0 | **Fanta Mango** | Zoom out of the flash. Ice cubes tumble down and land on the beat, with cold mist, floor fog, slow-motion mango juice drops and a vapour burst on the can crack. Then a vertical whip down | Ice clinks synced to each landing, cold wind, can opening |
| 21.0–24.0 | **Fanta Crimson Sour Cherry** | Vertical whip in with a dark-red liquid burst and shockwave. Flames rise behind the can with embers and smoke, then a fire burst floods the frame | Fire roar, crackle, boom |
| 24.0–28.5 | **End card** | "EZY PZY" drops in letter by letter with a gold flare. The "CONVENIENCE" subtitle, gold rule, full address and "STOP BY & GRAB YOUR FAVOURITES!" pill follow, with shine sweeps, gold dust and a slow push-in, then a fade to black. The complete card holds for about 3.5 s | Impact and chord, sparkles, final hit |

**How Reel 2 animates real photos.** These product photos have fruit and ice placed *in front of* each can. Cutting out a bare can would mean inventing the hidden label pixels. Instead, each photo is used as a 2.5D plate, and the photo itself is never altered:
1. `src/reel2_prep.py` builds a matte of the can plus the props touching its base, using U²-Net and ISNet through `rembg`, and measures the can's position.
2. `Plate.render()` in `src/reel2_render.py` draws the photo, then the "behind" effects (splashes, smoke, lightning, fire, retro sun…), then the hero can on top through the matte, so effects pass *behind* the can.
3. Foreground effects, camera moves (push-ins, close-ups, zoom-outs, whips), light sweeps, rim lights and the transitions are layered on top.

Files: `src/timeline2.py` (timing), `src/reel2_audio.py` (soundtrack), `src/reel2_render.py` (scenes), `src/reel2_thumbnail.py`, `build_reel2.sh`.

### Two Locations edition (`src/reel2_two_locations.py`)

Only the closing store-promotion scene is new. Everything before it is reused, not re-rendered:
- **Video:** the original export's compressed video stream is copied bit-for-bit up to its last keyframe before the end card (frames 0–698, up to 23.3 s). Frames 699–719 are encoded from the identical original source frames. All 699 frames decode pixel-identical to the original file.
- **Audio:** the original mix is untouched (decoded samples identical) up to 26.99 s. To give the addresses more reading time, the end card's last groove bar (25.5–27.0 s) plays twice before the final hit, joined with a 12 ms crossfade just before the downbeat. That's a standard music edit, so the soundtrack stays the original and the ending gets 1.5 s longer (30.0 s total).

The new closing scene (24.0–30.0 s) is synced to the original end-card hits:

| Time | Element | Synced to |
|---|---|---|
| 24.15 s | The uploaded Ezy Pzy Convenience logo picture, shown unaltered on a rounded card with a gold rim, glow, shadow and reflection. It reveals with a light-wipe from the centre outward, then gently floats and pushes in | Low hit |
| 24.30 s | "VISIT US AT TWO LOCATIONS!" and a gold rule | Whoosh |
| 24.75 / 24.93 s | 📍 153 Great Northern Rd, Sault Ste. Marie, ON P6B 4Y9 and 📍 216 Second Line West, Sault Ste. Marie, ON. Each sits on a glass panel and its pin drops in | Whoosh |
| 25.20 s | "STOP BY & GRAB YOUR FAVOURITES!" gold pill | Bell |
| 26.0 / 27.5 s | Shine sweeps | Sparkles |
| 28.5 s | Final punch and glow | Final hit |
| 29.6–30.0 s | Fade to black | Ring-out |

The complete card is on screen together for about 4 s, and the addresses are readable for about 4.4 s. No postal code was added for the second address. Rebuild with `TWO_LOCATIONS=1 ./build_reel2.sh`, or run `python3 src/reel2_two_locations.py` after Reel 2's frames exist.

### Final edition (`src/reel2_final.py`)

This is a text-only correction to the Two Locations edition. The line under each product name in the seven drink scenes changes from "NOW AT EZY PZY" to **"NOW AT EZY PZY CONVENIENCE"**, with the same font, size, spacing, position and animation. No other on-screen text contained the store name: the end card already shows the full name through the logo picture.
- Only the 244 frames where that line is visible were re-rendered. The renderer is deterministic: with the old text it reproduces the original frames pixel-for-pixel, and the script checks this before building.
- The Two Locations MP4 is split at its keyframes. GOPs (groups of frames between keyframes) without a changed frame are copied bit-for-bit: 469 frames, including the whole end card. GOPs containing a changed frame are re-encoded with the original settings and matched to the original GOP's bit budget, so compression quality stays consistent.
- The audio is identical (decoded samples match). Duration (30.0 s), resolution, frame rate and file size (48.3 MB) are unchanged.

### Logo Final edition (`src/reel2_logo_final.py`)

The text line "NOW AT EZY PZY CONVENIENCE" under each product name, in all seven drink scenes, is replaced by the supplied store logo (`assets/brand/ezy_pzy_logo.jpg`), centred where the text was. The logo is shown exactly as designed on a small rounded white badge, with only its empty white margin trimmed. A white badge keeps the black "CONVENIENCE" lettering readable on the dark scenes. It has a soft shadow and a thin gold edge, matching the end card. It uses the line's existing timing with a subtle zoom-in and a single light shine.

The build method is the same as the Final edition. The script checks that the renderer reproduces the Final Reel's frames exactly, then re-renders only the 244 frames showing the line. GOPs without a changed frame are copied bit-for-bit (469 frames, including the whole end card), and the rest are re-encoded to the original bit budget. The audio is identical, and duration, resolution and size are unchanged.

---

## Reel 1: Bucked Up protein (120 BPM, 4 s per drink)

| Time | Scene | Visuals | Sound |
|---|---|---|---|
| 0.0–3.0 | **Intro** | Dark stage with a volumetric light cone and dust. Light sweeps reveal the first can during a slow push-in while "THE ULTIMATE / REFRESHMENT / IS HERE." slams in | Dark pad, sub hits, riser, snare roll, reverse cymbal |
| 3.0–7.0 | **Cherry Limeade** | Bass drop with an explosive lime and cherry liquid splash in slow motion, a cylinder orbit, a close-up tilt down the label, and a can crack with fizz | Impact, splash, can opening, fizz |
| 6.6–7.5 | *Transition* | A rainbow liquid wave fully covers the frame, then drains to reveal the next can | Liquid surge, splash, gurgle |
| 7.0–11.0 | **Rainbow Candy** | Candy orbs orbiting the can, rainbow droplets, a close-up pan up the candy art, and a whip pan out | Arp, candy "bloops", whoosh |
| 11.0–15.0 | **Green Apple** | Tumbling ice landing on the beat, cold mist, vapour, a can-crack vapour burst, and a zoom-blur rush | Ice clinks, cold wind, can crack |
| 15.0–19.0 | **Banana Pineapple** | Lightning strike reveal, electric arcs, energy rings, sparks, punch-ins, and a vertical whip out | Thunder, zaps, electric hum, riser |
| 19.0–24.5 | **End card** | Store name, full address and call to action with gold lighting. The complete card holds for about 3.5 s | Impact, chord, sparkles, final hit |

These photos had clean backgrounds, so each can was cut out (`src/extract_cans.py`) and rendered with a gentle cylindrical rotation, a specular sweep and a floor reflection.

### Reel 1, Updated (`src/reel1_two_locations.py`)

Only the closing scene changes. It's the same ending design as Reel 2's Two Locations edition, in Reel 1's own lighting:
1. The store logo, revealed with a light-wipe at 19.15 s.
2. "VISIT US AT TWO LOCATIONS!"
3. 📍 153 Great Northern Rd, Sault Ste. Marie, ON P6B 4Y9
4. 📍 216 Second Line W, Sault Ste. Marie, ON P6C 2J3
5. "STOP BY & GRAB YOUR FAVOURITES!"

Each element lands on the original end-card hits, and the complete card stays on screen for about 5.5 s.

Everything before it is reused, not re-rendered:
- **Video:** the original Reel 1 video stream is copied bit-for-bit up to its last keyframe before the ending (frames 0–544, up to 18.1 s). Frames 545–569 are re-encoded from the identical original source frames.
- **Audio:** the original mix is untouched (decoded samples identical) up to 22.99 s. The end card's last groove bar (21.0–23.0 s) then plays twice before the final hit, so the ending is 2 s longer (26.5 s total).

Rebuild with `python3 src/reel1_two_locations.py` after Reel 1's frames exist (`./build.sh`).

### Reel 1, Final (`src/reel1_final.py`)

Only the opening headline changes, to **"POWER EVERY REP WITH BUCKED UP!"**. It keeps the original three slams on the same three bass hits: "POWER EVERY REP" / "WITH" / "BUCKED UP!". The fonts, white and lime-gradient colours, glow, slam-in and the burst on the bass drop are all unchanged, and the lime highlight now lands on BUCKED UP!.
- Only the 82 opening frames showing the headline (0.27–2.97 s) were re-rendered, after the script checks that the renderer reproduces the current frames exactly.
- Every other GOP of the Updated Reel 1 is copied bit-for-bit: 705 of 795 frames, including all four drink scenes and the whole ending.
- The audio is identical. The generic patch helpers live in `src/patching.py`.

---

## Shared engine

No stock footage, templates or licensed audio were used. Everything is generated by code.

- **`src/fx.py`** is the effects toolkit:
  - a metaball liquid simulator with surface normals, for glossy splashes and waves
  - refractive ice cubes, fractal-noise mist and smoke, procedural lightning and particles
  - bloom, film grain, chromatic aberration, motion blur and zoom blur
  - text rendering
- **`src/synth.py`** synthesizes every instrument and sound effect from scratch: kick, 808, claps, supersaw hooks, pads, arps, splash, fizz, can crack, ice clinks, zaps, thunder, whooshes, risers and impacts. Reel 2 adds vinyl scratch, fire roar and crackle. Both soundtracks are mastered to about −10 LUFS with a −1 dBTP ceiling. **All audio is 100% original**, with no copyright claims.
- **`src/render.py`** holds Reel 1's scenes plus the shared `end_card()`.

## Rebuild

```bash
pip install -r requirements.txt
./build.sh                         # Reel 1 (~5 min on 4 cores)
./build_reel2.sh                   # Reel 2 (~7 min on 4 cores)
python3 src/reel2_render.py --test 3.5,8.0   # preview single frames into output/build2/test/
```

## Notes and limitations

- **Store logo:** the supplied logo picture (`assets/brand/ezy_pzy_logo.jpg`) appears unaltered in the Two Locations edition. Reel 1 and the original Reel 2 still use the typography-only end card.
- **Product fidelity:** the cans are the supplied photos, never redrawn or relabelled. Movement is 2.5D: camera moves, parallax effect layers and light sweeps. Real 3D turnarounds aren't possible from a single front photo.
- **On-screen copy:** product names exactly as printed on the cans, plus "NOW AT EZY PZY" and the store details. No nutrition or health claims are added.
- Fonts: Inter / Inter Display (SIL Open Font License), included in `assets/fonts/`.

---

## Storefront reveal (`src/storefront_reveal.py`)

`output/storefront/Ezy_Pzy_Cinematic_Storefront_Reveal.mp4` is a 9.5 s, 1080×1920, 30 fps cinematic storefront reveal made from the real store video (`assets/storefront/store_video_original.MOV`).

**What it is and isn't:** the source is a 5.5 s, 360×480, night-time phone clip filmed from ground level, zooming toward the store. It contains no aerial footage, so a real drone descent (sky → down → forward) **cannot** be made from it honestly. Any "from above" view would have to be invented, which would mean inventing the building and its surroundings. This video is instead an honest cinematic treatment of the real footage. Nothing in the frame is AI-generated.
- **Camera tracking:** SIFT + RANSAC places every frame in one shared space, so the move is re-timed into a single eased approach. It starts with a wide hold, then a smooth push to the storefront, then a gentle push-in and a hold on the hero shot. The original speed jumps, handheld shake and the iPhone lens-switch exposure jump are all removed.
- **Cleanup:** the filmer's shadow is relit (asphalt texture kept), the parked car is cloned out, and small litter specks on the lot are removed. The building, signage, windows, door and neighbouring businesses are untouched.
- **Look:** denoising, multi-frame averaging on the hero hold, a night cinematic grade (navy sky, warm store light, bloom, vignette, grain) and subtle zoom blur on the push.
- **Sound** (`src/storefront_audio.py`): original and synthesized. It has night ambience and a pad, a build-up and aerial whooshes during the approach, an impact on the reveal, and a warm ending.

The source resolution limits sharpness: the storefront is about 360 px wide in the original, so the hero shot is softer than native 1080p. A real drone clip (ideally at golden hour), or a higher-resolution 4K phone video, would remove both limits.


---

## Reel 3: Alani Nu, 2 FOR $5.99 (`src/reel3_render.py`, 160 BPM, 6 beats = 2.25 s per drink, 60 fps)

| Time | Scene | Visuals | Transition out |
|---|---|---|---|
| 0.00–3.00 | **Intro** | Dark stage; light sweeps reveal the first can during a push-in; "ICE COLD." / "UNBEATABLE" / "DEALS." slam in on three bass hits; energy gathers behind the can | Bass drop |
| 3.00–5.25 | **Cotton Candy** | Pink + teal liquid explosion in slow motion, shockwave, foreground drops, close-up, can crack | Purple/teal liquid wave |
| 5.25–7.50 | **Purple Cotton Candy** | Carnival string lights chasing, confetti | Vertical whip up |
| 7.50–9.75 | **Witch's Brew** | Green cauldron glow, rising bubbles, purple smoke, orbiting embers | Green smoke wipe |
| 9.75–12.00 | **Hawaiian Shaved Ice** | Ice cubes landing on the beat, cold mist, frost sparkles | Vertical whip down |
| 12.00–14.25 | **Orange Kiss** | Sunburst rays, orange/coral splash, fizz | Sun-flare burst |
| 14.25–16.50 | **Cherry Slush** | Teal lightning on the beat, red sparks | Blue liquid wave |
| 16.50–18.75 | **Breezeberry** | Slow-motion blue droplets, frost | Hyperspace warp flash |
| 18.75–21.00 | **Cosmic Stardust** | Warp streaks into a starfield and nebula, orbiting stardust | Vertical whip up |
| 21.00–23.25 | **Watermelon Wave** | Pink/teal splash, can crack | Flash + whip |
| 23.25–27.75 | **End card** | Store logo, 📍 153 Great Northern Rd, Sault Ste. Marie, ON P6B 4Y9, "2 FOR $5.99 / BEFORE TAXES" (lands on a hit), and "STOP BY & GRAB YOUR FAVOURITES!" | Final hit, fade |

**Price label:** a red retail tag in the left margin reading "2 FOR / $5.99 / BEFORE TAXES", composited last on every frame, intro and end card included. It pulses with the beat and gets a shine sweep every 3 s. The layout guarantees no can ever touches it: close-ups are framed right of centre, all whips are vertical, and camera shake is mostly vertical. The build was verified on the decoded MP4: the label was intact in all 1,665 frames, and the closest any can came was 21 px from its edge.

**Photos:** fruit, ice and splashes sit in front of the cans, so each photo is used as a 2.5D plate. The packaging is untouched, a matte lets effects pass behind the can, and `src/reel2_prep.py reel3 --robust` measures each can from its straight edges when props merge into the outline.

**Audio** (`src/reel3_audio.py`): an original 160 BPM trap track with an impact on every reveal and sound design matched to each scene (splash, can crack, fizz, carnival chimes, cauldron bubbles, ice clinks, thunder and zaps, warp riser, sparkles). It's mastered to −10.7 LUFS with a −1.0 dBTP ceiling.

Rebuild everything with `./build_reel3.sh` (about 12 minutes on 4 cores).
