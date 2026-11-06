# Ezy Pzy Convenience — Cinematic Product Reels

Two 9:16 Instagram Reel ads for **Ezy Pzy Convenience**
(153 Great Northern Rd, Sault Ste. Marie, ON P6B 4Y9). In both, every drink gets
its own scene and only one product is ever on screen at a time.

| Reel | Products | Length | Files |
|---|---|---|---|
| **Reel 1** | Bucked Up Lightly Carbonated Protein: Cherry Limeade, Rainbow Candy, Green Apple, Banana Pineapple | 24.5 s | `output/ezy_pzy_reel_1080x1920.mp4`, `output/ezy_pzy_reel_thumbnail.jpg`, `output/storyboard.jpg` |
| **Reel 2** | Crush Pineapple, Crush Big Troppa Punch, Crush Peach, Fanta Strawberry, Fanta Pineapple, Fanta Mango, Fanta Crimson Sour Cherry | 28.5 s | `output/reel2/ezy_pzy_reel2_1080x1920.mp4`, `output/reel2/ezy_pzy_reel2_thumbnail.jpg`, `output/reel2/storyboard.jpg` |

Both videos are 1080×1920, 30 fps, H.264 High at about 12–15 Mbps, yuv420p, with AAC 320 kbps / 48 kHz stereo audio and faststart. Each is under 50 MB.

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

- **Store logo:** none was supplied, so the end card uses typography only (Inter Display). No logo was invented. To add the real logo, drop a transparent PNG in `assets/` and composite it in `end_card()` in `src/render.py`.
- **Product fidelity:** the cans are the supplied photos, never redrawn or relabelled. Movement is 2.5D: camera moves, parallax effect layers and light sweeps. Real 3D turnarounds aren't possible from a single front photo.
- **On-screen copy:** product names exactly as printed on the cans, plus "NOW AT EZY PZY" and the store details. No nutrition or health claims are added.
- Fonts: Inter / Inter Display (SIL Open Font License), included in `assets/fonts/`.
