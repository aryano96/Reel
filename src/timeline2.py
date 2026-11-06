"""Reel 2 timing - single source of truth for picture and sound.

160 BPM half-time trap: 1 beat = 0.375 s, 1 bar = 1.5 s.
Every drink gets exactly 2 bars (3.0 s); every cut lands on the beat grid.
"""
FPS = 30
W, H = 1080, 1920
BPM = 160
BEAT = 60.0 / BPM          # 0.375 s
BAR = 4 * BEAT             # 1.5 s
S16 = BEAT / 4

DROP = 3.0
SCENE = 3.0
DRINK_STARTS = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
END_START = 24.0
DURATION = 28.5

INTRO_WORDS = [(0.375, "THE ULTIMATE"), (1.125, "REFRESHMENT"), (1.875, "IS HERE.")]

PRODUCTS = [
    dict(key="01_crush_pineapple",           name="CRUSH PINEAPPLE",       fx="golden splash"),
    dict(key="02_crush_big_troppa_punch",    name="CRUSH BIG TROPPA PUNCH", fx="neon liquid wave"),
    dict(key="03_crush_peach",               name="CRUSH PEACH",           fx="fizz bubbles"),
    dict(key="04_fanta_strawberry",          name="FANTA STRAWBERRY",      fx="crimson smoke + embers"),
    dict(key="05_fanta_pineapple",           name="FANTA PINEAPPLE",       fx="electric lightning"),
    dict(key="06_fanta_mango",               name="FANTA MANGO",           fx="ice + cold mist"),
    dict(key="07_fanta_crimson_sour_cherry", name="FANTA CRIMSON SOUR CHERRY", fx="dark fire"),
]

# per-scene beat offsets (relative to each drink's start)
CLOSE = 1.5        # cut to close-up (beat 4)
MEDIUM = 2.25      # cut to medium (beat 6)
EXIT = 2.625       # exit transition begins (beat 7)

CAN_OPENS = [4.5, 10.5, 19.5]            # crack + fizz moments
STRIKES_S5 = [15.0, 15.75, 16.5, 16.875, 17.25]
ICE_HITS_S6 = [18.28, 18.5, 18.72, 19.05, 19.42]

# end card
END_NAME = 24.15
END_SUB = 24.45
END_RULE = 24.7
END_ADDR = 24.85
END_CTA = 25.2
END_SHINES = [26.0, 27.0]
END_FINAL = 27.0
END_FADE = 28.1
