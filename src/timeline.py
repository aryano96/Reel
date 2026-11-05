"""Single source of truth for timing - shared by the audio and the video renderer.

Music is 120 BPM (1 beat = 0.5 s, 1 bar = 2 s). Every cut, reveal and
transition below lands on the beat grid so picture and sound stay locked.
"""
FPS = 30
W, H = 1080, 1920
BPM = 120
BEAT = 60.0 / BPM          # 0.5 s
BAR = 4 * BEAT             # 2.0 s
S16 = BEAT / 4             # 16th note = 0.125 s

DROP = 3.0                 # bass drop / first product reveal
SCENE_LEN = 4.0            # 2 bars per drink
DRINK_STARTS = [3.0, 7.0, 11.0, 15.0]
END_START = 19.0
DURATION = 24.5

# intro text slams
INTRO_WORDS = [(0.25, "THE ULTIMATE"), (1.0, "REFRESHMENT"), (1.75, "IS HERE.")]

PRODUCTS = [
    dict(key="01_cherry_limeade",   name="CHERRY LIMEADE"),
    dict(key="02_rainbow_candy",    name="RAINBOW CANDY"),
    dict(key="03_green_apple",      name="GREEN APPLE"),
    dict(key="04_banana_pineapple", name="BANANA PINEAPPLE"),
]

# in-scene cut points (absolute seconds)
CUT_S1 = [4.5, 5.5]            # close-up, medium
CAN_OPEN_S1 = 5.5
WAVE_START = 6.62              # rainbow wave rises over scene 1 -> fully covered at 7.0
WAVE_CLEAR = 7.5               # scene 2 can fully revealed
CUT_S2 = [9.0, 10.0]
WHIP_OUT_S2 = 10.78
CUT_S3 = [13.0, 14.0]
CAN_OPEN_S3 = 13.5
ICE_HITS = [11.22, 11.47, 11.66, 11.95, 12.34, 12.71]
ZOOM_OUT_S3 = 14.72
STRIKES_S4 = [15.0, 15.5, 16.0, 16.75, 17.0, 17.5, 18.0]
CUT_S4 = [17.0, 17.5, 18.0]
WHIP_OUT_S4 = 18.75

# end card text entrances
END_NAME = 19.15
END_SUB = 19.45
END_RULE = 19.7
END_ADDR = 19.85
END_CTA = 20.2
END_SHINES = [21.0, 23.0]
END_FADE = 24.1
