"""Reel 3 timing (Alani Nu x9) - single source of truth for picture and sound.

160 BPM half-time trap: 1 beat = 0.375 s, 1 bar = 1.5 s.
Each drink gets 6 beats (2.25 s); every cut and transition lands on a beat.
Rendered at 60 fps.
"""
FPS = 60
W, H = 1080, 1920
BPM = 160
BEAT = 60.0 / BPM          # 0.375 s
BAR = 4 * BEAT
S16 = BEAT / 4

DROP = 3.0
SCENE = 2.25               # 6 beats per drink
N_DRINKS = 9
DRINK_STARTS = [DROP + i * SCENE for i in range(N_DRINKS)]
END_START = DROP + N_DRINKS * SCENE        # 23.25
DURATION = END_START + 3 * BAR             # 27.75

INTRO_WORDS = [(0.375, "ICE COLD."), (1.125, "UNBEATABLE"), (1.875, "DEALS.")]

PRODUCTS = [
    dict(key="01_cotton_candy",        name="COTTON CANDY"),
    dict(key="02_purple_cotton_candy", name="PURPLE COTTON CANDY"),
    dict(key="03_witchs_brew",         name="WITCH'S BREW"),
    dict(key="04_hawaiian_shaved_ice", name="HAWAIIAN SHAVED ICE"),
    dict(key="05_orange_kiss",         name="ORANGE KISS"),
    dict(key="06_cherry_slush",        name="CHERRY SLUSH"),
    dict(key="07_breezeberry",         name="BREEZEBERRY"),
    dict(key="08_cosmic_stardust",     name="COSMIC STARDUST"),
    dict(key="09_watermelon_wave",     name="WATERMELON WAVE"),
]

# per-scene beat offsets (relative to each drink's start)
CLOSE = 3 * BEAT           # 1.125  cut to hero close-up
MEDIUM = 5 * BEAT          # 1.875  back to the full can
EXIT = SCENE - 0.27        # exit transition starts

CAN_OPENS = [DRINK_STARTS[0] + CLOSE, DRINK_STARTS[8] + CLOSE]
STRIKES_S6 = [DRINK_STARTS[5] + k * BEAT for k in (0, 2, 4, 5)]
ICE_HITS_S4 = [DRINK_STARTS[3] + d for d in (0.30, 0.52, 0.75, 1.02, 1.30)]

# end card
END_LOGO = END_START + 0.15
END_ADDR = END_START + 0.45
END_OFFER = END_START + 2 * BEAT
END_CTA = END_START + 3 * BEAT
END_SHINES = (END_START + 5 * BEAT, END_START + 7 * BEAT)
END_FINAL = END_START + 8 * BEAT           # 26.25
END_FADE = DURATION - 0.45

# price label (every frame)
PRICE_MAIN = ("2 FOR", "$5.99")
PRICE_SUB = "BEFORE TAXES"
