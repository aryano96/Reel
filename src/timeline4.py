"""Reel 4 timing (Red Bull Editions x13) - single source of truth for picture and sound.

160 BPM half-time trap: 1 beat = 0.375 s, 1 bar = 1.5 s.
Each drink gets 5 beats (1.875 s): full can -> hero close-up -> full can -> transition.
Every reveal, cut and transition lands on a beat. Rendered at 60 fps.
"""
FPS = 60
W, H = 1080, 1920
BPM = 160
BEAT = 60.0 / BPM          # 0.375 s
BAR = 4 * BEAT
S16 = BEAT / 4

DROP = 3.0
SCENE = 5 * BEAT           # 1.875 s per drink
N_DRINKS = 13
DRINK_STARTS = [DROP + i * SCENE for i in range(N_DRINKS)]
END_START = DROP + N_DRINKS * SCENE        # 27.375 (beat 65 after the drop)
DURATION = END_START + 3 * BAR             # 31.875

INTRO_WORDS = [(0.375, "ICE COLD."), (1.125, "UNBEATABLE"), (1.875, "DEALS.")]

PRODUCTS = [
    dict(key="01_summer_sudachi_lime",       name="SUDACHI LIME",          edition="THE SUMMER EDITION"),
    dict(key="02_ice_iced_vanilla_berry",    name="ICED VANILLA BERRY",    edition="THE ICE EDITION"),
    dict(key="03_yellow_tropical",           name="TROPICAL",              edition="THE YELLOW EDITION"),
    dict(key="04_spring_cherry_sakura",      name="CHERRY SAKURA",         edition="THE SPRING EDITION"),
    dict(key="05_peach_white_peach",         name="WHITE PEACH",           edition="THE PEACH EDITION"),
    dict(key="06_red_watermelon",            name="WATERMELON",            edition="THE RED EDITION"),
    dict(key="07_sea_blue_juneberry",        name="JUNEBERRY",             edition="THE SEA BLUE EDITION"),
    dict(key="08_amber_strawberry_apricot",  name="STRAWBERRY APRICOT",    edition="THE AMBER EDITION"),
    dict(key="09_pink_wild_berries",         name="WILD BERRIES",          edition="THE PINK EDITION"),
    dict(key="10_lilac_grapefruit_blossom",  name="GRAPEFRUIT & BLOSSOM",  edition="THE LILAC EDITION"),
    dict(key="11_winter_pistachio_berries",  name="PISTACHIO & BERRIES",   edition="THE WINTER EDITION"),
    dict(key="12_coconut_berry",             name="COCONUT BERRY",         edition="THE COCONUT EDITION"),
    dict(key="13_apple_fuji_ginger",         name="FUJI APPLE & GINGER",   edition="THE APPLE EDITION"),
]

# per-scene offsets (relative to each drink's start)
CLOSE = 2 * BEAT           # 0.75   cut to hero close-up
MEDIUM = 3.5 * BEAT        # 1.3125 back to the full can
EXIT = SCENE - 0.27        # exit transition starts

S = DRINK_STARTS
CAN_OPENS = [S[0] + CLOSE, S[12] + CLOSE]                       # Summer, Apple
ICE_HITS = [S[1] + d for d in (0.22, 0.42, 0.62, 0.95)]         # Ice Edition: cubes land
STRIKES = [S[11] + k * BEAT for k in (0, 2, 3.5, 4)]            # Coconut Edition: lightning

# end card (same structure as Reel 3)
END_LOGO = END_START + 0.15
END_ADDR = END_START + 0.45
END_OFFER = END_START + 2 * BEAT
END_CTA = END_START + 3 * BEAT
END_SHINES = (END_START + 5 * BEAT, END_START + 7 * BEAT)
END_FINAL = END_START + 8 * BEAT           # 30.375
END_FADE = DURATION - 0.45
