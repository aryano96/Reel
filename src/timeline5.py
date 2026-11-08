"""Reel 5 timing - "Red Bull Energy Universe" (13 editions). Single source of truth for picture + sound.

128 BPM techno/breakbeat: 1 beat = 0.46875 s, 1 bar = 1.875 s. Every can gets exactly one bar;
every reveal lands on a downbeat. Rendered at 60 fps.
"""
FPS = 60
W, H = 1080, 1920
BPM = 128
BEAT = 60.0 / BPM          # 0.46875 s
BAR = 4 * BEAT             # 1.875 s
S16 = BEAT / 4

INTRO_BEATS = 6
DROP = INTRO_BEATS * BEAT  # 2.8125 - first reveal (bass impact)
SCENE = BAR
N_DRINKS = 13
DRINK_STARTS = [DROP + i * SCENE for i in range(N_DRINKS)]
S = DRINK_STARTS
END_START = DROP + N_DRINKS * SCENE        # 27.1875
DURATION = END_START + 10 * BEAT           # 31.875

HERO2 = 2 * BEAT           # 0.9375 s into each bar: cut to the second (hero) shot
EXIT = SCENE - 0.28        # exit transition window starts
TEXT_IN, TEXT_DONE = 1.5 * BEAT, 3.6 * BEAT  # intro headline decode

# per-edition identity: environment, entrance, exit, second shot
PRODUCTS = [
    dict(key="01_summer_sudachi_lime",      edition="THE SUMMER EDITION",   flavour="SUDACHI LIME",         env="chamber",  enter="awaken",     exit="lightspeed", shot2="push"),
    dict(key="02_ice_iced_vanilla_berry",   edition="THE ICE EDITION",      flavour="ICED VANILLA BERRY",   env="tunnel",   enter="tunnel",     exit="shutter",    shot2="float"),
    dict(key="03_yellow_tropical",          edition="THE YELLOW EDITION",   flavour="TROPICAL",             env="carbon",   enter="shockwave",  exit="crt",        shot2="track"),
    dict(key="04_spring_cherry_sakura",     edition="THE SPRING EDITION",   flavour="CHERRY SAKURA",        env="hexgrid",  enter="hologram",   exit="dissolve",   shot2="push"),
    dict(key="05_peach_white_peach",        edition="THE PEACH EDITION",    flavour="WHITE PEACH",          env="nebula",   enter="assemble",   exit="glitch",     shot2="float"),
    dict(key="06_red_watermelon",           edition="THE RED EDITION",      flavour="WATERMELON",           env="circuit",  enter="headlights", exit="trails",     shot2="track"),
    dict(key="07_sea_blue_juneberry",       edition="THE SEA BLUE EDITION", flavour="JUNEBERRY",            env="rings",    enter="em_pulse",   exit="surge",      shot2="push"),
    dict(key="08_amber_strawberry_apricot", edition="THE AMBER EDITION",    flavour="STRAWBERRY APRICOT",   env="vortex",   enter="vortex",     exit="defocus",    shot2="float"),
    dict(key="09_pink_wild_berries",        edition="THE PINK EDITION",     flavour="WILD BERRIES",         env="prism",    enter="refocus",    exit="prism",      shot2="track"),
    dict(key="10_lilac_grapefruit_blossom", edition="THE LILAC EDITION",    flavour="GRAPEFRUIT & BLOSSOM", env="tesla",    enter="arc",        exit="whiteout",   shot2="push"),
    dict(key="11_winter_pistachio_berries", edition="THE WINTER EDITION",   flavour="PISTACHIO & BERRIES",  env="wind",     enter="scan",       exit="overtake",   shot2="float"),
    dict(key="12_coconut_berry",            edition="THE COCONUT EDITION",  flavour="COCONUT BERRY",        env="void",     enter="dropin",     exit="blackout",   shot2="float"),
    dict(key="13_apple_fuji_ginger",        edition="THE APPLE EDITION",    flavour="FUJI APPLE & GINGER",  env="turbine",  enter="ignition",   exit="launch",     shot2="push"),
]

# end card
E = END_START
END_LOGO = E + 0.5 * BEAT
END_ADDR = (E + 1.5 * BEAT, E + 2.0 * BEAT)
END_OFFER = E + 3 * BEAT
END_CTA = E + 4.5 * BEAT
END_SHINE = E + 6 * BEAT
END_FINAL = E + 8 * BEAT                   # 30.9375
END_FADE = DURATION - 0.5
