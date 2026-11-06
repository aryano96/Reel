"""Reel 2 cover/thumbnail (1080x1920): Crush Pineapple hero (the reel's opener)
with golden splash, headline and store info. Key content sits inside the centre
1080x1440 so it survives Instagram's 3:4 profile-grid crop."""
import os
import numpy as np
import cv2

import fx
from fx import render_liquid, draw_text, fit_size, draw_sprite, sphere_sprite, finish, add_glow
import render as R
import reel2_render as R2

OUT = os.path.join(fx.ROOT, "output", "reel2")
os.makedirs(OUT, exist_ok=True)


def pill(canvas, text, cy, size, maxw):
    size = fit_size(text, "InterDisplay-Black.otf", maxw, size)
    tw = fx.font("InterDisplay-Black.otf", size).getlength(text)
    pw, ph = int(tw + 90), int(size * 1.9)
    m = np.zeros((ph + 60, pw + 60), np.float32)
    rr = ph // 2
    cv2.rectangle(m, (30 + rr, 30), (30 + pw - rr, 30 + ph), 1.0, -1, cv2.LINE_AA)
    cv2.circle(m, (30 + rr, 30 + rr), rr, 1.0, -1, cv2.LINE_AA)
    cv2.circle(m, (30 + pw - rr, 30 + rr), rr, 1.0, -1, cv2.LINE_AA)
    gy = np.linspace(0, 1, m.shape[0], dtype=np.float32)[:, None, None]
    col = np.array([1.0, 0.86, 0.45], np.float32) * (1 - gy) + np.array([0.98, 0.60, 0.16], np.float32) * gy
    x0, y0 = int(540 - m.shape[1] / 2), int(cy - m.shape[0] / 2)
    fx.blend_add(canvas, (cv2.GaussianBlur(m, (0, 0), 16) * 0.5)[..., None] * np.array(R.GOLD, np.float32), x0, y0)
    fx.blend_over(canvas, col * m[..., None], m, x0, y0)
    draw_text(canvas, text, "InterDisplay-Black.otf", size, 540, cy, color=(0.07, 0.05, 0.10))


def main():
    p = R2.plate(0)
    cam = p.hero_cam(z=0.9)
    cam.ay = 1000
    cxp, cyp = p.cx, p.top + 480
    sim = R.s1_sim(0.9)

    def behind(cv):
        R.god_rays(cv, 0.6, *cam.p(cxp, cyp), (1.0, 0.85, 0.45), 0.24, n=12, speed=0.05)
        streams, drops, red, front = R.SPLASH
        sd = R.cam_discs(cam, R.splash_prims(sim, cxp, cyp, streams, drops))
        render_liquid(cv, discs=[d for d in sd if d[2] > 4], color=(1.0, 0.80, 0.14), dark=(0.55, 0.28, 0.0),
                      sigma=4.0, thr=(0.30, 0.52), bump=40)
        for (x, y, r) in sd:
            if 1.0 < r <= 4:
                draw_sprite(cv, sphere_sprite((1.0, 0.85, 0.25)), x, y, r * 2.2)
    canvas = p.render(cam, behind, spec=-0.25, spec_s=0.16, rim_r=(0.5, 0.42, 0.15), rim_l=(0.3, 0.28, 0.2))
    # darken the top for the headline
    yy = np.arange(fx.H, dtype=np.float32)
    canvas *= (0.55 + 0.45 * np.clip((yy - 150) / 450, 0, 1))[:, None, None]
    draw_text(canvas, "THE ULTIMATE", "InterDisplay-Black.otf", fit_size("THE ULTIMATE", "InterDisplay-Black.otf", 700, 190),
              540, 285, glow=(1.0, 0.75, 0.2), glow_r=24, glow_s=0.7, shadow=0.6)
    draw_text(canvas, "REFRESHMENT", "InterDisplay-BlackItalic.otf", fit_size("REFRESHMENT", "InterDisplay-BlackItalic.otf", 940, 200),
              540, 400, grad=((1, 1, 1), (1.0, 0.88, 0.4)), glow=(1.0, 0.75, 0.2), glow_r=26, glow_s=0.9, shadow=0.6)
    # bottom scrim + store info
    canvas *= (1 - 0.55 * np.clip((yy - 1480) / 280, 0, 1))[:, None, None]
    pill(canvas, "NOW AT EZY PZY CONVENIENCE", 1618, 60, 760)
    addr = "153 Great Northern Rd, Sault Ste. Marie, ON P6B 4Y9"
    draw_text(canvas, addr, "InterDisplay-SemiBold.otf", fit_size(addr, "InterDisplay-SemiBold.otf", 900, 40), 540, 1708, shadow=0.8)
    img = (finish(canvas, 0, grain=0.012) * 255 + 0.5).astype(np.uint8)
    path = os.path.join(OUT, "ezy_pzy_reel2_thumbnail.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(path.replace(".jpg", ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote", path)


if __name__ == "__main__":
    main()
