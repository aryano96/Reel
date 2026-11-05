"""9:16 promotional cover/thumbnail for the reel (1080x1920).

Single hero product (Cherry Limeade, the reel's opening drink) with the
splash effect, the opening headline and the store name/address. Key content is
kept inside the centre 1080x1440 area so it survives Instagram's 3:4 grid crop.
"""
import os
import numpy as np
import cv2

import timeline as TL
from timeline import W, H
import fx
from fx import Cam, render_liquid, draw_text, fit_size, radial_bg, draw_sprite, sphere_sprite, finish, add_glow
import render as R

OUT = os.path.join(fx.ROOT, "output")


def main():
    c = R.can(0)
    cam = Cam(z=0.76, fy=R.CAN_Y, ay=1050)
    canvas = radial_bg((0.50, 0.86, 0.18), (0.015, 0.10, 0.02), 540, 1000, 1000, 1350, 1.35).copy()
    R.god_rays(canvas, 0.6, *cam.p(540, 900), (0.75, 1.0, 0.45), 0.24, n=11, speed=0.05)
    R.bokeh(canvas, 1.0, 3, 22, [(0.8, 1.0, 0.3), (1.0, 0.95, 0.5), (0.5, 1.0, 0.4)], cam, inten=0.10, rmin=25, rmax=90)
    fy = cam.p(0, R.FLOOR)[1]
    yy = np.arange(H, dtype=np.float32)
    canvas *= 1 - 0.45 * np.clip((yy - fy) / 300, 0, 1)[:, None, None]

    streams, drops, red, front = R.SPLASH
    sim = R.s1_sim(0.9)
    discs = R.splash_prims(sim, 540, 990, streams, drops)
    sd = R.cam_discs(cam, discs)
    render_liquid(canvas, discs=[d for d in sd if d[2] > 4], color=(0.58, 0.95, 0.12), dark=(0.10, 0.42, 0.02),
                  sigma=4.0, thr=(0.30, 0.52), bump=40)
    for (x, y, r) in sd:
        if 1.0 < r <= 4:
            draw_sprite(canvas, sphere_sprite((0.6, 0.98, 0.2)), x, y, r * 2.2)
    for (x, y, r) in R.cam_discs(cam, R.splash_prims(sim, 540, 1010, [], red, drag=2.0)):
        draw_sprite(canvas, sphere_sprite((0.85, 0.04, 0.09)), x, y, r * 2.0)
    add_glow(canvas, *cam.p(540, 960), 650, (0.7, 1.0, 0.35), 0.18)

    c.draw(canvas, cam, 540, R.CAN_Y, rot=0.03, spec=-0.25, spec_s=0.18, rim_l=(0.35, 0.40, 0.30), rim_r=(0.55, 0.9, 0.3),
           reflect_floor=R.FLOOR, refl_strength=0.3, shadow=0.55)

    # headline
    draw_text(canvas, "THE ULTIMATE", "InterDisplay-Black.otf", fit_size("THE ULTIMATE", "InterDisplay-Black.otf", 700, 190),
              540, 300, glow=(0.45, 1.0, 0.25), glow_r=24, glow_s=0.7, shadow=0.5)
    draw_text(canvas, "REFRESHMENT", "InterDisplay-BlackItalic.otf", fit_size("REFRESHMENT", "InterDisplay-BlackItalic.otf", 940, 200),
              540, 420, grad=((1, 1, 1), (0.82, 1.0, 0.5)), glow=(0.45, 1.0, 0.25), glow_r=26, glow_s=0.9, shadow=0.5)
    # store strip
    s1 = "NOW AT EZY PZY CONVENIENCE"
    size = fit_size(s1, "InterDisplay-Black.otf", 760, 60)
    f = fx.font("InterDisplay-Black.otf", size)
    tw = f.getlength(s1)
    pw, ph, cy = int(tw + 90), int(size * 1.9), 1590
    pill = np.zeros((ph + 60, pw + 60), np.float32)
    rr = ph // 2
    cv2.rectangle(pill, (30 + rr, 30), (30 + pw - rr, 30 + ph), 1.0, -1, cv2.LINE_AA)
    cv2.circle(pill, (30 + rr, 30 + rr), rr, 1.0, -1, cv2.LINE_AA)
    cv2.circle(pill, (30 + pw - rr, 30 + rr), rr, 1.0, -1, cv2.LINE_AA)
    gy = np.linspace(0, 1, pill.shape[0], dtype=np.float32)[:, None, None]
    col = np.array([1.0, 0.86, 0.45], np.float32) * (1 - gy) + np.array([0.98, 0.60, 0.16], np.float32) * gy
    x0, y0 = int(540 - pill.shape[1] / 2), int(cy - pill.shape[0] / 2)
    fx.blend_add(canvas, (cv2.GaussianBlur(pill, (0, 0), 16) * 0.5)[..., None] * np.array(R.GOLD, np.float32), x0, y0)
    fx.blend_over(canvas, col * pill[..., None], pill, x0, y0)
    draw_text(canvas, s1, "InterDisplay-Black.otf", size, 540, cy, color=(0.07, 0.05, 0.10))
    addr = "153 Great Northern Rd, Sault Ste. Marie, ON P6B 4Y9"
    draw_text(canvas, addr, "InterDisplay-SemiBold.otf", fit_size(addr, "InterDisplay-SemiBold.otf", 900, 40),
              540, cy + 92, shadow=0.7, glow=(0, 0, 0), glow_r=12, glow_s=0.0)

    img = finish(canvas, 0, grain=0.012)
    img = (img * 255 + 0.5).astype(np.uint8)
    path = os.path.join(OUT, "ezy_pzy_reel_thumbnail.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(path.replace(".jpg", ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote", path)


if __name__ == "__main__":
    main()
