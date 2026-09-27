"""Generate the Cappycat app icons with the Python standard library only.

Produces in ../icons:
  32x32.png, 128x128.png, 128x128@2x.png (256), icon.png (512), icon.ico

Design: flat blue (#1D4ED8) rounded square with a near-white "C" ring.
Rendered once at 1024px with 2x2 supersampling, then box-downsampled.

Usage:  python scripts/gen_icons.py   (from src-tauri)
"""
from __future__ import annotations

import math
import os
import struct
import sys
import zlib

BG = (0x1D, 0x4E, 0xD8)      # blue (#1D4ED8)
FG = (0xEA, 0xF2, 0xFF)      # near-white "C"
BASE = 1024                  # master render size
SS = 2                       # supersampling factor per axis


def rounded_rect_inside(x: float, y: float, size: float, radius: float) -> bool:
    """True when (x, y) lies inside a rounded square of `size` with corner `radius`."""
    if x < 0 or y < 0 or x >= size or y >= size:
        return False
    cx = min(max(x, radius), size - radius)
    cy = min(max(y, radius), size - radius)
    dx, dy = x - cx, y - cy
    return dx * dx + dy * dy <= radius * radius


def c_glyph_inside(x: float, y: float, size: float) -> bool:
    """A thick ring with a 70-degree opening on the right: the letter C."""
    cx = cy = size / 2
    r_out = size * 0.34
    r_in = size * 0.19
    dx, dy = x - cx, y - cy
    d = math.hypot(dx, dy)
    if not (r_in <= d <= r_out):
        return False
    ang = math.degrees(math.atan2(dy, dx))  # -180..180, 0 = right
    return abs(ang) > 38.0


def render_master() -> list[list[tuple[int, int, int, int]]]:
    n = BASE
    radius = n * 0.22
    rows = []
    inv = 1.0 / (SS * SS)
    for py in range(n):
        row = []
        for px in range(n):
            bg_hits = 0
            fg_hits = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) / SS
                    y = py + (sy + 0.5) / SS
                    if rounded_rect_inside(x, y, n, radius):
                        bg_hits += 1
                        if c_glyph_inside(x, y, n):
                            fg_hits += 1
            if bg_hits == 0:
                row.append((0, 0, 0, 0))
                continue
            a = bg_hits * inv
            f = fg_hits / bg_hits
            r = BG[0] * (1 - f) + FG[0] * f
            g = BG[1] * (1 - f) + FG[1] * f
            b = BG[2] * (1 - f) + FG[2] * f
            row.append((int(round(r)), int(round(g)), int(round(b)), int(round(a * 255))))
        rows.append(row)
    return rows


def downsample(img, factor: int):
    """Box-filter downsample by an integer factor (premultiplied alpha aware)."""
    n = len(img)
    m = n // factor
    out = []
    for oy in range(m):
        row = []
        for ox in range(m):
            r = g = b = a = 0.0
            for sy in range(factor):
                for sx in range(factor):
                    pr, pg, pb, pa = img[oy * factor + sy][ox * factor + sx]
                    r += pr * pa
                    g += pg * pa
                    b += pb * pa
                    a += pa
            cnt = factor * factor
            if a == 0:
                row.append((0, 0, 0, 0))
            else:
                row.append((int(round(r / a)), int(round(g / a)), int(round(b / a)), int(round(a / cnt))))
        out.append(row)
    return out


def png_bytes(img) -> bytes:
    h = len(img)
    w = len(img[0])
    raw = bytearray()
    for row in img:
        raw.append(0)  # filter: none
        for (r, g, b, a) in row:
            raw += bytes((r, g, b, a))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def ico_bytes(entries: list[tuple[int, bytes]]) -> bytes:
    """ICO container with PNG-compressed images (Vista+)."""
    header = struct.pack("<HHH", 0, 1, len(entries))
    dir_size = 6 + 16 * len(entries)
    offset = dir_size
    directory = b""
    payload = b""
    for size, data in entries:
        s = 0 if size >= 256 else size
        directory += struct.pack("<BBBBHHII", s, s, 0, 0, 1, 32, len(data), offset)
        payload += data
        offset += len(data)
    return header + directory + payload


def main() -> int:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "icons")
    os.makedirs(out_dir, exist_ok=True)
    print("rendering master 1024px ...", flush=True)
    master = render_master()
    sizes = {
        512: downsample(master, 2),
    }
    sizes[256] = downsample(sizes[512], 2)
    sizes[128] = downsample(sizes[256], 2)
    sizes[64] = downsample(sizes[128], 2)
    sizes[32] = downsample(sizes[64], 2)
    sizes[16] = downsample(sizes[32], 2)

    pngs = {s: png_bytes(img) for s, img in sizes.items()}
    files = {
        "32x32.png": pngs[32],
        "128x128.png": pngs[128],
        "128x128@2x.png": pngs[256],
        "icon.png": pngs[512],
        "icon.ico": ico_bytes([(256, pngs[256]), (128, pngs[128]), (64, pngs[64]), (32, pngs[32]), (16, pngs[16])]),
    }
    for name, data in files.items():
        path = os.path.join(out_dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        print(f"wrote {path} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
