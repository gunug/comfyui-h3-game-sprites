"""H3 → Game Sprites, as ComfyUI nodes.

Port of gary149/h3-game-sprites (sprite_cut.py / build_atlas.py) so the whole
video → sprite-sheet pipeline runs inside a workflow instead of a shell:

    VHS_LoadVideo / LoadVideo  →  Sprite Auto Key  →  Sprite Select Frames
                                                   →  Sprite Move
                                                   →  Sprite Atlas Build

The two mistakes this pack exists to prevent are the ones the upstream skill
calls out: keying against a FIXED backdrop colour (H3's magenta drifts hue
across a 5 s clip), and scaling each frame to a fixed on-screen height (a
crouch has a smaller bbox, so per-frame scaling renders it BIGGER than idle).
"""

import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

import folder_paths


# ---------------------------------------------------------------- helpers

def _to_np(images):
    """IMAGE tensor [B,H,W,3] 0..1 -> uint8 array [B,H,W,3]."""
    return (images.detach().cpu().numpy().clip(0, 1) * 255.0).round().astype(np.uint8)


def _to_image(arr):
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def _key_auto(rgb, tol, despill):
    """Key whatever flat colour the frame's border actually IS.

    H3 does not hold one saturated backdrop colour across a clip — the hue
    drifts (magenta → blue → green → brown). Sampling the border per frame
    keys every frame regardless of that drift.
    """
    a = rgb.astype(np.float32)
    h, w, _ = a.shape
    m = max(2, min(h, w) // 40)
    border = np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3),
                             a[:, :m].reshape(-1, 3), a[:, -m:].reshape(-1, 3)])
    key = np.median(border, axis=0)
    dist = np.sqrt(((a - key) ** 2).sum(axis=2))
    bg = dist < tol
    fringe = (~bg) & (dist < tol * 1.7)
    if despill > 0 and fringe.any():
        px = a[fringe]
        a[fringe] = np.clip(px + (px - key) * despill, 0, 255)
    return a.clip(0, 255).astype(np.uint8), bg


def _key_fixed(rgb, color):
    """Upstream's magenta / green key. Despill CLAMPS the key channels against
    the off-channel — blending toward it visibly tints the character."""
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == "magenta":
        score, off = r + b - 2 * g, g
        bg = (score > 120) & (r > 90) & (b > 90)
    else:
        score, off = 2 * g - r - b, np.maximum(r, b)
        bg = (score > 120) & (g > 90)
    cap = off * 1.18 + 30
    if color == "magenta":
        a[..., 0], a[..., 2] = np.minimum(r, cap), np.minimum(b, cap)
    else:
        a[..., 1] = np.minimum(g, cap)
    return a.clip(0, 255).astype(np.uint8), bg


def _pick_arclength(energy, n):
    """Equal steps of CUMULATIVE motion, so frames land on pose extremes.
    Uniform sampling looks mushy; local-maxima picking over-samples the idle
    bounce at the head of the clip."""
    cum = np.cumsum(energy)
    if cum[-1] <= 0:
        return sorted(set(np.linspace(0, len(energy) - 1, n).round().astype(int).tolist()))
    targets = np.linspace(0, cum[-1], n)
    return sorted({int(np.argmin(np.abs(cum - t))) for t in targets} | {0, len(energy) - 1})


def _pick_loop(small, n, lo_frac=0.2, hi_frac=0.85, min_gap=12, max_gap=45):
    """Find the closest-matching frame pair (the cycle seam) and sample inside
    it. Whole-clip picks include the start-up and the settle back to idle,
    which is exactly why a walk visibly resets instead of looping."""
    total = len(small)
    idxs = range(int(total * lo_frac), max(int(total * hi_frac), int(total * lo_frac) + 1))
    best = None
    for i in idxs:
        for j in idxs:
            if not (min_gap <= j - i <= max_gap):
                continue
            d = float(np.abs(small[i] - small[j]).mean())
            if best is None or d < best[0]:
                best = (d, i, j)
    if best is None:
        return None, sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))
    d, i, j = best
    return ({"i": i, "j": j, "diff": round(d, 3)},
            sorted({i + round(k * (j - i) / n) for k in range(n)}))


def _stats(alpha):
    """bbox, feet baseline, stance anchor x.

    The anchor comes from the centroid of the bottom ~22% of the mask (the
    ankles), NOT the bbox center — a big arm swing moves the bbox and centering
    on it makes the character slide sideways between poses.
    """
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    thresh = ys.max() - int((ys.max() - ys.min()) * 0.22)
    leg = ys >= thresh
    return bbox, int(ys.max()), float(xs[leg].mean() if leg.any() else xs.mean())


# ---------------------------------------------------------------- nodes

class SpriteAutoKey:
    """Chroma-key a clip to alpha. 'auto' re-samples the backdrop every frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "key_mode": (["auto", "magenta", "green"], {"default": "auto"}),
            "tolerance": ("FLOAT", {"default": 52.0, "min": 4.0, "max": 200.0, "step": 1.0}),
            "despill": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "alpha", "report")
    FUNCTION = "run"
    CATEGORY = "sprites"

    def run(self, images, key_mode, tolerance, despill):
        frames = _to_np(images)
        out, alphas, purity = [], [], []
        for f in frames:
            if key_mode == "auto":
                rgb, bg = _key_auto(f, tolerance, despill)
            else:
                rgb, bg = _key_fixed(f, key_mode)
            out.append(rgb)
            alphas.append(~bg)
            purity.append(float(bg.mean() + (~bg).mean()))
        alpha = np.stack(alphas).astype(np.float32)
        report = json.dumps({"frames": len(frames), "key_mode": key_mode,
                             "worst_bg_purity": round(min(purity), 4)}, indent=1)
        return (_to_image(np.stack(out)), torch.from_numpy(alpha), report)


class SpriteSelectFrames:
    """Pick N pose extremes by cumulative motion arc-length (or a loop seam)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "alpha": ("MASK",),
            "frames": ("INT", {"default": 12, "min": 2, "max": 64}),
            "mode": (["arclength", "loop"], {"default": "arclength"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "alpha", "report")
    FUNCTION = "run"
    CATEGORY = "sprites"

    def run(self, images, alpha, frames, mode):
        rgb = _to_np(images)
        al = (alpha.detach().cpu().numpy() > 0.5)
        n = len(rgb)
        small = []
        for i in range(n):
            im = Image.fromarray(np.dstack([rgb[i], (al[i] * 255).astype(np.uint8)]), "RGBA")
            s = im.resize((max(1, im.width // 8), max(1, im.height // 8)))
            small.append(np.asarray(s).astype(np.int16))
        energy = [0.0] + [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, n)]

        loop_info = None
        if mode == "loop":
            loop_info, picks = _pick_loop(small, frames)
        else:
            picks = _pick_arclength(energy, frames)
        picks = [p for p in picks if 0 <= p < n][:frames]

        base = [_stats(al[i]) for i in range(n)]
        base = [b for b in base if b]
        report = {
            "frames": n, "picked": picks, "loop": loop_info,
            "baseline_drift_px": (max(b[1] for b in base) - min(b[1] for b in base)) if base else None,
            "loop_diff": round(loop_info["diff"], 3) if loop_info else
                         (round(float(np.abs(small[0] - small[-1]).mean()), 3) if n > 1 else 0.0),
        }
        idx = torch.tensor(picks, dtype=torch.long)
        return (images[idx], alpha[idx], json.dumps(report, indent=1))


class SpriteMove:
    """Bundle one move's keyed frames under a name, for the atlas builder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "alpha": ("MASK",),
            "name": ("STRING", {"default": "idle"}),
        }}

    RETURN_TYPES = ("SPRITE_MOVE",)
    RETURN_NAMES = ("move",)
    FUNCTION = "run"
    CATEGORY = "sprites"

    def run(self, images, alpha, name):
        return ({"name": name.strip() or "move",
                 "rgb": _to_np(images),
                 "alpha": (alpha.detach().cpu().numpy() > 0.5)},)


class SpriteAtlasBuild:
    """Pack moves into one atlas at a SINGLE global scale, + a contact sheet.

    Per-frame normalization is the trap: a crouch has a smaller bbox, so
    scaling each frame to a fixed height renders it BIGGER than standing. One
    scale comes from the reference move's first frame and applies to everything.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "move_1": ("SPRITE_MOVE",),
                "ref_move": ("STRING", {"default": "idle"}),
                "height": ("INT", {"default": 340, "min": 32, "max": 2048}),
                "filename": ("STRING", {"default": "sprites/atlas.json"}),
                "ground_y": ("INT", {"default": 420, "min": 0, "max": 4096}),
            },
            "optional": {
                "move_2": ("SPRITE_MOVE",), "move_3": ("SPRITE_MOVE",),
                "move_4": ("SPRITE_MOVE",), "move_5": ("SPRITE_MOVE",),
                "move_6": ("SPRITE_MOVE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("contact_sheet", "atlas_path", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "sprites"

    def run(self, move_1, ref_move, height, filename, ground_y, **kw):
        moves = [move_1] + [kw[k] for k in sorted(kw) if kw.get(k)]
        by_name = {m["name"]: m for m in moves}
        ref = by_name.get(ref_move.strip(), moves[0])

        st = _stats(ref["alpha"][0])
        if st is None:
            raise ValueError("reference frame is fully transparent — check the key")
        ref_bbox, ref_base, ref_anchor = st
        scale = height / (ref_bbox[3] - ref_bbox[1])

        atlas, lines, sheet_rows = {}, [], []
        for m in moves:
            frames, tiles = [], []
            for i in range(len(m["rgb"])):
                s = _stats(m["alpha"][i])
                if s is None:
                    continue
                bbox, base, anchor = s
                rgba = np.dstack([m["rgb"][i], (m["alpha"][i] * 255).astype(np.uint8)])
                crop = Image.fromarray(rgba, "RGBA").crop(bbox)
                w = max(1, round(crop.width * scale))
                h = max(1, round(crop.height * scale))
                small = crop.resize((w, h), Image.LANCZOS)
                buf = io.BytesIO()
                small.save(buf, "PNG", optimize=True)
                frames.append({
                    "b64": base64.b64encode(buf.getvalue()).decode(), "w": w, "h": h,
                    "footOffset": round((base - ref_base) * scale),
                    "anchorOffset": round((anchor - bbox[0]) * scale),
                    "driftX": round((anchor - ref_anchor) * scale),
                })
                tiles.append((small, frames[-1]))
            atlas[m["name"]] = frames
            if frames:
                hs = [f["h"] for f in frames]
                lines.append(f"{m['name']:12s} {len(frames):3d} frames   on-screen height {min(hs)}-{max(hs)}px")
            sheet_rows.append(tiles)

        out_dir = folder_paths.get_output_directory()
        path = os.path.normpath(os.path.join(out_dir, filename))
        if not path.startswith(os.path.normpath(out_dir)):
            raise ValueError("filename escapes the output directory")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(atlas, fh)

        sheet = self._sheet(sheet_rows, height, ground_y)
        report = (f"reference {ref['name']}: scale {scale:.4f}, ground {ref_base}, "
                  f"anchor_x {ref_anchor:.1f}\n" + "\n".join(lines) +
                  f"\nwrote {path} ({os.path.getsize(path) // 1024} KB)\n"
                  f"sanity: crouch/tuck poses must be SHORTER than {ref['name']} ({height}px).")
        return (sheet, path, report)

    def _sheet(self, rows, height, ground_y):
        """Draw every frame on a shared ground line, using the same placement
        math the runtime uses — this is the pixel check that the offsets are
        right, not a metadata assertion."""
        cell_w = max(120, int(height * 0.9))
        cell_h = int(height * 1.35)
        cols = max((len(r) for r in rows), default=1)
        W, H = cell_w * max(cols, 1), cell_h * max(len(rows), 1)
        canvas = Image.new("RGBA", (max(W, 1), max(H, 1)), (28, 32, 41, 255))
        ground = int(cell_h * 0.86)
        for ri, tiles in enumerate(rows):
            y0 = ri * cell_h
            for ci, (im, f) in enumerate(tiles):
                x = ci * cell_w + cell_w // 2 + int(f["driftX"]) - int(f["anchorOffset"])
                y = y0 + ground + int(f["footOffset"]) - int(f["h"])
                canvas.alpha_composite(im, (max(x, 0), max(y, 0)))
            for ci in range(cols):
                for xx in range(ci * cell_w, (ci + 1) * cell_w):
                    if 0 <= xx < canvas.width and 0 <= y0 + ground < canvas.height:
                        canvas.putpixel((xx, y0 + ground), (90, 105, 140, 255))
        arr = np.asarray(canvas.convert("RGB"))
        return _to_image(arr[None, ...])



class SpriteAtlasPlayer:
    """Play an atlas back — as an IMAGE batch AND as a live player in the node.

    The batch is composited with the runtime's own placement math
    (translate driftX, drawImage at footOffset - height), so what you watch is
    what the engine will draw. Feed it to SaveAnimatedWEBP for a shareable loop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "atlas_path": ("STRING", {"default": "sprites/atlas.json", "forceInput": False}),
            "move": ("STRING", {"default": ""}),
            "canvas_width": ("INT", {"default": 560, "min": 64, "max": 4096}),
            "canvas_height": ("INT", {"default": 480, "min": 64, "max": 4096}),
            "ground_y": ("INT", {"default": 420, "min": 0, "max": 4096}),
            "flip": ("BOOLEAN", {"default": False}),
            "draw_ground_line": ("BOOLEAN", {"default": True}),
            "background": ("STRING", {"default": "#1c2029"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "sprites"

    def _resolve(self, atlas_path):
        if os.path.isabs(atlas_path) and os.path.isfile(atlas_path):
            return atlas_path
        cand = os.path.join(folder_paths.get_output_directory(), atlas_path)
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"atlas not found: {atlas_path}")

    def run(self, atlas_path, move, canvas_width, canvas_height, ground_y,
            flip, draw_ground_line, background):
        path = self._resolve(atlas_path)
        with open(path) as fh:
            atlas = json.load(fh)
        names = list(atlas.keys())
        want = [move.strip()] if move.strip() in atlas else names

        try:
            bg = tuple(int(background.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
        except Exception:
            bg = (28, 32, 41, 255)

        out, counts = [], []
        for name in want:
            frames = atlas[name]
            counts.append(f"{name}: {len(frames)} frames")
            for f in frames:
                canvas = Image.new("RGBA", (canvas_width, canvas_height), bg)
                if draw_ground_line and 0 <= ground_y < canvas_height:
                    for x in range(canvas_width):
                        canvas.putpixel((x, ground_y), (90, 105, 140, 255))
                im = Image.open(io.BytesIO(base64.b64decode(f["b64"]))).convert("RGBA")
                facing = -1 if flip else 1
                if flip:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                x = canvas_width // 2 + int(f["driftX"]) * facing
                # drawImage's y is the TOP edge while footOffset positions the FEET
                y = ground_y + int(f["footOffset"]) - int(f["h"])
                ax = int(f["anchorOffset"]) if not flip else int(f["w"]) - int(f["anchorOffset"])
                canvas.alpha_composite(im, (x - ax, y))
                out.append(np.asarray(canvas.convert("RGB")))

        payload = {"filename": os.path.basename(path),
                   "subfolder": os.path.dirname(atlas_path).replace("\\", "/"),
                   "type": "output", "moves": names, "move": move.strip(),
                   "ground_y": ground_y, "flip": bool(flip)}
        report = f"{path}\n" + "\n".join(counts)
        return {"ui": {"sprite_player": [payload]},
                "result": (_to_image(np.stack(out)), report)}


NODE_CLASS_MAPPINGS = {
    "SpriteAutoKey": SpriteAutoKey,
    "SpriteSelectFrames": SpriteSelectFrames,
    "SpriteMove": SpriteMove,
    "SpriteAtlasBuild": SpriteAtlasBuild,
    "SpriteAtlasPlayer": SpriteAtlasPlayer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpriteAutoKey": "Sprite Auto Key",
    "SpriteSelectFrames": "Sprite Select Frames",
    "SpriteMove": "Sprite Move",
    "SpriteAtlasBuild": "Sprite Atlas Build",
    "SpriteAtlasPlayer": "Sprite Atlas Player",
}
