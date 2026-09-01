# ComfyUI-H3-Game-Sprites

**This is nothing more than a ComfyUI custom-node port of [gary149/h3-game-sprites](https://github.com/gary149/h3-game-sprites).**

All of the actual method — the "Mortal Kombat" approach of filming an AI-generated
actor (MiniMax H3) and cutting the video into a 2D sprite sheet — is that project's
work. This repository only wraps its `sprite_cut.py` / `build_atlas.py` logic in
ComfyUI nodes, so the pipeline runs inside a workflow graph instead of a shell.
No new technique is claimed here.

Upstream is MIT licensed; this port is released under the same license and keeps
the original copyright notice.

## What it does

```
LoadVideo → GetVideoComponents → Sprite Auto Key → Sprite Select Frames
                                                 → Sprite Move
                                                 → Sprite Atlas Build → Sprite Atlas Player → SaveAnimatedWEBP
```

## Nodes

| Node | Purpose |
| --- | --- |
| `Sprite Auto Key` | Chroma-key a clip to alpha. `auto` re-samples the backdrop colour **every frame**. |
| `Sprite Select Frames` | Pick N pose extremes by cumulative motion arc-length, or find a loop seam. |
| `Sprite Move` | Bundle one move's keyed frames under a name (`idle`, `walk`, `punch`, …). |
| `Sprite Atlas Build` | Pack up to 6 moves into one atlas JSON at a **single global scale**, plus a contact sheet. |
| `Sprite Atlas Player` | Play an atlas back as an IMAGE batch and as a live player widget inside the node. |

## The two traps it exists to avoid

Both are called out by upstream, and both are preserved here:

1. **Do not key against a fixed backdrop colour.** H3's magenta drifts hue across a
   5-second clip (magenta → blue → green → brown). `key_mode=auto` samples the
   frame border per frame instead, so every frame keys regardless of that drift.
2. **Do not scale each frame to a fixed on-screen height.** A crouch has a smaller
   bounding box, so per-frame normalization renders it *bigger* than idle. One
   scale is derived from the reference move's first frame and applied to everything.

Two smaller details that follow from the same reasoning:

- The stance anchor is the centroid of the bottom ~22% of the mask (the ankles),
  not the bbox centre — a big arm swing moves the bbox and makes the character
  slide sideways between poses.
- The contact sheet draws every frame on a shared ground line using the *runtime's
  own* placement math, so it is a pixel check of the offsets rather than a
  metadata assertion.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gunug/comfyui-h3-game-sprites
```

Restart ComfyUI. Dependencies (`numpy`, `torch`, `Pillow`) all ship with ComfyUI.

## Atlas format

`Sprite Atlas Build` writes JSON into the ComfyUI **output** directory:

```jsonc
{
  "idle": [
    {
      "b64": "<base64 PNG, RGBA, cropped and scaled>",
      "w": 210, "h": 340,
      "footOffset": 0,      // feet position relative to the reference baseline
      "anchorOffset": 96,   // ankle anchor x, relative to the tile's left edge
      "driftX": -4          // stance drift from the reference anchor
    }
  ]
}
```

Runtime placement (this is what `Sprite Atlas Player` and the JS widget both do):

```js
x = canvasWidth / 2 + driftX * facing - anchorOffset
y = groundY + footOffset - h   // drawImage's y is the TOP edge; footOffset positions the FEET
```

## Credits

- Method, keying/selection/atlas logic, and the original scripts: **[gary149/h3-game-sprites](https://github.com/gary149/h3-game-sprites)** (MIT)
- This repo: ComfyUI node wrappers around that logic, plus a canvas player widget.

## License

MIT — see [LICENSE](LICENSE).
