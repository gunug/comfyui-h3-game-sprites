import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* Live sprite-atlas player inside the node.
   Draws with the runtime's own math: translate(driftX), drawImage at
   footOffset - height. drawImage's y is the TOP edge, so the subtraction is
   what puts the FEET on the ground line - omit it and the character sinks. */

const DEFAULT_MS = 90;

function build(node, payload) {
  const el = node.__spritePlayerEl;
  if (!el) return;
  const state = (node.__spriteState ||= { playing: true, frame: 0, move: null, acc: 0, last: 0 });

  const url = api.apiURL(
    `/view?filename=${encodeURIComponent(payload.filename)}` +
    `&subfolder=${encodeURIComponent(payload.subfolder || "")}` +
    `&type=${payload.type || "output"}&rand=${Math.random()}`);

  fetch(url).then(r => r.json()).then(async atlas => {
    node.__atlas = atlas;
    node.__imgs = {};
    for (const m of Object.keys(atlas)) {
      node.__imgs[m] = await Promise.all(atlas[m].map(f => new Promise(res => {
        const im = new Image();
        im.onload = () => res(im);
        im.src = "data:image/png;base64," + f.b64;
      })));
    }
    state.move = (payload.move && atlas[payload.move]) ? payload.move : Object.keys(atlas)[0];
    state.frame = 0;
    render(node, payload);
  }).catch(err => {
    el.querySelector(".sp-info").textContent = "atlas load failed: " + err;
  });
}

function render(node, payload) {
  const el = node.__spritePlayerEl, state = node.__spriteState;
  const bar = el.querySelector(".sp-bar");
  bar.innerHTML = "";
  for (const m of Object.keys(node.__atlas || {})) {
    const b = document.createElement("button");
    b.textContent = `${m} (${node.__atlas[m].length})`;
    b.className = "sp-btn" + (m === state.move ? " on" : "");
    b.onclick = () => { state.move = m; state.frame = 0; state.acc = 0; render(node, payload); };
    bar.appendChild(b);
  }
  const play = document.createElement("button");
  play.className = "sp-btn on";
  play.textContent = state.playing ? "pause" : "play";
  play.onclick = () => { state.playing = !state.playing; render(node, payload); };
  bar.appendChild(play);

  const step = document.createElement("button");
  step.className = "sp-btn";
  step.textContent = "step";
  step.onclick = () => { state.playing = false; advance(node); render(node, payload); };
  bar.appendChild(step);

  const flip = document.createElement("button");
  flip.className = "sp-btn" + (state.flip ? " on" : "");
  flip.textContent = "flip";
  flip.onclick = () => { state.flip = !state.flip; render(node, payload); };
  bar.appendChild(flip);
}

function advance(node) {
  const s = node.__spriteState, frames = node.__atlas?.[s.move];
  if (!frames) return;
  s.frame = (s.frame + 1) % frames.length;
}

function draw(node, payload) {
  const el = node.__spritePlayerEl;
  if (!el) return;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d");
  const s = node.__spriteState, frames = node.__atlas?.[s.move];
  ctx.fillStyle = "#1c2029";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!frames) return;

  const ground = Math.round(cv.height * 0.86);
  ctx.strokeStyle = "#39425a";
  ctx.beginPath(); ctx.moveTo(0, ground + 0.5); ctx.lineTo(cv.width, ground + 0.5); ctx.stroke();

  const f = frames[s.frame], im = node.__imgs?.[s.move]?.[s.frame];
  if (!f || !im) return;

  // fit the tallest frame into the canvas so the whole move stays visible
  const tallest = Math.max(...frames.map(x => +x.h));
  const k = Math.min(1, (ground - 8) / tallest);
  const facing = s.flip ? -1 : 1;

  ctx.save();
  ctx.translate(cv.width / 2 + (+f.driftX) * k * facing, ground);
  ctx.scale(facing * k, k);                       // p2 is a flip, never a mirrored render
  ctx.drawImage(im, -(+f.anchorOffset), (+f.footOffset) - (+f.h), +f.w, +f.h);
  ctx.restore();

  el.querySelector(".sp-info").textContent =
    `${s.move}  ${s.frame + 1}/${frames.length}   h ${f.h}  foot ${f.footOffset}  drift ${f.driftX}`;
}

app.registerExtension({
  name: "h3.sprite.player",
  async nodeCreated(node) {
    if (node.comfyClass !== "SpriteAtlasPlayer") return;

    const el = document.createElement("div");
    el.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:4px;";
    el.innerHTML = `
      <canvas width="360" height="260" style="width:100%;background:#1c2029;border:1px solid #2b3140;border-radius:4px"></canvas>
      <div class="sp-bar" style="display:flex;flex-wrap:wrap;gap:4px"></div>
      <div class="sp-info" style="font:11px ui-monospace,monospace;color:#8fa0bd">run the node to load an atlas</div>`;
    const style = document.createElement("style");
    style.textContent = `.sp-btn{background:#232a36;color:#cfd6e4;border:1px solid #39425a;
      border-radius:3px;padding:2px 7px;font-size:11px;cursor:pointer}
      .sp-btn.on{background:#3a5cff;border-color:#3a5cff;color:#fff}`;
    el.appendChild(style);

    node.__spritePlayerEl = el;
    node.__spriteState = { playing: true, frame: 0, move: null, acc: 0, last: 0, flip: false };
    node.addDOMWidget("player", "div", el, { serialize: false, hideOnZoom: false });
    node.size = [400, 460];

    const tick = (t) => {
      const s = node.__spriteState;
      if (s.last && s.playing) {
        s.acc += t - s.last;
        while (s.acc >= DEFAULT_MS) { s.acc -= DEFAULT_MS; advance(node); }
      }
      s.last = t;
      draw(node, node.__spritePayload || {});
      node.__raf = requestAnimationFrame(tick);
    };
    node.__raf = requestAnimationFrame(tick);

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
      cancelAnimationFrame(node.__raf);
      return onRemoved?.apply(this, arguments);
    };

    const onExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const payload = message?.sprite_player?.[0];
      if (payload) { node.__spritePayload = payload; build(node, payload); }
      return onExecuted?.apply(this, arguments);
    };
  },
});
