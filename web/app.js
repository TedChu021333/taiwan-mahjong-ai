// 網頁版牌桌：畫面與操作。牌局、規則、AI 都在 worker.js（Pyodide 跑和電腦版同一套 Python）。
// 版面沿用 Tk 版（play/__main__.py）的 1400×900 座標。
"use strict";

const W = 1400, H = 900;
const ME = 0;
const DISCARD_DELAY = 1000; // 電腦打牌：從開始想到打出至少這麼久
const CLAIM_DELAY = 700;    // 電腦吃碰槓胡之後停一下
const DRAW_DELAY = 180;
const HAND_W = 60, HAND_H = 100, RIVER_W = 32, RIVER_H = 52, MELD_W = 32, MELD_H = 52;
const COMPASS = { x: 550, y: 330, w: 300, h: 150 };
const SEATS = {
  2: { avatar: [380, 10], backs: [600, 14, 16, 28, 44], melds: [1090, 14, 1390], river: [550, 330 - RIVER_H - 3, 8, "up"] },
  3: { avatar: [20, 226], backs: [20, 300, 8, 24, 36], melds: [20, 392, 320], river: [336, 330, 6, "down"] },
  1: { avatar: [1080, 226], backs: [1080, 300, 8, 24, 36], melds: [1080, 392, 1390], river: [860, 330, 6, "down"] },
};
const MY_RIVER = [550, 330 + 150 + 8, 8, "down"];
const RANKS = "一二三四五六七八九", SUITS = "萬筒條", HONORS = "東南西北中發白", FLOWERS = "春夏秋冬梅蘭竹菊";
const CLAIMS = new Set(["chow", "pung", "kong", "win"]);

const table = document.getElementById("table");
const overlay = document.getElementById("overlay");
const home = document.getElementById("home");
const statusLine = document.getElementById("status");

const ui = {
  view: null, actions: null, selected: [], declareMode: false, messages: [], resolve: null, mode: "full",
};

// ---- 和 worker 講話 ----
const worker = new Worker("worker.js");
const waiting = new Map();
let nextId = 1;
worker.onmessage = (event) => {
  const data = event.data;
  if (data.progress) { statusLine.textContent = data.progress; return; }
  const entry = waiting.get(data.id);
  if (!entry) return;
  waiting.delete(data.id);
  if (data.error) entry.reject(new Error(data.error)); else entry.resolve(data.result);
};
worker.onerror = (event) => { statusLine.textContent = "載入失敗：" + (event.message || "請重新整理"); };
function call(cmd, args = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, args });
  });
}
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));

// ---- 小工具 ----
function tileName(tile) {
  if (tile < 27) return RANKS[tile % 9] + SUITS[Math.floor(tile / 9)];
  if (tile < 34) return HONORS[tile - 27];
  return FLOWERS[tile - 34];
}
function esc(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function tileHtml(x, y, w, h, tile, options = {}) {
  const classes = ["tile"];
  let inner = "", style = `left:${x}px;top:${y}px;width:${w}px;height:${h}px;`;
  if (tile < 0) classes.push("back");
  else if (tile >= 34) {
    classes.push("flower");
    style += `font-size:${Math.round(h * 0.38)}px;`;
    inner = `${FLOWERS[tile - 34]}<small>花</small>`;
  } else style += `background-image:url(tiles/${tile}.png);`;
  for (const name of ["gold", "dim", "last", "pick", "declare", "hand", "can", "picked"]) if (options[name]) classes.push(name);
  const data = options.index != null ? ` data-index="${options.index}"` : "";
  return `<div class="${classes.join(" ")}" style="${style}"${data} title="${tile >= 0 ? tileName(tile) : ""}">${inner}</div>`;
}
function text(x, y, value, size = 13, color = "#fff", options = {}) {
  let style = `left:${x}px;top:${y}px;font-size:${size}px;color:${color};`;
  if (options.bold) style += "font-weight:bold;";
  if (options.anchor === "center") style += "transform:translate(-50%,-50%);";
  if (options.anchor === "right") style += "transform:translateX(-100%);";
  if (options.width) style += `white-space:normal;width:${options.width}px;`;
  return `<div class="txt" style="${style}">${esc(value)}</div>`;
}
function panel(x, y, w, h, alpha = 0.75, gold = false) {
  return `<div class="panel${gold ? " gold-border" : ""}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px;background:rgba(8,28,18,${alpha});"></div>`;
}
function isActive(view, seat) { return view.turn === seat; }
function signed(n) { return (n >= 0 ? "+" : "") + n; }

// ---- 畫牌桌 ----
function plate(view, seat, x, y, size = 64) {
  const p = view.players[seat], on = isActive(view, seat);
  let html = `<div class="avatar${on ? " on" : ""}" style="left:${x}px;top:${y}px;width:${size}px;height:${size}px;font-size:${Math.round(size * 0.5)}px;">${p.wind}</div>`;
  const tx = x + size + 10;
  html += text(tx, y + 2, p.name, 17, on ? "#ffd84d" : "#fff", { bold: true });
  html += text(tx, y + 32, `${signed(p.total)} 台（${signed(p.total * view.money_per_tai)}）`, 13, p.total >= 0 ? "#bff5d3" : "#ffb3b3");
  let bx = tx + 64;
  if (view.dealer === seat) {
    html += `<div class="badge dealer" style="left:${bx}px;top:${y + 2}px;">莊</div>`;
    if (view.streak) html += text(bx + 34, y + 6, `連${view.streak}`, 13, "#ffd0d0", { bold: true });
    bx += 66;
  }
  if (p.declared) html += `<div class="badge ting" style="left:${bx}px;top:${y + 2}px;">聽</div>`;
  return html;
}
function river(layout, tiles, gold, highlightLast) {
  const [x, y, perRow, direction] = layout;
  const step = direction === "down" ? RIVER_H + 3 : -(RIVER_H + 3);
  let html = "";
  tiles.forEach((tile, i) => {
    const row = Math.floor(i / perRow), col = i % perRow;
    const last = highlightLast && i === tiles.length - 1;
    const tx = x + col * (RIVER_W + 2), ty = y + row * step;
    html += tileHtml(tx, ty, RIVER_W, RIVER_H, tile, { gold: gold.has(tile), last });
    if (last) html += `<div class="arrow" style="left:${tx + RIVER_W / 2 - 7}px;top:${ty - 11}px;"></div>`;
  });
  return html;
}
function melds(x, y, list, w, h, right, gold) {
  const x0 = x;
  let html = "";
  for (const meld of list) {
    const width = meld.tiles.length * (w + 1) + 10;
    if (x + width > right && x > x0) { x = x0; y += h + 8; }
    for (const tile of meld.tiles) {
      html += tileHtml(x, y, w, h, tile, { gold: tile >= 0 && gold.has(tile) });
      x += w + 1;
    }
    x += 10;
  }
  return { html, x, y };
}
function flowers(x, y, list, w, h) {
  let html = "";
  for (const flower of list) { html += tileHtml(x, y, w, h, flower); x += w + 1; }
  return html;
}
function drawInfo(view) {
  const x = 16, y = 10;
  let html = panel(x, y, 340, 200);
  const dealer = view.players[view.dealer].name + (view.streak ? `（連 ${view.streak}）` : "");
  const title = view.mode === "simple" ? "單局簡化" : `東風圈　第 ${view.number} 局`;
  html += text(x + 18, y + 10, title, 18, "#ffe08a", { bold: true });
  html += text(x + 18, y + 42, `莊家 ${dealer}　·　牌牆剩 ${view.wall} 張`, 13, "#d8efe2");
  if (view.gold.length) {
    html += text(x + 18, y + 80, "金牌", 15, "#ffd84d", { bold: true });
    view.gold.forEach((tile, i) => { html += tileHtml(x + 66 + i * 34, y + 66, 32, 50, tile, { gold: true }); });
  }
  ui.messages.forEach((message, i) => { html += text(x + 18, y + 126 + i * 22, message, 14, "#cfe9da"); });
  return html;
}
function drawCompass(view) {
  const { x, y, w, h } = COMPASS, cx = x + w / 2, cy = y + h / 2;
  let html = panel(x, y, w, h, 0.6);
  const spots = { 0: [cx, y + h - 22], 1: [x + w - 24, cy], 2: [cx, y + 22], 3: [x + 24, cy] };
  for (const seat of [0, 1, 2, 3]) {
    const on = isActive(view, seat) || view.discard_player === seat;
    html += text(spots[seat][0], spots[seat][1], view.players[seat].wind, on ? 26 : 18, on ? "#ffd84d" : "#9fbfae",
      { anchor: "center", bold: on });
  }
  if (view.last_discard != null) html += tileHtml(cx - 22, cy - 38, 44, 72, view.last_discard, { last: true });
  else {
    html += text(cx, cy - 10, String(view.wall), 30, "#e8f5ec", { anchor: "center", bold: true });
    html += text(cx, cy + 24, "牌牆剩餘", 12, "#9fbfae", { anchor: "center" });
  }
  return html;
}
function drawOpponent(view, seat) {
  const layout = SEATS[seat], p = view.players[seat], gold = new Set(view.gold);
  let html = plate(view, seat, layout.avatar[0], layout.avatar[1]);
  const [bx, by, perRow, bw, bh] = layout.backs;
  for (let i = 0; i < p.count; i++) {
    const row = Math.floor(i / perRow), col = i % perRow;
    const tx = bx + col * (bw + 2), ty = by + row * (bh + 4);
    html += p.hand ? tileHtml(tx, ty, bw, bh, p.hand[i], { gold: gold.has(p.hand[i]) }) : tileHtml(tx, ty, bw, bh, -1);
  }
  const [mx, my, right] = layout.melds;
  const m = melds(mx, my, p.melds, MELD_W, MELD_H, right, gold);
  html += m.html;
  html += flowers(mx, p.melds.length ? m.y + MELD_H + 8 : my, p.flowers, MELD_W, MELD_H);
  html += river(layout.river, p.discards, gold, view.discard_player === seat);
  return html;
}
function prompt(view) {
  const actions = ui.actions;
  if (!actions) return view.players[ME].declared ? "已聽牌：系統代打中" : "";
  if (actions[0].kind === "swap") return "換三張：點 0～3 張要換掉的牌，再按右下按鈕";
  if (view.phase === "response") return `${view.players[view.discard_player].name}打出 ${tileName(view.last_discard)}：要吃碰嗎？`;
  if (ui.declareMode) return "聽牌：點藍框的牌打出並宣告";
  return "輪到你：點一張牌打出";
}
function drawMe(view) {
  const me = view.players[ME], gold = new Set(view.gold);
  let html = river(MY_RIVER, me.discards, gold, view.discard_player === ME);
  html += plate(view, ME, 20, 664);
  const m = melds(300, 664, me.melds, 40, 66, 900, gold);
  html += m.html + flowers(m.x, m.y, me.flowers, 40, 66);
  const actions = ui.actions || [];
  const swapping = actions.length > 0 && actions[0].kind === "swap";
  const wanted = ui.declareMode ? "declare" : "discard";
  const playable = new Set(actions.filter((a) => a.kind === wanted).map((a) => a.tile));
  const choosing = actions.length > 0 && (swapping || view.phase === "discard");
  const hand = me.hand || [];
  const drew = me.drew;
  const x0 = 150;
  hand.forEach((tile, i) => {
    const x = x0 + i * (HAND_W + 2) + (drew && i === hand.length - 1 ? 18 : 0);
    const can = choosing && (swapping || playable.has(tile));
    html += tileHtml(x, 780, HAND_W, HAND_H, tile, {
      index: i, hand: true, can, gold: gold.has(tile), dim: choosing && !swapping && !playable.has(tile),
      declare: ui.declareMode && playable.has(tile), pick: ui.selected.includes(i), picked: ui.selected.includes(i),
    });
  });
  const line = prompt(view);
  if (line) html += text(x0, 742, line, 15, "#ffe08a", { bold: true });
  html += drawButtons(view, actions, swapping);
  return html;
}
function drawButtons(view, actions, swapping) {
  if (!actions.length) return "";
  const buttons = [];
  if (swapping) buttons.push([ui.selected.length ? `換牌（${ui.selected.length} 張）` : "不換", "swap", "swap"]);
  else {
    let hasDeclare = false;
    for (const action of actions) {
      const name = String(action.index);
      if (action.kind === "win") buttons.push([action.self_draw ? "自摸" : "胡", name, "win"]);
      else if (action.kind === "pung") buttons.push(["碰", name, "claim"]);
      else if (action.kind === "kong") buttons.push([`槓 ${tileName(action.tile)}`, name, "claim"]);
      else if (action.kind === "chow") {
        buttons.push(["吃 " + action.tiles.map((t) => RANKS[t % 9]).join("") + SUITS[Math.floor(action.tiles[0] / 9)], name, "claim"]);
      } else if (action.kind === "pass") buttons.push(["過", name, "pass"]);
      else if (action.kind === "declare") hasDeclare = true;
    }
    if (hasDeclare) buttons.push([ui.declareMode ? "取消聽" : "聽", "declare", "declare"]);
  }
  return `<div class="buttons">` + buttons.map(([label, name, color]) =>
    `<button class="btn ${color}" data-button="${name}">${esc(label)}</button>`).join("") + `</div>`;
}
function draw() {
  const view = ui.view;
  if (!view) { table.innerHTML = ""; return; }
  table.innerHTML = drawInfo(view) + drawCompass(view) + [1, 2, 3].map((seat) => drawOpponent(view, seat)).join("") + drawMe(view);
}

// ---- 結算畫面 ----
function drawSummary(view, summary) {
  const px = 110, py = 40, gold = new Set(view.gold);
  let html = `<div class="shade"></div>` + panel(px, py, 1180, 820, 0.93, true);
  const dealer = view.players[summary.dealer].name + (summary.streak ? `（連 ${summary.streak}）` : "");
  html += text(px + 36, py + 22, `第 ${summary.number} 局　莊家：${dealer}`, 18, "#d8efe2");
  const outcome = summary.outcome + (summary.tai != null ? `　${summary.tai} 台` : "");
  html += text(px + 36, py + 50, outcome, 32, "#ffd84d", { bold: true });
  html += text(px + 36, py + 100, summary.lines.join("　"), 15, "#ffe9a8", { width: 1100 });
  for (const seat of [0, 1, 2, 3]) {
    const y = py + 150 + seat * 150, p = view.players[seat], winner = seat === summary.winner;
    if (winner) html += `<div class="abs" style="left:${px + 20}px;top:${y - 10}px;width:1140px;height:140px;border:2px solid #e6be50;border-radius:8px;"></div>`;
    html += `<div class="avatar${winner ? " on" : ""}" style="left:${px + 36}px;top:${y}px;width:56px;height:56px;font-size:28px;">${p.wind}</div>`;
    const pay = summary.pay[seat];
    html += text(px + 104, y, p.name, 17, winner ? "#ffd84d" : "#fff", { bold: true });
    html += text(px + 104, y + 28, `這局 ${signed(pay)} 台（${signed(pay * view.money_per_tai)}）　合計 ${signed(summary.totals[seat])} 台`, 14,
      pay > 0 ? "#bff5d3" : pay < 0 ? "#ffb3b3" : "#fff");
    if (p.declared) html += `<div class="badge ting" style="left:${px + 170}px;top:${y - 2}px;width:26px;height:26px;font-size:15px;">聽</div>`;
    const m = melds(px + 104, y + 58, p.melds, 34, 56, px + 1150, gold);
    html += m.html;
    let x = m.x;
    const tiles = (p.hand || []).slice();
    const winTile = winner ? summary.win_tile : null;
    if (winTile != null && tiles.includes(winTile)) { tiles.splice(tiles.lastIndexOf(winTile), 1); tiles.push(winTile); }
    tiles.forEach((tile, i) => {
      const isWin = winTile != null && i === tiles.length - 1;
      html += tileHtml(x + (isWin ? 10 : 0), m.y, 34, 56, tile, { gold: gold.has(tile), last: isWin });
      x += 35;
    });
    html += flowers(x + 24, m.y, p.flowers, 34, 56);
  }
  const label = summary.over ? (view.mode === "simple" ? "看結果" : "看整圈結果") : "下一局";
  html += `<button class="btn continue" data-button="continue" style="left:${px + 1180 - 200}px;top:${py + 820 - 70}px;">${label}</button>`;
  overlay.innerHTML = html;
}
function drawRoundEnd(totals) {
  const view = ui.view, px = 400, py = 210, pw = 600, ph = 460;
  let html = `<div class="shade"></div>` + panel(px, py, pw, ph, 0.94, true);
  html += text(px + pw / 2, py + 52, view.mode === "simple" ? "這一局結束" : "一圈結束", 34, "#ffd84d", { anchor: "center", bold: true });
  const order = [0, 1, 2, 3].sort((a, b) => totals[b] - totals[a]);
  order.forEach((seat, rank) => {
    const y = py + 100 + rank * 64, color = seat === ME ? "#ffd84d" : "#fff";
    html += text(px + 70, y, String(rank + 1), 28, color, { bold: true });
    html += text(px + 120, y + 4, view.players[seat].name, 24, color, { bold: true });
    html += text(px + pw - 60, y + 6, `${signed(totals[seat])} 台（${signed(totals[seat] * view.money_per_tai)}）`, 22,
      totals[seat] >= 0 ? "#bff5d3" : "#ffb3b3", { anchor: "right" });
  });
  html += `<button class="btn swap" data-button="again" style="left:${px + 90}px;top:${py + ph - 84}px;">再來一次</button>`;
  html += `<button class="btn pass" data-button="home" style="left:${px + pw - 250}px;top:${py + ph - 84}px;">回首頁</button>`;
  overlay.innerHTML = html;
}

// ---- 操作 ----
function waitFor() { return new Promise((resolve) => { ui.resolve = resolve; }); }
function reply(value) {
  const resolve = ui.resolve;
  ui.resolve = null;
  if (resolve) resolve(value);
}
function clearAsk() { ui.actions = null; ui.selected = []; ui.declareMode = false; }
document.getElementById("stage").addEventListener("click", (event) => {
  const button = event.target.closest("[data-button]");
  if (button) { press(button.dataset.button); return; }
  const tile = event.target.closest(".tile.hand");
  if (tile) tileClicked(Number(tile.dataset.index));
});
function press(name) {
  if (name === "continue" || name === "again" || name === "home") { reply(name); return; }
  if (!ui.actions) return;
  if (name === "swap") {
    const hand = ui.view.players[ME].hand;
    reply({ swap: ui.selected.map((i) => hand[i]) });
  } else if (name === "declare") {
    ui.declareMode = !ui.declareMode;
    draw();
  } else reply({ index: Number(name) });
}
function tileClicked(index) {
  if (!ui.actions) return;
  if (ui.actions[0].kind === "swap") {
    const at = ui.selected.indexOf(index);
    if (at >= 0) ui.selected.splice(at, 1);
    else if (ui.selected.length < 3) ui.selected.push(index);
    draw();
    return;
  }
  if (ui.view.phase !== "discard") return;
  const tile = ui.view.players[ME].hand[index], kind = ui.declareMode ? "declare" : "discard";
  const action = ui.actions.find((a) => a.kind === kind && a.tile === tile);
  if (action) reply({ index: action.index });
}

// ---- 牌局迴圈 ----
function say(message) {
  if (!message) return;
  ui.messages.push(message);
  if (ui.messages.length > 3) ui.messages.shift();
}
async function playSession(mode) {
  ui.mode = mode;
  await call("new", { mode });
  ui.messages = [];
  while (true) {
    const started = performance.now();
    const out = await call("step");
    if (out.type === "round_end") {
      drawRoundEnd(out.totals);
      const choice = await waitFor();
      overlay.innerHTML = "";
      return choice;
    }
    if (out.type === "ask") {
      ui.view = out.view;
      ui.actions = out.actions;
      draw();
      const answer = await waitFor();
      clearAsk();
      draw();
      await call("answer", answer);
      continue;
    }
    if (out.type === "hand_end") {
      if (out.last) {
        say(`${out.view.players[out.last.player].name} ${out.last.kind === "win" ? "胡！" : ""}`.trim());
        await sleep(CLAIM_DELAY);
      }
      ui.view = out.view;
      draw();
      drawSummary(out.view, out.summary);
      await waitFor();
      overlay.innerHTML = "";
      ui.messages = [];
      continue;
    }
    // deal / move
    const bot = out.player != null && out.player !== ME;
    if (out.type === "move" && bot && (out.kind === "discard" || out.kind === "declare")) {
      await sleep(DISCARD_DELAY - (performance.now() - started));
    }
    ui.view = out.view;
    say(out.message);
    draw();
    if (out.type === "deal") await sleep(400);
    else if (bot && CLAIMS.has(out.kind)) await sleep(CLAIM_DELAY);
    else if (out.kind === "draw") await sleep(DRAW_DELAY);
    else await sleep(60);
  }
}
async function start(mode) {
  home.style.display = "none";
  try {
    let choice = "again";
    while (choice === "again") choice = await playSession(mode);
  } catch (error) {
    statusLine.textContent = "發生錯誤：" + error.message;
  }
  ui.view = null;
  draw();
  home.style.display = "";
}

// ---- 縮放與啟動 ----
function fit() {
  const scale = Math.min(window.innerWidth / W, window.innerHeight / H);
  const stage = document.getElementById("stage");
  stage.style.transform = `scale(${scale})`;
  stage.style.left = `${(window.innerWidth - W * scale) / 2}px`;
  stage.style.top = `${(window.innerHeight - H * scale) / 2}px`;
}
window.addEventListener("resize", fit);
fit();
for (const button of document.querySelectorAll(".mode")) {
  button.addEventListener("click", () => start(button.dataset.mode));
}
call("ready").then(() => {
  statusLine.textContent = "準備好了，選一種玩法開始";
  for (const button of document.querySelectorAll(".mode")) button.disabled = false;
}).catch((error) => { statusLine.textContent = "載入失敗：" + error.message; });
