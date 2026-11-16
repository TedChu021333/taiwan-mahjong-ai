// 網頁版牌桌：畫面與操作。牌局、規則、AI 都在 worker.js（Pyodide 跑和電腦版同一套 Python）。
// 版面沿用 Tk 版（play/__main__.py）的 1400×900 座標。
"use strict";

const W = 1400, H = 900;
const ME = 0;
// 節奏（11/16 使用者：吃碰要講出來、不要那麼快，不然看不懂發生什麼事）。三段速度可切換，預設「正常」。
const SPEEDS = {
  slow:   { label: "慢", discard: 2000, claim: 2200, draw: 450, win: 2600 },
  normal: { label: "正常", discard: 1400, claim: 1700, draw: 300, win: 2200 },
  fast:   { label: "快", discard: 700, claim: 1000, draw: 120, win: 1400 },
};
const MESSAGES = 4;
const BANNER = {
  chow: ["吃", "#9b4dd8"], pung: ["碰", "#2e9e5b"], kong: ["槓", "#e08a1e"], win: ["胡", "#e03434"],
  self_draw: ["自摸", "#e03434"], declare: ["聽", "#2f78e0"],
};
const BANNER_SPOTS = { 0: [700, 610], 1: [1010, 420], 2: [700, 190], 3: [390, 420] };
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
  lastDiscard: null, // {player, tile}：最近一張打出、還在牌河裡的牌（中央亮出、牌河加箭頭）
  banner: null,      // {seat, kind, tiles}：吃碰槓胡聽的大字
  fresh: [],         // 換三張換到的牌（打出第一張前標綠框）
  swapBefore: null,  // 換三張送出時的手牌與換出的牌
  declaredAt: {},    // 座位 → 宣告聽牌時打的是牌河第幾張
  speed: loadSpeed(),
};
function loadSpeed() {
  try { const v = localStorage.getItem("mahjong-speed"); if (v && SPEEDS[v]) return v; } catch (e) { /* 無痕視窗等 */ }
  return "normal";
}
function pace() { return SPEEDS[ui.speed]; }

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
  else style += `background-image:url(tiles/${tile}.png);`; // 34～41 是花牌（從遊戲結算截圖裁的）
  for (const name of ["gold", "dim", "last", "pick", "declare", "hand", "can", "picked", "fresh", "swap"]) if (options[name]) classes.push(name);
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
function lastIsFrom(view, seat) {
  const last = ui.lastDiscard;
  if (!last || last.player !== seat) return false;
  const river = view.players[seat].discards;
  return river.length > 0 && river[river.length - 1] === last.tile;
}
function signed(n) { return (n >= 0 ? "+" : "") + n; }

// ---- 畫牌桌 ----
function plate(view, seat, x, y, size = 64) {
  const p = view.players[seat], on = isActive(view, seat);
  let html = `<div class="avatar${on ? " on" : ""}${p.declared ? " ting" : ""}" style="left:${x}px;top:${y}px;width:${size}px;height:${size}px;font-size:${Math.round(size * 0.5)}px;">${p.wind}</div>`;
  const tx = x + size + 10;
  html += text(tx, y + 2, p.name, 17, on ? "#ffd84d" : "#fff", { bold: true });
  html += text(tx, y + 32, `${signed(p.total)} 台（${signed(p.total * view.money_per_tai)}）`, 13, p.total >= 0 ? "#bff5d3" : "#ffb3b3");
  let bx = tx + 64;
  if (view.dealer === seat) {
    html += `<div class="badge dealer" style="left:${bx}px;top:${y + 2}px;">莊</div>`;
    if (view.streak) html += text(bx + 34, y + 6, `連${view.streak}`, 13, "#ffd0d0", { bold: true });
    bx += 66;
  }
  if (p.declared) html += `<div class="badge ting wide${seat === ME ? "" : " alert"}" style="left:${bx}px;top:${y + 2}px;">聽牌</div>`;
  return html;
}
function river(layout, tiles, gold, highlightLast, declaredAt = null) {
  const [x, y, perRow, direction] = layout;
  const step = direction === "down" ? RIVER_H + 3 : -(RIVER_H + 3);
  let html = "";
  tiles.forEach((tile, i) => {
    const row = Math.floor(i / perRow), col = i % perRow;
    const last = highlightLast && i === tiles.length - 1;
    const tx = x + col * (RIVER_W + 2), ty = y + row * step;
    html += tileHtml(tx, ty, RIVER_W, RIVER_H, tile, { gold: gold.has(tile), last, declare: i === declaredAt && !last });
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
  let html = panel(x, y, 340, 210);
  const dealer = view.players[view.dealer].name + (view.streak ? `（連 ${view.streak}）` : "");
  const title = view.mode === "simple" ? "單局簡化" : `東風圈　第 ${view.number} 局`;
  html += text(x + 18, y + 10, title, 18, "#ffe08a", { bold: true });
  html += text(x + 18, y + 42, `莊家 ${dealer}　·　牌牆剩 ${view.wall} 張`, 13, "#d8efe2");
  if (view.gold.length) {
    html += text(x + 18, y + 80, "金牌", 15, "#ffd84d", { bold: true });
    view.gold.forEach((tile, i) => { html += tileHtml(x + 66 + i * 34, y + 66, 32, 50, tile, { gold: true }); });
  }
  const newest = ui.messages.length - 1;
  ui.messages.forEach((message, i) => {
    html += text(x + 18, y + 118 + i * 21, message, i === newest ? 15 : 13, i === newest ? "#fff3c4" : "#9fbfae",
      { bold: i === newest });
  });
  return html;
}
function drawCompass(view) {
  const { x, y, w, h } = COMPASS, cx = x + w / 2, cy = y + h / 2;
  let html = panel(x, y, w, h, 0.6);
  const spots = { 0: [cx, y + h - 15], 1: [x + w - 24, cy], 2: [cx, y + 15], 3: [x + 24, cy] };
  for (const seat of [0, 1, 2, 3]) {
    const on = isActive(view, seat) || view.discard_player === seat;
    html += text(spots[seat][0], spots[seat][1], view.players[seat].wind, on ? 26 : 18, on ? "#ffd84d" : "#9fbfae",
      { anchor: "center", bold: on });
  }
  const shown = view.last_discard != null ? { player: view.discard_player, tile: view.last_discard }
    : (ui.lastDiscard && lastIsFrom(view, ui.lastDiscard.player) ? ui.lastDiscard : null);
  if (shown) {
    html += tileHtml(cx - 22, cy - 42, 44, 68, shown.tile, { last: true });
    html += text(cx, cy + 37, `${view.players[shown.player].name}打出`, 13, "#ffe08a", { anchor: "center", bold: true });
  } else {
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
  html += river(layout.river, p.discards, gold, lastIsFrom(view, seat), ui.declaredAt[seat]);
  return html;
}
function prompt(view) {
  const actions = ui.actions;
  if (!actions) return view.players[ME].declared ? "已聽牌：系統代打中" : "";
  if (actions[0].kind === "swap") return "換三張：點 0～3 張要換掉的牌，再按右下按鈕";
  if (view.phase === "response") return `${view.players[view.discard_player].name}打出 ${tileName(view.last_discard)}：要吃碰嗎？`;
  if (ui.declareMode) return "聽牌：點藍框的牌打出並宣告";
  const hand = view.players[ME].hand || [];
  const playable = new Set(actions.filter((a) => a.kind === "discard").map((a) => a.tile));
  if (hand.some((tile) => !playable.has(tile))) return "輪到你：點一張牌打出（變暗的牌是吃碰後這手不能打的）";
  return "輪到你：點一張牌打出";
}
function drawMe(view) {
  const me = view.players[ME], gold = new Set(view.gold);
  let html = river(MY_RIVER, me.discards, gold, lastIsFrom(view, ME), ui.declaredAt[ME]);
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
  const fresh = ui.fresh.slice();
  hand.forEach((tile, i) => {
    const freshAt = fresh.indexOf(tile);
    if (freshAt >= 0) fresh.splice(freshAt, 1);
    const x = x0 + i * (HAND_W + 2) + (drew && i === hand.length - 1 ? 18 : 0);
    const can = choosing && (swapping || playable.has(tile));
    html += tileHtml(x, 780, HAND_W, HAND_H, tile, {
      index: i, hand: true, can, gold: gold.has(tile), dim: choosing && !swapping && !playable.has(tile),
      declare: ui.declareMode && playable.has(tile), pick: ui.selected.includes(i), picked: ui.selected.includes(i),
      fresh: freshAt >= 0, swap: swapping,
    });
  });
  const choice = drawButtons(view, actions, swapping);
  const line = prompt(view);
  if (choice) html += choice;
  else if (line) {
    const ready = [1, 2, 3].filter((seat) => view.players[seat].declared).map((seat) => view.players[seat].name);
    const warn = ready.length && actions.length && view.phase === "discard" && !swapping;
    if (warn) html += text(x0, 742, `${line}　⚠ ${ready.join("、")}已聽牌，小心放槍`, 17, "#ff8a80", { bold: true });
    else html += text(x0, 742, line, 17, "#ffe08a", { bold: true });
  }
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
  // 要自己決定時，選擇框放在畫面正中間、手牌正上方（11/16 使用者：按鈕在右邊怕沒看到、以為卡住）
  let title = "", hint = "", tile = null;
  if (swapping) {
    title = "換三張";
    hint = "點 0～3 張要換掉的牌（會浮起來），再按下面的按鈕";
  } else if (view.phase === "response") {
    tile = view.last_discard;
    title = `${view.players[view.discard_player].name}打出 ${tileName(tile)}`;
    hint = actions.some((a) => a.kind === "win") ? "可以胡這張！" : "要吃碰嗎？不要就按「過」";
  } else {
    if (actions.some((a) => a.kind === "win")) title = "可以自摸！";
    else if (ui.declareMode) title = "宣告聽牌";
    else if (actions.some((a) => a.kind === "declare")) title = "你聽牌了，要宣告嗎？";
    else title = "可以槓";
    hint = ui.declareMode ? "點藍框的牌打出並宣告（多 1 台，之後系統代打）"
      : "不用的話，直接點下面的手牌打出";
  }
  if (!buttons.length) return ""; // 一般打牌：沒有按鈕，只顯示提示文字
  const row = buttons.map(([label, name, color]) =>
    `<button class="btn big ${color}${color === "pass" ? "" : " pulse"}" data-button="${name}">${esc(label)}</button>`).join("");
  const tileBox = tile != null ? `<div class="choice-tile">${tileHtml(0, 0, 52, 84, tile, { last: true })}</div>` : "";
  return `<div class="choice">${tileBox}<div class="choice-body"><div class="choice-title">${esc(title)}</div>
    <div class="choice-hint">${esc(hint)}</div><div class="choice-row">${row}</div></div></div>`;
}
function drawBanner(view) {
  const banner = ui.banner;
  if (!banner) return "";
  const [word, color] = BANNER[banner.kind];
  const [x, y] = BANNER_SPOTS[banner.seat];
  let tiles = "";
  for (const tile of banner.tiles) tiles += tileHtml(0, 0, 46, 74, tile, { gold: tile >= 0 && view.gold.includes(tile) });
  const width = Math.max(150, banner.tiles.length * 50);
  return `<div class="banner" style="left:${x}px;top:${y}px;--c:${color};">
    <div class="word">${word}</div><div class="who">${esc(view.players[banner.seat].name)}</div>
    <div class="tiles" style="width:${width - 4}px;height:80px;">${tiles}</div></div>`;
}
function drawSpeed() {
  // 右下角（手牌最多 17 張，排到 x≈1222）
  let html = text(1262, 826, "出牌速度", 13, "#9fbfae");
  let x = 1262;
  for (const [key, speed] of Object.entries(SPEEDS)) {
    html += `<button class="speed${ui.speed === key ? " on" : ""}" data-button="speed:${key}" style="left:${x}px;top:850px;">${speed.label}</button>`;
    x += key === "normal" ? 54 : 38;
  }
  return html;
}
function draw() {
  const view = ui.view;
  if (!view) { table.innerHTML = ""; return; }
  table.innerHTML = drawInfo(view) + drawCompass(view) + [1, 2, 3].map((seat) => drawOpponent(view, seat)).join("")
    + drawMe(view) + drawSpeed() + drawBanner(view);
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
  if (name.startsWith("speed:")) {
    ui.speed = name.slice(6);
    try { localStorage.setItem("mahjong-speed", ui.speed); } catch (e) { /* 存不了就算了 */ }
    draw();
    return;
  }
  if (name === "continue" || name === "again" || name === "home") { reply(name); return; }
  if (!ui.actions) return;
  if (name === "swap") {
    const hand = ui.view.players[ME].hand;
    const given = ui.selected.map((i) => hand[i]);
    ui.swapBefore = { hand: hand.slice(), given };
    reply({ swap: given });
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
  while (ui.messages.length > MESSAGES) ui.messages.shift();
}
function tilesText(tiles) { return tiles.map(tileName).join(" "); }
function swapResult(view) {
  // 換三張結束（換三張要等四家都換完才生效）：比較送出前後的手牌，標出換到的牌
  const before = ui.swapBefore;
  if (!before || view.phase === "swap") return;
  ui.swapBefore = null;
  const kept = before.hand.slice();
  for (const tile of before.given) kept.splice(kept.indexOf(tile), 1);
  const received = view.players[ME].hand.slice();
  for (const tile of kept) {
    const at = received.indexOf(tile);
    if (at >= 0) received.splice(at, 1);
  }
  ui.fresh = before.given.length ? received.slice(0, before.given.length) : [];
  say(before.given.length ? `換三張：換出 ${tilesText(before.given)}，換到 ${tilesText(ui.fresh)}` : "換三張：沒有換牌");
}
async function showBanner(seat, kind, tiles, ms) {
  ui.banner = { seat, kind, tiles };
  draw();
  await sleep(ms);
  ui.banner = null;
}
function claimMessage(view, out) {
  // 「上家 碰 對家打的 九條」：吃碰槓要講清楚拿了誰的哪張牌
  const name = view.players[out.player].name;
  const from = ui.lastDiscard && ui.lastDiscard.player !== out.player ? ui.lastDiscard : null;
  const source = from ? `${view.players[from.player].name}打的 ` : "";
  if (out.kind === "pung") return `${name} 碰 ${source}${tileName(out.tile)}`;
  if (out.kind === "chow") return `${name} 吃 ${source}${tileName(from ? from.tile : out.tile)}`;
  if (out.kind === "kong") return from ? `${name} 槓 ${source}${tileName(out.tile)}` : `${name} 槓 ${tileName(out.tile)}`;
  return out.message;
}
function meldTiles(view, seat) {
  const melds = view.players[seat].melds;
  const meld = melds[melds.length - 1];
  return meld ? meld.tiles : [];
}
async function playSession(mode) {
  ui.mode = mode;
  await call("new", { mode });
  ui.messages = [];
  ui.lastDiscard = null;
  ui.fresh = [];
  ui.declaredAt = {};
  while (true) {
    const started = performance.now();
    const out = await call("step");
    if (out.type === "round_end") {
      drawRoundEnd(out.totals);
      const choice = await waitFor();
      overlay.innerHTML = "";
      return choice;
    }
    if (out.view) swapResult(out.view);
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
      const summary = out.summary;
      if (summary.winner != null) {
        // 先在牌桌上講誰胡了、胡哪張，停一下再出結算
        const selfDraw = summary.discarder == null;
        say(selfDraw ? `${out.view.players[summary.winner].name} 自摸 ${tileName(summary.win_tile)}！`
          : `${out.view.players[summary.winner].name} 胡 ${out.view.players[summary.discarder].name}打的 ${tileName(summary.win_tile)}！`);
        await showBanner(summary.winner, selfDraw ? "self_draw" : "win", [summary.win_tile], pace().win);
      } else {
        say("流局：牌摸完了，沒有人胡");
        draw();
        await sleep(pace().claim);
      }
      ui.view = out.view;
      draw();
      drawSummary(out.view, summary);
      await waitFor();
      overlay.innerHTML = "";
      ui.messages = [];
      ui.lastDiscard = null;
      ui.fresh = [];
  ui.declaredAt = {};
      continue;
    }
    if (out.type === "deal") {
      ui.view = out.view;
      ui.lastDiscard = null;
      say(mode === "simple" ? "新的一局開始" : `第 ${out.view.number} 局開始，莊家是${out.view.players[out.view.dealer].name}`);
      draw();
      await sleep(800);
      continue;
    }
    // move：一步一步照節奏顯示
    const bot = out.player !== ME;
    const kind = out.kind;
    if (kind === "discard" || kind === "declare") {
      if (bot) await sleep(pace().discard - (performance.now() - started));
      if (out.player === ME) ui.fresh = [];
      ui.view = out.view;
      ui.lastDiscard = { player: out.player, tile: out.tile };
      say(out.message);
      if (kind === "declare") {
        ui.declaredAt[out.player] = out.view.players[out.player].discards.length - 1;
        if (out.player !== ME) say(`⚠ ${out.view.players[out.player].name}聽牌了！`);
        await showBanner(out.player, "declare", [out.tile], pace().claim + 600);
        draw();
      } else {
        draw();
        await sleep(bot ? 250 : 120);
      }
      continue;
    }
    if (kind === "chow" || kind === "pung" || kind === "kong") {
      ui.view = out.view;
      say(claimMessage(out.view, out));
      if (ui.lastDiscard && ui.lastDiscard.player !== out.player) ui.lastDiscard = null; // 那張牌被拿走了
      await showBanner(out.player, kind, meldTiles(out.view, out.player), pace().claim);
      draw();
      continue;
    }
    ui.view = out.view;
    draw();
    await sleep(kind === "draw" ? pace().draw : 40);
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
