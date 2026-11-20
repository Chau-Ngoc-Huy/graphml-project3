"use strict";
/* MMXAI Lab — giao diện một trang. Không framework; Chart.js + Cytoscape.js từ cdnjs. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* giữ statusText */ }
    throw new Error(msg);
  }
  return r.json();
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 3500);
}

/* ---------------- biến thể (A = đang xem, B = so sánh) ---------------- */
let VARS = [], curVar = null, DATASET = "VG";
const varTitle = (id) => (VARS.find((v) => v.id === id) || {}).title || id;
const vq = () => (curVar ? `variant=${encodeURIComponent(curVar)}` : "");

async function loadVariants() {
  const r = await api("/api/variants");
  VARS = r.items;
  if (r.dataset) {  // VG hoặc PSG: chữ hiển thị + kích thước pool theo bundle đang mở
    DATASET = r.dataset;
    document.title = `MMXAI · ${r.dataset}`;
    const pl = r.pool || {};
    $("#pool").innerHTML = `<option value="test">test (${pl.test ?? "?"})</option><option value="val">val (${pl.val ?? "?"})</option>`;
    document.querySelectorAll(".ds-name").forEach((e) => { e.textContent = r.dataset; });
  }
  let saved = null;
  try { saved = localStorage.getItem("mmxai.variant"); } catch (_) { /* bộ nhớ trình duyệt bị chặn */ }
  const hv = new URLSearchParams(location.hash.split("?")[1] || "").get("v");
  curVar = [hv, saved, r.default].find((x) => x && VARS.some((v) => v.id === x));
  const opts = (sel) => VARS.map((v) => `<option value="${esc(v.id)}"${v.id === sel ? " selected" : ""}>${esc(v.title)}${v.loaded ? "" : " (đang nạp…)"}</option>`).join("");
  $("#varSel").innerHTML = opts(curVar);
  const cmp = $("#cmpSel").value;
  $("#cmpSel").innerHTML = `<option value="">— không so sánh (hiện “chỉ tín hiệu đồ thị” của A) —</option>` + opts(cmp);
  $("#cmpSel").value = cmp;
  showVarNote();
  if (VARS.some((v) => !v.loaded)) setTimeout(loadVariants, 8000);  // cập nhật trạng thái nạp nền
}
function showVarNote() {
  const v = VARS.find((x) => x.id === curVar);
  $("#varNote").textContent = v ? `${v.note}${v.alpha != null ? ` · α=${v.alpha}` : ""}` : "";
}
function setVariant(id) {
  curVar = id; $("#varSel").value = id; showVarNote();
  try { localStorage.setItem("mmxai.variant", id); } catch (_) { /* bỏ qua */ }
  // làm mới trang đang mở theo biến thể mới
  if (lastQuery) runSearch(lastQuery);
  if (loaded.builder && buildRow != null) loadBuild(buildRow);
  if (loaded.graph && $("#labelInput").value) showLabel($("#labelInput").value);
  if (loaded.data) initData();
  toast(`Biến thể: ${varTitle(id)}`);
}

/* ---------------- điều hướng (hash) ---------------- */
const loaded = {};
function route() {
  const v = (location.hash || "#search").slice(1).split("?")[0];
  const view = ["search", "data", "builder", "graph", "ablation"].includes(v) ? v : "search";
  $$(".view").forEach((s) => (s.hidden = s.id !== `view-${view}`));
  $$(".nav a").forEach((a) => { if (a.dataset.view !== view) a.removeAttribute("aria-current"); else a.setAttribute("aria-current", "page"); });
  $("#main").focus({ preventScroll: true });
  if (!loaded[view]) { loaded[view] = true; ({ search: initSearch, data: initData, builder: initBuilder, graph: initGraph, ablation: initAblation })[view](); }
  else if (view === "builder") builderFromHash();
}
window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", async () => {
  Chart.defaults.color = css("--fg-2");
  Chart.defaults.borderColor = "rgba(148,163,184,.15)";
  Chart.defaults.font.family = css("--sans");
  $("#varSel").addEventListener("change", (e) => setVariant(e.target.value));
  $("#cmpSel").addEventListener("change", () => lastQuery && runSearch(lastQuery));
  try { await loadVariants(); } catch (e) { $("#varNote").textContent = "Lỗi nạp danh sách biến thể: " + e.message; }
  route();
  $("#modelInfo").textContent = `${VARS.length} biến thể · dữ liệu thật`;
});

/* ---------------- THỬ MÔ HÌNH ---------------- */
let sampleGroup = "", sampleSeed = 0, lastSearch = null, lastQuery = null;

function initSearch() {
  $("#searchForm").addEventListener("submit", (e) => { e.preventDefault(); runSearch({ text: $("#q").value }); });
  $$(".seg .chip-btn").forEach((b) => b.addEventListener("click", () => {
    $$(".seg .chip-btn").forEach((x) => x.classList.toggle("on", x === b));
    sampleGroup = b.dataset.group; loadSamples();
  }));
  $("#btnShuffle").addEventListener("click", () => { sampleSeed++; loadSamples(); });
  $("#btnCloseExplain").addEventListener("click", () => { $("#explain").hidden = true; $$(".icard.sel").forEach((c) => c.classList.remove("sel")); });
  loadSamples();
  // link chia sẻ được: #search?caption=<id>[&row=<ảnh>][&b=<biến thể B>]  hoặc  #search?q=<câu>
  const qs = new URLSearchParams(location.hash.split("?")[1] || "");
  if (qs.get("b")) $("#cmpSel").value = qs.get("b");
  if (qs.get("caption")) runSearch({ caption_id: +qs.get("caption") }).then(() => qs.get("row") && openExplain(+qs.get("row"), curVar));
  else if (qs.get("q")) { $("#q").value = qs.get("q"); runSearch({ text: qs.get("q") }); }
}

async function loadSamples() {
  const ul = $("#sampleList");
  ul.innerHTML = `<li class="small" style="padding:10px">Đang tải…</li>`;
  try {
    const xs = await api(`/api/captions?split=test&group=${sampleGroup}&n=12&seed=${sampleSeed}`);
    ul.innerHTML = xs.map((c) => `<li><button type="button" data-id="${c.caption_id}">
      <span>${esc(c.text)}</span><span class="tag ${esc(c.group)}">${esc(c.group)}</span>
      <span class="cc">khái niệm: ${esc(c.concepts.join(", ") || "—")}</span></button></li>`).join("");
    $$("button", ul).forEach((b) => b.addEventListener("click", () => {
      const c = xs.find((x) => String(x.caption_id) === b.dataset.id);
      $("#q").value = c.text; $("#concepts").value = c.concepts.join(", ");
      runSearch({ caption_id: c.caption_id });
    }));
  } catch (e) { ul.innerHTML = `<li class="err">${esc(e.message)}</li>`; }
}

async function runSearch(base) {
  const btn = $("#btnSearch"); const err = $("#searchErr"); err.textContent = "";
  const cTxt = $("#concepts").value.trim();
  const compare = $("#cmpSel").value || null;
  const body = { ...base, split: $("#pool").value, k: 12, variant: curVar, compare };
  if (cTxt) body.concepts = cTxt.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  if (base.caption_id == null && !(base.text || "").trim()) { err.textContent = "Nhập câu truy vấn hoặc chọn một caption mẫu."; $("#q").focus(); return; }
  lastQuery = base;
  btn.disabled = true; btn.lastChild.textContent = "Đang tìm…";
  ["clip", "A", "B"].forEach((k) => ($(`#col-${k}`).innerHTML = Array(6).fill('<div class="skel"></div>').join("")));
  $("#results").hidden = false; $("#explain").hidden = true;
  try {
    const r = await api("/api/search", body);
    lastSearch = r;
    if (!cTxt) $("#concepts").value = r.variants.A.concepts.join(", ");
    if (!$("#q").value) $("#q").value = r.text;
    const tail = `${compare ? `&b=${encodeURIComponent(compare)}` : ""}&v=${encodeURIComponent(curVar)}`;
    history.replaceState(null, "", base.caption_id != null ? `#search?caption=${base.caption_id}${tail}` : `#search?q=${encodeURIComponent(r.text)}${tail}`);
    renderQueryBar(r);
    const A = r.variants.A, B = r.variants.B;
    renderCol("clip", r.clip, r, null);
    $("#hdr-A").innerHTML = `<span class="dot final"></span>A · ${esc(A.variant.title)} <small>kết quả cuối · α=${A.alpha}</small>`;
    renderCol("A", A.final, r, A);
    if (B) {
      $("#hdr-B").innerHTML = `<span class="dot graph"></span>B · ${esc(B.variant.title)} <small>kết quả cuối · α=${B.alpha}</small>`;
      renderCol("B", B.final, r, B);
    } else {
      $("#hdr-B").innerHTML = `<span class="dot graph"></span>A · chỉ tín hiệu đồ thị <small>s<sub>G</sub> trên top-100</small>`;
      renderCol("B", A.graph, r, A);
    }
    if (r.gold) loadGoldAll(body); else $("#goldAll").hidden = true;
  } catch (e) {
    err.textContent = e.message + " — kiểm tra câu/khái niệm rồi thử lại.";
    $("#results").hidden = true;
  } finally { btn.disabled = false; btn.lastChild.textContent = "Tìm"; }
}

async function loadGoldAll(body) {
  const box = $("#goldAll"), out = $("#goldAllBody"); box.hidden = false;
  out.innerHTML = '<div class="skel" style="aspect-ratio:auto;height:80px"></div>';
  try {
    const r = await api("/api/gold_all", body);
    if (!r.in_top_k) { out.innerHTML = `<p class="small">Ảnh đúng nằm ngoài top-${r.K} của CLIP → không biến thể nào xếp lại được (cùng một tập ứng viên).</p>`; return; }
    const clipRank = r.rows[0].rank;
    out.innerHTML = r.rows.map((x) => {
      const pct = x.rank ? Math.max(4, 100 * (1 - (Math.log(x.rank) / Math.log(r.K)))) : 0;
      const d = x.rank && x.id !== "clip" ? clipRank - x.rank : null;
      const col = x.id === "clip" ? css("--c-clip") : d > 0 ? css("--accent") : d < 0 ? css("--danger") : css("--muted");
      const dTxt = d == null ? "" : d > 0 ? ` <span class="up">▲${d}</span>` : d < 0 ? ` <span class="down">▼${-d}</span>` : ' <span class="same">＝</span>';
      return `<button type="button" class="rankbar ${x.id === curVar ? "cur" : ""}" data-id="${esc(x.id)}" ${x.id === "clip" || x.pending ? "disabled" : ""}>
        <span class="tt">${esc(x.title)}${x.pending ? " <small>đang nạp…</small>" : x.note ? `<small>${esc(x.note)}</small>` : ""}</span>
        <span class="rkn">${x.rank ? "#" + x.rank : "—"}${dTxt}</span>
        <span class="track" aria-hidden="true"><i style="width:${pct}%;background:${col}"></i></span></button>`;
    }).join("");
    $$(".rankbar", out).forEach((b) => !b.disabled && b.addEventListener("click", () => setVariant(b.dataset.id)));
    if (r.rows.some((x) => x.pending)) setTimeout(() => lastSearch && loadGoldAll(body), 10000);
  } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

function moveHtml(a, b, label) {
  if (a == null || b == null) return `<span class="same">ngoài top-${lastSearch?.K ?? 100} của CLIP</span>`;
  const d = a - b;
  const cls = d > 0 ? "up" : d < 0 ? "down" : "same";
  const word = d > 0 ? `lên ${d}` : d < 0 ? `xuống ${-d}` : "giữ nguyên";
  return `<span class="${cls}">${label} #${b} (${word})</span>`;
}
function renderQueryBar(r) {
  const qb = $("#queryBar"); qb.hidden = false;
  const A = r.variants.A, B = r.variants.B;
  const chips = A.concepts.map((c) => `<span class="chip">${esc(c)}</span>`).join("") +
    A.concepts_unknown.map((c) => `<span class="chip bad" title="không có trong bộ nhãn">${esc(c)}</span>`).join("");
  const g = r.gold;
  qb.innerHTML = `<span class="qtext">“${esc(r.text)}” ${r.group ? `<span class="tag ${esc(r.group)}">${esc(r.group)}</span>` : ""}</span>
    <span class="chips" aria-label="Khái niệm dùng để chấm đồ thị">${chips || '<span class="chip plain">không có khái niệm → s_G = 0</span>'}</span>
    ${g ? `<span class="gold-move" aria-label="Thứ hạng ảnh đúng">Ảnh đúng: <span>CLIP #${g.clip_rank ?? "—"}</span> →
      ${moveHtml(g.clip_rank, A.gold_rank, "A")}${B ? ` · ${moveHtml(g.clip_rank, B.gold_rank, "B")}` : ""}</span>` : ""}
    <span class="small mono">pool ${r.pool} (${r.pool_size}) · top-${r.K} · w<sub>A</sub>=${fmt(A.w)}</span>`;
}

function imgTag(c, cls = "thumb") {
  const alt = `Ảnh #${c.idx}: ${c.labels.slice(0, 6).join(", ")}`;
  return `<img class="${cls}" loading="lazy" width="320" height="240" referrerpolicy="no-referrer" alt="${esc(alt)}"
    src="${esc(c.url)}" data-alt="${esc(c.coco_url || "")}" onerror="if(this.dataset.alt){this.src=this.dataset.alt;this.dataset.alt=''}else{this.style.visibility='hidden'}">`;
}
function renderCol(key, items, r, V) {
  const el = $(`#col-${key}`);
  const gvals = items.map((x) => x.graph).filter((x) => x != null);
  const maxG = Math.max(1e-6, ...gvals);
  el.innerHTML = items.map((c) => {
    const delta = V && c.final_rank != null && c.rank === c.final_rank ? c.clip_rank - c.final_rank : null;
    const dTxt = delta == null ? "" : delta > 0 ? `<span class="delta up">▲${delta}</span>` : delta < 0 ? `<span class="delta down">▼${-delta}</span>` : `<span class="delta same">＝</span>`;
    const gTxt = c.graph != null ? ` · s<sub>G</sub> ${fmt(c.graph, 3)}` : "";
    return `<button type="button" class="icard ${c.is_gold ? "gold" : ""}" data-row="${c.row}" data-v="${esc(V ? V.variant.id : curVar)}"
      aria-label="Hạng ${c.rank}${c.is_gold ? ", ảnh đúng" : ""}. Xem giải thích">
      ${imgTag(c)}<span class="rk">#${c.rank}</span>${c.is_gold ? '<span class="gbadge">ẢNH ĐÚNG</span>' : ""}
      <div class="meta">${dTxt} cos ${fmt(c.clip, 3)}${gTxt}
        ${c.graph != null ? `<div class="bar" aria-hidden="true"><i style="width:${Math.max(0, (c.graph / maxG) * 100)}%"></i></div>` : ""}</div></button>`;
  }).join("");
  $$(".icard", el).forEach((b) => b.addEventListener("click", () => {
    $$(".icard.sel").forEach((x) => x.classList.remove("sel")); b.classList.add("sel");
    openExplain(+b.dataset.row, b.dataset.v);
  }));
}

let cy = null;
async function openExplain(row, variant) {
  const box = $("#explain"); box.hidden = false;
  $("#exMatches").innerHTML = '<div class="skel" style="aspect-ratio:auto;height:120px"></div>';
  box.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  try {
    const V = variant || curVar;
    const slot = lastSearch && lastSearch.variants.B && lastSearch.variants.B.variant.id === V ? lastSearch.variants.B : lastSearch?.variants.A;
    const r = await api("/api/explain", { row, concepts: slot?.concepts || [], variant: V });
    $("#h-explain").textContent = `Vì sao ảnh này được điểm như vậy? — ${r.variant.title}`;
    const c = r.card;
    const img = $("#exImg"); img.src = c.url; img.alt = `Ảnh #${c.idx}: ${c.labels.slice(0, 8).join(", ")}`;
    img.onerror = () => { if (c.coco_url && img.src !== c.coco_url) img.src = c.coco_url; };
    const hit = slot ? [...slot.final, ...slot.graph].find((x) => x.row === row) : null;
    $("#exMeta").innerHTML = `ảnh #${c.idx} · ${c.split} · ${r.counts.labels_total} nhãn (vẽ ${r.counts.labels_shown}) · s<sub>G</sub> = ${fmt(r.graph_score, 3)}` +
      (hit ? ` · CLIP #${hit.clip_rank} → cuối #${hit.final_rank}` : "");
    $("#exMatches").innerHTML = r.matches.length ? r.matches.map((m) => `
      <div class="match"><div class="mh"><span>“${esc(m.concept)}”</span><span class="mono">m = ${fmt(m.m, 3)}</span></div>
      <div class="why">${m.in_image ? "Có đúng nhãn này trong ảnh → khớp trực tiếp." :
        m.path ? `Không có trong ảnh; nối qua đồ thị: ${m.path.nodes.map((n) => esc(n.name)).join(" → ")}.` :
        "Không có trong ảnh; khớp gần nhất theo nhúng nhãn sau GNN."}</div>
      <table><thead><tr><th>nhãn của ảnh</th><th class="num">cos</th><th class="num">trọng số</th></tr></thead><tbody>
      ${m.top.map((t, i) => `<tr class="${i === 0 ? "hl" : ""}"><td>${esc(t.label)}</td><td class="num">${fmt(t.cos, 3)}</td><td class="num">${fmt(t.weight * 100, 1)}%</td></tr>`).join("")}
      </tbody></table></div>`).join("") : '<p class="small">Câu không có khái niệm nào trong bộ nhãn của biến thể này → điểm đồ thị = 0, chỉ còn CLIP + adapter.</p>';
    drawGraph("#cy", r.elements, "img:" + row);
    $("#exEdges").innerHTML = imageEdgeTable(r, row);
    $("#toBuilder").href = `#builder?row=${row}`;
  } catch (e) { $("#exMatches").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

/* Cạnh nối với ảnh khi truy vấn: mọi cạnh contains của ảnh + cạnh nhãn–nhãn giữa chúng, đánh dấu cạnh mà truy vấn dùng. */
function imageEdgeTable(r, row) {
  const el = r.elements;
  const name = Object.fromEntries(el.nodes.map((n) => [n.id, n.label]));
  const kindOf = Object.fromEntries(el.nodes.map((n) => [n.id, n.kind]));
  const usedBy = {};  // nhãn → các khái niệm khớp tốt nhất với nó
  r.matches.forEach((m) => { const t = m.top[0]; if (t) (usedBy[t.label] = usedBy[t.label] || []).push(m.concept); });
  const img = `img:${row}`;
  const cont = el.edges.filter((e) => e.kind === "contains" && e.target === img);
  const other = el.edges.filter((e) => e.kind !== "contains");
  const contRows = cont.map((e) => {
    const lab = name[e.source]; const u = usedBy[lab];
    return `<tr class="${u ? "hl" : ""}"><td>${esc(lab)}</td><td class="mono">contains</td><td>${esc(name[img])}</td>
      <td>${u ? `<span class="ok">✓ khớp “${u.map(esc).join("”, “")}”</span>` : '<span class="no">—</span>'}</td></tr>`;
  }).join("");
  const otherRows = other.map((e) => {
    const q = kindOf[e.source] === "query" || kindOf[e.target] === "query";
    const note = e.kind === "match" ? `cos ${fmt(e.cos, 3)}` : q || e.hot ? "đường khớp của truy vấn" : "";
    return `<tr class="${e.hot ? "hl" : ""}"><td>${esc(name[e.source])}</td><td class="mono">${esc(e.kind)}${e.rel ? " · " + esc(e.rel) : ""}</td>
      <td>${esc(name[e.target])}</td><td class="small">${esc(note)}</td></tr>`;
  }).join("");
  return `<p class="small">${cont.length} cạnh <b>contains</b> vào ảnh (mỗi nhãn của ảnh một cạnh) · ${other.length} cạnh khác trong hình
    (nhãn–nhãn giữa các nhãn của ảnh, và đường từ khái niệm của câu). Dòng tô xanh = cạnh truy vấn này dùng.</p>
    <div class="tbl-scroll"><table><thead><tr><th>từ</th><th>loại cạnh</th><th>tới</th><th>với truy vấn</th></tr></thead>
    <tbody>${contRows}${otherRows}</tbody></table></div>`;
}

function edgeTable(el) {
  const name = Object.fromEntries(el.nodes.map((n) => [n.id, n.label]));
  return `<div class="tbl-scroll"><table><thead><tr><th>từ</th><th>loại cạnh</th><th>tới</th></tr></thead><tbody>${
    el.edges.map((e) => `<tr class="${e.hot ? "hl" : ""}"><td>${esc(name[e.source])}</td><td class="mono">${esc(e.kind)}${e.rel ? " · " + esc(e.rel) : ""}</td><td>${esc(name[e.target])}</td></tr>`).join("")}</tbody></table></div>`;
}

function drawGraph(sel, el, center) {
  const C = { image: css("--n-image"), label: css("--n-label"), concept: css("--n-concept"), query: css("--n-query"), hop2: "#94a3b8" };
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const inst = cytoscape({
    container: $(sel),
    elements: [...el.nodes.map((n) => ({ data: n })), ...el.edges.map((e, i) => ({ data: { id: "e" + i, ...e } }))],
    wheelSensitivity: 0.25,
    style: [
      { selector: "node", style: { "background-color": (n) => C[n.data("kind")] || "#94a3b8", label: "data(label)", color: "#e2e8f0",
        "font-size": 10, "font-family": "Fira Sans", "text-valign": "bottom", "text-margin-y": 4, width: 14, height: 14,
        "text-outline-color": "#0b1120", "text-outline-width": 2 } },
      { selector: 'node[kind = "image"]', style: { shape: "round-rectangle", width: 34, height: 26, "font-size": 11, "font-weight": 600 } },
      { selector: 'node[kind = "query"]', style: { shape: "round-rectangle", width: 22, height: 22, "font-size": 12, "font-weight": 600, color: "#fde68a" } },
      { selector: 'node[kind = "concept"]', style: { shape: "diamond", width: 16, height: 16 } },
      { selector: "node[?hot]", style: { width: 20, height: 20, "border-width": 2, "border-color": "#f59e0b", "font-weight": 600 } },
      { selector: "edge", style: { width: 1, "line-color": "#334155", "curve-style": "bezier", opacity: 0.55 } },
      { selector: 'edge[kind = "contains"]', style: { width: 2, "line-color": "#60a5fa", opacity: 0.55, "target-arrow-shape": "triangle", "target-arrow-color": "#60a5fa", "arrow-scale": 0.7 } },
      { selector: 'node[kind = "hop2"]', style: { width: 10, height: 10, "font-size": 9, color: "#94a3b8" } },
      { selector: 'edge[kind = "sg_rel"], edge[kind = "concept"]', style: { "line-style": "dashed", "line-color": "#64748b", opacity: 0.6 } },
      { selector: "edge[?hot]", style: { width: 3, "line-color": "#f59e0b", opacity: 1, "target-arrow-shape": "triangle", "target-arrow-color": "#f59e0b" } },
    ],
    // ego-graph của một nhãn: vòng đồng tâm (tâm = nhãn đang xem); đồ thị con của ảnh: lực đẩy (cose)
    layout: sel === "#cyBuild"
      ? { name: "concentric", concentric: (n) => ({ image: 3, label: 2 }[n.data("kind")] || 1), levelWidth: () => 1,
          minNodeSpacing: 14, padding: 24, animate: false }
      : sel === "#cyLabel"
      ?{ name: "concentric", concentric: (n) => (n.id() === center ? 2 : 1), levelWidth: () => 1, minNodeSpacing: 18, padding: 24, animate: false }
      : { name: "cose", animate: !reduce, animationDuration: 400, nodeRepulsion: 12000, idealEdgeLength: 80, padding: 24, randomize: true },
  });
  inst.on("tap", "node", (e) => {
    const k = e.target.data("kind");
    if (k === "label" || k === "concept" || k === "query") { location.hash = "#graph"; setTimeout(() => showLabel(e.target.data("label")), 50); }
  });
  if (sel === "#cy") { if (cy) cy.destroy(); cy = inst; }
  return inst;
}

/* ---------------- DỮ LIỆU ---------------- */
async function initData() {
  try {
    const o = await api(`/api/overview?${vq()}`);
    $("#steps").innerHTML = o.steps.map((s) => `<li><h3>${esc(s.title)}</h3><span class="who">${esc(s.who)}</span>
      <dl>${s.items.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(typeof v === "number" ? v.toLocaleString("vi-VN") : v)}</dd></div>`).join("")}</dl></li>`).join("");
    $("#learned").innerHTML = `<span><b>w</b>${fmt(o.learned.w, 3)}</span><span><b>τ_match</b>${fmt(o.learned.tau_match, 3)}</span>
      <span><b>gate</b>${fmt(o.learned.gate, 3)}</span><span><b>α</b>${o.alpha}</span>
      <span><b>graph</b>${esc(JSON.stringify(o.config.graph))}</span><span><b>extra</b>${esc(o.config.extra || "graph_extra")}</span>`;
    hbar("#chLabels", o.label_degree_top.map((x) => x[0]), o.label_degree_top.map((x) => x[1]), "số ảnh", css("--n-label"));
    hbar("#chRels", o.rel_hist.map((x) => x[0]), o.rel_hist.map((x) => x[1]), "số cặp nhãn (hai chiều)", css("--c-clip"));
  } catch (e) { $("#steps").innerHTML = `<li class="err">${esc(e.message)}</li>`; }
}
const charts = {};
function hbar(sel, labels, data, label, color) {
  if (charts[sel]) charts[sel].destroy();
  return (charts[sel] = new Chart($(sel), { type: "bar", data: { labels, datasets: [{ label, data, backgroundColor: color, borderRadius: 4 }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: label } }, y: { ticks: { autoSkip: false, font: { size: 11 } } } } } }));
}

/* ---------------- GRAPH BUILDER ---------------- */
let cyBuild = null, buildRow = null, pickSeed = 0;
function initBuilder() {
  const inp = $("#pickLabel"); let t;
  inp.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(async () => {
      const xs = await api(`/api/labels?q=${encodeURIComponent(inp.value)}&${vq()}`).catch(() => []);
      $("#labelList2").innerHTML = xs.map((x) => `<option value="${esc(x)}">`).join("");
    }, 200);
  });
  $("#pickForm").addEventListener("submit", (e) => { e.preventDefault(); pickSeed++; loadPick(); });
  $("#hop2N").addEventListener("change", () => buildRow != null && loadBuild(buildRow));
  loadPick().then(builderFromHash);
}
function builderFromHash() {
  const row = new URLSearchParams(location.hash.split("?")[1] || "").get("row");
  if (row != null && +row !== buildRow) loadBuild(+row);
}
async function loadPick() {
  const strip = $("#pickStrip"); $("#pickErr").textContent = "";
  strip.innerHTML = Array(8).fill('<div class="skel"></div>').join("");
  try {
    const r = await api(`/api/images?split=${$("#pickSplit").value}&label=${encodeURIComponent($("#pickLabel").value.trim())}&n=16&seed=${pickSeed}&${vq()}`);
    strip.innerHTML = r.items.map((c) => `<button type="button" data-row="${c.row}" aria-pressed="${c.row === buildRow}" aria-label="Xem ảnh #${c.idx}">
      ${imgTag(c)}<span>#${c.idx}</span></button>`).join("") || '<p class="small">Không có ảnh nào khớp.</p>';
    $$("button", strip).forEach((b) => b.addEventListener("click", () => {
      history.replaceState(null, "", `#builder?row=${b.dataset.row}`); loadBuild(+b.dataset.row);
    }));
    if (buildRow == null && r.items.length && !location.hash.includes("row=")) loadBuild(r.items[0].row);
  } catch (e) { strip.innerHTML = ""; $("#pickErr").textContent = e.message; }
}
async function loadBuild(row) {
  buildRow = row;
  $$("#pickStrip button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.row === row)));
  try {
    const r = await api(`/api/build/${row}?two_hop=${$("#hop2N").value}&${vq()}`);
    $("#buildOut").hidden = false;
    const s = r.stats, c = r.card;
    const st = [["đối tượng gốc (lần xuất hiện)", s.objects_raw], ["tên đối tượng khác nhau", s.objects_unique],
      ["→ giữ thành nhãn", s.labels_kept], ["→ bỏ (ngoài bộ nhãn)", s.objects_dropped], ["quan hệ gốc", s.relations_raw],
      ["→ góp vào sg_rel", s.relations_used], ["cạnh contains vào ảnh", s.contains], ["cạnh nhãn–nhãn trong ảnh", s.label_label_edges],
      ["cạnh ConceptNet", s.concept_edges], ["nhãn láng giềng 2 bước", s.hop2_labels]];
    $("#buildStats").innerHTML = st.map(([k, v]) => `<div class="stat"><b>${v.toLocaleString("vi-VN")}</b><span>${esc(k)}</span></div>`).join("");
    const img = $("#bImg"); img.src = c.url; img.alt = `Ảnh #${c.idx}`;
    img.onerror = () => { if (c.coco_url && img.src !== c.coco_url) img.src = c.coco_url; };
    $("#bMeta").innerHTML = `ảnh #${c.idx} · row ${c.row} · tập <b>${esc(c.split)}</b>${r.is_train ? " (quan hệ của ảnh này được gộp vào sg_rel)" :
      " (ảnh không thuộc train: quan hệ của nó KHÔNG tạo cạnh — chống rò rỉ)"} · min_rel_count = ${r.min_rel_count}`;
    $("#bObjects").innerHTML = `<table><thead><tr><th>tên gốc</th><th>chuẩn hóa</th><th class="num">số lần</th><th>kết quả</th><th class="num">ảnh có nhãn</th></tr></thead><tbody>${
      r.objects.map((o) => `<tr><td>${esc(o.raw)}</td><td class="mono">${esc(o.norm)}</td><td class="num">${o.count}</td>
        <td>${o.kept ? '<span class="ok">✓ nhãn → cạnh contains</span>' : '<span class="no">✗ bỏ</span>'}</td><td class="num">${o.kept ? o.images_with_label : "—"}</td></tr>`).join("")}</tbody></table>`;
    $("#bContains").innerHTML = `<table><thead><tr><th>nguồn (nhãn)</th><th>cạnh</th><th>đích</th><th class="num">nhãn có trong # ảnh</th></tr></thead><tbody>${
      r.contains.map((x) => `<tr><td>${esc(x.label)}</td><td class="mono">contains</td><td>ảnh #${c.idx}</td><td class="num">${x.images_with_label}</td></tr>`).join("")}</tbody></table>`;
    $("#bRels").innerHTML = r.relations.length ? `<table><thead><tr><th>chủ thể</th><th>vị từ</th><th>đối tượng</th><th>kết quả</th></tr></thead><tbody>${
      r.relations.map((x) => `<tr class="${x.used ? "hl" : ""}"><td>${esc(x.s)}</td><td class="mono">${esc(x.p)}</td><td>${esc(x.o)}</td>
        <td class="small">${x.used ? '<span class="ok">✓</span> ' : ""}${esc(x.why)}${x.graph_rel ? ` · cạnh trong đồ thị mang nhãn “${esc(x.graph_rel)}”` : ""}</td></tr>`).join("")}</tbody></table>`
      : '<p class="small">Ảnh không có quan hệ nào.</p>';
    $("#bAmong").innerHTML = r.among.length ? `<p class="small">Cạnh nhãn–nhãn giữa các nhãn của ảnh này. Cạnh do <b>mọi ảnh train</b> tạo ra; quan hệ ghi trên cạnh là vị từ phổ biến nhất.</p>
      <table><thead><tr><th>nhãn</th><th>cạnh</th><th>quan hệ</th><th>nhãn</th></tr></thead><tbody>${
      r.among.map((x) => `<tr><td>${esc(x.a)}</td><td class="mono">${esc(x.type)}</td><td>${esc(x.rel)}</td><td>${esc(x.b)}</td></tr>`).join("")}</tbody></table>`
      : '<p class="small">Không có cạnh nào giữa các nhãn của ảnh.</p>';
    $("#bConcept").innerHTML = r.concept.length ? `<div class="chips">${r.concept.slice(0, 40).map((x) => `<span class="chip plain">${esc(x.label)} —${esc(x.rel)}→ ${esc(x.concept)}</span>`).join("")}</div>`
      : '<p class="small">Mô hình đang nạp chưa có ConceptNet (hoặc nhãn của ảnh không có cạnh ConceptNet).</p>';
    $("#bHop2").innerHTML = r.hop2.map((h) => `<button type="button" class="chip plain" data-n="${esc(h.name)}" title="nối tới ${h.links} nhãn của ảnh">${esc(h.name)} · ${h.links}</button>`).join("");
    $$("#bHop2 button").forEach((b) => b.addEventListener("click", () => { location.hash = "#graph"; setTimeout(() => showLabel(b.dataset.n), 50); }));
    if (cyBuild) cyBuild.destroy();
    cyBuild = drawGraph("#cyBuild", r.elements, `img:${row}`);
  } catch (e) { $("#pickErr").textContent = e.message; }
}

/* ---------------- ĐỒ THỊ ---------------- */
let cyLabel = null;
function initGraph() {
  const inp = $("#labelInput");
  let t;
  inp.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(async () => {
      const xs = await api(`/api/labels?q=${encodeURIComponent(inp.value)}&${vq()}`).catch(() => []);
      $("#labelList").innerHTML = xs.map((x) => `<option value="${esc(x)}">`).join("");
    }, 200);
  });
  $("#labelForm").addEventListener("submit", (e) => { e.preventDefault(); showLabel(inp.value); });
  if (!inp.value) showLabel("dog");
}
async function showLabel(name) {
  $("#labelInput").value = name; $("#labelErr").textContent = "";
  try {
    const r = await api(`/api/label/${encodeURIComponent(name.trim())}?${vq()}`);
    $("#labelOut").hidden = false;
    $("#labelInfo").innerHTML = `<h2 class="h2">${esc(r.name)} <span class="tag ${r.kind === "label" ? "direct" : "indirect"}">${r.kind === "label" ? `nhãn ảnh ${DATASET}` : "khái niệm (ConceptNet)"}</span></h2>
      <p class="small">Có trong ${r.images.toLocaleString("vi-VN")} ảnh · ${r.n_neighbors} láng giềng (hiện 60 nhiều ảnh nhất)</p>`;
    $("#labelNear").innerHTML = r.nearest_after_gnn.map(([n, s]) => `<button type="button" class="chip plain" data-n="${esc(n)}">${esc(n)} · ${fmt(s, 2)}</button>`).join("");
    $$("#labelNear button").forEach((b) => b.addEventListener("click", () => showLabel(b.dataset.n)));
    $("#labelNbrs").innerHTML = `<table><thead><tr><th>láng giềng</th><th>cạnh</th><th>quan hệ</th><th class="num">số ảnh</th></tr></thead><tbody>${
      r.neighbors.map((n) => `<tr><td><a href="#graph" data-n="${esc(n.name)}">${esc(n.name)}</a></td><td class="mono">${esc(n.type)}</td><td>${esc(n.rel)}</td><td class="num">${n.images}</td></tr>`).join("")}</tbody></table>`;
    $$("#labelNbrs a").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); showLabel(a.dataset.n); }));
    const nodes = [{ id: "c", label: r.name, kind: "query" }, ...r.neighbors.slice(0, 40).map((n, i) => ({ id: "n" + i, label: n.name, kind: n.type === "concept" ? "concept" : "label" }))];
    const edges = r.neighbors.slice(0, 40).map((n, i) => ({ source: "c", target: "n" + i, kind: n.type }));
    if (cyLabel) cyLabel.destroy();
    cyLabel = drawGraph("#cyLabel", { nodes, edges }, "c");
  } catch (e) { $("#labelErr").textContent = e.message; }
}

/* ---------------- ABLATION ---------------- */
async function initAblation() {
  try {
    const { story } = await api("/api/ablation");
    const base = story.find((s) => s.exp === "vg_qa_ablation_no_edge" && s.section === "main");
    const bm = base?.full?.MRR;
    const SECT = { main: "Bảng chính (3 seed, Kaggle)", gnn: "So sánh GNN (3 seed)", history: "Lịch sử thí nghiệm (1 seed, máy) — vì sao chuyển sang QA-Graph" };
    const pm = (v, sd) => `${fmt(v)}${sd ? `<small class="no"> ±${fmt(sd)}</small>` : ""}`;
    const rowHtml = (s, i) => {
      const d = bm != null && s.full.MRR != null ? s.full.MRR - bm : null;
      const cls = d == null ? "" : d > 0.3 ? "up" : d < -0.3 ? "down" : "same";
      return `<tr class="${s === base ? "hl" : ""}"><td class="num">${i + 1}</td><td><b>${esc(s.title)}</b><div class="small mono">${esc(s.exp)}${s.tag ? " · " + esc(s.tag) : ""}</div></td>
        <td class="num">${pm(s.full["R@1"], s.full_std?.["R@1"])}</td><td class="num">${fmt(s.full["R@10"])}</td><td class="num"><b>${pm(s.full.MRR, s.full_std?.MRR)}</b></td>
        <td class="num ${cls}">${d == null || s === base ? "—" : (d > 0 ? "+" : "") + fmt(d)}</td>
        <td class="num">${fmt(s.by_group?.direct)}</td><td class="num">${fmt(s.by_group?.indirect)}</td><td class="num">${s.seeds || "—"}</td>
        <td class="small">${esc(s.note)}</td></tr>`;
    };
    let i = 0;
    $("#abTable").innerHTML = ["main", "gnn", "history"].map((sec) => {
      const xs = story.filter((s) => s.section === sec);
      if (!xs.length) return "";
      return `<h2 class="h2" style="margin-top:8px">${SECT[sec]}</h2><table><thead><tr><th>#</th><th>Mô hình</th><th class="num">R@1</th><th class="num">R@10</th><th class="num">MRR</th>
        <th class="num">Δ MRR vs Adapter</th><th class="num">direct</th><th class="num">indirect</th><th class="num">seed</th><th>Ý nghĩa</th></tr></thead>
        <tbody>${xs.map((s) => rowHtml(s, i++)).join("")}</tbody></table>`;
    }).join("") + `<p class="help">MRR/R@k: toàn pool test (${$("#pool").options[0]?.text.match(/\d+/)?.[0] || "?"} ảnh), mean ± std theo seed. direct/indirect: MRR gallery 1000. Δ so với CLIP + Adapter (3 seed).
      Bảng đầy đủ kèm khoảng tin cậy bootstrap: <code>results/tables/${DATASET === "PSG" ? "psg_official.md" : "vg_official.md"}</code>.</p>`;
    story.sort((a, b) => ["main", "gnn", "history"].indexOf(a.section) - ["main", "gnn", "history"].indexOf(b.section));
    const short = story.map((s, i) => `${i + 1}. ${s.title.replace(/\s*\(#\d+\)/, "")}`);
    const colorOf = (s) => (s.exp === "vg_clip" ? css("--c-clip") : s.exp.startsWith("vg_qa") ? css("--c-final") : css("--c-graph"));
    new Chart($("#chMRR"), { type: "bar",
      data: { labels: short, datasets: [{ label: "MRR toàn pool", data: story.map((s) => s.full.MRR), backgroundColor: story.map(colorOf), borderRadius: 4 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { min: 58, title: { display: true, text: "MRR (%)" } } } } });
    new Chart($("#chGroup"), { type: "bar",
      data: { labels: short, datasets: [
        { label: "direct", data: story.map((s) => s.by_group?.direct), backgroundColor: css("--n-label"), borderRadius: 3 },
        { label: "indirect", data: story.map((s) => s.by_group?.indirect), backgroundColor: css("--n-concept"), borderRadius: 3 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, scales: { x: { min: 65, title: { display: true, text: "MRR (%)" } } } } });
    const trained = story.filter((s) => s.history && s.history.length);
    const pal = ["#60a5fa", "#c084fc", "#22c55e", "#f59e0b", "#f472b6", "#2dd4bf", "#a3e635", "#fb7185", "#38bdf8", "#e879f9", "#facc15"];
    new Chart($("#chCurve"), { type: "line",
      data: { labels: [...Array(Math.max(...trained.map((s) => s.history.length))).keys()],
        datasets: trained.map((s, i) => ({ label: s.title.replace(/\s*\(#\d+\)/, ""), data: s.history.map((h) => (h.val_MRR ?? h.val_MRR_graph) * 100),
          borderColor: pal[i % pal.length], backgroundColor: pal[i % pal.length], tension: 0.25, pointRadius: 3, hidden: !s.exp.startsWith("vg_qa") })) },
      options: { maintainAspectRatio: false, interaction: { mode: "nearest", intersect: false },
        scales: { x: { title: { display: true, text: "epoch" } }, y: { title: { display: true, text: "MRR val (%)" } } },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 11 } } } } } });
    const withCurve = story.filter((s) => s.alpha_curve && s.alpha_curve.length);
    $("#alphaRun").innerHTML = withCurve.map((s, i) => `<option value="${i}">${esc(s.title)}</option>`).join("");
    let ch;
    const drawAlpha = () => {
      const s = withCurve[+$("#alphaRun").value]; if (!s) return;
      const pts = s.alpha_curve.map((p) => Array.isArray(p) ? { x: p[0], y: p[1] } : { x: p.alpha, y: p.MRR ?? p.score });
      if (ch) ch.destroy();
      ch = new Chart($("#chAlpha"), { type: "line", data: { datasets: [{ label: "MRR val", data: pts, borderColor: css("--c-final"), pointRadius: 2, tension: 0.2 }] },
        options: { maintainAspectRatio: false, parsing: false, plugins: { legend: { display: false } },
          scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "α (1 = chỉ CLIP)" } }, y: { title: { display: true, text: "MRR val (%)" } } } } });
    };
    $("#alphaRun").addEventListener("change", drawAlpha);
    if (withCurve.length) { $("#alphaRun").value = String(Math.max(0, withCurve.findIndex((s) => s.title.startsWith("QA: Adapter")))); drawAlpha(); }
  } catch (e) { $("#abTable").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
