/* Trang tổng hợp Nhiệm vụ 1 — vẽ từ window.T1DATA (data.js), không phụ thuộc thư viện ngoài. */
(function () {
  'use strict';
  const D = window.T1DATA;
  if (!D) { document.getElementById('lead').textContent = 'Không tìm thấy data.js — chạy build_site.py trước.'; return; }
  const METRICS = ['R@1', 'R@5', 'R@10', 'MRR', 'nDCG@10'];
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const $ = (id) => document.getElementById(id);
  const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : '—');
  const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : '—');
  const fi = (x) => (Number.isFinite(x) ? x.toLocaleString('vi-VN') : '—');
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const where = (rows, cond) => rows.filter((r) => Object.entries(cond).every(([k, v]) => String(r[k]) === String(v)));
  const one = (rows, cond) => where(rows, cond)[0];
  const el = (tag, attrs, html) => { const e = document.createElement(tag); if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (html != null) e.innerHTML = html; return e; };
  const svgEl = (tag, attrs, text) => { const e = document.createElementNS(SVG_NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; };

  // ---------- tooltip
  const tip = $('tip');
  const showTip = (ev, html) => {
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY - h - 12;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y < 8) y = ev.clientY + 16;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const hideTip = () => { tip.hidden = true; };

  // ---------- legends
  const SERIES = {
    protocol: [['seed0', 'fold seed 0', 's0'], ['seed1', 'fold seed 1', 's1'], ['seed2', 'fold seed 2', 's2'], ['full', 'toàn pool', 's3']],
    group: [['direct', 'direct', 's0'], ['indirect', 'indirect', 's1'], ['no_concept', 'no_concept', 's2']],
    alpha: [['MRR', 'MRR', 's0'], ['R@1', 'R@1', 's1'], ['R@10', 'R@10', 's2']],
    compare: [['clip', 'CLIP thuần', 's0'], ['clip_lex', 'CLIP + khớp tên (α*)', 's1']],
  };
  const COLOR_VAR = { s0: '--s0', s1: '--s1', s2: '--s2', s3: '--s3', s4: '--s4' };
  const cssColor = (cls) => getComputedStyle(document.documentElement).getPropertyValue(COLOR_VAR[cls]).trim();
  document.querySelectorAll('[data-legend]').forEach((lg) => {
    for (const [, label, cls] of SERIES[lg.dataset.legend]) { const s = el('span'); s.style.setProperty('--sw', `var(${COLOR_VAR[cls]})`); s.textContent = label; lg.appendChild(s); }
  });

  // ---------- grouped bar chart
  function groupedBars(container, { groups, series, value, yMax = 100, height = 240, fmt = f1 }) {
    container.replaceChildren();
    const W = Math.max(300, container.clientWidth || 600), H = height, m = { t: 14, r: 8, b: 30, l: 36 };
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img' });
    const innerW = W - m.l - m.r, innerH = H - m.t - m.b;
    const y = (v) => m.t + innerH * (1 - v / yMax);
    const grid = svgEl('g', { class: 'grid' });
    const step = yMax <= 30 ? 5 : yMax <= 70 ? 10 : 20;
    for (let v = 0; v <= yMax; v += step) {
      grid.appendChild(svgEl('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      grid.appendChild(svgEl('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, String(v)));
    }
    svg.appendChild(grid);
    svg.appendChild(svgEl('line', { class: 'axis', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }));
    const gw = innerW / groups.length, pad = gw * 0.18, bw = Math.min(32, (gw - 2 * pad) / series.length * 0.88), gap = ((gw - 2 * pad) - bw * series.length) / Math.max(1, series.length - 1);
    groups.forEach((g, gi) => {
      const x0 = m.l + gi * gw + pad;
      svg.appendChild(svgEl('text', { x: m.l + gi * gw + gw / 2, y: H - m.b + 16, 'text-anchor': 'middle' }, g));
      series.forEach(([key, label, cls], si) => {
        const v = value(g, key);
        if (!Number.isFinite(v)) return;
        const x = x0 + si * (bw + gap);
        const r = svgEl('rect', { class: 'bar ' + cls, x, y: y(v), width: bw, height: y(0) - y(v) });
        r.addEventListener('mousemove', (ev) => showTip(ev, `<span class="k">${esc(g)}</span><br><span class="sw" style="background:${cssColor(cls)}"></span>${esc(label)}: <b>${fmt(v)}</b>`));
        r.addEventListener('mouseleave', hideTip);
        svg.appendChild(r);
        if (bw >= 20) svg.appendChild(svgEl('text', { class: 'lbl', x: x + bw / 2, y: y(v) - 3, 'text-anchor': 'middle' }, fmt(v)));
      });
    });
    container.appendChild(svg);
  }

  // ---------- line chart (alpha curve)
  function alphaChart(container) {
    container.replaceChildren();
    const pts = D.alpha_curve.slice().sort((a, b) => a.alpha - b.alpha);
    const keys = SERIES.alpha;
    const W = Math.max(300, container.clientWidth || 600), H = 240, m = { t: 16, r: 12, b: 32, l: 36 };
    const vals = pts.flatMap((p) => keys.map(([k]) => p[k]));
    const yMin = Math.floor(Math.min(...vals) / 10) * 10, yMax = Math.ceil(Math.max(...vals) / 10) * 10;
    const x = (a) => m.l + (W - m.l - m.r) * a, y = (v) => m.t + (H - m.t - m.b) * (1 - (v - yMin) / (yMax - yMin));
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img' });
    const grid = svgEl('g', { class: 'grid' });
    for (let v = yMin; v <= yMax; v += 10) { grid.appendChild(svgEl('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) })); grid.appendChild(svgEl('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, String(v))); }
    svg.appendChild(grid);
    svg.appendChild(svgEl('line', { class: 'axis', x1: m.l, x2: W - m.r, y1: y(yMin), y2: y(yMin) }));
    for (let a = 0; a <= 1.001; a += 0.25) svg.appendChild(svgEl('text', { x: x(a), y: H - m.b + 16, 'text-anchor': 'middle' }, a.toFixed(2)));
    svg.appendChild(svgEl('text', { x: W - m.r, y: H - 2, 'text-anchor': 'end' }, 'α (1 = CLIP thuần)'));
    const aStar = D.meta.alpha;
    svg.appendChild(svgEl('line', { class: 'ref', x1: x(aStar), x2: x(aStar), y1: m.t, y2: y(yMin) }));
    svg.appendChild(svgEl('text', { x: x(aStar) - 5, y: m.t + 10, 'text-anchor': 'end' }, 'α* = ' + f2(aStar)));
    keys.forEach(([k, , cls]) => {
      svg.appendChild(svgEl('path', { class: 'line ' + cls, d: pts.map((p, i) => (i ? 'L' : 'M') + x(p.alpha).toFixed(1) + ' ' + y(p[k]).toFixed(1)).join(' ') }));
      const last = pts[pts.length - 1];
      svg.appendChild(svgEl('text', { x: x(last.alpha) - 4, y: y(last[k]) - 6, 'text-anchor': 'end' }, `${k} ${f1(last[k])}`));
    });
    const hover = svgEl('g', { visibility: 'hidden' });
    const rule = svgEl('line', { class: 'axis', y1: m.t, y2: y(yMin) }); hover.appendChild(rule);
    const dots = keys.map(([, , cls]) => { const c = svgEl('circle', { class: cls, r: 4 }); hover.appendChild(c); return c; });
    svg.appendChild(hover);
    const hit = svgEl('rect', { class: 'hit', x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b });
    hit.addEventListener('mousemove', (ev) => {
      const box = svg.getBoundingClientRect(); const a = Math.min(1, Math.max(0, ((ev.clientX - box.left) / box.width * W - m.l) / (W - m.l - m.r)));
      let p = pts[0]; for (const q of pts) if (Math.abs(q.alpha - a) < Math.abs(p.alpha - a)) p = q;
      hover.setAttribute('visibility', 'visible'); rule.setAttribute('x1', x(p.alpha)); rule.setAttribute('x2', x(p.alpha));
      dots.forEach((d, i) => { d.setAttribute('cx', x(p.alpha)); d.setAttribute('cy', y(p[keys[i][0]])); });
      showTip(ev, `<span class="k">α = ${f2(p.alpha)}</span><br>` + keys.map(([k, , cls]) => `<span class="sw" style="background:${cssColor(cls)}"></span>${k}: <b>${f2(p[k])}</b>`).join('<br>'));
    });
    hit.addEventListener('mouseleave', () => { hover.setAttribute('visibility', 'hidden'); hideTip(); });
    svg.appendChild(hit);
    container.appendChild(svg);
  }

  // ---------- header + tiles + facts
  const T = { method: 'clip', direction: 't2i', split: 'test' };
  const mean = one(D.metrics, { ...T, protocol: 'mean3' }), std = one(D.metrics, { ...T, protocol: 'std3' }), full = one(D.metrics, { ...T, protocol: 'full' });
  $('lead').innerHTML = `CLIP ${esc(D.meta.model)} (${esc(D.meta.pretrained.split('_')[0])}) đóng băng, truy vấn caption → ảnh đạt R@1 = <b>${f1(mean['R@1'])} %</b> trên gallery ${fi(D.meta.gallery_size)} ảnh (trung bình ${D.meta.seeds.length} fold) và <b>${f1(full['R@1'])} %</b> trên toàn bộ ${fi(full.gallery)} ảnh test. Đây là mốc mà CLIP + Graph (Huy) và phương pháp mới (Khánh) phải vượt trên cùng split, cùng top-${D.meta.top_k}.`;
  const tiles = $('tiles');
  for (const k of METRICS) {
    const t = el('div', { class: 'tile' });
    t.appendChild(el('span', { class: 'lbl' }, k === 'R@1' ? 'Recall@1' : k === 'R@5' ? 'Recall@5' : k === 'R@10' ? 'Recall@10' : k));
    t.appendChild(el('span', { class: 'big' }, f2(mean[k])));
    t.appendChild(el('span', { class: 'sub' }, `± ${f2(std[k])} · test 1K, ${D.meta.seeds.length} fold · toàn pool ${f2(full[k])}`));
    tiles.appendChild(t);
  }
  const facts = [[fi(D.meta.split.train), `ảnh train (lấy mẫu từ pool ${fi(D.meta.pool.train)})`], [fi(D.meta.split.val), 'ảnh val (toàn pool)'], [fi(D.meta.split.test), 'ảnh test (toàn pool)'],
    [fi(D.meta.n_captions), 'caption COCO (5 / ảnh)'], [`${fi(D.meta.n_images_ok)} / ${fi(D.meta.n_images)}`, 'ảnh mã hoá thành công'], [fi(D.meta.dim), 'chiều đặc trưng CLIP'],
    [fi(D.meta.gallery_size), 'ảnh mỗi fold gallery'], [fi(D.meta.top_k), 'ứng viên top-K dùng chung']];
  for (const [v, k] of facts) { const d = el('div'); d.appendChild(el('span', { class: 'v' }, esc(v))); d.appendChild(el('span', { class: 'k' }, esc(k))); $('facts').appendChild(d); }
  $('protocol-notes').innerHTML = [
    `Split: ${esc(D.meta.source)}; seed split = ${D.meta.seed_data}; tạo lúc ${esc(D.meta.created)} (giờ máy Kaggle, UTC).`,
    `CLIP đóng băng là tất định, nên <b>seed ${D.meta.seeds.join('/')}</b> định nghĩa ${D.meta.seeds.length} fold gallery ${fi(D.meta.gallery_size)} ảnh lấy ngẫu nhiên từ pool test (giao thức test-1K của COCO); <b>toàn pool</b> = mọi ảnh test, khó hơn vì nhiều ảnh gây nhiễu hơn.`,
    'Chống rò rỉ: từ vựng đối tượng và nhóm truy vấn chỉ tính trên train; α của mọi phương pháp chỉ chọn trên val; caption val/test không tạo bất kỳ thống kê nào.',
  ].map((s) => `<li>${s}</li>`).join('');

  // ---------- charts
  const protoValue = (direction) => (g, key) => { const r = one(D.metrics, { method: 'clip', direction, split: 'test', protocol: key }); return r ? r[g] : NaN; };
  const draw = () => {
    groupedBars($('chart-t2i'), { groups: METRICS, series: SERIES.protocol, value: protoValue('t2i') });
    groupedBars($('chart-i2t'), { groups: METRICS, series: SERIES.protocol, value: protoValue('i2t') });
    const bins = D.rank_dist.full.map((b) => b.bin);
    groupedBars($('chart-rank'), { groups: bins, series: SERIES.protocol, yMax: 70, value: (g, key) => { const b = (D.rank_dist[key] || []).find((x) => x.bin === g); return b ? b.pct : NaN; } });
    groupedBars($('chart-group'), { groups: ['R@1', 'R@5', 'R@10', 'MRR'], series: SERIES.group, value: (g, key) => { const r = one(D.by_group, { method: 'clip', split: 'test', group: key }); return r ? r[g] : NaN; } });
    alphaChart($('chart-alpha'));
    groupedBars($('chart-compare'), { groups: ['direct R@1', 'indirect R@1', 'direct MRR', 'indirect MRR'], series: SERIES.compare,
      value: (g, key) => { const [grp, met] = g.split(' '); const r = one(D.by_group, { method: key, split: 'test', group: grp }); return r ? r[met] : NaN; } });
  };
  draw();
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 120); });

  // ---------- notes with numbers
  const b = D.meta.bootstrap;
  const pf = (p) => (p < 0.001 ? '< 0.001' : p.toFixed(3));
  $('alpha-note').innerHTML = `score = α·z(cos) + (1 − α)·z(s<sub>G</sub>), với s<sub>G</sub> = tỉ lệ khái niệm của truy vấn có trong scene graph của ảnh (chỉ khớp tên, không dùng quan hệ; <b>không phải</b> mô hình đồ thị của nhóm, chỉ minh hoạ cách cắm vào module). α* chọn theo MRR trên val = <b>${f2(D.meta.alpha)}</b>.`;
  $('boot-note').innerHTML = `Cùng fold, cùng top-${D.meta.top_k}. Bootstrap ghép cặp trên toàn pool test: ΔMRR = <b>${b.MRR.delta >= 0 ? '+' : ''}${f2(b.MRR.delta)}</b> (CI95 ${f2(b.MRR.ci95[0])} … ${f2(b.MRR.ci95[1])}, p ${pf(b.MRR.p_value)}); ΔR@1 = <b>${b['R@1'].delta >= 0 ? '+' : ''}${f2(b['R@1'].delta)}</b> (p ${pf(b['R@1'].p_value)}); ΔR@10 = <b>${b['R@10'].delta >= 0 ? '+' : ''}${f2(b['R@10'].delta)}</b>. Khớp tên đơn giản đã giúp, nhất là nhóm direct; GNN cần vượt mức này.`;

  // ---------- metrics table with filters
  const LABEL_P = { seed0: 'fold seed 0', seed1: 'fold seed 1', seed2: 'fold seed 2', mean3: 'mean 3 fold', std3: 'std 3 fold', full: 'toàn pool' };
  const ORDER_P = ['seed0', 'seed1', 'seed2', 'mean3', 'std3', 'full'];
  function renderTable() {
    const dir = $('sel-dir').value, split = $('sel-split').value;
    const rows = where(D.metrics, { method: 'clip', direction: dir, split }).sort((a, c) => ORDER_P.indexOf(a.protocol) - ORDER_P.indexOf(c.protocol));
    const t = $('tbl-metrics');
    t.innerHTML = '<thead><tr><th>giao thức</th><th class="num">gallery</th><th class="num">n truy vấn</th>' + METRICS.map((m) => `<th class="num">${m}</th>`).join('') + '<th class="num">MedR</th></tr></thead>';
    const tb = el('tbody');
    for (const r of rows) {
      const tr = el('tr', { class: r.protocol === 'mean3' || r.protocol === 'full' ? 'hl' : '' });
      tr.innerHTML = `<td>${esc(LABEL_P[r.protocol])}</td><td class="num">${fi(r.gallery)}</td><td class="num">${fi(r.n)}</td>` + METRICS.map((m) => `<td class="num">${f2(r[m])}</td>`).join('') + `<td class="num">${r.MedR == null ? '' : f1(r.MedR)}</td>`;
      tb.appendChild(tr);
    }
    t.appendChild(tb);
  }
  $('sel-dir').addEventListener('change', renderTable); $('sel-split').addEventListener('change', renderTable); renderTable();

  // ---------- examples
  const exBox = $('example-cards');
  for (const r of D.examples) {
    const ok = Number(r.rank) === 1;
    const c = el('div', { class: 'ex' });
    const fig = (url, title, objs) => `<figure>${url ? `<img src="${esc(url)}" alt="${esc(title)}" loading="lazy">` : ''}<figcaption><b>${esc(title)}</b><br>${esc(objs)}</figcaption></figure>`;
    c.innerHTML = `<div class="head"><span class="pill ${ok ? 'ok' : 'bad'}">${esc(r.kind)}</span><span>hạng ảnh đúng: <b>${fi(Number(r.rank))}</b></span><span>nhóm: ${esc(r.group)}</span></div>
      <div class="q">“${esc(r.query)}”</div>
      <div class="note">khái niệm liên kết: ${esc(r.query_concepts || '(không)')}</div>
      <div class="pair">${fig(r.gt_url, 'ảnh đúng ' + r.gt_image, r.gt_objects)}${fig(r.top1_url, 'top-1 của CLIP ' + r.top1_image, r.top1_objects)}</div>`;
    exBox.appendChild(c);
  }

  $('foot').textContent = `Nguồn: notebook task1_clip_baseline_kaggle.ipynb chạy trên Kaggle (T4), split tạo lúc ${D.meta.created} UTC · model ${D.meta.model} / ${D.meta.pretrained} · data.js sinh bởi build_site.py từ kaggle_results/.`;
})();
