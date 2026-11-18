"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const BUCKETS = [["study", "Study closely"], ["read", "Read"], ["skim", "Skim"], ["skip", "Skip on first read"]];
const PRIORITY = { claim: 0, complex: 1, key: 2, definition: 3 };
const VIEWS = ["home", "loading", "error", "reader"];

let DOC = null;
let pinned = null;
let run = 0;          // bumps on every navigation so stale polls stop
let lastInput = null; // for "Try again"
let CONFIG = null;

/* ---------- views and navigation ---------- */

function show(view) {
  for (const v of VIEWS) $("#" + (v === "reader" ? "readerView" : v)).classList.toggle("hidden", v !== view);
  document.body.className = document.body.className.replace(/\bview-\w+/g, "").trim() + " view-" + view;
  if (view !== "reader") { closePanel(); hideTip(); }
  window.scrollTo(0, 0);
}

function goHome(push = true) {
  run++;
  DOC = null;
  pinned = null;
  setLive(null);
  $("#reader").innerHTML = "";
  $("#docTitle").textContent = "";
  $("#formError").textContent = "";
  document.title = "Kami";
  if (push && location.search) history.pushState(null, "", "/");
  show("home");
  loadRecent();
  $("#urlInput").focus({ preventScroll: true });
}

function fail(msg) {
  $("#errorMsg").textContent = msg;
  $("#retryBtn").classList.toggle("hidden", !lastInput);
  show("error");
}

function describe(input) {
  if (input.file) return input.file.name;
  if (input.sample) return "Attention Is All You Need";
  return input.url;
}

async function submit(input) {
  lastInput = input;
  const my = ++run;
  $("#loadingSource").textContent = describe(input);
  $("#docTitle").textContent = "";
  setStage(input.url ? "Fetching the link" : "Uploading", 0.01);
  show("loading");
  let res;
  try {
    if (input.sample) res = await fetch("/api/sample/" + encodeURIComponent(input.sample), { method: "POST" });
    else {
      const fd = new FormData();
      if (input.file) fd.append("file", input.file); else fd.append("url", input.url);
      res = await fetch("/api/analyze", { method: "POST", body: fd });
    }
  } catch (e) { if (my === run) fail("Could not reach the server. Is it still running?"); return; }
  const body = await res.json().catch(() => ({}));
  if (my !== run) return;
  if (!res.ok) return fail(typeof body.detail === "string" ? body.detail : "Something went wrong.");
  history.pushState(null, "", "?doc=" + body.doc_id);
  poll(body.doc_id, my);
}

function setStage(stage, p) { $("#stage").textContent = stage; $("#barFill").style.width = Math.max(2, Math.round(100 * p)) + "%"; }

function setLive(s) {
  const live = $("#live");
  if (!live) return;
  if (!s || s.status === "done") {
    live.classList.add("hidden");
    return;
  }
  live.classList.remove("hidden");
  $("#liveStage").textContent = s.stage || "Analyzing";
  $("#liveFill").style.width = Math.max(2, Math.round(100 * (s.progress || 0))) + "%";
}

function docSig(doc) {
  return `${(doc.highlights || []).length}:${(doc.claims || []).length}:${((doc.plan || {}).order || []).length}:${doc.partial ? 1 : 0}`;
}

async function poll(id, my = ++run) {
  let opened = false;
  let lastSig = "";
  for (;;) {
    let r, s;
    try { r = await fetch("/api/status/" + id); s = await r.json().catch(() => ({})); }
    catch (e) { if (my === run) fail("Lost the connection to the server."); return; }
    if (my !== run) return;
    if (!r.ok) return fail(r.status === 404 ? "This document is not on the server any more. Start again from the home page." : (s.detail || "Unknown document."));
    if (s.status === "error") { setLive(null); return fail(s.error || "Analysis failed."); }
    setStage(s.stage || "Working", s.progress || 0);
    if (s.ready || s.status === "done") {
      const dr = await fetch("/api/doc/" + id);
      if (my !== run) return;
      if (dr.ok) {
        const doc = await dr.json();
        const next = docSig(doc);
        if (!opened) { render(doc); opened = true; }
        else if (next !== lastSig) updateDoc(doc);
        lastSig = next;
        setLive(s.status === "done" ? null : s);
      }
    }
    if (s.status === "done") { setLive(null); break; }
    await new Promise((ok) => setTimeout(ok, 500));
    if (my !== run) return;
  }
  if (!opened) {
    const r = await fetch("/api/doc/" + id);
    if (my !== run) return;
    if (!r.ok) return fail("Could not load the result.");
    render(await r.json());
  }
}

/* ---------- rendering ---------- */

function pct(v, total) { return (100 * v / total).toFixed(4) + "%"; }

function diffDots(n) { return `<span class="diff" title="Difficulty ${n} of 5">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= n ? "f" : ""}"></i>`).join("")}</span>`; }

function badgeHtml(s) {
  const d = s.analysis ? `<span class="d">${s.analysis.difficulty}/5</span>` : "";
  return `<span class="badge ${s.bucket}" data-s="${s.id}">${esc(s.bucket)}${d}</span>`;
}

function render(doc) {
  DOC = doc;
  pinned = null;
  document.title = (doc.title || "Document") + " | Kami";
  $("#docTitle").textContent = doc.title || "";
  const reader = $("#reader");
  reader.innerHTML = "";
  const isArticle = doc.kind === "html";
  $("#readerView").classList.toggle("article-mode", isArticle);
  const nLabels = isArticle ? renderArticle(doc, reader) : renderPages(doc, reader);
  paintMeta(doc, nLabels);
  renderPanel(doc);
  show("reader");
}

function updateDoc(doc) {
  DOC = doc;
  const y = window.scrollY;
  let nLabels;
  if (doc.kind === "html") {
    $("#reader").innerHTML = "";
    nLabels = renderArticle(doc, $("#reader"));
  } else {
    nLabels = paintPdfOverlays(doc);
  }
  paintMeta(doc, nLabels);
  renderPanel(doc);
  window.scrollTo(0, y);
}

function paintMeta(doc, nLabels) {
  const counts = {};
  for (const h of doc.highlights || []) counts[h.type] = (counts[h.type] || 0) + 1;
  for (const c of $$(".chip")) $(".n", c).textContent = c.dataset.type === "badge" ? nLabels : counts[c.dataset.type] || 0;
  $("#analyzerTag").textContent = doc.mock ? "Demo analyzer, no API key" : "";
  const banner = $("#banner");
  if (doc.mock) {
    banner.textContent = "Offline demo analyzer. Highlights are picked by simple heuristics and the notes are placeholders. Add an API key in .env for real analysis.";
    banner.classList.remove("hidden");
  } else banner.classList.add("hidden");
}

function renderPages(doc, reader) {
  for (const p of doc.pages) {
    const el = document.createElement("div");
    el.className = "page";
    el.id = "page-" + p.index;
    el.style.aspectRatio = `${p.width} / ${p.height}`;
    el.innerHTML = `<img loading="${p.index < 2 ? "eager" : "lazy"}" alt="Page ${p.index + 1}" src="${doc.page_image.replace("{n}", p.index)}"><span class="pageno">${p.index + 1}</span>`;
    reader.appendChild(el);
  }
  return paintPdfOverlays(doc);
}

function paintPdfOverlays(doc) {
  const pages = $$(".page");
  for (const el of pages) {
    for (const n of [...el.children]) {
      if (n.tagName === "IMG" || n.classList.contains("pageno")) continue;
      n.remove();
    }
  }
  let n = 0;
  for (const s of doc.sections) {
    const r = s.heading_rect;
    if (!r || !s.bucket) continue;
    const P = doc.pages[r.page], el = pages[r.page];
    if (!P || !el) continue;
    const bar = document.createElement("div");
    bar.className = "hbar " + s.bucket;
    bar.id = "sec-" + s.id;
    Object.assign(bar.style, { left: pct(r.x0 - 8, P.width), top: pct(r.y0 - 1, P.height), width: "0", height: pct(r.y1 - r.y0 + 2, P.height) });
    el.appendChild(bar);
    const wrap = document.createElement("div");
    wrap.innerHTML = badgeHtml(s);
    const b = wrap.firstChild;
    Object.assign(b.style, { left: pct(r.x1 + 10, P.width), top: pct((r.y0 + r.y1) / 2, P.height) });
    el.appendChild(b);
    n++;
  }
  for (const h of doc.highlights || []) {
    for (const r of h.rects || []) {
      const P = doc.pages[r.page];
      if (!P || !pages[r.page]) continue;
      const el = document.createElement("div");
      el.className = "hl " + h.type;
      el.dataset.h = h.id;
      Object.assign(el.style, { left: pct(r.x0, P.width), top: pct(r.y0, P.height), width: pct(r.x1 - r.x0, P.width), height: pct(r.y1 - r.y0, P.height) });
      pages[r.page].appendChild(el);
    }
  }
  return n;
}

function renderArticle(doc, reader) {
  const tok = doc.tokens;
  const owner = new Array(tok.length).fill(null);
  for (const h of [...doc.highlights].sort((a, b) => PRIORITY[a.type] - PRIORITY[b.type])) {
    for (let i = h.token_range[0]; i < h.token_range[1] && i < tok.length; i++) if (!owner[i]) owner[i] = h;
  }
  const secAt = {};
  for (const s of doc.sections) if (s.token_range) secAt[s.token_range[0]] = s;
  const refStart = Math.min(...doc.sections.filter((s) => s.kind === "references" || s.kind === "appendix").map((s) => s.token_range[0]), Infinity);

  const text = (a, b) => {
    let out = "", i = a;
    while (i < b) {
      const h = owner[i];
      let j = i + 1;
      while (j < b && owner[j] === h) j++;
      const words = esc(tok.slice(i, j).join(" "));
      out += (out ? " " : "") + (h ? `<mark class="hl ${h.type}" data-h="${h.id}">${words}</mark>` : words);
      i = j;
    }
    return out;
  };

  const a = doc.article || {};
  let host = "";
  try { host = new URL(a.url || doc.source).hostname.replace(/^www\./, ""); } catch (e) { /* not a URL */ }
  const kicker = [a.site || host, a.author, a.date].filter(Boolean).map(esc).join("<span aria-hidden='true'> &middot; </span>");
  let html = `<header class="article-head">${kicker ? `<p class="kicker">${kicker}</p>` : ""}<h1>${esc(doc.title || "Untitled article")}</h1>` +
    (a.url ? `<div class="source"><a class="linkish" href="${esc(a.url)}" target="_blank" rel="noopener noreferrer">Open the original</a></div>` : "") + `</header>`;
  let n = 0;
  for (const b of doc.blocks) {
    const ref = b.start >= refStart ? " refs" : "";
    if (b.tag === "pre") { html += `<pre><code>${esc(b.text)}</code></pre>`; continue; }
    if (b.end <= b.start) continue;
    const part = secAt[b.start];
    if (part && part.synthetic && b.tag !== "h") {
      if (part.bucket) n++;
      html += `<div class="part" id="sec-${part.id}"><span>${esc(part.title)}</span>${part.bucket ? badgeHtml(part) : ""}</div>`;
    }
    const t = text(b.start, b.end);
    if (b.tag === "h") {
      const lvl = Math.min(5, (b.level || 1) + 1);
      const s = secAt[b.start];
      const badge = s && s.bucket ? badgeHtml(s) : "";
      if (badge) n++;
      html += `<h${lvl} class="sh ${s && s.bucket ? s.bucket : ""}"${s ? ` id="sec-${s.id}"` : ""}><span>${t}</span>${badge}</h${lvl}>`;
    } else if (b.tag === "li") html += `<div class="li d${Math.min(3, b.depth || 0)}${ref}">${t}</div>`;
    else if (b.tag === "quote") html += `<blockquote><p>${t}</p></blockquote>`;
    else if (b.tag === "row") html += `<p class="row">${t}</p>`;
    else html += `<p class="${ref.trim()}">${t}</p>`;
  }
  const art = document.createElement("article");
  art.className = "article";
  art.innerHTML = html;
  reader.appendChild(art);
  return n;
}

/* ---------- highlights, labels, tooltips ---------- */

function hlById(id) { return DOC && DOC.highlights.find((h) => h.id === id); }
function secById(id) { return DOC && DOC.sections.find((s) => s.id === id); }
function activate(id, on) { for (const el of $$(`[data-h="${id}"]`)) el.classList.toggle("active", on); }

function tipHighlight(h, e) {
  const sec = secById(h.section_id);
  let sub = sec ? `In: ${esc(sec.title)}` : "";
  if (h.evidence_hint) sub += (sub ? " &middot; " : "") + "Evidence: " + esc(h.evidence_hint);
  $("#tip").innerHTML = `<div class="lbl ${h.type}">${esc(h.label)}</div><div>${esc(h.note)}</div>${sub ? `<div class="sub">${sub}</div>` : ""}`;
  $("#tip").classList.remove("hidden");
  moveTip(e);
}

function tipSection(s, e) {
  const a = s.analysis;
  const tip = $("#tip");
  tip.classList.remove("pinned");
  tip.innerHTML = `<div class="lbl section">${esc(s.bucket)}${a ? " &middot; difficulty " + a.difficulty + " of 5" : ""}</div><div><b>${esc(s.title)}</b></div>` +
    (s.why ? `<div>${esc(s.why)}</div>` : "") +
    (a && a.prerequisites.length ? `<div class="sub">Needs: ${a.prerequisites.map(esc).join(", ")}</div>` : "");
  tip.classList.remove("hidden");
  moveTip(e);
}

function moveTip(e) {
  const tip = $("#tip");
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}
function hideTip() { $("#tip").classList.add("hidden"); $("#tip").classList.remove("pinned"); }

const reader = $("#reader");
reader.addEventListener("mouseover", (e) => {
  const hl = e.target.closest("[data-h]");
  if (hl) { activate(hl.dataset.h, true); if (!pinned) { const h = hlById(hl.dataset.h); if (h) tipHighlight(h, e); } return; }
  const b = e.target.closest(".badge[data-s]");
  if (b && !pinned) { const s = secById(b.dataset.s); if (s) tipSection(s, e); }
});
reader.addEventListener("mousemove", (e) => { if (!pinned && e.target.closest("[data-h], .badge")) moveTip(e); });
reader.addEventListener("mouseout", (e) => {
  const hl = e.target.closest("[data-h]");
  if (hl && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest(`[data-h="${hl.dataset.h}"]`))) {
    activate(hl.dataset.h, false);
    if (!pinned) hideTip();
  } else if (e.target.closest(".badge") && !pinned) hideTip();
});
reader.addEventListener("click", (e) => {
  const hl = e.target.closest("[data-h]");
  if (hl && !document.body.classList.contains("off-" + (hlById(hl.dataset.h) || {}).type)) {
    e.stopPropagation();
    pinned = hl.dataset.h;
    tipHighlight(hlById(pinned), e);
    $("#tip").classList.add("pinned");
    return;
  }
  const b = e.target.closest(".badge[data-s]");
  if (b) { e.stopPropagation(); openPanel(); selectTab("plan"); focusSection(b.dataset.s, false); }
});

/* ---------- side panel ---------- */

function renderPanel(doc) {
  const sec = Object.fromEntries(doc.sections.map((s) => [s.id, s]));
  const plan = doc.plan || { order: [] };
  let html = plan.summary ? `<p class="summary">${esc(plan.summary)}</p>` : "";
  const list = (arr) => `<ul>${arr.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
  for (const [b, label] of BUCKETS) {
    const items = plan.order.filter((o) => o.bucket === b && sec[o.section_id]);
    if (!items.length) continue;
    html += `<div class="group"><h4><span class="dot" style="background:var(--${b})"></span>${label}<span class="c">${items.length}</span></h4>`;
    for (const o of items) {
      const s = sec[o.section_id];
      const a = s.analysis || {};
      html += `<div class="sec" data-s="${s.id}" tabindex="0"><div class="t"><span>${esc(s.title)}</span>${a.difficulty ? diffDots(a.difficulty) : ""}</div>` +
        (o.why ? `<div class="why">${esc(o.why)}</div>` : "") + `<div class="more">` +
        (a.difficulty_reason ? `<p>${esc(a.difficulty_reason)}</p>` : "") +
        (a.supports_claims && a.supports_claims.length ? `<b>Supports</b><p>${a.supports_claims.map(esc).join(", ")}</p>` : "") +
        (a.prerequisites && a.prerequisites.length ? `<b>Prerequisites</b>${list(a.prerequisites)}` : "") +
        (a.self_check && a.self_check.length ? `<b>Check yourself</b>${list(a.self_check)}` : "") + `</div></div>`;
    }
    html += `</div>`;
  }
  $("#tab-plan").innerHTML = html || (doc.partial
    ? "<p class='summary'>Reading plan will show up after every section is analyzed.</p>"
    : "<p class='summary'>No reading plan for this document.</p>");

  $("#tab-claims").innerHTML = (doc.claims && doc.claims.length)
    ? doc.claims.map((c) => `<div class="claimcard" data-c="${esc(c.id)}" tabindex="0"><div class="id">${esc(c.id)}</div><div class="c">${esc(c.claim)}</div>${c.evidence_hint ? `<div class="ev">Evidence: ${esc(c.evidence_hint)}</div>` : ""}</div>`).join("")
    : (doc.partial ? "<p class='summary'>Looking for the main claims…</p>" : "<p class='summary'>No main claims were found.</p>");

  const pre = plan.prerequisites || [];
  $("#tab-prereq").innerHTML = pre.length
    ? `<p class="summary">Worth knowing before you start, most basic first.</p><ol class="prereq">${pre.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>`
    : "<p class='summary'>No prerequisites listed.</p>";
}

$("#panel").addEventListener("click", (e) => {
  const s = e.target.closest(".sec");
  if (s) return focusSection(s.dataset.s, true);
  const c = e.target.closest(".claimcard");
  if (c && DOC) {
    const h = DOC.highlights.find((x) => x.claim_id === c.dataset.c);
    if (h) scrollToHighlight(h.id);
  }
});
$("#panel").addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.matches(".sec, .claimcard")) e.target.click(); });

function focusSection(id, scroll) {
  for (const el of $$("#tab-plan .sec")) {
    if (el.dataset.s === id) el.classList.toggle("open", scroll ? !el.classList.contains("open") : true);
    else el.classList.remove("open");
  }
  const card = $(`#tab-plan .sec[data-s="${id}"]`);
  if (card && !scroll) card.scrollIntoView({ block: "nearest" });
  if (scroll) {
    const s = secById(id);
    const target = document.getElementById("sec-" + id) || (s && document.getElementById("page-" + s.page));
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      if (innerWidth <= 1100) closePanel();
    }
  }
}

function scrollToHighlight(id) {
  const el = $(`[data-h="${id}"]`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  if (innerWidth <= 1100) closePanel();
}

function selectTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
  $$(".tabbody").forEach((b) => b.classList.toggle("hidden", b.id !== "tab-" + name));
}
function openPanel() { document.body.classList.add("panel-open"); $("#panelToggle").setAttribute("aria-expanded", "true"); }
function closePanel() { document.body.classList.remove("panel-open"); $("#panelToggle").setAttribute("aria-expanded", "false"); }

$$(".tab").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));
$("#panelToggle").addEventListener("click", () => (document.body.classList.contains("panel-open") ? closePanel() : openPanel()));
$("#panelClose").addEventListener("click", closePanel);
$("#scrim").addEventListener("click", closePanel);
$$(".chip").forEach((c) => c.addEventListener("click", () => {
  c.classList.toggle("on");
  document.body.classList.toggle("off-" + c.dataset.type, !c.classList.contains("on"));
}));
document.addEventListener("click", (e) => { if (pinned && !e.target.closest("#tip")) { pinned = null; hideTip(); } });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { pinned = null; hideTip(); closePanel(); } });

/* ---------- home ---------- */

function looksLikeLink(v) { return /^(https?:\/\/|www\.)\S+$/i.test(v) || /^[\w-]+(\.[\w-]+)+(\/\S*)?$/.test(v) || /^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$/i.test(v); }

$("#intake").addEventListener("submit", (e) => {
  e.preventDefault();
  const url = $("#urlInput").value.trim();
  const err = $("#formError");
  if (!url) { err.textContent = "Paste a link first, or drop a file below."; $("#urlInput").focus(); return; }
  if (!looksLikeLink(url)) { err.textContent = "That does not look like a link. Paste a full address such as https://arxiv.org/abs/1706.03762"; return; }
  err.textContent = "";
  submit({ url });
});
$("#urlInput").addEventListener("input", () => { $("#formError").textContent = ""; });
$("#fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) submit({ file: f }); });
$$("[data-url]").forEach((b) => b.addEventListener("click", () => { $("#urlInput").value = b.dataset.url; submit({ url: b.dataset.url }); }));
$$("[data-sample]").forEach((b) => b.addEventListener("click", () => submit({ sample: b.dataset.sample })));
$("#retryBtn").addEventListener("click", () => { if (lastInput) submit(lastInput); });

for (const a of [$("#homeLink"), $("#newLink"), $("#cancelLink"), $("#error a[href='/']")]) {
  a.addEventListener("click", (e) => { if (e.metaKey || e.ctrlKey || e.shiftKey) return; e.preventDefault(); goHome(); });
}
window.addEventListener("popstate", route);

let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault(); dragDepth = 0; document.body.classList.remove("dragging");
  const f = e.dataTransfer.files[0]; if (f) submit({ file: f });
});
window.addEventListener("paste", (e) => {
  if ($("#home").classList.contains("hidden")) return;
  const f = [...(e.clipboardData?.files || [])][0];
  if (f) { e.preventDefault(); submit({ file: f }); }
});

function ago(t) {
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " h ago";
  if (s < 86400 * 7) return Math.round(s / 86400) + " d ago";
  return new Date(t * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
const KIND = { pdf: "PDF", html: "Article", image: "Image" };

async function loadRecent() {
  let items = [];
  try { const r = await fetch("/api/recent"); if (r.ok) items = await r.json(); } catch (e) { /* offline */ }
  const box = $("#recent");
  box.classList.toggle("hidden", !items.length);
  box.dataset.loaded = "1";
  $("#recentList").innerHTML = items.map((it) => `<li><a href="?doc=${it.doc_id}" data-doc="${it.doc_id}"><span class="r-title">${esc(it.title)}</span><span class="r-kind">${KIND[it.kind] || ""}</span><span class="r-time">${ago(it.time)}</span></a></li>`).join("");
}
$("#recentList").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-doc]");
  if (!a || e.metaKey || e.ctrlKey) return;
  e.preventDefault();
  history.pushState(null, "", "?doc=" + a.dataset.doc);
  lastInput = null;
  $("#loadingSource").textContent = $(".r-title", a).textContent;
  setStage("Loading", 0.05);
  show("loading");
  poll(a.dataset.doc);
});

async function loadConfig() {
  try { const r = await fetch("/api/config"); CONFIG = await r.json(); } catch (e) { CONFIG = null; }
  const foot = $("#foot");
  if (!CONFIG) { foot.textContent = ""; return; }
  foot.innerHTML = CONFIG.mock
    ? "Running the offline demo analyzer. Add an API key such as <code>GROQ_API_KEY</code> or <code>GEMINI_API_KEY</code> to <code>.env</code> for real analysis."
    : "";
}

function route() {
  const p = new URLSearchParams(location.search);
  if (p.get("doc")) {
    $("#loadingSource").textContent = "Opening your document";
    setStage("Loading", 0.05);
    show("loading");
    poll(p.get("doc"));
  } else if (p.get("url")) {
    $("#urlInput").value = p.get("url");
    submit({ url: p.get("url") });
  } else goHome(false);
}

loadConfig();
route();

function isDark() { return document.documentElement.getAttribute("data-theme") === "dark"; }

function applyTheme(dark, persist) {
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  if (persist) {
    try { localStorage.setItem("ph-theme", dark ? "dark" : "light"); } catch (e) { /* private mode */ }
  }
  const btn = $("#themeToggle");
  if (!btn) return;
  btn.setAttribute("aria-pressed", dark ? "true" : "false");
  const label = dark ? "Switch to light theme" : "Switch to dark theme";
  btn.title = label;
  btn.setAttribute("aria-label", label);
  const word = $(".theme-label", btn);
  if (word) word.textContent = dark ? "LIGHT" : "DARK";
}

applyTheme(isDark(), false);
$("#themeToggle").addEventListener("click", () => applyTheme(!isDark(), true));
