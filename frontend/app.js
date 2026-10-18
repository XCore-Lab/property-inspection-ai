"use strict";
const $ = (id) => document.getElementById(id);
const API = "/api/v1";
const STATE_LABEL = { DETECTED: "Detected", NEEDS_REVIEW: "Needs review", UNKNOWN: "Unknown",
                      MANUAL_CHECK_REQUIRED: "Manual check", NOT_DETECTED: "No visible damage" };
const LOST = "Lost connection to the server — this job is no longer being tracked. Reload the page once the server is reachable again.";
const STAGES = ["queued", "scene", "localization", "verification", "reporting"];
let current = null, pollTimer = null, pollingId = null, clockTimer = null, lastHealth = null;
// One progress card, two possible jobs (a photo inspection and a video walkthrough can run at the same time):
// each job keeps its own state and the card is repainted for whichever job owns the current view.
const runState = { photo: { t0: 0 }, walk: { t0: 0 } };
let panelOwner = null;
const STAGE_LABEL = { queued: "Waiting in queue", scene: "Gemini: identifying room & components", localization: "LocateAnything: locating defects",
                      verification: "Gemma 4: verifying close-ups", validation: "Gemini: second opinion",
                      reporting: "Risk engine & report" };
const PHOTO_NOTE = "Gemini identifies the room, then LocateAnything and Gemma 4 find and verify defects on the local GPU — usually 20–60 s per image (longer if the queue is busy).";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pill = (st) => `<span class="pill ${esc(st)}">${esc(STATE_LABEL[st] || st)}</span>`;
const icon = (name, size) => `<svg${size ? ` width="${size}" height="${size}"` : ""} aria-hidden="true"><use href="#i-${name}"/></svg>`;

// ---- API key: only needed when the server has auth on (health.auth) --------------------------
const KEY_LS = "pi_api_key";
let apiKey = "";
try { apiKey = localStorage.getItem(KEY_LS) || ""; } catch { /* storage blocked */ }
function setApiKey(k) {
  apiKey = String(k || "").trim();
  try { if (apiKey) localStorage.setItem(KEY_LS, apiKey); else localStorage.removeItem(KEY_LS); } catch { /* storage blocked */ }
}
// URL for <img src> / <a href> / <video src>: these cannot send headers, so the key goes in the query string.
function withKey(url) {
  url = String(url);
  if (!apiKey || !url.startsWith("/api/")) return url;
  return `${url}${url.includes("?") ? "&" : "?"}key=${encodeURIComponent(apiKey)}`;
}
// fetch() for the backend: `path` is relative to API ("/health") or absolute ("/api/..."); adds X-API-Key.
function api(path, opts = {}) {
  const headers = new Headers(opts.headers || {});
  if (apiKey) headers.set("X-API-Key", apiKey);
  return fetch(path.startsWith("/api/") ? path : API + path, { cache: "no-store", ...opts, headers });
}
function errText(r, j) {
  let d = j && j.detail;
  if (d && typeof d !== "string") d = JSON.stringify(d);
  if (r.status === 401 || r.status === 403) { $("key-box").hidden = false; return "API key required or invalid — enter it in the sidebar (System status)."; }
  if (r.status === 502) return "Gemini error: " + (d || "the model did not answer — try again.");
  return d || r.statusText || `HTTP ${r.status}`;
}
async function apiJson(path, opts) {
  const r = await api(path, opts);
  let j = null;
  try { j = await r.json(); } catch { /* empty or non-JSON body */ }
  if (!r.ok) throw Object.assign(new Error(errText(r, j)), { status: r.status });
  return j;
}
const listOf = (j) => Array.isArray(j) ? j : (j && (j.items || j.inspections || j.rows)) || [];

// ---- formatting helpers (risk, money, dates) -----------------------------------------------------
const GRADES = ["A", "B", "C", "D", "E"];
const gcls = (g) => GRADES.includes(g) ? g : "x";
const gradeBadge = (g) => g ? `<span class="gb g-${gcls(g)}" title="Condition grade ${esc(g)}">${esc(g)}</span>` : `<span class="gb g-x" title="Not graded">–</span>`;
const decisionCls = (d) => /^PROCEED WITH|CONDITION/i.test(d || "") ? "cond" : /^PROCEED/i.test(d || "") ? "go" : /^REFER/i.test(d || "") ? "refer" : "cond";
const decShort = (d) => String(d || "—").replace(/\s+-\s+/, " — ");
function money(v, cur) {
  if (v == null || v === "" || isNaN(v)) return "—";
  try { return new Intl.NumberFormat(undefined, { style: "currency", currency: cur || "USD", maximumFractionDigits: 0 }).format(v); }
  catch { return `${Math.round(v).toLocaleString()} ${cur || ""}`.trim(); }
}
const moneyRange = (a, b, cur) => a == null && b == null ? "—" : (b == null || a === b) ? money(a ?? b, cur) : `${money(a, cur)} – ${money(b, cur)}`;
const pctRange = (r) => Array.isArray(r) ? (r[0] === r[1] ? `${r[0]}%` : `${r[0]}% to ${r[1]}%`) : r == null ? "—" : `${r}%`;
function fmtDate(v, short) {
  if (v == null || v === "") return "";
  const d = new Date(typeof v === "number" && v < 1e12 ? v * 1000 : v);
  return isNaN(d) ? String(v) : d.toLocaleString(undefined, short ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    : { dateStyle: "medium", timeStyle: "short" });
}
const flagIcon = (lvl) => lvl === "high" || lvl === "medium"
  ? `<span class="flag ${lvl}" title="Evidence integrity: ${esc(lvl)}" aria-label="Evidence integrity ${esc(lvl)}">⚑</span>` : "";
const covPct = (c) => c == null ? null : typeof c === "object" ? c.inspectable_pct ?? null : c;
function prettyModel(m) {
  const s = String(m || "").split(/\s+/)[0];
  if (/^gemini/i.test(s)) return s.split("-").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
  if (/^gemma-?4/i.test(s)) return s.replace(/^gemma-?4-?/i, "Gemma 4 ");
  return s || "?";
}

// Gemini answers: small, safe markdown subset - escape first, then headings, bullets (- * 1.), bold, italics, rules.
function mdLite(s) {
  let out = "", list = "";
  const inline = t => t.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, "$1<i>$2</i>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  const close = () => { if (list) { out += `</${list}>`; list = ""; } };
  for (const raw of esc(s).split(/\r?\n/)) {
    let m;
    if ((m = raw.match(/^\s*[-*•]\s+(.*)$/)) || (m = raw.match(/^\s*\d+[.)]\s+(.*)$/))) {
      const kind = /^\s*\d/.test(raw) ? "ol" : "ul";
      if (list !== kind) { close(); out += `<${kind}>`; list = kind; }
      out += `<li>${inline(m[1])}</li>`; continue;
    }
    close();
    if ((m = raw.match(/^\s*#{1,6}\s+(.*)$/))) { out += `<p class="md-h"><b>${inline(m[1])}</b></p>`; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(raw)) { out += "<hr>"; continue; }
    if (raw.trim()) out += `<p>${inline(raw)}</p>`;
  }
  close();
  return out;
}

function toast(html, ms = 8000) {
  const t = $("toast"); t.innerHTML = html; t.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}

// ---- theme ---------------------------------------------------------------------------------------
const THEME_LS = "pi_theme";
const darkMQ = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
function effectiveTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "dark" || t === "light" ? t : darkMQ && darkMQ.matches ? "dark" : "light";
}
function paintThemeBtn() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  $("theme-btn").innerHTML = `${icon(next === "dark" ? "moon" : "sun")}<span>${next === "dark" ? "Dark" : "Light"} mode</span>`;
  $("theme-btn").setAttribute("aria-label", `Switch to ${next} theme`);
}
function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem(THEME_LS, next); } catch { /* storage blocked */ }
  paintThemeBtn();
}

// ---- init ------------------------------------------------------------------------------------------
async function init() {
  $("api-key").value = apiKey;
  $("api-key").onchange = () => { setApiKey($("api-key").value); refreshAll(); };
  paintThemeBtn();
  $("theme-btn").onclick = toggleTheme;
  if (darkMQ && darkMQ.addEventListener) darkMQ.addEventListener("change", paintThemeBtn);
  // Lists first, in parallel - they must not wait for (or fail with) the taxonomy request.
  health(); setInterval(health, 15000);
  walks(); setInterval(walks, 20000);

  const wd = $("walk-drop");
  wd.onclick = () => $("walk-file").click();
  wd.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("walk-file").click(); } };
  wd.ondragover = (e) => { e.preventDefault(); wd.classList.add("over"); };
  wd.ondragleave = () => wd.classList.remove("over");
  wd.ondrop = (e) => {
    e.preventDefault(); wd.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) { const dt = new DataTransfer(); dt.items.add(f); $("walk-file").files = dt.files; pickWalk(); }
  };
  $("walk-file").onchange = pickWalk;
  $("walk-form").onsubmit = submitWalk;
  // Scanning view readout: which moment of the walkthrough video is on screen.
  const rv = $("run-vid"), mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  rv.addEventListener("timeupdate", () => {
    $("scan-hud").hidden = rv.hidden || !rv.duration;
    if (rv.duration) $("scan-hud").textContent = `${mmss(rv.currentTime)} / ${mmss(rv.duration)}`;
  });

  document.querySelectorAll(".viewer-tabs button").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".viewer-tabs button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
    if (current) $("r-img").src = withKey(`${API}/inspection/${current.inspection_id}/image/${b.dataset.kind}?t=${Date.now()}`);
  });
  $("btn-json").onclick = () => { const d = $("json-box"); d.open = !d.open; if (d.open) d.scrollIntoView({ behavior: "smooth" }); };
  $("btn-summary").onclick = genSummary;
  $("ask-form").onsubmit = ask;
  $("suggest").querySelectorAll("button").forEach((b) => b.onclick = () => { $("ask-q").value = b.textContent; $("ask-form").requestSubmit(); });
  $("pf-refresh").onclick = portfolio;
  window.onhashchange = route;
  route();
}

function refreshAll() { health(); walks(); if (currentView === "dashboard") portfolio(); }

// ---- routing: #/dashboard, #/walkthroughs, #/inspection/INS-0001 (direct links only) -------------
let currentView = null, currentRouteId = null;
function parseRoute() {
  let h = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  if (/^INS-\d+$/i.test(h)) h = "inspection/" + h;            // legacy links (#INS-0006)
  if (h === "portfolio") h = "dashboard";
  if (h === "inspect") h = "walkthroughs";                     // the photo upload page was removed
  const [view, id] = h.split("/");
  return { view: ["dashboard", "walkthroughs", "inspection"].includes(view) ? view : "dashboard", id: id || null };
}
function go(hash) { if (location.hash !== hash) location.hash = hash; else route(); }

function route() {
  const r = parseRoute();
  const legacy = location.hash.replace(/^#\/?/, "");
  if (/^INS-\d+$/i.test(legacy) || legacy === "portfolio") { window.history.replaceState(null, "", `#/${r.view}${r.id ? "/" + r.id : ""}`); }
  if (r.view === "inspection" && !r.id) { go("#/dashboard"); return; }
  const changed = r.view !== currentView || r.id !== currentRouteId;
  currentView = r.view; currentRouteId = r.id;
  for (const v of ["dashboard", "walkthroughs", "inspection"]) $("view-" + v).hidden = v !== r.view;
  for (const v of ["dashboard", "walkthroughs"]) {
    const on = v === r.view || (v === "dashboard" && r.view === "inspection");
    $("nav-" + v).classList.toggle("on", on);
    if (on) $("nav-" + v).setAttribute("aria-current", "page"); else $("nav-" + v).removeAttribute("aria-current");
  }
  if (changed) window.scrollTo(0, 0);
  hideError();
  if (r.view === "dashboard") { portfolio(); renderWalkCards(); }
  if (r.view === "walkthroughs") { renderWalkCards(); if (r.id === "new") showUploadForm(); }
  if (r.view === "inspection") openInspection(r.id);
  document.title = { dashboard: "Dashboard", walkthroughs: "Video walkthroughs",
                     inspection: r.id || "Inspection" }[r.view] + " · Property Inspection AI";
  placeRunning();
}

// "New walkthrough" (#/walkthroughs/new): bring the upload form into view and put focus on the drop zone,
// instead of leaving the visitor on the list of finished walkthroughs.
function showUploadForm() {
  requestAnimationFrame(() => {
    $("walk-form").scrollIntoView({ behavior: "smooth", block: "start" });
    $("walk-drop").focus({ preventScroll: true });
  });
}

// The shared progress card lives in whichever view owns the running job.
function placeRunning() {
  const run = $("running");
  let slot = null;
  if (pollingId && pollingId === currentRouteId && currentView === "inspection") slot = "ins-run-slot";
  else if (watching && currentView === "walkthroughs") slot = "walk-run-slot";
  $("nav-walk-live").hidden = !watching;
  if (!slot) { run.hidden = true; panelOwner = null; return; }
  const kind = slot === "walk-run-slot" ? "walk" : "photo";
  if (kind !== panelOwner) paintOwner(kind);
  if (run.parentElement.id !== slot) $(slot).appendChild(run);
  run.hidden = false;
}

async function health() {
  const el = $("health");
  try {
    const r = await api("/health");
    const h = await r.json();
    lastHealth = h;
    const v = h.vlm || {}, c = h.cloud || {}, sc = h.scene || {}, val = h.validator || null, loc = h.localizer || {};
    const mode = typeof c.gemini === "string" && c.gemini ? c.gemini : sc.mode;
    el.className = "health ok"; el.textContent = "Online";
    // A walkthrough runs its models in its own process, so the server's "loaded" flags stay false meanwhile.
    const busy = (walkItems || []).some((w) => w.status === "running");
    const gpuWhere = (loaded) => busy ? "GPU · in use" : loaded ? "GPU · loaded" : "GPU";
    const rows = [
      [prettyModel(sc.model || "gemini"), sc.backend && sc.backend !== "gemini" ? gpuWhere(sc.running) : "cloud", true, `Scene: ${sc.model || "?"}${mode ? " via " + mode : ""}`],
      [prettyModel(loc.model || "LocateAnything-3B"), gpuWhere(loc.loaded), loc.available !== false, `Localizer: ${loc.model || "?"} (${loc.backend || ""})`],
      [prettyModel(v.model || "Gemma 4"), gpuWhere(v.running), v.available !== false, `Verifier: ${v.model || "?"} (${v.backend || ""})`],
    ];
    // Local mode: Gemma does the scene step too. One model, two jobs: show it once.
    if (rows[0][0] === rows[2][0]) {
      rows[2] = [rows[2][0], rows[2][1], rows[2][2], `Scene + verifier: ${v.model || "?"}`];
      rows.shift();
    }
    if (val && val.model && val.model !== sc.model) rows.push([prettyModel(val.model), "cloud", true, `Second opinion: ${val.model}`]);
    $("models").innerHTML = rows.map(([n, where, on, tip]) =>
      `<li title="${esc(tip)}"><span class="dot ${on ? "on" : ""}"></span><span class="nm">${esc(n)}</span><span class="where">${esc(where)}</span></li>`).join("");
    if (h.vram_total_mb) {
      $("vram").hidden = false;
      $("gpu-name").textContent = String(h.gpu || "GPU").replace(/^NVIDIA\s+(GeForce\s+)?/i, "").replace(/\s+Laptop GPU$/i, "");
      $("vram-txt").textContent = `${((h.vram_used_mb || 0) / 1024).toFixed(1)} / ${(h.vram_total_mb / 1024).toFixed(1)} GB`;
      $("vram-bar").style.width = `${Math.min(100, 100 * (h.vram_used_mb || 0) / h.vram_total_mb).toFixed(1)}%`;
    } else $("vram").hidden = true;
    const running = Array.isArray(h.running) ? h.running.length : h.running || 0;
    paintVideoQueue();
    const on = (label, val2) => `${label}: ${val2 ? `on (${val2})` : "off"}`;
    el.title = ["Google Cloud integrations", on("• Cloud Storage bucket", c.gcs_bucket), on("• BigQuery dataset", c.bigquery_dataset),
      on("• Pub/Sub topic", c.pubsub_topic), c.project ? `• Project: ${c.project}` : null,
      `• Gemini: ${sc.model || "?"}${mode ? " via " + mode : ""}`,
      h.gpu ? `GPU: ${h.gpu}` : "GPU: none (video walkthroughs need a GPU)",
      `API key: ${h.auth ? "required" : "not required"}`, running ? `Running jobs: ${running}` : null].filter(Boolean).join("\n");
    if (h.auth) $("key-box").hidden = false; else if (!apiKey) $("key-box").hidden = true;
    $("pf-bq").textContent = c.bigquery_dataset ? ` (on: ${c.bigquery_dataset})` : " (currently off)";
  } catch { el.className = "health bad"; el.textContent = "Offline"; el.title = "Backend not reachable"; }
}

// ---- photo inspection progress (inspection detail view) ---------------------------------------------
function startRunning() {
  runState.photo = { t0: Date.now(), img: runState.photo.img, prog: null, stageText: "" };
  photoStage = null; setProgress("photo", 0, "Starting", 0, null);
  if (panelOwner === "photo") paintOwner("photo");
  placeRunning();
  startClock();
}

function paintOwner(kind) {
  panelOwner = kind;
  const st = runState[kind], walk = kind === "walk", v = $("run-vid");
  $("run-title").textContent = walk ? `Processing “${st.name || "walkthrough"}”` : "Inspecting photo";
  $("run-note").textContent = walk ? WALK_NOTE : PHOTO_NOTE;
  $("photo-steps").hidden = walk; $("walk-steps").hidden = !walk;
  $("run-log").hidden = !walk; $("run-log").textContent = st.log || "starting…";
  $("run-img").hidden = walk;
  if (!walk && st.img && $("run-img").getAttribute("src") !== st.img) $("run-img").src = st.img;
  // A stored video that can't be loaded (older server, file removed) hides the preview instead of a black box.
  v.onerror = () => { if (panelOwner === "walk" && runState.walk.videoUrl) { runState.walk.videoUrl = null; paintOwner("walk"); } };
  if (walk && st.videoUrl) { if (v.getAttribute("src") !== st.videoUrl) v.src = st.videoUrl; v.hidden = false; v.play().catch(() => {}); }
  else { v.hidden = true; v.pause(); }
  $("run-scan").hidden = walk && !st.videoUrl;
  $("run-stage").textContent = st.stageText || (walk ? "Starting" : "Queued");
  $("scan-tag").textContent = st.stageText || (walk ? "Processing" : "AI analysing photo");
  if (st.prog) paintProgress(...st.prog); else resetProgress();
  tickClock();
}

function setStageText(kind, txt) {
  runState[kind].stageText = txt;
  if (panelOwner === kind) { $("run-stage").textContent = txt; $("scan-tag").textContent = txt; }
}

const fmtDur = (s) => {
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
};

// Fill the progress bar with a real percentage + elapsed / remaining time (stored per job, painted for the owner).
function setProgress(kind, pct, label, elapsedS, etaS) {
  runState[kind].prog = [pct, label, elapsedS, etaS];
  if (panelOwner === kind) paintProgress(pct, label, elapsedS, etaS);
}
function paintProgress(pct, label, elapsedS, etaS) {
  pct = Math.max(0, Math.min(100, pct));
  const bar = $("run-bar");
  bar.classList.add("det"); bar.querySelector("span").style.width = pct.toFixed(1) + "%";
  const eta = etaS == null ? "" : etaS > 0 ? `<span>about <b>${fmtDur(etaS)}</b> left</span>` : "";
  $("run-progress").innerHTML = `<span><b>${Math.floor(pct)}%</b></span>` + (label ? `<span>${esc(label)}</span>` : "") +
    `<span>${fmtDur(elapsedS)} elapsed</span>` + eta;
}

function resetProgress() {
  const bar = $("run-bar");
  bar.classList.remove("det"); bar.querySelector("span").style.width = "";
  $("run-progress").innerHTML = "";
}

// Photo inspections: % from the stage + typical stage durations, interpolated inside the running stage.
const PHOTO_STAGE_S = { queued: 1, scene: 4, localization: 12, verification: 15, validation: 5, reporting: 1 };
let photoStage = null, photoStageT = 0;
function photoProgress(stage) {
  if (stage === "preprocessed") stage = "scene";
  if (stage !== photoStage) { photoStage = stage; photoStageT = Date.now(); }
  const i = Math.max(0, STAGES.indexOf(stage));
  const total = STAGES.reduce((a, s) => a + PHOTO_STAGE_S[s], 0);
  const before = STAGES.slice(0, i).reduce((a, s) => a + PHOTO_STAGE_S[s], 0);
  const inStage = (Date.now() - photoStageT) / 1000, exp = PHOTO_STAGE_S[STAGES[i]];
  const frac = Math.min(0.95, inStage / exp);
  const left = exp * (1 - frac) + total - before - exp + Math.max(0, inStage - exp) * 0.5;
  setProgress("photo", 100 * (before + frac * exp) / total, `Step ${i + 1} of ${STAGES.length}`, (Date.now() - runState.photo.t0) / 1000, left);
}

function tickClock() {
  if (panelOwner) $("run-elapsed").textContent = fmtDur((Date.now() - runState[panelOwner].t0) / 1000) + " elapsed";
}
function startClock() { if (!clockTimer) clockTimer = setInterval(tickClock, 1000); tickClock(); }
function stopClockIfIdle() { if (!pollingId && !watching) { clearInterval(clockTimer); clockTimer = null; } }

function stopRunning() {
  clearTimeout(pollTimer); pollingId = null; stopClockIfIdle();
  runState.photo.prog = null;
  placeRunning();
}

function poll(id) {
  clearTimeout(pollTimer);
  pollingId = id;
  placeRunning();
  let fails = 0;
  const tick = async () => {
    if (pollingId !== id) return;
    try {
      const r = await api(`/inspection/${id}`);
      if (!r.ok) throw new Error(r.statusText);
      const ins = await r.json();
      fails = 0;
      if (ins.status === "done") {
        setStage("done"); stopRunning();
        if (currentView === "inspection" && currentRouteId === id) render(ins);
        else toast(`Inspection <b>${esc(id)}</b> is ready — <a href="#/inspection/${esc(id)}">open results</a>`);
        return;
      }
      if (ins.status === "failed") { stopRunning(); showError("Inspection failed: " + ins.error); return; }
      setStage(ins.stage || "queued");
      photoProgress(ins.stage || "queued");
    } catch (e) {                       // server gone (stopped / crashed): stop the animation
      if (++fails >= 8) { stopRunning(); showError(LOST); return; }
    }
    pollTimer = setTimeout(tick, 1000);
  };
  tick();
}

function setStage(stage) {
  const idx = stage === "done" ? STAGES.length : Math.max(0, STAGES.indexOf(stage === "preprocessed" ? "scene" : stage));
  document.querySelectorAll("#photo-steps li").forEach((li, i) => {
    li.classList.toggle("done", i < idx); li.classList.toggle("active", i === idx);
  });
  const key = STAGES[Math.min(idx, STAGES.length - 1)];
  setStageText("photo", STAGE_LABEL[key] || key);
}

function showPendingHero(id) {
  $("r-crumb").textContent = id; $("r-title").textContent = `Inspection ${id}`;
  $("r-meta").innerHTML = `<span>${icon("clock")}Processing — results appear here automatically</span>`;
  $("r-chips").innerHTML = ""; $("r-grade").hidden = true; $("results").hidden = true;
}

function openInspection(id) {
  if (current && current.inspection_id === id && current.status === "done") { $("results").hidden = false; return; }
  if (pollingId === id) { showPendingHero(id); placeRunning(); return; }
  load(id);
}

async function load(id) {
  showPendingHero(id);
  $("r-meta").innerHTML = `<span><span class="mini-spin"></span>Loading…</span>`;
  let ins;
  try { ins = await apiJson(`/inspection/${id}`); }
  catch (e) { if (currentRouteId === id) { showError(e.message); $("r-meta").textContent = "Not available."; } return; }
  if (currentRouteId !== id) return;
  if (ins.status === "done") render(ins);
  else if (ins.status === "failed") { showError("Inspection failed: " + (ins.error || "unknown error")); $("r-meta").textContent = "This inspection failed."; }
  else if (pollingId !== id) {
    if (pollingId && pollingId !== "upload") { showPendingHero(id); return; }
    showPendingHero(id);
    runState.photo.img = withKey(`${API}/inspection/${id}/image/original`);
    pollingId = id; startRunning(); poll(id);
  }
}

// ---- inspection detail -----------------------------------------------------------------------------
function render(ins) {
  current = ins;
  hideError();
  $("results").hidden = false;
  const risk = ins.risk || null;
  const t = ins.timings || {}, f1 = (x) => x == null ? "?" : Number(x).toFixed(1);
  const cur = risk && risk.repair_cost && risk.repair_cost.currency;
  $("r-crumb").textContent = ins.inspection_id;
  $("r-title").textContent = ins.source_image || `${ins.room_label || "Room"} — ${ins.inspection_id}`;
  const meta = [
    [ "home", ins.room_label || "Room" ],
    ins.property_id ? [ "pin", `Property ${ins.property_id}` ] : null,
    ins.property_value ? [ "coin", `Value ${money(ins.property_value, cur)}` ] : null,
    ins.created_at ? [ "clock", fmtDate(ins.created_at) ] : null,
    t.total_s != null ? [ "spark", `${f1(t.total_s)} s` + (t.scene_s != null ? ` (scene ${f1(t.scene_s)} · locate ${f1(t.localization_s)} · verify ${f1(t.verification_s)})` : "") ] : null,
  ].filter(Boolean);
  $("r-meta").innerHTML = meta.map(([i, txt]) => `<span>${icon(i)}${esc(txt)}</span>`).join("");
  const dets = ins.detections || [], rejected = ins.rejected || [];
  const nDet = dets.filter((d) => d.state === "DETECTED").length, nRev = dets.filter((d) => d.state === "NEEDS_REVIEW" || d.state === "UNKNOWN").length;
  const ig = ins.integrity;
  $("r-chips").innerHTML = [
    nDet ? `<span class="pill DETECTED">${nDet} detected</span>` : "",
    nRev ? `<span class="pill NEEDS_REVIEW">${nRev} needs review</span>` : "",
    (ins.manual_checks || []).length ? `<span class="pill MANUAL_CHECK_REQUIRED">${ins.manual_checks.length} manual check${ins.manual_checks.length > 1 ? "s" : ""}</span>` : "",
    !dets.length ? `<span class="pill NOT_DETECTED">No visible damage</span>` : "",
    risk && risk.risk_level ? `<span class="lvl ${esc(risk.risk_level)}">${esc(risk.risk_level)} risk</span>` : "",
    ig ? (IG_BADGE[ig.level] ? `<span class="ig ${esc(ig.level)}">${icon("shield", 12)} ${IG_BADGE[ig.level]}</span>` : `<span class="ig ok">Evidence OK</span>`) : "",
  ].join("");
  $("r-grade").hidden = !risk;
  if (risk) $("r-grade").innerHTML = `<div class="grade-big g-${gcls(risk.grade)}" aria-label="Condition grade ${esc(risk.grade || "?")}">${esc(risk.grade || "?")}</div>
    <div><div class="gl">Condition grade</div><div class="gs">${esc(risk.condition_score ?? "—")}<small> / 100</small></div><div class="gl">${esc(risk.grade_label || "")}</div></div>`;
  $("r-decision").innerHTML = risk ? decisionHtml(risk) : "";

  const q = ins.image_quality;
  $("quality").hidden = !(q && q.warnings && q.warnings.length);
  if (q && q.warnings) $("quality").textContent = "Image quality: " + q.warnings.join(" ");
  document.querySelectorAll(".viewer-tabs button").forEach((x) => { x.classList.toggle("on", x.dataset.kind === "annotated"); x.setAttribute("aria-selected", x.dataset.kind === "annotated"); });
  const base = `${API}/inspection/${ins.inspection_id}`;
  $("r-img").src = withKey(`${base}/image/annotated?t=${Date.now()}`);
  $("btn-report").href = withKey(`${base}/report?format=html`);
  $("btn-md").href = withKey(`${base}/report?format=md`);

  $("risk-card").hidden = !risk;
  $("risk").innerHTML = risk ? riskHtml(risk) : "";
  $("integrity-card").hidden = !ig;
  $("integrity").innerHTML = ig ? integrityHtml(ig) : "";
  resetAsk(ins);

  $("issues-count").textContent = dets.length ? `· ${dets.length}` : "";
  $("issues").innerHTML = dets.length ? dets.map((d) => issueHtml(d, base)).join("")
    : `<div class="empty-state">${icon("check", 22)}<br>No visible defects were confirmed.${rejected.length
      ? ` ${rejected.length} candidate${rejected.length > 1 ? "s were" : " was"} rejected by verification — see “With rejected”.` : ""}</div>`;

  $("components").innerHTML = (ins.components || []).map((c) =>
    `<tr><td class="cell-main">${esc(c.label)}</td><td>${esc(c.visible_summary)}</td><td>${pill(c.visible_state)}</td></tr>`).join("")
    || `<tr><td colspan="3" class="empty-state">No components identified.</td></tr>`;
  const manual = ins.manual_checks || [], unknown = ins.unknown_items || [];
  $("manual").innerHTML = manual.length
    ? manual.map((m) => `<li>${pill("MANUAL_CHECK_REQUIRED")}<span><b>${esc(m.component_label)}</b> — ${esc(m.check)}${m.reason ? `<br><span class="muted small">${esc(m.reason)}</span>` : ""}</span></li>`).join("")
    : `<li class="muted">None for the visible components.</li>`;
  $("unk-h").hidden = !unknown.length;
  $("unknown").innerHTML = unknown.map((u) => `<li>${esc(u)}</li>`).join("");
  $("json").textContent = "";
  apiJson(`/inspection/${ins.inspection_id}/result`)
    .then((j) => { $("json").textContent = JSON.stringify(j, null, 2); }).catch(() => {});
  $("disclaimer").textContent = ins.disclaimer || "";
}

function secondOpinionHtml(v) {
  if (!v) return "";
  const ok = v.agrees === true, bad = v.agrees === false;
  const head = ok ? "Gemini second opinion agrees" : bad ? "Gemini second opinion disputes this" : "Gemini second opinion inconclusive";
  const why = [v.description, bad && v.alternative ? `Likely: ${v.alternative}` : ""].filter(Boolean).map(esc).join(" · ");
  return `<div class="so ${ok ? "agree" : bad ? "dispute" : ""}"><span class="so-i" aria-hidden="true">${ok ? "✔" : bad ? "✖" : "?"}</span>
    <span><b>${head}</b>${why ? ` — ${why}` : ""}</span></div>`;
}

function issueHtml(d, base) {
  const no = esc(String(d.issue_id).split("-").pop());
  const sev = d.defect && d.defect.severity;
  return `<article class="issue ${esc(d.state)}">
    <img src="${esc(withKey(`${base}/image/crop-${d.issue_id}`))}" alt="Close-up of finding ${no}: ${esc(d.component_label)} ${esc(d.defect.label)}" loading="lazy">
    <div style="min-width:0">
      <div class="issue-top"><h4><span class="no">#${no}</span>${esc(d.component_label)} · ${esc(d.defect.label)}</h4>
        <div class="chips">${pill(d.state)}${sev ? `<span class="chip sev ${esc(sev)}">${esc(sev)}</span>` : ""}</div></div>
      <p class="desc">${esc(d.description || "—")}</p>
      ${(d.review_reasons || []).length ? `<p class="why"><b>Why review:</b> ${esc(d.review_reasons.join(" "))}</p>` : ""}
      ${(d.professional_review || []).length ? `<p class="why"><b>Professional assessment:</b> ${esc(d.professional_review.join(", "))}</p>` : ""}
      ${secondOpinionHtml(d.validation)}
    </div></article>`;
}

// ---- Risk & valuation ----------------------------------------------------------------------------
function decisionHtml(risk) {
  const cls = decisionCls(risk.decision);
  const ico = cls === "go" ? "check" : cls === "refer" ? "refer" : "alert";
  const rc = risk.repair_cost || {};
  const loading = (risk.insurance || {}).premium_loading_pct;
  const why = [rc.low != null || rc.high != null ? `Repair cost ${moneyRange(rc.low, rc.high, rc.currency)}` : "",
    loading != null ? `premium loading ${loading > 0 ? "+" : ""}${loading}%` : "",
    risk.risk_level ? `${risk.risk_level} risk` : ""].filter(Boolean).join(" · ");
  return `<div class="decision ${cls}" role="status"><span class="dico">${icon(ico)}</span><span class="k">Underwriting decision</span>
    <b>${esc(decShort(risk.decision))}</b>${why ? `<span class="why">${esc(why)}</span>` : ""}</div>`;
}

function riskHtml(risk) {
  const rc = risk.repair_cost || {}, val = risk.valuation || {}, ins = risk.insurance || {};
  const mort = risk.mortgage || {}, ev = risk.evidence || {}, cur = rc.currency || "USD";
  const tile = (k, v, s) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
  const ret = mort.repair_retention;
  const retTxt = typeof ret === "number" ? money(ret, cur) : typeof ret === "boolean" ? (ret ? "required" : "none") : ret ? esc(ret) : "";
  const valSub = [Array.isArray(val.impact_amount) ? "≈ " + moneyRange(val.impact_amount[0], val.impact_amount[1], cur) : "",
    Array.isArray(val.adjusted_value) ? "Adjusted " + moneyRange(val.adjusted_value[0], val.adjusted_value[1], cur) : "",
    val.basis ? esc(val.basis) : ""].filter(Boolean).join("<br>");
  const loading = ins.premium_loading_pct;
  const flags = (ev.integrity_flags || []).map((f) => typeof f === "string" ? f : f.message || f.code);
  const perils = ins.perils || [];
  const nFind = (x) => Array.isArray(x) ? x.length : x ?? "—";
  return `
    <div class="kpis compact">
      ${tile("Repair cost", moneyRange(rc.low, rc.high, cur))}
      ${tile("Valuation impact", pctRange(val.impact_pct), valSub)}
      ${tile("Premium loading", loading == null ? "—" : `${loading > 0 ? "+" : ""}${loading}%`)}
      ${tile("Collateral risk", esc(mort.collateral_risk || "—"), retTxt ? "Retention " + retTxt : "")}
      ${ev.coverage_pct != null || ev.complete != null ? tile("Evidence", ev.coverage_pct != null ? `${ev.coverage_pct}%` : ev.complete ? "Complete" : "Incomplete",
        ev.complete === false ? "more photos needed" : "") : ""}
    </div>
    ${flags.length ? `<div class="warn" style="margin-top:12px">Evidence integrity: ${flags.map(esc).join(" · ")}</div>` : ""}
    ${(risk.reasons || []).length ? `<h3 class="sec-h">Why this decision</h3><ul class="reasons">${risk.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}
    ${perils.length ? `<h3 class="sec-h">Insurance perils</h3><div class="tbl-wrap boxed"><table class="tbl perils"><thead><tr>
        <th>Peril</th><th class="num">Findings</th><th class="num">Risk pts</th><th class="num">Repair cost</th></tr></thead><tbody>
        ${perils.map((p) => `<tr><td><span class="cell-main">${esc(p.label || p.peril)}</span>${p.label && p.peril ? `<br><span class="cell-sub">${esc(p.peril)}</span>` : ""}</td>
          <td class="num">${esc(nFind(p.findings))}</td><td class="num">${esc(p.risk_points ?? "—")}</td>
          <td class="num">${Array.isArray(p.repair_cost) ? moneyRange(p.repair_cost[0], p.repair_cost[1], cur) : "—"}</td></tr>`).join("")}
      </tbody></table></div>` : ""}
    ${risk.disclaimer ? `<p class="disclaimer" style="margin-top:14px">${esc(risk.disclaimer)}</p>` : ""}`;
}

// ---- Evidence integrity --------------------------------------------------------------------------
const AUTH_LABEL = { real_photo: "Real camera photo", screen_photo: "Photo of a screen", rendering: "3D rendering / CGI",
                     ai_generated: "AI-generated image", unsure: "Unsure" };
const IG_BADGE = { high: "Possible fraud — refer", medium: "Check evidence", low: "Minor notes" };
function integrityHtml(g) {
  const flags = g.flags || [], lvl = g.level || (flags.length ? "low" : "none"), ex = g.exif || {};
  const match = (m) => {
    if (!m) return "";
    const id = typeof m === "string" ? m : m.inspection_id || m.id;
    const prop = typeof m === "object" && m.property_id ? ` (property ${esc(m.property_id)})` : "";
    return /^INS-/.test(id || "") ? ` — matches <a href="#/inspection/${esc(id)}">${esc(id)}</a>${prop}`
      : ` — matches ${esc(typeof m === "string" ? m : JSON.stringify(m))}`;
  };
  const camera = [ex.Make, ex.Model].filter(Boolean).join(" ");
  const kv = [["Authenticity", g.authenticity ? esc(AUTH_LABEL[g.authenticity] || g.authenticity) : ""],
    ["Camera", esc(camera)], ["Software", esc(ex.Software || "")], ["Taken", esc(ex.DateTimeOriginal || "")],
    ["Image hash", g.dhash ? `<code>${esc(g.dhash)}</code>` : ""]].filter((x) => x[1]);
  return `
    <div class="ig-head"><span class="ig ${IG_BADGE[lvl] ? esc(lvl) : "ok"}">${IG_BADGE[lvl] || "No integrity concerns"}</span>
      ${g.authenticity && g.authenticity !== "real_photo" ? `<span class="muted small">${esc(AUTH_LABEL[g.authenticity] || g.authenticity)}</span>` : ""}</div>
    ${flags.length ? `<ul class="flags">${flags.map((f) => `<li><span class="ig ${esc(f.level)}">${esc(f.level)}</span>
      <span style="flex:1;min-width:0">${esc(f.message)}${match(f.match)} <span class="muted small">${esc(f.code)}</span></span></li>`).join("")}</ul>`
      : `<p class="muted small" style="margin-top:10px">No duplicate across properties, no editing software, no screen / AI-image signs found.</p>`}
    ${kv.length ? `<dl class="kv">${kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` : ""}`;
}

// ---- Ask Gemini (underwriting assistant) ---------------------------------------------------------
function busy(b, on, label) {
  b.disabled = on; b.classList.toggle("busy", on);
  const svg = b.querySelector("svg");
  b.textContent = label; if (svg && !on) b.prepend(svg);
}

function resetAsk(ins) {
  $("qa").innerHTML = ""; $("ask-err").hidden = true; $("ask-q").value = "";
  const s = ins.ai_summary && (typeof ins.ai_summary === "string" ? ins.ai_summary : ins.ai_summary.summary);
  $("summary").hidden = !s; $("summary").innerHTML = s ? mdLite(s) : "";
  $("btn-summary").hidden = !!s;
  busy($("btn-summary"), false, "Generate underwriter summary");
  $("suggest").hidden = false;
}

async function genSummary() {
  if (!current) return;
  const id = current.inspection_id, b = $("btn-summary");
  $("ask-err").hidden = true;
  busy(b, true, "Gemini is writing…");
  try {
    const j = await apiJson(`/inspection/${id}/summary`, { method: "POST" });
    if (!current || current.inspection_id !== id) return;
    current.ai_summary = j.summary;
    $("summary").innerHTML = mdLite(j.summary || "(empty summary)"); $("summary").hidden = false; b.hidden = true;
  } catch (e) {
    if (current && current.inspection_id === id) { $("ask-err").textContent = e.message; $("ask-err").hidden = false; }
  } finally { busy(b, false, "Generate underwriter summary"); }
}

async function ask(e) {
  e.preventDefault();
  const q = $("ask-q").value.trim();
  if (!current || !q) return;
  const id = current.inspection_id;
  const item = document.createElement("div");
  item.className = "qa-item";
  item.innerHTML = `<div class="q">${esc(q)}</div><div class="a md"><span class="mini-spin"></span>Gemini is thinking…</div>`;
  $("qa").appendChild(item);
  $("suggest").hidden = true;
  $("ask-q").value = ""; $("ask-q").disabled = true; busy($("ask-go"), true, "Ask");
  try {
    const j = await apiJson(`/inspection/${id}/ask`, { method: "POST", headers: { "Content-Type": "application/json" },
                                                      body: JSON.stringify({ question: q }) });
    item.querySelector(".a").innerHTML = mdLite(j.answer || "(no answer)");
  } catch (err) { item.querySelector(".a").innerHTML = `<span class="error">${esc(err.message)}</span>`; }
  finally { $("ask-q").disabled = false; busy($("ask-go"), false, "Ask"); $("ask-q").focus(); }
}

// ---- Portfolio / dashboard -------------------------------------------------------------------------
let pf = null, pfSort = { key: "created_at", dir: -1 }, pfBusy = false;
const PF_COLS = [["label", "Walkthrough"], ["property_id", "Property"], ["created_at", "Date"], ["grade", "Grade"],
  ["condition_score", "Score"], ["risk_level", "Risk"], ["decision", "Decision"], ["repair_high", "Repair cost"],
  ["valuation_pct", "Value impact"], ["coverage_pct", "Coverage"]];

async function portfolio() {
  if (pfBusy) return;
  pfBusy = true;
  $("pf-err").hidden = true;
  if (!pf) $("pf-kpis").innerHTML = ["Walkthroughs", "Repair exposure", "Referred", "Average coverage"].map((k) =>
    `<div class="kpi"><div class="k">${k}</div><div class="v muted">…</div></div>`).join("");
  busy($("pf-refresh"), true, "Refresh");
  try { pf = await apiJson("/portfolio"); renderPortfolio(); }
  catch (e) { $("pf-err").textContent = "Cannot load portfolio: " + e.message; $("pf-err").hidden = false; }
  finally { busy($("pf-refresh"), false, "Refresh"); pfBusy = false; }
}

function renderPortfolio() {
  const p = pf || {}, rows = p.rows || [], ex = p.repair_exposure || {}, cur = ex.currency || "USD";
  const byDec = p.by_decision || {}, byGrade = p.by_grade || {}, byRisk = p.by_risk_level || {};
  const n = p.n ?? rows.length;
  const referred = Object.entries(byDec).filter(([k]) => /^REFER/i.test(k)).reduce((a, [, c]) => a + c, 0);
  const nProps = new Set(rows.map((r) => r.property_id || r.id)).size;
  const covs = rows.map((r) => covPct(r.coverage_pct)).filter((c) => c != null);
  const avgCov = covs.length ? Math.round(covs.reduce((a, c) => a + c, 0) / covs.length) : null;
  const pct = (a) => n ? `${Math.round(100 * a / n)}% of records` : "";
  const tile = (ico, tone, k, v, s) => `<div class="kpi"><div class="k"><span class="ico ${tone}">${icon(ico)}</span>${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
  $("pf-kpis").innerHTML = tile("video", "", "Walkthroughs", esc(n), `${nProps} ${nProps === 1 ? "property" : "properties"} inspected by video`) +
    tile("coin", "amber", "Repair exposure", moneyRange(ex.low, ex.high, cur), "low – high estimate, whole book") +
    tile("refer", "red", "Referred", esc(referred), pct(referred) || "decision = REFER") +
    tile("scan", "", "Average coverage", avgCov != null ? `${avgCov}%` : "—", "of surfaces filmed close enough to inspect");

  const max = Math.max(1, ...GRADES.map((g) => byGrade[g] || 0));
  $("pf-grades").innerHTML = GRADES.map((g) => `<div class="gbar"><span class="gb g-${g}">${g}</span>
    <div class="track" role="img" aria-label="Grade ${g}: ${byGrade[g] || 0}"><i class="g-${g}" style="width:${(100 * (byGrade[g] || 0) / max).toFixed(1)}%"></i></div><b>${byGrade[g] || 0}</b></div>`).join("");

  const LVL = [["Low", "c-ok"], ["Moderate", "c-rev"], ["Medium", "c-rev"], ["High", "c-det"], ["Critical", "c-det"]];
  const riskKeys = Object.keys(byRisk).sort((a, b) => LVL.findIndex((x) => x[0] === a) - LVL.findIndex((x) => x[0] === b));
  const tot = riskKeys.reduce((a, k) => a + byRisk[k], 0) || 1;
  const lc = (k) => (LVL.find((x) => x[0] === k) || [0, "c-mute"])[1];
  $("pf-risk").innerHTML = riskKeys.length ? `<div class="stackbar" role="img" aria-label="Risk level breakdown">${riskKeys.map((k) =>
      `<i class="${lc(k)}" style="width:${(100 * byRisk[k] / tot).toFixed(1)}%" title="${esc(k)}: ${byRisk[k]}"></i>`).join("")}</div>
    <ul class="legend-list">${riskKeys.map((k) => `<li><span class="sw ${lc(k)}"></span>${esc(k)}<b>${byRisk[k]}</b><span class="pct">${Math.round(100 * byRisk[k] / tot)}%</span></li>`).join("")}</ul>`
    : `<div class="empty-state">No graded records yet.</div>`;

  const decs = Object.entries(byDec).sort((a, b) => b[1] - a[1]);
  const dmax = Math.max(1, ...decs.map(([, c]) => c));
  const dc = (k) => ({ go: "c-ok", cond: "c-rev", refer: "c-det" }[decisionCls(k)]);
  $("pf-dec").innerHTML = decs.map(([k, c]) => `<div class="dec-bar"><div class="top"><span title="${esc(k)}">${esc(decShort(k))}</span><b>${c}</b></div>
      <div class="track"><i class="${dc(k)}" style="width:${(100 * c / dmax).toFixed(1)}%"></i></div></div>`).join("")
    || `<div class="empty-state">No decisions yet.</div>`;

  const { key, dir } = pfSort;
  const val = (r) => { const v = key === "coverage_pct" ? covPct(r[key]) : r[key]; return Array.isArray(v) ? v[0] : v; };
  const sorted = rows.slice().sort((a, b) => {
    const x = val(a), y = val(b);
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * dir;
  });
  $("pf-head").innerHTML = `<tr>${PF_COLS.map(([k, l]) =>
    `<th data-k="${k}" tabindex="0" aria-sort="${k === key ? (dir > 0 ? "ascending" : "descending") : "none"}" class="${k === key ? "sorted" + (dir > 0 ? " asc" : "") : ""}${["condition_score", "repair_high", "valuation_pct", "coverage_pct"].includes(k) ? " num" : ""}">${l}</th>`).join("")}</tr>`;
  $("pf-head").querySelectorAll("th").forEach((th) => {
    th.onclick = () => {
      pfSort = { key: th.dataset.k, dir: pfSort.key === th.dataset.k ? -pfSort.dir : (th.dataset.k === "created_at" ? -1 : 1) };
      renderPortfolio();
    };
    th.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); th.click(); } };
  });
  $("pf-rows").innerHTML = sorted.map((r) => {
    const cov = covPct(r.coverage_pct);
    return `<tr class="click" tabindex="0" data-kind="${esc(r.kind)}" data-id="${esc(r.id)}">
    <td><div class="cell-main" style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.label || r.id)}</div></td>
    <td>${esc(r.property_id || "—")}</td>
    <td class="nowrap muted">${esc(fmtDate(r.created_at, true))}</td>
    <td>${gradeBadge(r.grade)}</td><td class="num">${esc(r.condition_score ?? "—")}</td>
    <td>${r.risk_level ? `<span class="lvl ${esc(r.risk_level)}">${esc(r.risk_level)}</span>` : "—"}</td>
    <td><span class="dec-txt ${decisionCls(r.decision)}" title="${esc(r.decision || "")}">${esc(String(r.decision || "—").split(" - ")[0])}</span></td>
    <td class="num">${moneyRange(r.repair_low, r.repair_high, cur)}</td><td class="num">${pctRange(r.valuation_pct)}</td>
    <td class="num">${cov != null ? esc(cov) + "%" : "—"}</td></tr>`;
  }).join("") || `<tr><td colspan="${PF_COLS.length}" class="empty-state">No walkthroughs yet. Upload a video to inspect a property.</td></tr>`;
  $("pf-rows").querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.onclick = () => {
      if (tr.dataset.kind === "walkthrough") window.open(`/walkthroughs/${encodeURIComponent(tr.dataset.id)}/`, "_blank", "noopener");
      else go(`#/inspection/${tr.dataset.id}`);
    };
    tr.onkeydown = (e) => { if (e.key === "Enter") tr.click(); };
  });
}

// ---- Walkthroughs ----------------------------------------------------------------------------------
let watching = null, walkItems = null;
const WALK_STEPS = ["upload", "slam", "coverage", "ai", "render"];
const WALK_NOTE = "Depth Anything 3 + OpenCV SLAM run on the GPU, then the AI inspects the keyframes — usually several minutes. You can leave this page open; the results open automatically when finished.";

// Walkthroughs uploaded from this browser: only those are re-attached automatically, so other people's
// jobs on a shared demo server don't take over this page (their cards stay clickable).
const MY_WALKS_LS = "pi_my_walks";
function myWalks() { try { return JSON.parse(localStorage.getItem(MY_WALKS_LS) || "[]"); } catch { return []; } }
function rememberWalk(name) {
  try { localStorage.setItem(MY_WALKS_LS, JSON.stringify([...myWalks().filter((n) => n !== name), name].slice(-20))); }
  catch { /* storage blocked */ }
}

async function walks() {
  try {
    walkItems = listOf(await apiJson("/walkthroughs"));
    renderWalkCards();
    // Re-attach to this browser's walkthrough that is still queued or processing (page reload, second tab, ...).
    const mine = myWalks();
    const run = walkItems.find((w) => w.status === "running" && mine.includes(w.name))
      || walkItems.find((w) => w.status === "queued" && mine.includes(w.name));
    if (run && !watching) watchWalk(run.name);
  } catch {                             // offline: don't leave stale "running" spinners in the list
    document.querySelectorAll("#walks .mini-spin, #dash-walks .mini-spin").forEach((x) => x.remove());
    document.querySelectorAll(".wcard.running .wstat-status").forEach((x) => { x.textContent = "server offline"; });
  }
}

function walkCardHtml(w) {
  if (!w.url) {
    const thumb = w.status === "running" ? `<span><span class="mini-spin"></span>Processing…</span>`
      : w.status === "queued" ? `<span>${icon("clock", 16)} Waiting for the GPU…</span>` : icon("video", 28);
    const note = { running: "processing — click to follow progress",
                   queued: "queued: waits for the walkthrough ahead of it — click to follow",
                   interrupted: "interrupted by a server restart — click for details",
                   failed: "failed — click for details" }[w.status] || `${w.status} — click for details`;
    return `<button type="button" class="wcard running" data-walk="${esc(w.name)}" style="text-align:left;font:inherit;padding:0;background:var(--surface);color:var(--text)">
      <div class="wthumb">${thumb}</div>
      <div class="wbody"><h3>${esc(w.name)}</h3><span class="muted small wstat-status">${esc(note)}</span></div></button>`;
  }
  const cov = covPct(w.coverage);
  const nf = Array.isArray(w.findings) ? w.findings.length : w.findings ?? 0;
  return `<a class="wcard" href="${esc(w.url)}" target="_blank" rel="noopener" aria-label="Open 3D walkthrough ${esc(w.name)}">
    <div class="wthumb"><img src="${esc(w.url)}coverage_map.png" alt="Coverage map of ${esc(w.name)}" loading="lazy" onerror="this.remove()">
      ${w.grade ? gradeBadge(w.grade) : ""}<span class="badge3d">3D · ${esc(w.duration_s != null ? Math.round(w.duration_s) + " s" : "video")}</span></div>
    <div class="wbody">
      <div style="display:flex;justify-content:space-between;gap:8px;align-items:center"><h3>${esc(w.name)} <span class="mode-tag">${w.mode === "lite" ? "Lite" : "Advanced"}</span></h3>${icon("ext", 15)}</div>
      ${w.decision ? `<span class="dec-txt ${decisionCls(w.decision)}" style="align-self:flex-start">${esc(decShort(w.decision))}</span>` : ""}
      <div class="wstats"><div><div class="k">Coverage</div><div class="v">${cov != null ? esc(cov) + "%" : "—"}</div></div>
        <div><div class="k">Findings</div><div class="v">${esc(nf)}</div></div>
        <div><div class="k">Rooms</div><div class="v">${Array.isArray(w.rooms) ? w.rooms.length : "—"}</div></div></div>
      ${Array.isArray(w.rooms) && w.rooms.length ? `<span class="muted small">${esc(w.rooms.join(" · "))}</span>` : ""}
    </div></a>`;
}

// Sidebar status line: what the GPU is doing with videos (the product's only job queue).
function paintVideoQueue() {
  const items = walkItems || [];
  const nRun = items.filter((w) => w.status === "running").length, nWait = items.filter((w) => w.status === "queued").length;
  const txt = nRun ? `Processing ${nRun} video${nRun > 1 ? "s" : ""}${nWait ? ` · ${nWait} waiting` : ""}`
    : nWait ? `${nWait} video${nWait > 1 ? "s" : ""} waiting` : lastHealth && !lastHealth.cuda ? "No GPU: videos can't be processed" : "GPU idle";
  $("queue-txt").textContent = txt;
  $("queue-txt").classList.toggle("busy", nRun > 0);
}

function renderWalkCards() {
  paintVideoQueue();
  if (!walkItems) {
    $("walks").innerHTML = $("dash-walks").innerHTML = `<div class="card empty-state">Loading walkthroughs…</div>`;
    return;
  }
  const done = walkItems.filter((w) => w.url).length;
  $("nav-walk-count").textContent = done || "";
  const html = walkItems.map(walkCardHtml).join("");
  $("walks").innerHTML = html || `<div class="card empty-state">No walkthroughs yet — upload a video to build the first 3D inspection.</div>`;
  $("dash-walks").innerHTML = walkItems.filter((w) => w.url).slice(0, 6).map(walkCardHtml).join("")
    || `<div class="card empty-state">No walkthroughs yet. <a href="#/walkthroughs/new">Upload a video</a>.</div>`;
  document.querySelectorAll("[data-walk]").forEach((b) => b.onclick = () => { watchWalk(b.dataset.walk); go("#/walkthroughs"); });
}

function pickWalk() {
  const f = $("walk-file").files[0];
  $("walk-go").disabled = !f || !!watching;
  $("walk-drop-text").textContent = f ? f.name : "Drop a video here or browse";
  if (f) $("walk-drop").classList.add("has-file");
}

function setWalkStep(step) {
  const idx = WALK_STEPS.indexOf(step);
  document.querySelectorAll("#walk-steps li").forEach((li, i) => {
    li.classList.toggle("done", i < idx); li.classList.toggle("active", i === idx);
  });
  const li = $("walk-steps").querySelector(`[data-step="${step}"] .st`);
  setStageText("walk", li ? li.firstChild.textContent : step);
}

// Infer the current step from the log of scripts/inspect_walkthrough.py.
function walkStepFromLog(lines) {
  const txt = lines.join("\n");
  if (/rendering live map|building 3D GLBs|findings placed in 3D/.test(txt)) return "render";
  if (/AI keyframes|INS-\d+/.test(txt)) return "ai";
  if (/coverage over|SLAM run:|converting SLAM videos/.test(txt)) return "coverage";
  return "slam";
}

const cleanLine = (l) => l.replace(/\u001b\[[0-9;]*m/g, "").trim();
const isNoise = (l) => !l || /triton|gsplat|Traceback|File "|^\s*\^|import triton|ModuleNotFoundError|warnings\.warn/i.test(l);

function startWalkPanel(name, videoUrl) {
  runState.walk = { t0: Date.now(), name, videoUrl: videoUrl || null, log: "starting…", prog: null, stageText: "" };
  $("walk-go").disabled = true; $("walk-go").classList.add("busy");
  setProgress("walk", 0, "Uploading video", 0, null);
  if (panelOwner === "walk") paintOwner("walk");
  placeRunning();
  startClock();
}

function stopWalkPanel() {
  watching = null; stopClockIfIdle();
  $("walk-go").disabled = !$("walk-file").files.length; $("walk-go").classList.remove("busy");
  runState.walk.prog = null;
  if (panelOwner === "walk") $("run-vid").pause();
  placeRunning();
}

function watchWalk(name, videoUrl) {
  if (watching === name) return;
  watching = name;
  // No local copy of the video (page reloaded, other tab, started elsewhere): play the stored upload.
  startWalkPanel(name, videoUrl || withKey(`${API}/walkthrough/${encodeURIComponent(name)}/video`));
  setWalkStep("slam");
  const view = `/walkthroughs/${name}/`;
  let misses = 0, fails = 0;
  const tick = async () => {
    if (watching !== name) return;
    let l;
    try {
      const r = await api(`/walkthrough/${name}/log`);
      if (r.status === 404) {                       // log file not created yet: job is still starting
        if (++misses < 60) { setTimeout(tick, 3000); return; }
        throw new Error("no progress log");
      }
      if (!r.ok) throw new Error(r.statusText);
      l = await r.json();
      fails = 0;
    } catch (err) {                     // transient blip: retry; server gone: stop the animation
      if (++fails < 4) { setTimeout(tick, 4000); return; }
      stopWalkPanel(); $("walk-status").textContent = LOST; return;
    }
    const lines = (l.tail || []).map(cleanLine);
    const useful = lines.filter((x) => !isNoise(x));
    runState.walk.log = useful.slice(-6).join("\n") || "starting…";
    if (panelOwner === "walk") $("run-log").textContent = runState.walk.log;
    $("walk-status").innerHTML = `<span class="mini-spin"></span>${esc(name)}: ${esc(l.status)}`;
    if (l.status === "queued") {                  // another walkthrough has the GPU: say so instead of a silent wait
      const msg = queueText(l.queue || {});
      setWalkStep("slam"); setStageText("walk", "Waiting for the GPU");
      $("walk-status").innerHTML = `<span class="mini-spin"></span>${esc(name)}: ${esc(msg)}`;
      const waited = (l.queue && l.queue.waiting_s) || 0;
      runState.walk.t0 = Date.now() - waited * 1000;
      setProgress("walk", 0, msg, waited, null);
      setTimeout(tick, 4000); return;
    }
    if (l.status === "running") {
      setWalkStep(walkStepFromLog(lines));
      const p = l.progress;
      if (p) {
        runState.walk.t0 = Date.now() - p.elapsed_s * 1000;   // server clock: survives page reloads
        setProgress("walk", p.percent, p.steps ? `Step ${p.step}/${p.steps}: ${p.phase}` : p.phase, p.elapsed_s,
                    p.elapsed_s < 20 && !p.step ? null : p.eta_s);
      }
      setTimeout(tick, 4000); return;
    }
    if (l.progress) setProgress("walk", 100, "Done", l.progress.elapsed_s, 0);
    stopWalkPanel(); walks();
    if (l.status === "done") {
      $("walk-status").innerHTML = `${esc(name)}: done — <a href="${esc(view)}" target="_blank" rel="noopener">` +
        `<strong>Open results (3D model, live map, risk report) ↗</strong></a>`;
      if (currentView === "walkthroughs") {
        $("walk-status").innerHTML += " — opening…";
        setTimeout(() => { location.href = view; }, 1500);   // a popup would be blocked after the long wait
      } else toast(`Walkthrough <b>${esc(name)}</b> is ready — <a href="${esc(view)}">open 3D results</a>`, 20000);
    } else if (l.status === "interrupted" || l.status === "failed") {
      $("walk-status").innerHTML = (l.status === "interrupted"
        ? `${esc(name)}: stopped before finishing (the server was restarted). `
        : `${esc(name)}: failed — ${esc(useful.slice(-1)[0] || "")} `) +
        `<a href="#" data-retry="${esc(name)}"><strong>Process it again</strong></a> (no re-upload; finished 3D steps are reused).`;
      $("walk-status").querySelector("[data-retry]").onclick = (e) => { e.preventDefault(); retryWalk(name); };
    } else {
      $("walk-status").textContent = l.status === "unknown"
        ? `${name}: stopped before finishing (the server was restarted). Upload the video again — finished 3D steps are reused.`
        : `${name}: ${l.status} — ${useful.slice(-1)[0] || ""}`;
    }
  };
  tick();
}

// "Waiting: another walkthrough is processing, started 7 min ago (about 4 min left)"
function queueText(q) {
  if (!q.running) return "Waiting for the GPU — starting shortly";
  const el = q.running_elapsed_s || 0;
  let s = `Waiting: another walkthrough is processing, started ${el < 60 ? "just now" : `${Math.round(el / 60)} min ago`}`;
  if (q.running_eta_s > 0) s += ` (about ${Math.max(1, Math.round(q.running_eta_s / 60))} min left)`;
  if (q.ahead > 1) s += ` · ${q.ahead - 1} more queued before yours`;
  return s;
}

async function retryWalk(name) {
  $("walk-status").innerHTML = `<span class="mini-spin"></span>Restarting ${esc(name)}…`;
  try {
    const r = await api(`/walkthrough/${encodeURIComponent(name)}/retry`, { method: "POST" });
    let j = null; try { j = await r.json(); } catch { /* non-JSON body */ }
    if (!r.ok && r.status !== 409) throw new Error(errText(r, j));
    rememberWalk(name); watchWalk(name); go("#/walkthroughs");
  } catch (err) { $("walk-status").textContent = "Failed: " + err.message; }
}

async function submitWalk(e) {
  e.preventDefault();
  const f = $("walk-file").files[0]; if (!f) return;
  const fd = new FormData(); fd.append("video", f);
  const prop = $("walk-property").value.trim();
  if (prop) fd.append("property_id", prop);
  const mode = (document.querySelector('input[name="walk-mode"]:checked') || {}).value || "advanced";
  fd.append("mode", mode);
  const videoUrl = URL.createObjectURL(f);
  const base = (f.name.replace(/\.[^.]*$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 55)) || "walkthrough";
  const name = mode === "lite" ? base + "_lite" : base;   // same naming as the server
  watching = null;
  startWalkPanel(name, videoUrl);
  watching = "upload:" + name; placeRunning();
  setWalkStep("upload");
  $("walk-status").innerHTML = `<span class="mini-spin"></span>Uploading…`;
  $("running").scrollIntoView({ behavior: "smooth", block: "start" });   // the progress card sits above the form
  try {
    const r = await api("/walkthrough", { method: "POST", body: fd });
    let j = null; try { j = await r.json(); } catch { /* non-JSON body */ }
    watching = null;
    if (r.status === 409) { rememberWalk(name); watchWalk(name, videoUrl); return; }   // already queued/running: show its progress
    if (!r.ok) throw new Error(errText(r, j));
    rememberWalk(j.name);
    watchWalk(j.name, videoUrl);
  } catch (err) { stopWalkPanel(); $("walk-status").textContent = "Failed: " + err.message; }
}

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = false; }
function hideError() { $("error").hidden = true; }
init();
