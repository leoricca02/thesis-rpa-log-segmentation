/* RPA Log Segmentation — demo UI.
 * Vanilla JS, no build step. The backend streams pipeline events over SSE;
 * this file turns them into the four views: Input → Routines → Pipeline → Result.
 */
"use strict";

// ------------------------------------------------------------------ constants
const VIEWS = ["input", "routines", "pipeline", "result"];
const VIEW_LABELS = { input: "Input log", routines: "Routines", pipeline: "Pipeline", result: "Result" };

// The three phases the UI presents. The backend still streams its finer-grained
// steps (`parts`); each one is folded into the phase it belongs to.
const PHASES = [
  { id: "0", code: "0", title: "Cleaning & narrative", kind: "llm",
    parts: ["load", "0A", "0B", "0B2", "serialize", "0A2"],
    desc: "Phase 0 cleans the raw log and turns it into a natural-language narrative.",
    policy: "Every event becomes one sentence with a stable node id, the only input Phases 1 and 2 read." },
  { id: "1", code: "1", title: "Shared actions", kind: "llm",
    parts: ["1", "review"],
    desc: "For every shared action — login, module openers, logout — the model infers which subset of routines depends on it. Assumption A1′: an action may be shared by any subset, not only by all routines. Reasoning is written before the answer; with human review on, you approve or edit the subsets before Phase 2.",
    policy: "Fail-fast on an API error. Hallucinated node ids and unknown routine names are dropped." },
  { id: "2", code: "2", title: "Execution routing", kind: "llm",
    parts: ["2"],
    desc: "The remaining events are distributed into exactly the execution buckets you declared: routine meaning decides which routine, payload continuity decides which execution. The result is validated as a cover, and each trace receives only the shared actions its subset allows.",
    policy: "A node in two executions aborts the run; a node routed nowhere goes to the Noise sheet." },
];
const PHASE_BY_ID = Object.fromEntries(PHASES.map((p) => [p.id, p]));
const PHASE_OF_PART = Object.fromEntries(PHASES.flatMap((p) => p.parts.map((part) => [part, p.id])));
const CASE_STUDY_ROUTINES = [
  { name: "Travel Authorization", executions: 2 },
  { name: "Expense Reimbursement", executions: 2 },
  { name: "Purchase Order Approval", executions: 2 },
  { name: "Student Grant Disbursement", executions: 2 },
];
const MAX_ROUTINES = 8;
// Size factor applied to every measure in styles.css; the untangle geometry uses it too.
const UI_SCALE = 1.35;

// ---------------------------------------------------------------------- state
// Only `review` is a visible choice; the other options are fixed for every live run.
// No cache reuse: every call is a fresh inference, so the model's thinking is always shown.
function defaultOptions(model = "") {
  return { review: true, relevance_filter: true, include_thoughts: true, use_cache: false, model };
}

const S = {
  view: "input",
  maxView: 0,
  status: { api_key: false, model: "" },
  recordings: [],
  log: null,          // preview of the selected log
  source: null,       // {kind: 'sample'|'upload'|'recording', id}
  recording: null,    // recording details when replaying
  routines: [{ name: "", executions: 1 }],
  options: defaultOptions(),
  run: null,
  focus: null,
  follow: true,
  activeTrace: 0,
};

// -------------------------------------------------------------------- helpers
const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtInt = (n) => (n ?? 0).toLocaleString("en-US");
const fmtSec = (s) => (s < 10 ? s.toFixed(1) : Math.round(s)) + "s";
const seriesVar = (i) => `var(--series-${(i % 8) + 1})`;

function mdLite(text) {
  const html = esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return html.split(/\n\s*\n/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

function toast(msg) {
  const t = h("div", { class: "toast" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 2600);
}

const ICON = {
  file: '<svg viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 5v14l11-7z"/></svg>',
};

// ------------------------------------------------------------------- tooltip
const tip = $("#tooltip");
function showTip(e, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };

// --------------------------------------------------------------------- modal
function openModal(title, body) {
  $("#modalTitle").textContent = title;
  const b = $("#modalBody");
  b.innerHTML = "";
  b.append(body);
  $("#modal").hidden = false;
}
$("#modalClose").addEventListener("click", () => ($("#modal").hidden = true));
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#modal").hidden = true; });

// ---------------------------------------------------------------- typewriter
// Text reveals progressively and survives re-renders: progress is kept per key.
const TW = { shown: {}, full: {} };
function typewriter(key, text, cls = "") {
  TW.full[key] = text;
  if (!(key in TW.shown)) TW.shown[key] = 0;
  const el = h("div", { class: "tw " + cls, "data-tw": key });
  paintTw(el, key);
  return el;
}
function paintTw(el, key) {
  const full = TW.full[key] || "";
  const n = Math.min(TW.shown[key], full.length);
  el.innerHTML = mdLite(full.slice(0, n)) + (n < full.length ? '<span class="caret"></span>' : "");
}
setInterval(() => {
  document.querySelectorAll("[data-tw]").forEach((el) => {
    const key = el.dataset.tw;
    const full = TW.full[key] || "";
    if (TW.shown[key] >= full.length) return;
    TW.shown[key] += Math.max(3, Math.ceil(full.length / 260));
    paintTw(el, key);
  });
}, 30);

// ---------------------------------------------------------------- navigation
function go(view) {
  const idx = VIEWS.indexOf(view);
  S.view = view;
  S.maxView = Math.max(S.maxView, idx);
  for (const v of VIEWS) $(`#view-${v}`).hidden = v !== view;
  renderStepper();
  if (view === "routines") renderRoutines();
  if (view === "pipeline") renderPipeline();
  if (view === "result") renderResult();
  scrollTo({ top: 0, behavior: "smooth" });
}

function renderStepper() {
  const nav = $("#stepper");
  nav.innerHTML = "";
  VIEWS.forEach((v, i) => {
    if (i) nav.append(h("span", { class: "step-sep" }));
    const done = i <= S.maxView && v !== S.view;
    nav.append(
      h("button", {
        class: `step ${v === S.view ? "active" : ""} ${done ? "done" : ""}`,
        onclick: () => { if (i <= S.maxView) go(v); },
      }, h("span", { class: "num" }, i + 1), VIEW_LABELS[v])
    );
  });
}

// ------------------------------------------------------------------ theme
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("rpa-ui-theme", theme); } catch (_) { /* private mode */ }
}
$("#themeToggle").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  if (S.view === "result") renderUntangle(false);
});
(() => {
  let theme = "dark";
  try { theme = localStorage.getItem("rpa-ui-theme") || "dark"; } catch (_) { /* ignore */ }
  applyTheme(theme);
})();

// ================================================================ 1 · INPUT
async function loadInputs() {
  try {
    S.status = await api("/api/status");
    S.options.model = S.status.model;
    $("#modelBadge").textContent = S.status.model;
  } catch (_) { /* server offline */ }
  await refreshRecordings();
}

async function refreshRecordings() {
  try { S.recordings = await api("/api/recordings"); } catch (_) { S.recordings = []; }
  renderRecordingList();
}

function renderRecordingList() {
  const list = $("#recordingList");
  list.innerHTML = "";
  for (const r of S.recordings) {
    const selected = S.source && S.source.kind === "recording" && S.source.id === r.id;
    const n = (r.routines || []).length;
    list.append(
      h("button", { class: `list-item ${selected ? "selected" : ""}`, onclick: () => selectRecording(r) },
        h("span", { class: "li-icon", html: ICON.play }),
        h("span", { class: "li-main" },
          h("div", { class: "li-name" }, r.title || r.id),
          h("div", { class: "li-sub" }, `${r.log_name} · ${n} routines · ${r.model} · ${fmtSec(r.duration || 0)}`)))
    );
  }
  if (!S.recordings.length) {
    list.append(h("div", { class: "muted-empty" }, "No recordings yet. Finish a live run, then “Save as recording”."));
  }
}

async function selectRecording(rec) {
  try {
    const detail = await api(`/api/recordings/${encodeURIComponent(rec.id)}`);
    S.recording = detail;
    setLog(detail.preview, { kind: "recording", id: rec.id });
    const start = detail.run_start;
    S.routines = (start.routines || []).map((r) => ({ name: r.routine_name, executions: r.executions }));
    S.options = { ...S.options, ...(start.options || {}), model: start.model };
  } catch (err) { toast(err.message); }
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  try {
    const preview = await api("/api/logs", { method: "POST", body: form });
    setLog(preview, { kind: "upload", id: preview.log_id });
  } catch (err) { toast(err.message); }
}

function setLog(preview, source) {
  const wasRecording = S.source && S.source.kind === "recording";
  S.log = preview;
  S.source = source;
  if (source.kind !== "recording") {
    S.recording = null;
    if (wasRecording) {
      S.routines = [{ name: "", executions: 1 }];
      S.options = defaultOptions(S.status.model);
    }
  }
  S.maxView = 0;
  renderRecordingList();
  renderPreview();
}

function renderPreview() {
  const card = $("#previewCard");
  const L = S.log;
  card.innerHTML = "";
  const apps = Object.keys(L.apps || {});
  const kind = { sample: "Thesis log", upload: "Uploaded", recording: "Recording" }[S.source.kind];
  card.append(
    h("div", { class: "preview-head" },
      h("h2", { title: L.name }, L.name),
      h("span", { class: `badge ${S.source.kind === "recording" ? "replay" : "subtle"}` }, kind)),
    h("div", { class: "kpis inset" },
      kpi(fmtInt(L.rows), "events"),
      kpi(`${L.populated_columns}/${L.columns}`, "columns populated"),
      kpi(apps.length, apps.length === 1 ? "application" : "applications"),
      kpi(L.span || "—", "time span")),
  );

  const ribbon = h("div", { class: "ribbon" });
  for (const ev of L.events) {
    const tick = h("div", { class: "tick" });
    tick.addEventListener("mousemove", (e) => showTip(e, eventTip(ev)));
    tick.addEventListener("mouseleave", hideTip);
    ribbon.append(tick);
  }
  card.append(
    h("div", { class: "section-title" }, "The raw, interleaved log"),
    h("div", { class: "ribbon-wrap" }, ribbon,
      h("div", { class: "ribbon-caption" },
        h("span", {}, "Each bar is one event, in recorded order. Nothing says which execution it belongs to."),
        h("span", {}, `${fmtInt(L.rows)} events`))),
  );

  const tbody = h("tbody");
  for (const ev of L.events) {
    tbody.append(h("tr", {},
      h("td", { class: "num" }, ev.i),
      h("td", { class: "mono" }, ev.time),
      h("td", {}, h("span", { class: "app-chip" }, ev.app)),
      h("td", {}, h("span", { class: "type-chip" }, ev.type)),
      h("td", {}, ev.label)));
  }
  card.append(
    h("div", { class: "table-wrap" },
      h("table", {}, h("thead", {}, h("tr", {}, h("th", { class: "num" }, "#"), h("th", {}, "Time"), h("th", {}, "App"), h("th", {}, "Event"), h("th", {}, "Content"))), tbody)),
    h("div", { class: "preview-actions" },
      h("button", { class: "btn primary", onclick: () => go("routines") }, "Continue to the routines →")),
  );
}

function kpi(value, label) {
  return h("div", { class: "kpi" }, h("div", { class: "kpi-value" }, value), h("div", { class: "kpi-label" }, label));
}

function eventTip(ev) {
  return `<div class="tt-title">${esc(ev.type)}${ev.label ? " · " + esc(ev.label) : ""}</div>
    <div class="tt-sub">#${ev.i} · ${esc(ev.time)} · ${esc(ev.app)}${ev.where ? " · " + esc(ev.where) : ""}</div>`;
}

// Drag & drop
const dz = $("#dropzone");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("over");
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});
$("#fileInput").addEventListener("change", (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ""; });

// =============================================================== 2 · ROUTINES
const isReplay = () => S.source && S.source.kind === "recording";

function renderRoutines() {
  const replay = isReplay();
  const list = $("#routineList");
  list.innerHTML = "";
  S.routines.forEach((r, i) => {
    const input = h("input", {
      class: "name", value: r.name, placeholder: "e.g. Process Refund", maxlength: 80,
      disabled: replay, "aria-label": `Name of routine ${i + 1}`,
      oninput: (e) => { r.name = e.target.value; renderRunSummary(); },
    });
    list.append(
      h("div", { class: "routine" },
        h("span", { class: "swatch", style: { background: seriesVar(i) } }),
        h("div", {}, h("div", { class: "rname-label" }, `Routine ${i + 1}`), input),
        h("div", { class: "counter-wrap" },
          h("div", { class: "counter" },
            h("button", { disabled: replay, "aria-label": "Fewer executions", onclick: () => { r.executions = Math.max(1, r.executions - 1); renderRoutines(); } }, "−"),
            h("span", { class: "val" }, r.executions),
            h("button", { disabled: replay, "aria-label": "More executions", onclick: () => { r.executions = Math.min(50, r.executions + 1); renderRoutines(); } }, "+")),
          h("span", { class: "counter-label" }, r.executions === 1 ? "execution" : "executions")),
        h("button", {
          class: "remove-btn", title: "Remove routine", disabled: replay || S.routines.length === 1,
          onclick: () => { S.routines.splice(i, 1); renderRoutines(); },
        }, "✕"))
    );
  });
  $("#addRoutine").hidden = replay;
  $("#addRoutine").disabled = S.routines.length >= MAX_ROUTINES;
  const isCase = S.log && /^caso_studio_trasferte/.test(S.log.name);
  $("#fillCaseStudy").hidden = replay || !isCase;
  $("#replayNote").hidden = !replay;
  renderOptions();
  renderRunSummary();
}

$("#addRoutine").addEventListener("click", () => {
  if (S.routines.length < MAX_ROUTINES) S.routines.push({ name: "", executions: 1 });
  renderRoutines();
  const inputs = document.querySelectorAll("#routineList input.name");
  inputs[inputs.length - 1]?.focus();
});
$("#fillCaseStudy").addEventListener("click", () => {
  S.routines = CASE_STUDY_ROUTINES.map((r) => ({ ...r }));
  renderRoutines();
});

function renderOptions() {
  const replay = isReplay();
  const box = $("#options");
  box.innerHTML = "";
  const input = h("input", { type: "checkbox", disabled: replay, onchange: (e) => { S.options.review = e.target.checked; } });
  input.checked = !!S.options.review;
  box.append(h("label", { class: "option" },
    h("div", {},
      h("div", { class: "option-title" }, "Human review of the shared actions"),
      h("div", { class: "option-desc" }, "Pause after Phase 1 to approve or edit which routines share each action.")),
    h("span", { class: "switch" }, input, h("span"))));
}

function renderRunSummary() {
  const replay = isReplay();
  const named = S.routines.filter((r) => r.name.trim()).length;
  const execs = S.routines.reduce((a, r) => a + r.executions, 0);
  const box = $("#runSummary");
  box.innerHTML = "";
  box.append(kpi(S.routines.length, S.routines.length === 1 ? "routine" : "routines"), kpi(execs, "executions expected"));
  if (!replay && !S.status.api_key) {
    box.append(h("div", { class: "key-warning", style: { gridColumn: "1 / -1" } },
      "GEMINI_API_KEY is not set, so live runs are disabled. Add it to .env, or replay a recording."));
  }
  if (replay && S.recording?.meta?.source) {
    box.append(h("div", { class: "note", style: { gridColumn: "1 / -1" } }, S.recording.meta.source));
  }
  $("#speedRow").hidden = !replay;
  const btn = $("#runBtn");
  btn.textContent = replay ? "▶ Start replay" : "Run segmentation";
  btn.disabled = !replay && (!S.status.api_key || named !== S.routines.length);
}

$("#runBtn").addEventListener("click", startRun);

async function startRun() {
  const err = $("#runError");
  err.hidden = true;
  try {
    let res;
    if (isReplay()) {
      res = await api("/api/replays", { method: "POST", body: JSON.stringify({ recording_id: S.source.id, speed: Number($("#speed").value) }) });
    } else {
      const names = S.routines.map((r) => r.name.trim().toLowerCase());
      if (new Set(names).size !== names.length) throw new Error("Routine names must be distinct.");
      res = await api("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          log_id: S.source.id,
          routines: S.routines.map((r) => ({ routine_name: r.name.trim(), executions: r.executions })),
          ...S.options,
        }),
      });
    }
    beginRun(res.run_id, isReplay());
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  }
}

// ============================================================= 3 · PIPELINE
function newRun(id, replay) {
  const phases = {};
  for (const part of Object.keys(PHASE_OF_PART)) phases[part] = { status: "pending", seconds: null, data: null, calls: [] };
  return {
    id, replay, lastSeq: -1, start: null, raw: null, phases, current: null,
    logs: [], nodes: [], nodeById: {}, review: null, error: null,
    done: false, success: false, result: null, topology: null,
    tokens: { input: 0, output: 0, thoughts: 0 }, calls: 0, hits: 0,
    startedAt: Date.now(), endedAt: null,
  };
}

let stream = null;
function beginRun(id, replay) {
  if (stream) stream.close();
  S.run = newRun(id, replay);
  S.focus = null;
  S.follow = true;
  S.activeTrace = 0;
  for (const k of Object.keys(TW.shown)) delete TW.shown[k];
  const badge = $("#modeBadge");
  badge.hidden = false;
  badge.className = `badge ${replay ? "replay" : "live"}`;
  badge.textContent = replay ? "REPLAY" : "LIVE";
  S.maxView = 2;
  go("pipeline");
  stream = new EventSource(`/api/runs/${id}/stream`);
  stream.onmessage = (m) => onEvent(JSON.parse(m.data));
  stream.addEventListener("end", () => stream.close());
}

function onEvent(e) {
  const R = S.run;
  if (!R || e.seq <= R.lastSeq) return;
  R.lastSeq = e.seq;
  const P = R.phases;
  let detailDirty = false;

  switch (e.type) {
    case "run_start":
      R.start = e;
      $("#modelBadge").textContent = e.model;
      if (!e.options.review) P.review.status = "skipped";
      break;
    case "raw_log":
      R.raw = e.data;
      P.load.data = e.data;
      detailDirty = true;
      break;
    case "phase": {
      const ph = P[e.phase];
      if (!ph) break;
      if (e.status === "start") {
        ph.status = "running";
        R.current = e.phase;
        if (e.phase === "2" && P.review.status === "pending") P.review.status = "skipped";
        if (S.follow) S.focus = PHASE_OF_PART[e.phase];
      } else {
        ph.status = e.status;
        ph.seconds = e.seconds ?? null;
      }
      detailDirty = true;
      break;
    }
    case "llm_call":
      (P[e.phase]?.calls || []).push({ ...e, status: "running", startedAt: Date.now() });
      R.calls += 1;
      if (e.cached) R.hits += 1;
      detailDirty = true;
      break;
    case "llm_result":
    case "llm_error": {
      const calls = P[e.phase]?.calls || [];
      const call = calls[calls.length - 1];
      if (call) {
        Object.assign(call, e, { status: e.type === "llm_error" ? "error" : "done" });
        call.type = "llm_call";
      }
      if (e.tokens) {
        R.tokens.input += e.tokens.input || 0;
        R.tokens.output += e.tokens.output || 0;
        R.tokens.thoughts += e.tokens.thoughts || 0;
      }
      detailDirty = true;
      break;
    }
    case "phase_data":
      if (P[e.phase]) P[e.phase].data = e.data;
      if (e.phase === "serialize") {
        R.nodes = e.data.nodes;
        R.nodeById = Object.fromEntries(R.nodes.map((n) => [n.id, n]));
      }
      if (e.phase === "1") R.topology = e.data.topology;
      if (e.phase === "2") R.result = e.data;
      detailDirty = true;
      break;
    case "review_required":
      R.review = {
        topology: e.topology, routines: e.routines, replay: !!e.replay, submitted: false,
        sel: Object.fromEntries(e.topology.map((a) => [a.node_id, new Set(a.shared_with)])),
        orig: Object.fromEntries(e.topology.map((a) => [a.node_id, new Set(a.shared_with)])),
      };
      P.review.status = "waiting";
      S.focus = "1";
      S.follow = true;
      if (S.view !== "pipeline") go("pipeline");
      detailDirty = true;
      break;
    case "review_resolved":
      if (R.review) {
        R.review.resolved = e;
        R.review.submitted = true;
      }
      R.topology = e.topology;
      detailDirty = true;
      break;
    case "error":
      R.error = e;
      if (P[e.phase]) P[e.phase].status = "error";
      S.focus = PHASE_OF_PART[e.phase] || S.focus;
      detailDirty = true;
      break;
    case "done":
      R.done = true;
      R.success = e.success;
      R.endedAt = Date.now();
      $("#modeBadge").hidden = true;
      if (e.success) S.maxView = 3;
      renderStepper();
      detailDirty = true;
      break;
  }
  if (S.view === "pipeline") {
    renderStats();
    renderTimeline();
    if (detailDirty) renderPhaseDetail();
  }
}

function renderPipeline() {
  renderStats();
  renderTimeline();
  renderPhaseDetail();
}

function elapsed(R) {
  return ((R.endedAt || Date.now()) - R.startedAt) / 1000;
}

function renderStats() {
  const R = S.run;
  const bar = $("#statsbar");
  if (!R) { bar.innerHTML = ""; return; }
  const routines = R.start?.routines || [];
  const execs = routines.reduce((a, r) => a + r.executions, 0);
  const stat = (v, l) => h("div", { class: "stat" }, h("span", { class: "stat-value" }, v), h("span", { class: "stat-label" }, l));
  bar.innerHTML = "";
  bar.append(
    h("div", { class: "run-title" }, R.start?.log_name || "Starting…",
      h("small", {}, `${routines.length} routines · ${execs} executions · ${R.start?.model || ""}`)),
    h("span", { class: "spacer" }),
    stat(h("span", { "data-elapsed": "1" }, fmtSec(elapsed(R))), "elapsed"),
    stat(R.calls, "LLM calls"),
    stat(R.hits, "cache hits"),
    stat(fmtInt(R.tokens.input), "tokens in"),
    stat(fmtInt(R.tokens.output + R.tokens.thoughts), "tokens out"),
  );
  if (R.done && R.success) {
    bar.append(h("button", { class: "btn primary", onclick: () => go("result") }, "View result →"));
  } else if (!R.done) {
    bar.append(h("button", { class: "btn ghost small", onclick: cancelRun }, "Stop"));
  }
}

async function cancelRun() {
  if (!S.run) return;
  try { await api(`/api/runs/${S.run.id}/cancel`, { method: "POST" }); } catch (_) { /* ignore */ }
}

setInterval(() => {
  const R = S.run;
  if (!R || R.done || S.view !== "pipeline") return;
  const el = document.querySelector("[data-elapsed]");
  if (el) el.textContent = fmtSec(elapsed(R));
  document.querySelectorAll("[data-wait]").forEach((w) => {
    w.textContent = fmtSec((Date.now() - Number(w.dataset.wait)) / 1000);
  });
}, 200);

function phaseTokens(calls) {
  return calls.reduce((a, c) => a + ((c.tokens?.input || 0) + (c.tokens?.output || 0) + (c.tokens?.thoughts || 0)), 0);
}

// Fold the backend steps of a phase into one status, duration and call list.
function phaseState(R, p) {
  const parts = p.parts.map((id) => R.phases[id]);
  const statuses = parts.map((ph) => ph.status);
  const later = PHASES.slice(PHASES.indexOf(p) + 1).some((q) => q.parts.some((id) => R.phases[id].status !== "pending"));
  let status = "pending";
  if (statuses.includes("error")) status = "error";
  else if (statuses.includes("waiting")) status = "waiting";
  else if (later || statuses.every((st) => st === "done" || st === "skipped")) status = "done";
  else if (statuses.some((st) => st !== "pending")) status = "running";
  const timed = parts.filter((ph, i) => p.parts[i] !== "review" && ph.seconds !== null);
  return {
    status,
    seconds: timed.length ? timed.reduce((a, ph) => a + ph.seconds, 0) : null,
    calls: parts.flatMap((ph) => ph.calls),
  };
}

function renderTimeline() {
  const R = S.run;
  const tl = $("#timeline");
  tl.innerHTML = "";
  if (!R) return;
  for (const p of PHASES) {
    const ph = phaseState(R, p);
    const meta = [];
    if (ph.status === "running") meta.push(h("span", {}, "running…"));
    if (ph.status === "waiting") meta.push(h("span", {}, "waiting for you"));
    if (ph.status === "error") meta.push(h("span", {}, "failed"));
    if (ph.seconds !== null && ph.status === "done") meta.push(h("span", {}, fmtSec(ph.seconds)));
    const call = ph.calls[ph.calls.length - 1];
    if (call && call.status !== "running") {
      const tok = phaseTokens(ph.calls);
      if (ph.calls.every((c) => c.cached)) meta.push(h("span", {}, "cache hit"));
      else if (tok) meta.push(h("span", {}, `${fmtInt(tok)} tok`));
    }
    const icon = ph.status === "done" ? "✓" : ph.status === "error" ? "!" : p.code;
    tl.append(
      h("div", {
        class: `tl-item ${ph.status} ${S.focus === p.id ? "focused" : ""}`,
        onclick: () => { S.focus = p.id; S.follow = R.done || p.id === PHASE_OF_PART[R.current]; renderTimeline(); renderPhaseDetail(); },
      },
        h("div", { class: "tl-dot" }, icon),
        h("div", {},
          h("div", { class: "tl-title" }, p.title, h("span", { class: `kind ${p.kind}` }, p.kind === "llm" ? "LLM" : p.kind)),
          h("div", { class: "tl-meta" }, h("span", { class: "mono" }, `Phase ${p.code}`), ...meta)))
    );
  }
}

// ------------------------------------------------------------ phase detail
function renderPhaseDetail() {
  const R = S.run;
  const box = $("#phaseDetail");
  const scroll = box.scrollTop;
  box.innerHTML = "";
  if (!R) return;
  const pid = S.focus || PHASE_OF_PART[R.current] || "0";
  const p = PHASE_BY_ID[pid];
  const ph = phaseState(R, p);

  box.append(h("div", { class: "pd-head" },
    h("div", { class: "pd-code" }, p.code),
    h("div", {},
      h("h2", {}, p.title, " ", h("span", { class: `kind ${p.kind}` }, p.kind === "llm" ? "LLM" : p.kind)),
      h("p", { class: "pd-desc" }, p.desc),
      h("p", { class: "pd-policy" }, p.policy))));

  if (ph.status === "pending") {
    box.append(h("div", { class: "section note" }, "Waiting for the earlier phases."));
  } else if (pid === "0" && ph.status === "running") {
    box.append(h("div", { class: "section waiting-model" },
      h("span", { class: "dots" }, h("i"), h("i"), h("i")), h("span", {}, "Processing trace events…")));
  }

  // Phase 0 shows only its output; Phases 1 and 2 also show their model call.
  if (pid !== "0") for (const call of ph.calls) box.append(renderCall(pid, call));

  const renderer = PHASE_RENDERERS[pid];
  if (renderer) {
    const content = renderer(R);
    if (content) box.append(content);
  }

  if (R.error && PHASE_OF_PART[R.error.phase] === pid) {
    box.append(h("div", { class: "error-box" },
      h("b", {}, "Run stopped. "), R.error.message,
      R.error.details?.length ? h("ul", {}, R.error.details.map((d) => h("li", {}, d))) : null));
  }
  if (pid === "2" && R.done && R.success) {
    box.append(h("div", { class: "done-cta" },
      h("div", {}, h("strong", {}, "Segmentation complete. "), "Every event is in a trace or in the Noise sheet."),
      h("button", { class: "btn primary", onclick: () => go("result") }, "View result →")));
  }
  box.scrollTop = scroll;
}

function renderCall(pid, call) {
  const wrap = h("div", { class: "section" });
  const thinkingLabel = call.thinking?.thinkingLevel
    ? `thinking: ${call.thinking.thinkingLevel}`
    : call.thinking?.thinkingBudget ? `thinking budget: ${call.thinking.thinkingBudget}` : "no thinking";
  const pills = [
    h("span", { class: "pill" }, call.model),
    h("span", { class: "pill" }, thinkingLabel),
    call.status === "error" ? h("span", { class: "pill err" }, "API error")
      : h("span", { class: `pill ${call.cached ? "hit" : "miss"}` }, call.cached ? "cache hit" : "live call"),
  ];
  if (call.status === "done" && !call.cached && call.tokens?.total) {
    pills.push(h("span", { class: "pill" }, `${fmtInt(call.tokens.input)} in · ${fmtInt(call.tokens.output)} out`
      + (call.tokens.thoughts ? ` · ${fmtInt(call.tokens.thoughts)} thinking` : "")));
  }
  if (call.status === "done" && call.seconds !== undefined) pills.push(h("span", { class: "pill" }, fmtSec(call.seconds)));
  wrap.append(
    h("div", { class: "section-title" }, "Model call"),
    h("div", { class: "call" }, ...pills, h("span", { style: { flex: 1 } }),
      h("button", { class: "link-btn", onclick: () => showPrompt(call) }, "View prompt")));

  const lowThinking = call.thinking?.thinkingLevel === "minimal" || call.thinking?.thinkingBudget === 0;
  const box = h("div", { class: "thinking" },
    h("div", { class: "thinking-head" }, "Model thought summary"));
  if (call.status === "running") {
    const nEvents = (call.user_prompt.match(/^Node \d+:/gm) || []).length;
    box.append(h("div", { class: "waiting-model" },
      h("span", { class: "dots" }, h("i"), h("i"), h("i")),
      h("span", {}, call.cached ? "Replaying a cached response…" : `Processing trace events… ${nEvents ? `${nEvents} events · ` : ""}${call.model} · `,
        h("span", { "data-wait": String(call.startedAt) }, "0.0s"))));
  } else if (call.thoughts) {
    box.append(typewriter(`${S.run.id}-${pid}-${call.seq}-thoughts`, call.thoughts, "t-body"));
  } else if (call.status === "error") {
    box.append(h("div", { class: "note" }, call.message || "The call failed."));
  } else {
    const why = call.cached && !lowThinking
      ? "Cached response: replayed without a new inference, and no thought summary was stored for it."
      : lowThinking ? "Thinking is off for this phase — a simple classification task, so the pipeline gives it no thinking budget. The answer is below."
        : S.run.start?.options?.include_thoughts === false ? "Thought summaries are off for this run."
          : "The model returned no thought summary for this call.";
    box.append(h("div", { class: "note" }, why));
  }
  wrap.append(box);
  return wrap;
}

function showPrompt(call) {
  openModal(`Prompt · Phase ${PHASE_BY_ID[PHASE_OF_PART[call.phase]]?.code || call.phase}`,
    h("div", {},
      h("div", { class: "section-title" }, "System instruction"), h("pre", {}, call.system_prompt),
      h("div", { class: "section-title" }, "User prompt"), h("pre", {}, call.user_prompt),
      call.answer !== undefined ? [h("div", { class: "section-title" }, "Parsed answer (schema-constrained JSON)"),
        h("pre", {}, JSON.stringify(call.answer, null, 2))] : null));
}

function section(title, ...children) {
  return h("div", { class: "section" }, h("div", { class: "section-title" }, title), ...children);
}

function nodeLine(n) {
  if (!n) return "";
  return [n.action, n.text ? `“${n.text}”` : "", n.where ? `· ${n.where}` : ""].filter(Boolean).join(" ");
}

function eventRow(nid, extraCls = "") {
  const n = S.run.nodeById[nid];
  return h("div", { class: `event-row ${extraCls}` },
    h("span", { class: "nid" }, `#${nid}`),
    h("span", { class: "ev-text", title: n?.narrative || "" }, n ? h("span", { class: "app-chip" }, n.app) : null, " ", nodeLine(n)));
}

function sentenceHtml(narrative) {
  const parts = esc(narrative).split(" | ");
  const head = parts.shift().replace(/^(\[APP: [^\]]*\])/, '<span class="anchor">$1</span>')
    .replace(/(action:)/, '<span class="key">$1</span>');
  const rest = parts.map((p) => p.replace(/^([A-Za-z_]+:)/, '<span class="key">$1</span>'));
  return [head, ...rest].join(' <span class="sep">|</span> ');
}

const PHASE_RENDERERS = {
  // Phase 0's output is the narrative Phase 1 receives: one sentence per event.
  "0"(R) {
    const data = R.phases.serialize.data;
    if (!data) return null;
    const kept = new Set(R.phases["0B2"].data?.kept || []);
    const out = h("div", {});
    out.append(section("From raw log to narrative",
      h("div", { class: "kpis inset" },
        kpi(fmtInt(R.raw?.rows ?? data.nodes.length), "raw events"),
        kpi(R.raw?.columns ?? "—", "raw columns"),
        kpi(fmtInt(data.nodes.length), "sentences"))));
    if (data.example) {
      const kv = h("div", { class: "kv" }, Object.entries(data.example.raw_fields).map(([k, v]) =>
        h("div", { class: kept.has(k) || k === "application" || k === "event_type" ? "" : "pruned" }, h("span", {}, k), h("span", { title: v }, v))));
      out.append(section(`Example · event #${data.example.node}`,
        h("div", { class: "serial" }, kv, h("div", { class: "arrow" }, "→"),
          h("div", { class: "sentence", html: sentenceHtml(data.example.narrative) }))));
    }
    const list = h("div", { class: "table-wrap", style: { maxHeight: "430px" } },
      h("table", {}, h("tbody", {}, data.nodes.map((n) =>
        h("tr", {}, h("td", { class: "num" }, n.id), h("td", { class: "mono", html: sentenceHtml(n.narrative) }))))));
    out.append(section(`The narrative Phase 1 receives · ${data.nodes.length} sentences`, list));
    return out;
  },

  "1"(R) {
    const data = R.phases["1"].data;
    if (!data) return null;
    const out = h("div", {});
    if (data.reasoning) {
      out.append(section("Reasoning written before the answer",
        typewriter(`${R.id}-1-reasoning`, data.reasoning, "reasoning")));
    }
    // With human review on, the review panel replaces the plain matrix.
    if (R.review) {
      out.append(reviewSection(R));
      return out;
    }
    const routines = (R.start?.routines || []).map((r) => r.routine_name);
    const shared = data.topology.filter((a) => a.shared_with.length > 1);
    out.append(section(`Inferred shared actions · ${shared.length}`
      + (data.proposed > data.topology.length ? ` (${data.proposed} proposed, sanitised)` : ""),
      topologyMatrix(shared, routines, { justify: true })));
    if (R.phases.review.status === "skipped") {
      out.append(h("div", { class: "section note" }, "Human review is off for this run: the inferred shared actions go straight to Phase 2."));
    }
    return out;
  },

  "2"(R) {
    const data = R.phases["2"].data;
    if (!data) return null;
    const routines = (R.start?.routines || []).map((r) => r.routine_name);
    const sharedIds = new Set(sharedActions(R).map((a) => a.node_id));
    const out = h("div", {});
    out.append(section("One reasoning note per execution",
      h("div", { class: "exec-reasons" }, data.traces.map((t, i) => {
        const k = routines.indexOf(t.routine);
        return h("div", { class: "exec-reason", style: { borderLeftColor: seriesVar(k) } },
          h("b", {}, `${t.routine} · execution ${t.execution}`),
          t.reasoning || h("span", { class: "note" }, "No reasoning returned."),
          h("div", { class: "n" }, `${t.nodes.length} events · ${t.shared.filter((n) => sharedIds.has(n)).length} shared`));
      }))));
    out.append(section("Validation", checksList(data.checks)));
    return out;
  },
};

function reviewSection(R) {
  const rv = R.review;
  if (!rv) return null;
  const routines = rv.routines;
  const out = h("div", { class: "section" }, h("div", { class: "section-title" }, "Human review of the shared actions"));
  if (!rv.submitted) {
    out.append(h("div", { class: "review-banner" }, h("div", {},
      h("strong", {}, rv.replay ? "Recorded review step. " : "Action required. "),
      rv.replay ? "In the recorded run the operator reviewed these shared actions here. Continue to replay their decision."
        : "Each row is an action performed once that several executions rely on. Toggle which routines depend on it, or clear a row to make it an ordinary step.")));
  }
  const resolved = rv.resolved;
  const topo = resolved ? resolved.topology : rv.topology;
  const editedCells = new Set();
  if (!resolved) {
    for (const a of rv.topology) for (const r of routines) {
      if (rv.sel[a.node_id].has(r) !== rv.orig[a.node_id].has(r)) editedCells.add(`${a.node_id}|${r}`);
    }
  } else {
    for (const c of resolved.changes || []) for (const r of routines) {
      if (c.before.includes(r) !== c.after.includes(r)) editedCells.add(`${c.node_id}|${r}`);
    }
  }
  const editable = !resolved && !rv.replay && !rv.submitted;
  // Actions Phase 1 gave to a single routine are not shown: they are not shared
  // actions. They are still sent back unchanged, so Phase 2 handles them as inferred.
  const isShared = (a) => rv.orig[a.node_id].size > 1;
  const rows = (resolved ? mergeRemoved(rv.topology, topo) : rv.topology).filter(isShared);
  out.append(topologyMatrix(rows, routines, {
    justify: true, editable, sel: resolved ? null : rv.sel, editedCells,
    onToggle: (nid, r) => {
      const set = rv.sel[nid];
      set.has(r) ? set.delete(r) : set.add(r);
      renderPhaseDetail();
    },
    onClear: (nid) => { rv.sel[nid].clear(); renderPhaseDetail(); },
  }));
  if (resolved) {
    const n = (resolved.changes || []).length;
    out.append(h("div", { class: "verdict" }, n ? `Approved with ${n} edited action(s).` : "Approved as inferred — no edits."));
  } else if (!rv.submitted) {
    const actions = h("div", { class: "review-actions" });
    if (rv.replay) {
      actions.append(h("button", { class: "btn primary", onclick: () => submitReview("accept") }, "Continue replay →"));
    } else {
      actions.append(
        h("button", { class: "btn danger", onclick: () => submitReview("abort") }, "Abort run"),
        h("button", { class: "btn ghost", disabled: !editedCells.size, onclick: () => {
          for (const a of rv.topology) rv.sel[a.node_id] = new Set(rv.orig[a.node_id]);
          renderPhaseDetail();
        } }, "Reset edits"),
        h("button", { class: "btn primary", onclick: () => submitReview("accept") },
          editedCells.size ? "Approve edits & run Phase 2 →" : "Approve & run Phase 2 →"));
    }
    out.append(actions);
  } else {
    out.append(h("div", { class: "verdict" }, h("span", { class: "spinner", style: { display: "inline-block", verticalAlign: "-2px", marginRight: "8px" } }), "Decision sent…"));
  }
  return out;
}

function mergeRemoved(original, final) {
  // Keep rows the operator cleared, so the edit stays visible.
  const byId = Object.fromEntries(final.map((a) => [a.node_id, a]));
  return original.map((a) => byId[a.node_id] || { ...a, shared_with: [] });
}

async function submitReview(action) {
  const rv = S.run.review;
  rv.submitted = true;
  renderPhaseDetail();
  const topology = rv.topology.map((a) => ({ node_id: a.node_id, shared_with: [...rv.sel[a.node_id]] }));
  try {
    await api(`/api/runs/${S.run.id}/review`, { method: "POST", body: JSON.stringify({ action, topology }) });
  } catch (err) {
    rv.submitted = false;
    toast(err.message);
    renderPhaseDetail();
  }
}

function topologyMatrix(topology, routines, opts = {}) {
  const head = h("tr", {}, h("th", {}, "Shared action"),
    routines.map((r, i) => h("th", { class: "rcol" }, h("div", { class: "rh" }, h("span", { class: "sw", style: { background: seriesVar(i) } }), r))),
    h("th", {}, "Scope"));
  const rows = topology.map((a) => {
    const set = opts.sel ? opts.sel[a.node_id] : new Set(a.shared_with);
    const cells = routines.map((r, i) => {
      const on = set.has(r);
      const edited = opts.editedCells?.has(`${a.node_id}|${r}`);
      const attrs = {
        class: `cell-toggle ${on ? "on" : ""} ${edited ? "edited" : ""}`,
        style: on ? { background: seriesVar(i) } : {},
        title: `${r}: ${on ? "depends on this action" : "does not depend on it"}`,
        "aria-pressed": on ? "true" : "false",
      };
      const el = opts.editable ? h("button", { ...attrs, onclick: () => opts.onToggle(a.node_id, r) }) : h("span", attrs);
      return h("td", { class: "cell" }, el);
    });
    const n = a.node_id;
    const node = { action: a.action, text: a.text, where: a.where };
    const scope = set.size === 0 ? h("span", { class: "scope" }, "not shared")
      : h("span", { class: "scope" }, `${set.size}/${routines.length}`, h("small", {}, set.size === routines.length ? " all" : ""));
    return h("tr", {},
      h("td", { class: "act" },
        h("div", { class: "act-main" }, h("span", { class: "mono", style: { color: "var(--text-3)" } }, `#${n} `), nodeLine(node)),
        h("div", { class: "act-sub" }, a.app),
        opts.justify && a.justification ? h("div", { class: "act-why" }, a.justification) : null,
        opts.editable ? h("button", { class: "link-btn", style: { marginTop: "4px" }, onclick: () => opts.onClear(n) }, "Make it an ordinary step") : null),
      cells, h("td", {}, scope));
  });
  return h("div", { style: { overflowX: "auto" } }, h("table", { class: "matrix" }, h("thead", {}, head), h("tbody", {}, rows)));
}

function checksList(checks) {
  return h("div", {}, checks.map((c) => h("div", { class: "check" },
    h("span", { class: `check-icon ${c.status}` }, c.status === "pass" ? "✓" : c.status === "warn" ? "!" : "✕"),
    h("div", {}, h("div", { class: "check-label" }, h("span", { class: "cid" }, c.id), c.label),
      c.detail ? h("div", { class: "check-detail" }, c.detail) : null))));
}

// =============================================================== 4 · RESULT
// A shared action is one that more than one routine depends on. An action Phase 1
// assigns to a single routine (e.g. 1/4) is shown throughout the UI as an ordinary step.
const sharedActions = (R) => (R.topology || []).filter((a) => a.shared_with.length > 1);

function renderResult() {
  const R = S.run;
  if (!R || !R.result) return;
  const res = R.result;
  const routines = (R.start.routines || []).map((r) => r.routine_name);
  const shared = sharedActions(R).length;
  const routed = new Set();
  const sharedIds = new Set(sharedActions(R).map((a) => a.node_id));
  for (const t of res.traces) for (const n of t.nodes) if (!sharedIds.has(n)) routed.add(n);
  const passed = res.checks.filter((c) => c.status === "pass").length;

  $("#resultLead").textContent = `${R.start.log_name} · ${routines.length} routines · ${res.traces.length} executions reconstructed`;
  const kp = $("#resultKpis");
  kp.innerHTML = "";
  kp.append(
    kpi(fmtInt(R.raw?.rows ?? R.nodes.length), "events in"),
    kpi(res.traces.length, "executions"),
    kpi(shared, "shared actions"),
    kpi(fmtInt(routed.size), "events routed"),
    kpi(res.noise.length, "diverted to Noise"),
    kpi(`${passed}/${res.checks.length}`, "checks passed"));

  const actions = $("#resultActions");
  actions.innerHTML = "";
  for (const f of res.files) {
    actions.append(h("a", { class: "btn ghost", href: `/api/runs/${R.id}/files/${encodeURIComponent(f)}`, download: f },
      f.endsWith(".xlsx") ? "⬇ Workbook (.xlsx)" : "⬇ Audit trail (.json)"));
  }
  if (!R.replay) actions.append(h("button", { class: "btn primary", onclick: saveRecording }, "Save as recording"));

  renderUntangle(true);
  renderTraces();
  $("#checks").innerHTML = "";
  $("#checks").append(checksList(res.checks));
  renderNoise();
  const topo = $("#topologyResult");
  topo.innerHTML = "";
  topo.append(topologyMatrix(sharedActions(R), routines, { justify: true }));
}

async function saveRecording() {
  const name = prompt("Name this recording:", `${S.run.start.log_name.replace(/\.csv$/, "")} — ${new Date().toLocaleString()}`);
  if (!name) return;
  try {
    await api(`/api/runs/${S.run.id}/save`, { method: "POST", body: JSON.stringify({ name }) });
    toast("Saved — it now appears under Recordings.");
    refreshRecordings();
  } catch (err) { toast(err.message); }
}

// The centrepiece: every event starts on one interleaved line and moves into
// the lane of its execution; shared actions are copied into every lane their
// subset entitles them to; noise drops to its own lane.
let untangleTimer = null;
function renderUntangle(animate) {
  const R = S.run;
  const res = R.result;
  const box = $("#untangle");
  box.innerHTML = "";
  box.classList.remove("settled");
  clearTimeout(untangleTimer);
  const routines = (R.start.routines || []).map((r) => r.routine_name);
  const W = box.clientWidth || 1000;
  const U = UI_SCALE;
  const G = Math.min(260 * U, Math.max(170 * U, W * 0.22));
  const x0 = G + 12 * U, x1 = W - 12 * U;
  const rawCount = R.raw?.rows || R.nodes.length;
  const xOf = (node) => {
    const pos = node.raw ?? node.id;
    return x0 + (rawCount > 1 ? (pos / (rawCount - 1)) * (x1 - x0) : 0);
  };
  const ROW = 30 * U, TOP = 14 * U, HALF = 11 * U;  // HALF: half a lane's CSS height
  const laneY = (i) => TOP + 46 * U + i * ROW;
  const noiseY = laneY(res.traces.length) + 10 * U;
  box.style.height = `${noiseY + 2 * HALF}px`;

  const label = (y, swatch, text, sub) => h("div", { class: "lane-label", style: { top: `${y - HALF}px`, width: `${G}px` } },
    swatch ? h("span", { class: "sw", style: { background: swatch } }) : null,
    h("span", { style: { overflow: "hidden", textOverflow: "ellipsis" } }, text, sub ? h("span", { class: "ex" }, ` · ${sub}`) : null));
  const laneBg = (y, striped) => h("div", { class: `lane-bg ${striped ? "striped" : ""}`, style: { top: `${y - HALF}px`, left: `${G}px` } });

  box.append(laneBg(TOP, false), label(TOP, null, "Interleaved log"));
  box.append(h("div", { class: "divider", style: { top: `${TOP + 23 * U}px`, left: `${G}px` } }));
  res.traces.forEach((t, i) => {
    const k = routines.indexOf(t.routine);
    box.append(laneBg(laneY(i), t.execution > 1), label(laneY(i), seriesVar(k), t.routine, `exec ${t.execution}`));
  });
  box.append(laneBg(noiseY, false), label(noiseY, "var(--noise)", "Noise"));

  // Lane membership per node.
  const lanes = {};
  res.traces.forEach((t, i) => t.nodes.forEach((n) => (lanes[n] ||= []).push(i)));
  const sharedIds = new Set(sharedActions(R).map((a) => a.node_id));
  const noise = new Set(res.noise);
  const dots = [];

  // Events Phase 0 removed as noise never became nodes: they fall into Noise too.
  const nodeRaw = new Set(R.nodes.map((n) => n.raw).filter((r) => r !== null && r !== undefined));
  const dropped = (R.phases["0A"].data?.removed || []).filter((r) => !nodeRaw.has(r));
  for (const r of dropped) {
    const ev = R.raw.events[r];
    dots.push({ x: xOf({ raw: r }), lane: null, noise: true, shared: false, first: true, tip: eventTip(ev) + '<div class="tt-sub">Removed by Phase 0 (noise event type)</div>' });
  }
  for (const n of R.nodes) {
    const x = xOf(n);
    const tipBase = `<div class="tt-title">#${n.id} · ${esc(nodeLine(n))}</div><div class="tt-sub">${esc(n.app)}${R.raw?.events[n.raw] ? " · " + esc(R.raw.events[n.raw].time) : ""}</div><div class="tt-mono">${esc(n.narrative)}</div>`;
    const ls = lanes[n.id] || [];
    if (!ls.length) {
      dots.push({ x, lane: null, noise: true, shared: false, first: true, tip: tipBase + `<div class="tt-sub" style="margin-top:6px">${noise.has(n.id) ? "Diverted to the Noise sheet" : "Not placed"}</div>` });
      continue;
    }
    const isShared = sharedIds.has(n.id);
    const where = ls.map((i) => `${res.traces[i].routine} · exec ${res.traces[i].execution}`).join("<br>");
    ls.forEach((li, j) => dots.push({
      x, lane: li, noise: false, shared: isShared, first: j === 0,
      tip: tipBase + `<div class="tt-sub" style="margin-top:6px">${isShared ? `Shared action → ${ls.length} traces:` : ls.length > 1 ? "Executions:" : "Execution:"}<br>${where}</div>`,
    }));
  }

  // Trace paths, revealed once the dots have landed.
  res.traces.forEach((t, i) => {
    const xs = t.nodes.map((n) => R.nodeById[n]).filter(Boolean).map(xOf);
    if (!xs.length) return;
    const k = routines.indexOf(t.routine);
    box.append(h("div", { class: "lane-line", style: { top: `${laneY(i) - 1}px`, left: `${Math.min(...xs)}px`, width: `${Math.max(...xs) - Math.min(...xs)}px`, background: seriesVar(k) } }));
  });

  const els = dots.map((d) => {
    const el = h("div", { class: `dot ${d.shared ? "shared" : ""}` });
    el.addEventListener("mousemove", (e) => showTip(e, d.tip));
    el.addEventListener("mouseleave", hideTip);
    box.append(el);
    return el;
  });
  const finalPos = (d) => {
    const y = d.noise ? noiseY : laneY(d.lane);
    return `translate(${d.x}px, ${y}px)${d.shared ? " rotate(45deg)" : ""}`;
  };
  const finalColor = (d) => d.noise ? "var(--noise)" : seriesVar(routines.indexOf(res.traces[d.lane].routine));

  if (!animate) {
    dots.forEach((d, i) => {
      els[i].style.transition = "none";
      els[i].style.transform = finalPos(d);
      els[i].style.background = finalColor(d);
    });
    box.classList.add("settled");
    return;
  }
  dots.forEach((d, i) => {
    els[i].style.transform = `translate(${d.x}px, ${TOP}px)${d.shared ? " rotate(45deg)" : ""}`;
    els[i].style.opacity = d.first ? "1" : "0";
  });
  const span = 1800;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    untangleTimer = setTimeout(() => {
      dots.forEach((d, i) => {
        const delay = ((d.x - x0) / Math.max(1, x1 - x0)) * span;
        els[i].style.transitionDelay = `${delay}ms`;
        els[i].style.transform = finalPos(d);
        els[i].style.background = finalColor(d);
        els[i].style.opacity = "1";
      });
      untangleTimer = setTimeout(() => box.classList.add("settled"), span + 1100);
    }, 700);
  }));

  const legend = $("#untangleLegend");
  legend.innerHTML = "";
  legend.append(
    h("span", {}, h("i"), "event"),
    h("span", {}, h("i", { class: "sq" }), "shared action — copied into every trace its subset allows"),
    h("span", {}, h("i", { style: { background: "var(--noise)" } }), "noise"),
    ...routines.map((r, i) => h("span", {}, h("i", { style: { background: seriesVar(i) } }), r)));
}
$("#replayAnim").addEventListener("click", () => renderUntangle(true));
let resizeTimer = null;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (S.view === "result" && S.run?.result) renderUntangle(false); }, 150);
});

function renderTraces() {
  const R = S.run;
  const res = R.result;
  const routines = (R.start.routines || []).map((r) => r.routine_name);
  const tabs = $("#traceTabs");
  tabs.innerHTML = "";
  res.traces.forEach((t, i) => {
    tabs.append(h("button", { class: `trace-tab ${i === S.activeTrace ? "active" : ""}`, onclick: () => { S.activeTrace = i; renderTraces(); } },
      h("span", { class: "sw", style: { background: seriesVar(routines.indexOf(t.routine)) } }),
      `${t.routine} · ${t.execution}`, h("span", { class: "cnt" }, t.nodes.length)));
  });
  const t = res.traces[S.activeTrace] || res.traces[0];
  const k = routines.indexOf(t.routine);
  const subsetSize = Object.fromEntries(sharedActions(R).map((a) => [a.node_id, a.shared_with.length]));
  const detail = $("#traceDetail");
  detail.innerHTML = "";
  detail.append(h("div", { class: "quote", style: { borderLeftColor: seriesVar(k) } },
    h("b", {}, "Model rationale: "), t.reasoning || "—"));
  const rows = t.nodes.map((nid) => {
    const n = R.nodeById[nid] || {};
    const ev = R.raw?.events[n.raw];
    return h("tr", {},
      h("td", { class: "num" }, nid),
      h("td", { class: "mono" }, ev?.time || ""),
      h("td", {}, h("span", { class: "app-chip" }, n.app || "")),
      h("td", {}, h("span", { class: "type-chip" }, n.action || "")),
      h("td", {}, n.text || "", n.where ? h("span", { style: { color: "var(--text-3)" } }, ` · ${n.where}`) : null),
      h("td", {}, subsetSize[nid] ? h("span", { class: "shared-tag" }, `shared · ${subsetSize[nid]}/${routines.length}`) : null));
  });
  detail.append(h("div", { class: "table-wrap", style: { maxHeight: "620px" } },
    h("table", {}, h("thead", {}, h("tr", {}, h("th", { class: "num" }, "#"), h("th", {}, "Time"), h("th", {}, "App"), h("th", {}, "Event"), h("th", {}, "Content"), h("th", {}, ""))),
      h("tbody", {}, rows))));
}

function renderNoise() {
  const R = S.run;
  const res = R.result;
  const box = $("#noiseList");
  box.innerHTML = "";
  const unrouted = new Set(res.noise_unrouted);
  if (!res.noise.length) {
    box.append(h("div", { class: "note" }, "No event was diverted."));
    return;
  }
  box.append(h("div", { class: "event-list" }, res.noise.map((nid) => {
    const row = eventRow(nid);
    row.title = unrouted.has(nid) ? "Not routed by Phase 2" : "Tagged by Phase 0: unrelated to the declared routines";
    return row;
  })));
}

// ---------------------------------------------------------------- bootstrap
renderStepper();
loadInputs();
