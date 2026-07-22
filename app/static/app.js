/* Dubbing — five-step wizard.

   Twelve technical stages collapse into five human decisions. The stepper is
   the spine; exactly one step renders at a time. Stage-level detail lives
   behind a disclosure, because an operator needs to know WHICH step needs them,
   not that `htdemucs` is queued.

   Steps are gated on readiness — you cannot cast voices before there is a
   transcript — and the wizard auto-advances to wherever the work actually is.
*/

const $ = (s, r = document) => r.querySelector(s);
const el = (t, c) => { const n = document.createElement(t); if (c) n.className = c; return n; };

let projects = [], project = null, poll = null;
let speakers = [], segments = [], translations = [], voices = [];
let stepIndex = 0, userPicked = false, activeLang = null;
let castingFor = null;
// "library" = the shelf of every video; "studio" = the wizard for one of them.
let view = "library";

const STEPS = [
  { key: "upload",    name: "Add video",    stages: ["probe", "proxy", "thumbnails", "shots"] },
  { key: "speakers",  name: "Speakers",     stages: ["separate", "diarize"] },
  { key: "script",    name: "Script",       stages: ["asr"] },
  { key: "translate", name: "Translation",  stages: ["translate", "refine"] },
  { key: "dub",       name: "Voices & dub", stages: ["synth", "fit", "mix"] },
];

// ── helpers ───────────────────────────────────────────────────────────────

function tc(sec) {
  if (sec == null || isNaN(sec)) return "—";
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function bytes(n) {
  if (!n) return "—";
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
  if (r.status === 204) return null;
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
  return body;
}
let toastTimer = null;
function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast " + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), kind === "bad" ? 9000 : 4000);
}
// `p` defaults to the open project so the wizard reads unchanged; the library
// passes each card's own project, so both share one definition of "done".
const jobOf = (stage, p = project) => (p?.jobs || []).find(j => j.stage === stage) || {};

/** A step's state, derived from the stages beneath it. */
function stepState(i, p = project) {
  const s = STEPS[i];
  if (s.key === "upload") {
    if (!p?.source) return "active";
    const js = s.stages.map(st => jobOf(st, p));
    if (js.some(j => j.state === "failed")) return "failed";
    if (js.some(j => j.state === "running" || j.state === "queued")) return "running";
    return js.every(j => j.state === "done") ? "done" : "active";
  }
  const js = s.stages.map(st => jobOf(st, p));
  if (js.some(j => j.state === "failed")) return "failed";
  if (js.some(j => j.state === "running" || j.state === "queued")) return "running";
  if (js.every(j => j.state === "done" || j.state === "skipped")) return "done";
  return "todo";
}
/** Can the user open this step yet? */
function stepReady(i) {
  if (!project) return i === 0;
  if (i === 0) return true;
  if (!project.source) return false;
  if (i === 1) return stepState(0) === "done";
  if (i === 2) return project.segment_count > 0;
  if (i === 3) return project.segment_count > 0;
  if (i === 4) return (project.translated_langs || []).length > 0;
  return false;
}
function subLabel(i) {
  const st = stepState(i);
  if (st === "running") return "working…";
  if (st === "failed") return "needs attention";
  const s = STEPS[i];
  if (s.key === "upload") return project?.source ? tc(project.source.duration) : "not started";
  if (s.key === "speakers") return project?.speaker_count ? `${project.speaker_count} found` : "not started";
  if (s.key === "script") return project?.segment_count ? `${project.segment_count} lines` : "not started";
  if (s.key === "translate") {
    const l = project?.translated_langs || [];
    return l.length ? l.join(", ") : "not started";
  }
  if (s.key === "dub") return Object.keys(project?.renders || {}).length ? "ready" : "not started";
  return "";
}
/** Where should the user actually be? */
function firstIncomplete() {
  for (let i = 0; i < STEPS.length; i++) {
    if (!stepReady(i)) return Math.max(0, i - 1);
    if (stepState(i) !== "done") return i;
  }
  return STEPS.length - 1;
}

// ── data ──────────────────────────────────────────────────────────────────

async function refresh(keepStep = true) {
  try {
    projects = await api("/api/projects");
  } catch { return; }
  const id = project?.id;
  project = projects.find(p => p.id === id) || projects[0] || null;
  $("#pickerTitle").textContent = project ? project.title : "No project";

  // The library renders from the project list alone. Skip the per-project
  // fetches so polling a shelf of videos stays cheap.
  if (view === "library") { render(); schedulePoll(); return; }

  if (project && (project.segment_count || project.speaker_count)) {
    try {
      [speakers, segments, voices] = await Promise.all([
        api(`/api/projects/${project.id}/speakers`),
        api(`/api/projects/${project.id}/segments`),
        api("/api/voices").catch(() => []),
      ]);
    } catch { /* keep last good */ }
  } else { speakers = []; segments = []; }

  const langs = project?.translated_langs || [];
  if (!activeLang || !langs.includes(activeLang)) activeLang = langs[0] || null;
  if (activeLang) {
    try { translations = await api(`/api/projects/${project.id}/translations?lang=${activeLang}`); }
    catch { translations = []; }
  } else translations = [];

  if (!keepStep || !userPicked) stepIndex = firstIncomplete();
  render();
  schedulePoll();
}

function schedulePoll() {
  const BUSY = ["ingesting", "analyzing", "translating", "dubbing"];
  // In the library any card can be moving, so watch them all — not just the
  // one that happens to be selected.
  const busy = view === "library"
    ? projects.some(p => BUSY.includes(p.status))
    : BUSY.includes(project?.status);
  clearInterval(poll);
  if (busy) poll = setInterval(() => refresh(true), 2500);
}

// ── render ────────────────────────────────────────────────────────────────

function render() {
  const lib = view === "library";
  $("#library").classList.toggle("hidden", !lib);
  $("#shell").classList.toggle("hidden", lib);
  $("#projectPicker").classList.toggle("hidden", lib);
  $("#libBtn").classList.toggle("hidden", lib);
  if (lib) { renderLibrary(); return; }
  renderSteps();
  renderStage();
}

// ── library ───────────────────────────────────────────────────────────────
// The shelf every video lands on. Poster, language pair, and how far through
// the five steps it got — enough to pick up where you left off without
// opening anything.

const langName = (c) => (LANGS.find(l => l[0] === c) || [c, c])[1];

function openProject(p) {
  project = p; activeLang = null; userPicked = false; view = "studio";
  refresh(false);
}
function showLibrary() { view = "library"; render(); schedulePoll(); }

/** Five bars — the same five steps as the wizard, at a glance. */
function progressRow(p) {
  const states = STEPS.map((_, i) => stepState(i, p));

  const row = el("span", "pc-prog");
  states.forEach(st => { const d = el("i"); d.dataset.state = st; row.append(d); });

  // Name the EARLIEST step that wants a human, not the latest one affected:
  // a failure at Speakers cascades to every step after it, and telling someone
  // "Voices & dub needs attention" sends them to the wrong end of the pipeline.
  const failed = states.indexOf("failed");
  const running = states.indexOf("running");
  const done = states.lastIndexOf("done");
  let label = "Not started";
  if (failed !== -1) label = `${STEPS[failed].name} — needs attention`;
  else if (running !== -1) label = `${STEPS[running].name} — working…`;
  else if (done !== -1) label = `${STEPS[done].name} done`;

  const t = el("span", "pc-prog-t");
  t.textContent = label;
  const wrap = el("span", "pc-progwrap");
  wrap.append(row, t);
  return wrap;
}

function projectCard(p) {
  const c = el("button", "pcard");
  c.type = "button";

  const th = el("span", "pc-thumb");
  if (p.media?.poster) th.style.backgroundImage = `url(${p.media.poster})`;
  else th.classList.add("empty");

  if (p.source?.duration) {
    const d = el("span", "pc-dur");
    d.textContent = tc(p.source.duration);
    th.append(d);
  }
  const busy = ["ingesting", "analyzing", "translating", "dubbing"].includes(p.status);
  if (busy) {
    const b = el("span", "pc-badge");
    b.textContent = p.status;
    th.append(b);
  }

  const body = el("span", "pc-body");
  const t = el("span", "pc-title"); t.textContent = p.title;
  const l = el("span", "pc-langs");
  const tgt = (p.target_langs || []).map(langName).join(", ");
  l.textContent = tgt ? `${langName(p.source_lang)} → ${tgt}` : langName(p.source_lang);
  body.append(t, l, progressRow(p));

  c.append(th, body);
  c.onclick = () => openProject(p);
  return c;
}

function renderLibrary() {
  const wrap = $("#library");
  wrap.textContent = "";

  const h = el("div", "lib-head");
  const left = el("div");
  const e = el("div", "eyebrow"); e.textContent = "Library";
  const t = el("h1");
  t.textContent = projects.length
    ? `${projects.length} ${projects.length === 1 ? "video" : "videos"}`
    : "Nothing here yet";
  left.append(e, t);
  const add = el("button", "btn"); add.type = "button";
  add.textContent = "Add video";
  add.onclick = () => openNew();
  h.append(left, add);
  wrap.append(h);

  if (!projects.length) {
    const p = el("p", "hint");
    p.textContent = "Every video you add lands here — with its transcript, "
      + "translations and dubs kept alongside it.";
    wrap.append(p);
    return;
  }
  const grid = el("div", "lib-grid");
  for (const p of projects) grid.append(projectCard(p));
  wrap.append(grid);
}

function renderSteps() {
  const nav = $("#steps");
  nav.textContent = "";
  STEPS.forEach((s, i) => {
    const b = el("button", "step");
    b.type = "button";
    const ready = stepReady(i);
    const st = stepState(i);
    b.dataset.state = i === stepIndex && st === "todo" ? "active" : st;
    b.setAttribute("aria-current", String(i === stepIndex));
    b.disabled = !ready;

    const num = el("span", "num");
    num.textContent = st === "done" ? "✓" : st === "failed" ? "!" : String(i + 1);
    const box = el("span");
    const nm = el("span", "name"); nm.textContent = s.name;
    const sub = el("span", "sub"); sub.textContent = subLabel(i);
    box.append(nm, sub);
    b.append(num, box);
    b.onclick = () => { stepIndex = i; userPicked = true; render(); };
    nav.append(b);
  });
}

function renderStage() {
  const st = $("#stage");
  st.textContent = "";
  if (!project) {
    const p = el("div", "pane");
    const h = el("div", "head");
    h.innerHTML = '<div class="eyebrow">Get started</div><h1>Create a project</h1>' +
      '<p>A project holds one video, its transcript, its translations and its dubs.</p>';
    const b = el("button", "btn big"); b.type = "button"; b.textContent = "New project";
    b.onclick = () => openNew();
    p.append(h, b);
    st.append(p);
    return;
  }
  ({ upload: paneUpload, speakers: paneSpeakers, script: paneScript,
     translate: paneTranslate, dub: paneDub })[STEPS[stepIndex].key](st);
}

function head(pane, eyebrow, title, sub) {
  const h = el("div", "head");
  const e = el("div", "eyebrow"); e.textContent = eyebrow;
  const t = el("h1"); t.textContent = title;
  h.append(e, t);
  if (sub) { const p = el("p"); p.textContent = sub; h.append(p); }
  pane.append(h);
  return h;
}

/** Notes are how a degraded stage tells the truth. Never hide them. */
function notes(pane, stages) {
  for (const stage of stages) {
    const j = jobOf(stage);
    if (j.error) {
      const n = el("div", "note bad");
      const b = el("b"); b.textContent = stage;
      const s = el("span"); s.textContent = j.error;
      n.append(b, s); pane.append(n);
    } else if (j.note) {
      const n = el("div", "note" + (j.state === "skipped" || /⚠️|cannot|no |not /i.test(j.note) ? " warn" : ""));
      const b = el("b"); b.textContent = stage;
      const s = el("span"); s.textContent = j.note;
      n.append(b, s); pane.append(n);
    }
  }
}

function techDetail(pane, stages, label = "Technical stages") {
  const d = el("details", "tech");
  const sm = el("summary"); sm.textContent = label;
  const grid = el("div", "stages");
  for (const stage of stages) {
    const j = jobOf(stage);
    const box = el("div", "stg");
    box.dataset.s = j.state || "pending";
    const dot = el("span", "dot");
    const n = el("span", "n"); n.textContent = stage;
    box.append(dot, n);
    if (j.note) box.title = j.note;
    grid.append(box);
  }
  d.append(sm, grid);
  pane.append(d);
}

// ── step 1 · upload ───────────────────────────────────────────────────────

function paneUpload(root) {
  const pane = el("div", "pane");
  root.append(pane);

  if (!project.source) {
    head(pane, "Step 1 of 5", "Add your video",
      "Drop a file in. We keep your original untouched and work from a 720p copy.");
    const z = el("div"); z.id = "drop";
    const h = el("h2"); h.textContent = "Drop a video here";
    const p = el("p"); p.textContent = "MP4, MKV, MOV — or click to choose";
    const b = el("button", "btn"); b.type = "button"; b.textContent = "Choose file";
    const prog = el("div", "bar-prog hidden"); const fill = el("i"); prog.append(fill);
    const stat = el("div", "lbl"); stat.style.marginTop = "10px";
    z.append(h, p, b, prog, stat);
    pane.append(z);

    const start = (file) => {
      if (!file) return;
      b.disabled = true; prog.classList.remove("hidden");
      const fd = new FormData(); fd.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/projects/${project.id}/source`);
      xhr.upload.onprogress = e => {
        if (!e.lengthComputable) return;
        fill.style.width = (e.loaded / e.total * 100) + "%";
        stat.textContent = `uploading ${bytes(e.loaded)} of ${bytes(e.total)}`;
      };
      xhr.onload = () => {
        if (xhr.status < 300) { toast("Upload complete — preparing your video", "good"); userPicked = false; refresh(false); }
        else {
          let m = xhr.statusText;
          try { m = JSON.parse(xhr.responseText).error || m; } catch {}
          toast("Upload failed — " + m, "bad"); b.disabled = false;
        }
      };
      xhr.onerror = () => { toast("Upload failed — connection lost", "bad"); b.disabled = false; };
      xhr.send(fd);
    };
    b.onclick = () => {
      const pick = $("#picker"); pick.value = "";
      pick.onchange = () => start(pick.files[0]); pick.click();
    };
    z.addEventListener("dragover", e => { e.preventDefault(); z.classList.add("hot"); });
    z.addEventListener("dragleave", () => z.classList.remove("hot"));
    z.addEventListener("drop", e => { e.preventDefault(); z.classList.remove("hot"); start(e.dataTransfer.files[0]); });
    return;
  }

  const s = project.source;
  head(pane, "Step 1 of 5", project.title,
    stepState(0) === "running" ? "Preparing your video — this takes a moment."
      : "Your video is ready. Move on to speakers.");

  if (project.media?.proxy) {
    const v = el("video", "player");
    v.controls = true; v.preload = "metadata"; v.src = project.media.proxy;
    if (project.media.poster) v.poster = project.media.poster;
    v.id = "player";
    pane.append(v);
  }

  const cards = el("div", "cards"); cards.style.marginTop = "16px";
  const add = (k, v, sm) => {
    const c = el("div", "card");
    const kk = el("div", "k"); kk.textContent = k;
    const vv = el("div", "v" + (sm ? " sm" : "")); vv.textContent = v;
    c.append(kk, vv); cards.append(c);
  };
  add("Length", tc(s.duration));
  add("Resolution", s.width ? `${s.width}×${s.height}` : "—");
  add("Shots", project.shot_count || "—");
  add("Size", bytes(s.size), true);
  pane.append(cards);

  notes(pane, ["probe", "proxy", "thumbnails", "shots"]);
  techDetail(pane, STEPS[0].stages);
  nextButton(pane, 1, "Find the speakers");
}

function nextButton(pane, i, label) {
  if (!stepReady(i)) return;
  const act = el("div", "act");
  const b = el("button", "btn big"); b.type = "button"; b.textContent = label;
  b.onclick = () => { stepIndex = i; userPicked = true; render(); };
  act.append(b);
  pane.append(act);
}

// ── step 2 · speakers ─────────────────────────────────────────────────────

function paneSpeakers(root) {
  const pane = el("div", "pane");
  root.append(pane);
  const st = stepState(1);

  head(pane, "Step 2 of 5", "Who is speaking?",
    "We separate the dialogue and group it by voice. Name each person — those names follow "
    + "them through the script, the translation and the casting.");

  const act = el("div", "act");
  const running = project.status === "analyzing";
  const b = el("button", "btn" + (project.speaker_count ? "" : " big"));
  b.type = "button";
  b.disabled = running;
  b.textContent = running ? "Listening…" : (project.speaker_count ? "Run again" : "Find speakers");
  b.onclick = async () => {
    b.disabled = true;
    try { await api(`/api/projects/${project.id}/analyze`, { method: "POST" }); toast("Listening to the dialogue…"); refresh(true); }
    catch (e) { toast(e.message, "bad"); b.disabled = false; }
  };
  act.append(b);
  if (!project.speaker_count) {
    const w = el("div", "why grow");
    w.textContent = "This also produces the transcript in the next step.";
    act.append(w);
  }
  pane.append(act);

  if (speakers.length) {
    const list = el("div", "spk-list");
    for (const s of speakers) {
      const row = el("div", "spk");
      const sw = el("span", "sw"); sw.style.background = s.color;

      const nm = el("input"); nm.type = "text";
      nm.value = s.display_name || ""; nm.placeholder = s.label;
      nm.setAttribute("aria-label", `Name for ${s.label}`);
      nm.onchange = async () => {
        try {
          await api(`/api/speakers/${s.id}`, {
            method: "PATCH", headers: { "content-type": "application/json" },
            body: JSON.stringify({ display_name: nm.value }),
          });
          s.display_name = nm.value.trim() || null; s.name = s.display_name || s.label;
          toast("Name saved", "good");
        } catch (e) { toast(e.message, "bad"); }
      };

      const meta = el("span", "meta");
      meta.textContent = `${Math.round(s.speech_seconds || 0)}s · ${s.segment_count} lines`;

      const cast = castButton(s);
      row.append(sw, nm, meta, cast);
      list.append(row);
    }
    pane.append(list);
  }

  notes(pane, ["separate", "diarize"]);
  techDetail(pane, STEPS[1].stages);
  if (project.segment_count) nextButton(pane, 2, "Check the script");
}

function castButton(s) {
  const v = voices.find(x => x.id === s.voice_id);
  const b = el("button", "cast-btn");
  b.type = "button";
  b.dataset.cast = String(!!v);
  const span = el("span", v ? "who" : "none");
  span.textContent = v ? v.display_name : "Cast a voice";
  b.append(span);
  b.onclick = () => openCast(s);
  return b;
}

// ── step 3 · script ───────────────────────────────────────────────────────

function paneScript(root) {
  const pane = el("div", "pane");
  root.append(pane);
  head(pane, "Step 3 of 5", "Check the script",
    "Fix anything misheard before it spreads. An error here becomes a translation error, then a "
    + "wrong line in someone's voice. Click a timecode to hear it.");

  const low = segments.filter(s => s.asr_confidence != null && s.asr_confidence < 0.6).length;
  const edited = segments.filter(s => s.edited).length;

  const cards = el("div", "cards");
  const add = (k, v) => {
    const c = el("div", "card");
    const kk = el("div", "k"); kk.textContent = k;
    const vv = el("div", "v"); vv.textContent = v;
    c.append(kk, vv); cards.append(c);
  };
  add("Lines", segments.length);
  add("Edited", edited);
  add("Low confidence", low);
  pane.append(cards);

  if (project.media?.proxy) {
    const v = el("video", "player");
    v.id = "player"; v.controls = true; v.preload = "metadata";
    v.src = project.media.proxy; v.style.marginTop = "16px"; v.style.maxHeight = "34vh";
    if (project.media.poster) v.poster = project.media.poster;
    pane.append(v);
  }

  const byId = Object.fromEntries(speakers.map(s => [s.id, s]));
  const lines = el("div", "lines"); lines.style.marginTop = "16px";
  for (const s of segments) {
    const row = el("div", "line");
    if (s.asr_confidence != null && s.asr_confidence < 0.6) row.classList.add("flag");
    const spk = byId[s.speaker_id];
    const stripe = el("span", "stripe");
    stripe.style.background = spk ? spk.color : "var(--idle)";

    const body = el("div");
    const top = el("div", "top");
    const t = el("button", "tcode"); t.type = "button"; t.textContent = tc(s.t_start);
    t.title = "Play from here"; t.onclick = () => seek(s.t_start);
    const who = el("span", "who-tag");
    who.textContent = spk ? spk.name : "—";
    if (spk) who.style.color = spk.color;
    top.append(t, who);
    if (s.asr_confidence != null) {
      const badge = el("span", "badge");
      badge.textContent = `${Math.round(s.asr_confidence * 100)}%`;
      if (s.edited) badge.textContent += " · edited";
      top.append(badge);
    }

    const ta = el("textarea", "edit");
    ta.value = s.text || ""; ta.rows = 1;
    ta.setAttribute("aria-label", `Line at ${tc(s.t_start)}`);
    autoGrow(ta);
    ta.onchange = async () => {
      try {
        const r = await api(`/api/segments/${s.id}`, {
          method: "PATCH", headers: { "content-type": "application/json" },
          body: JSON.stringify({ text_src_edited: ta.value }),
        });
        s.text = r.text; s.edited = r.edited;
      } catch (e) { toast(e.message, "bad"); }
    };
    body.append(top, ta);
    row.append(stripe, body);
    lines.append(row);
  }
  pane.append(lines);
  nextButton(pane, 3, "Translate");
}

function autoGrow(ta) {
  const grow = () => { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + "px"; };
  ta.oninput = grow;
  setTimeout(grow, 0);
}
function seek(t) {
  const v = $("#player");
  if (!v) return;
  v.currentTime = t; v.play().catch(() => {});
}

// ── step 4 · translation ──────────────────────────────────────────────────

function paneTranslate(root) {
  const pane = el("div", "pane");
  root.append(pane);
  head(pane, "Step 4 of 5", "Translate and trim",
    "Each line has to be speakable in the time the original takes. Lines marked over are too long — "
    + "shortening them is the fix that costs nothing.");

  const act = el("div", "act");
  const sel = el("select");
  for (const L of (project.languages || [])) {
    if (L.code === project.source_lang) continue;
    const o = el("option"); o.value = L.code; o.textContent = L.name;
    if (L.code === (activeLang || (project.target_langs || [])[0])) o.selected = true;
    sel.append(o);
  }
  const running = project.status === "translating";
  const b = el("button", "btn"); b.type = "button"; b.disabled = running;
  b.textContent = running ? "Translating…" : (translations.length ? "Translate again" : "Translate");
  b.onclick = async () => {
    b.disabled = true;
    try {
      await api(`/api/projects/${project.id}/translate`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ lang: sel.value }),
      });
      activeLang = sel.value;
      toast("Translating…"); refresh(true);
    } catch (e) { toast(e.message, "bad"); b.disabled = false; }
  };
  act.append(sel, b);

  if (translations.length) {
    const r = el("button", "ghost"); r.type = "button"; r.textContent = "Shorten with AI";
    r.title = "Re-run the AI editing pass over the machine translation";
    r.onclick = async () => {
      r.disabled = true;
      try {
        await api(`/api/projects/${project.id}/refine`, {
          method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ lang: activeLang }),
        });
        toast("Rewriting lines to fit…"); refresh(true);
      } catch (e) { toast(e.message, "bad"); r.disabled = false; }
    };
    act.append(r);
  }
  pane.append(act);

  if (translations.length) {
    const over = translations.filter(t => t.state === "needs_review").length;
    const cards = el("div", "cards");
    const add = (k, v, warn) => {
      const c = el("div", "card");
      const kk = el("div", "k"); kk.textContent = k;
      const vv = el("div", "v"); vv.textContent = v;
      if (warn) vv.style.color = "var(--tally)";
      c.append(kk, vv); cards.append(c);
    };
    add("Lines", translations.length);
    add("Fit the timing", translations.length - over, false);
    add("Run long", over, over > 0);
    pane.append(cards);

    const byId = Object.fromEntries(speakers.map(s => [s.id, s]));
    const lines = el("div", "lines"); lines.style.marginTop = "16px";
    for (const t of translations) {
      const row = el("div", "line");
      if (t.state === "needs_review") row.classList.add("flag");
      const spk = byId[t.speaker_id];
      const stripe = el("span", "stripe");
      stripe.style.background = spk ? spk.color : "var(--idle)";

      const body = el("div");
      const top = el("div", "top");
      const tb = el("button", "tcode"); tb.type = "button"; tb.textContent = tc(t.t_start);
      tb.onclick = () => seek(t.t_start);
      const who = el("span", "who-tag");
      who.textContent = spk ? spk.name : "—";
      if (spk) who.style.color = spk.color;
      const badge = el("span", "badge");
      const setBadge = (ratio) => {
        if (ratio == null) { badge.textContent = ""; return; }
        const pct = Math.round((ratio - 1) * 100);
        badge.textContent = pct > 0 ? `${pct}% too long` : "fits";
        badge.dataset.bad = String(Math.abs(ratio - 1) > (project.fit_tolerance || 0.15));
      };
      setBadge(t.fit_ratio);
      top.append(tb, who, badge);

      const src = el("div", "src"); src.textContent = t.source_text || "";
      const ta = el("textarea", "edit");
      ta.value = t.text || ""; ta.rows = 1; ta.dir = "auto";
      autoGrow(ta);
      ta.onchange = async () => {
        try {
          const r = await api(`/api/translations/${t.id}`, {
            method: "PATCH", headers: { "content-type": "application/json" },
            body: JSON.stringify({ text_edited: ta.value }),
          });
          t.fit_ratio = r.fit_ratio; t.state = r.state;
          setBadge(r.fit_ratio);
          row.classList.toggle("flag", r.state === "needs_review");
        } catch (e) { toast(e.message, "bad"); }
      };
      body.append(top, src, ta);

      if (t.refined && t.text_mt && t.text_mt !== t.text) {
        const d = el("details", "mt-was");
        const sm = el("summary");
        sm.textContent = t.llm_note ? `AI changed this — ${t.llm_note}` : "AI changed this";
        const dv = el("div"); dv.textContent = t.text_mt; dv.dir = "auto";
        d.append(sm, dv); body.append(d);
      }
      row.append(stripe, body);
      lines.append(row);
    }
    pane.append(lines);
  }

  notes(pane, ["translate", "refine"]);
  techDetail(pane, STEPS[3].stages);
  if ((project.translated_langs || []).length) nextButton(pane, 4, "Cast voices");
}

// ── step 5 · voices & dub ─────────────────────────────────────────────────

function paneDub(root) {
  const pane = el("div", "pane");
  root.append(pane);
  head(pane, "Step 5 of 5", "Cast voices and dub",
    "Give every speaker a voice, then produce the dub. AI voices are ready to use; a real "
    + "performer's voice needs consent on file first.");

  const uncast = speakers.filter(s => !s.voice_id);
  const list = el("div", "spk-list");
  for (const s of speakers) {
    const row = el("div", "spk");
    const sw = el("span", "sw"); sw.style.background = s.color;
    const nm = el("div");
    const t = el("div"); t.textContent = s.name; t.style.fontSize = "14px";
    const m = el("div", "meta"); m.textContent = `${s.segment_count} lines · ${Math.round(s.speech_seconds || 0)}s`;
    nm.append(t, m);
    const spacer = el("span");
    row.append(sw, nm, spacer, castButton(s));
    list.append(row);
  }
  pane.append(list);

  const act = el("div", "act");
  const running = project.status === "dubbing";
  const b = el("button", "btn big"); b.type = "button";
  b.disabled = running || uncast.length > 0 || !activeLang;
  b.textContent = running ? "Dubbing…" : (project.renders?.[activeLang] ? "Dub again" : "Produce the dub");
  b.onclick = async () => {
    b.disabled = true;
    try {
      await api(`/api/projects/${project.id}/dub`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ lang: activeLang }),
      });
      toast("Dubbing — this runs a while on CPU."); refresh(true);
    } catch (e) { toast(e.message, "bad"); b.disabled = false; }
  };
  act.append(b);
  if (uncast.length) {
    const w = el("div", "why grow");
    w.textContent = `Cast a voice for ${uncast.map(s => s.name).join(", ")} first.`;
    act.append(w);
  }
  pane.append(act);

  const url = project.renders?.[activeLang];
  if (url) {
    const h = el("div", "eyebrow"); h.textContent = "Your dub"; h.style.marginTop = "8px";
    const v = el("video", "player");
    v.controls = true; v.preload = "metadata"; v.src = url;
    if (project.media?.poster) v.poster = project.media.poster;
    const dl = el("a", "btn");
    dl.href = url; dl.download = `${project.title}-${activeLang}.mp4`;
    dl.textContent = "Download"; dl.style.textDecoration = "none";
    dl.style.display = "inline-block"; dl.style.marginTop = "12px";
    pane.append(h, v, dl);
  }

  notes(pane, ["synth", "fit", "mix"]);
  techDetail(pane, STEPS[4].stages);
}

// ── casting ───────────────────────────────────────────────────────────────

let voiceFilter = { cat: "all", q: "" };

function openCast(speaker) {
  castingFor = speaker;
  renderVoiceDialog();
  $("#voicesDlg").showModal();
}

function renderVoiceDialog() {
  const f = $("#voiceFilter");
  f.textContent = "";
  const cats = [["all", "All"], ["male", "Male"], ["female", "Female"], ["child", "Kids"], ["cloned", "Performers"]];
  for (const [key, label] of cats) {
    const c = el("button", "chip"); c.type = "button";
    c.textContent = label;
    c.setAttribute("aria-pressed", String(voiceFilter.cat === key));
    c.onclick = () => { voiceFilter.cat = key; renderVoiceDialog(); };
    f.append(c);
  }
  const q = el("input"); q.type = "text"; q.placeholder = "Search voices";
  q.value = voiceFilter.q;
  q.oninput = () => { voiceFilter.q = q.value; renderVoiceGrid(); };
  f.append(q);
  renderVoiceGrid();
}

function renderVoiceGrid() {
  const grid = $("#voiceGrid");
  grid.textContent = "";
  const term = voiceFilter.q.trim().toLowerCase();
  const rows = voices.filter(v => {
    if (voiceFilter.cat === "cloned") { if (v.kind === "synthetic") return false; }
    else if (voiceFilter.cat !== "all" && v.category !== voiceFilter.cat) return false;
    if (term && !(`${v.display_name} ${v.actor_name || ""}`.toLowerCase().includes(term))) return false;
    return true;
  });
  if (!rows.length) {
    const e = el("p", "hint"); e.textContent = "No voices match.";
    grid.append(e); return;
  }
  for (const v of rows) {
    const c = el("div", "vcard");
    if (!v.usable) c.classList.add("blocked");
    if (castingFor && castingFor.voice_id === v.id) c.classList.add("picked");

    const top = el("div", "vtop");
    const n = el("div", "vname"); n.textContent = v.display_name;
    const k = el("span", "vkind");
    k.dataset.k = v.kind === "synthetic" ? "ai" : "cloned";
    k.textContent = v.kind === "synthetic" ? "AI" : "Performer";
    top.append(n, k);
    c.append(top);

    const clip = (v.clips || [])[0];
    if (clip) {
      const a = el("audio"); a.controls = true; a.preload = "none"; a.src = clip.url;
      c.append(a);
    }
    if (!v.usable) {
      const w = el("div", "vwhy"); w.textContent = v.block_reason || "not usable";
      c.append(w);
      if (v.kind !== "synthetic") {
        const act = el("div", "vact");
        const cb = el("button", "btn"); cb.type = "button"; cb.textContent = "Add consent";
        cb.onclick = () => openConsent(v);
        act.append(cb); c.append(act);
      }
    } else if (castingFor) {
      const act = el("div", "vact");
      const pick = el("button", "btn"); pick.type = "button";
      pick.textContent = castingFor.voice_id === v.id ? "Cast" : "Use this voice";
      pick.onclick = async () => {
        try {
          await api(`/api/speakers/${castingFor.id}/cast`, {
            method: "POST", headers: { "content-type": "application/json" },
            body: JSON.stringify({ voice_id: v.id }),
          });
          castingFor.voice_id = v.id;
          toast(`${castingFor.name} → ${v.display_name}`, "good");
          $("#voicesDlg").close();
          refresh(true);
        } catch (e) { toast(e.message, "bad"); }
      };
      act.append(pick); c.append(act);
    }
    grid.append(c);
  }
}

// ── voice consent ─────────────────────────────────────────────────────────

let consentTarget = null;
function openConsent(v) {
  consentTarget = v;
  $("#cSig").value = v.consent?.signatory || v.actor_name || "";
  $("#cRef").value = v.consent?.agreement_ref || "";
  $("#cScope").value = v.consent?.scope || "";
  $("#cExp").value = (v.consent?.expires_at || "").slice(0, 10);
  $("#consentDlg").showModal();
}
$("#consentForm").addEventListener("submit", async e => {
  if (e.submitter?.value !== "ok") return;
  const body = {
    signatory: $("#cSig").value.trim(),
    agreement_ref: $("#cRef").value.trim() || null,
    scope: $("#cScope").value.trim() || null,
    expires_at: $("#cExp").value.trim() || null,
  };
  if (!body.signatory) { e.preventDefault(); toast("Who signed the agreement?", "bad"); return; }
  try {
    await api(`/api/voices/${consentTarget.id}/consent`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    voices = await api("/api/voices");
    renderVoiceGrid();
    toast("Consent recorded — the voice is now usable", "good");
  } catch (err) { toast(err.message, "bad"); }
});

// ── chrome ────────────────────────────────────────────────────────────────

const LANGS = [["hi","Hindi"],["en","English"],["bn","Bengali"],["ta","Tamil"],["te","Telugu"],
               ["mr","Marathi"],["gu","Gujarati"],["kn","Kannada"],["ml","Malayalam"],
               ["pa","Punjabi"],["or","Odia"],["as","Assamese"],["ur","Urdu"]];
function openNew() {
  const fill = (sel, def) => {
    sel.textContent = "";
    for (const [c, n] of LANGS) {
      const o = el("option"); o.value = c; o.textContent = n;
      if (c === def) o.selected = true;
      sel.append(o);
    }
  };
  fill($("#srcLang"), "hi"); fill($("#tgtLang"), "en");
  $("#ttl").value = "";
  $("#newDlg").showModal();
}
$("#newBtn").onclick = openNew;
$("#newForm").addEventListener("submit", async e => {
  if (e.submitter?.value !== "ok") return;
  const title = $("#ttl").value.trim();
  if (!title) { e.preventDefault(); return; }
  try {
    const p = await api("/api/projects", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({
        title, source_lang: $("#srcLang").value, target_langs: [$("#tgtLang").value],
      }),
    });
    stepIndex = 0;
    toast("Project created — add your video", "good");
    openProject(p);   // straight into the studio, at step 1
  } catch (err) { toast(err.message, "bad"); }
});

$("#projectPicker").onclick = () => {
  const m = $("#projectMenu");
  if (!m.classList.contains("hidden")) { m.classList.add("hidden"); return; }
  m.textContent = "";
  if (!projects.length) {
    const e = el("p", "hint"); e.style.padding = "10px"; e.textContent = "No projects yet.";
    m.append(e);
  }
  for (const p of projects) {
    const b = el("button"); b.type = "button";
    const th = el("span", "thumb");
    if (p.media?.poster) th.style.backgroundImage = `url(${p.media.poster})`;
    const t = el("span", "t"); t.textContent = p.title;
    const s = el("span", "lbl"); s.style.margin = "0"; s.textContent = p.status;
    b.append(th, t, s);
    b.onclick = () => { m.classList.add("hidden"); openProject(p); };
    m.append(b);
  }
  m.classList.remove("hidden");
};
document.addEventListener("click", e => {
  if (!e.target.closest("#projectPicker") && !e.target.closest("#projectMenu"))
    $("#projectMenu").classList.add("hidden");
});

$("#voicesBtn").onclick = async () => {
  castingFor = null;
  try { voices = await api("/api/voices"); } catch {}
  renderVoiceDialog();
  $("#voicesDlg").showModal();
};
$("#voicesClose").onclick = () => $("#voicesDlg").close();
$("#vCreate").onclick = async () => {
  const name = $("#vName").value.trim();
  if (!name) { toast("Give the voice a name", "bad"); return; }
  try {
    await api("/api/voices", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ display_name: name, actor_name: $("#vActor").value.trim() || null }),
    });
    $("#vName").value = ""; $("#vActor").value = "";
    voices = await api("/api/voices");
    renderVoiceGrid();
    toast("Voice created — add a recording, then consent", "good");
  } catch (e) { toast(e.message, "bad"); }
};

$("#modelsBtn").onclick = async () => {
  const dlg = $("#modelsDlg"), tb = $("#modelsTable tbody");
  tb.textContent = ""; $("#modelsIntro").textContent = "Loading…";
  dlg.showModal();
  let d;
  try { d = await api("/api/models"); }
  catch (e) { $("#modelsIntro").textContent = e.message; return; }
  $("#modelsIntro").textContent = d.offline
    ? "Running fully offline. Everything loads from this machine."
    : "Models load from this machine where available, and are downloaded once if not.";
  for (const m of d.models) {
    const tr = el("tr");
    for (const [v, col] of [[m.repo], [m.why], [m.gated ? "needs sign-in" : "open"],
                            [m.mirrored ? "yes" : "—"], [m.bytes ? `${Math.round(m.bytes / 1e6)} MB` : "—"]]) {
      const td = el("td"); td.textContent = v; tr.append(td);
    }
    tr.children[3].style.color = m.mirrored ? "var(--ok)" : "var(--text-faint)";
    tb.append(tr);
  }
  const missing = d.models.filter(m => m.gated && !m.mirrored).length;
  $("#modelsHelp").textContent =
    `${Math.round(d.mirrored_bytes / 1e6)} MB stored here · AI editing: ${d.llm}`
    + (missing ? ` · ${missing} optional model(s) not downloaded` : " · nothing else needed");
};
$("#modelsClose").onclick = () => $("#modelsDlg").close();

$("#libBtn").onclick = showLibrary;
$("#brand").onclick = showLibrary;

const themeBtn = $("#themeBtn");
themeBtn.onclick = () => {
  const now = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = now;
  localStorage.setItem("theme", now);
  themeBtn.textContent = now === "dark" ? "Light" : "Dark";
};
const saved = localStorage.getItem("theme");
if (saved) document.documentElement.dataset.theme = saved;
themeBtn.textContent = document.documentElement.dataset.theme === "dark" ? "Light" : "Dark";

refresh(false);
