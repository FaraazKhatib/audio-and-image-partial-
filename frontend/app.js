/* VeriTrust frontend. Same origin as the API, so paths are relative. */

const API = "/api/v1";

const UNCERTAIN = "uncertain";
const FACE_PATHWAY = "face_pathway";

const VERDICT_WORD_IMAGE = {
  likely_authentic: "Likely a real photograph",
  uncertain: "Not enough evidence either way",
  likely_ai_generated: "Likely AI generated",
};

const VERDICT_WORD_AUDIO = {
  likely_authentic: "Likely a real recording",
  uncertain: "Not enough evidence either way",
  likely_ai_generated: "Likely AI generated",
};

const el = (id) => document.getElementById(id);

const dom = {
  drop: el("drop"),
  dropTitle: el("dropTitle"),
  dropHint: el("dropHint"),
  file: el("file"),
  canvas: el("canvas"),
  audioPreview: el("audioPreview"),
  imageMeta: el("imageMeta"),
  clear: el("clear"),
  run: el("run"),
  stageNote: el("stageNote"),
  readoutEmpty: el("readoutEmpty"),
  report: el("report"),
  verdict: el("verdict"),
  verdictWord: el("verdictWord"),
  verdictSub: el("verdictSub"),
  scaleBand: el("scaleBand"),
  scaleTicks: el("scaleTicks"),
  needle: el("needle"),
  needleValue: el("needleValue"),
  scaleLegendLeft: el("scaleLegendLeft"),
  scaleLegendRight: el("scaleLegendRight"),
  scaleCaption: el("scaleCaption"),
  signalRows: el("signalRows"),
  signalNote: el("signalNote"),
  notesPanel: el("notesPanel"),
  notes: el("notes"),
  raw: el("raw"),
  status: el("status"),
  statusText: el("statusText"),
  footStatus: el("footStatus"),
  introHeading: el("introHeading"),
  introBody: el("introBody"),
  canvasFrame: el("canvasFrame"),
};

let selected = null;
let currentMode = "audio";
let thresholds = { authentic_max: 0.35, ai_min: 0.65 };

/* ---------- detector status ---------- */

async function loadStatus() {
  try {
    const response = await fetch(`${API}/models`);
    if (!response.ok) throw new Error(`status ${response.status}`);
    const data = await response.json();

    const ready = data.ensemble_size || 0;
    const failed = (data.failures || []).length;
    if (data.thresholds) thresholds = data.thresholds;

    dom.status.dataset.state = ready === 0 ? "down" : failed > 0 ? "degraded" : "ok";
    dom.statusText.textContent = `${ready} detector${ready === 1 ? "" : "s"} · ${data.device}`;

    const parts = [
      `${ready} loaded`,
      failed ? `${failed} unavailable` : null,
      `face: ${data.face_backend}`,
      data.calibrated ? `calibrated on ${data.calibration_source}` : "uncalibrated scores",
    ].filter(Boolean);
    dom.footStatus.textContent = parts.join(" · ");

    if (ready === 0) {
      note("No detector loaded. Run scripts/verify_models.py to see why.", "error");
    }
  } catch (error) {
    dom.status.dataset.state = "down";
    dom.statusText.textContent = "backend unreachable";
    dom.footStatus.textContent = "backend unreachable";
  }
}

/* ---------- file selection & mode ---------- */


function note(message, kind) {
  dom.stageNote.textContent = message || "";
  if (kind) dom.stageNote.dataset.kind = kind;
  else delete dom.stageNote.dataset.kind;
}

const AUDIO_EXTENSIONS = ["wav", "wave", "mp3", "flac", "ogg", "oga", "opus", "m4a", "m4b", "aac", "weba", "webm", "mp4"];
const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"];

/* Containers that hold an audio stream even though the browser labels them video. An mp4 or webm
   with audio in it is a legitimate thing to analyse, and the backend finds the stream or says it
   could not. */
const AUDIO_IN_VIDEO_TYPES = ["video/mp4", "video/webm", "video/quicktime", "video/x-m4v"];

/* file.type comes from the operating system's association table, not from the bytes, and it is
   routinely empty for .flac, .m4a and .opus and occasionally for .wav. The backend decides modality
   from magic bytes precisely because the client is not a reliable witness, so refusing here on a
   MIME string the browser guessed would reject files the server reads perfectly well. Extension is
   the fallback, and an unknown file is sent rather than blocked: the server sniffs it and returns a
   400 naming the real problem, which is better than a guess made here. */
function classify(file) {
  const type = (file.type || "").toLowerCase();
  if (type.startsWith("image/")) return "image";
  if (type.startsWith("audio/")) return "audio";
  if (AUDIO_IN_VIDEO_TYPES.includes(type)) return "audio";

  const extension = (file.name || "").split(".").pop().toLowerCase();
  if (IMAGE_EXTENSIONS.includes(extension)) return "image";
  if (AUDIO_EXTENSIONS.includes(extension)) return "audio";
  return "unknown";
}

function accept(file) {
  if (!file) return;

  selected = file;
  const url = URL.createObjectURL(file);

  dom.audioPreview.src = url;
  dom.audioPreview.hidden = false;
  dom.canvasFrame.classList.add("audio-mode");
  dom.imageMeta.textContent = `Audio · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  
  dom.canvas.hidden = false;
  dom.drop.hidden = true;

  dom.readoutEmpty.hidden = false;
  dom.report.hidden = true;
  dom.run.disabled = false;
  dom.run.classList.remove("busy");
  note("Ready. Press analyse.");
}

function resetReport() {
  dom.report.hidden = true;
  dom.readoutEmpty.hidden = false;
  dom.boxes.innerHTML = "";
  dom.overlayNote.textContent = "";
  dom.heat.hidden = true;
  dom.heat.removeAttribute("src");
  dom.toggleHeat.hidden = true;
  dom.toggleHeat.setAttribute("aria-pressed", "false");
  dom.toggleHeat.textContent = "Show attention map";
}

function clearAll() {
  selected = null;
  dom.file.value = "";
  dom.canvas.hidden = true;
  dom.drop.hidden = false;
  dom.run.disabled = true;
  dom.audioPreview.removeAttribute("src");
  dom.audioPreview.hidden = true;
  dom.run.textContent = "Analyse audio";
  resetReport();
  note("");
}

dom.file.addEventListener("change", (event) => accept(event.target.files[0]));

dom.drop.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    dom.file.click();
  }
});

["dragenter", "dragover"].forEach((type) =>
  dom.drop.addEventListener(type, (event) => {
    event.preventDefault();
    dom.drop.classList.add("over");
  })
);

["dragleave", "drop"].forEach((type) =>
  dom.drop.addEventListener(type, (event) => {
    event.preventDefault();
    dom.drop.classList.remove("over");
  })
);

dom.drop.addEventListener("drop", (event) => accept(event.dataTransfer.files[0]));

dom.clear.addEventListener("click", clearAll);



/* ---------- analysis ---------- */

dom.run.addEventListener("click", async () => {
  if (!selected) return;

  dom.run.classList.add("busy");
  dom.run.textContent = "Analysing";
  note("Running three checks. First run loads model weights and can take a while.");

  const body = new FormData();
  body.append("file", selected);

  try {
    const response = await fetch(`${API}/analyze`, { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || `status ${response.status}`);
    render(data);
    note(`Done in ${Object.values(data.timing_ms).reduce((a, b) => a + b, 0).toFixed(0)} ms`);
  } catch (error) {
    note(error.message || "Analysis failed.", "error");
  } finally {
    dom.run.classList.remove("busy");
    dom.run.textContent = "Analyse audio";
  }
});

/* ---------- rendering ---------- */

function render(data) {
  if (data.score_ai !== undefined) {
    data.signals = [{
      name: "Audio Ensemble",
      p_ai: data.score_ai,
      weight: 1.0,
      clamped: false,
      counted: true,
      detail: "Average fusion score"
    }];
  }

  dom.readoutEmpty.hidden = true;
  dom.report.hidden = false;

  dom.verdict.dataset.verdict = data.verdict;
  dom.verdictWord.textContent = verdictHeadline(data);
  dom.verdictSub.textContent = verdictFacts(data);

  renderScale(data);
  renderSignals(data);
  renderNotes(data.notes, data.errors);
  renderMediaMeta(data);

  dom.raw.textContent = JSON.stringify(data, (key, value) => (key === "heatmap" ? undefined : value), 2);
}

/* What the server actually analysed, which is not always what was uploaded. The meta line until now
   showed the file size the browser knew about, so a 10 minute recording cut to 600 s and a stereo
   file averaged to mono both read as a complete analysis. These come from the response. */
function renderMediaMeta(data) {
  const media = data.image || {};

  const parts = [];
  if (typeof media.duration === "number") parts.push(`${media.duration.toFixed(1)} s analysed`);
  if (media.truncated && typeof media.original_duration === "number") {
    parts.push(`of ${media.original_duration.toFixed(0)} s`);
  }
  if (media.windows_scored) parts.push(`${media.windows_scored} window(s)`);
  if (media.sample_rate) {
    const rate = `${(media.sample_rate / 1000).toFixed(1)} kHz`;
    parts.push(
      media.original_sample_rate && media.original_sample_rate !== media.sample_rate
        ? `${(media.original_sample_rate / 1000).toFixed(1)} to ${rate}`
        : rate
    );
  }
  if (media.downmixed && media.original_channels) parts.push(`${media.original_channels}ch to mono`);
  if (parts.length) dom.imageMeta.textContent = `Audio · ${parts.join(" · ")}`;
}

/* Which modality the server actually analysed, read from the response rather than from the open
   tab. The server decides modality from the file header, so the tab can be wrong: an audio file
   whose type the browser did not declare is sent from whichever tab was open, and taking the
   wording from the tab would put "photograph" in the headline of an audio verdict. */
/* The API reports two different situations as the same word. A flat uncertain means nothing was
   decisive. An escalated uncertain means one check was decisive and the rest outvoted it, which is
   the more useful fact and used to be visible only in the notes. */
function verdictHeadline(data) {
  if (data.verdict === UNCERTAIN && data.escalated_by) {
    return `Disputed: ${prettyName(data.escalated_by)} reads AI generated`;
  }
  const dict = VERDICT_WORD_AUDIO;
  return dict[data.verdict] || data.verdict;
}

function verdictFacts(data) {
  const facts = [`score ${data.score_ai.toFixed(3)}`];

  if (data.overridden_by === FACE_PATHWAY) {
    // The margin is deliberately not shown here. The score is one model's raw reading, so a face at
    // 1.00 computes a 100 percent margin, and printing that would claim a certainty no checkpoint
    // in this ensemble has earned.
    facts.push("decided by the face pathway alone");
  } else if (data.overridden_by) {
    facts.push(`set by ${prettyName(data.overridden_by)}`);
  } else if (data.escalated_by) {
    facts.push(`held at uncertain by ${prettyName(data.escalated_by)}`);
  } else if (data.verdict === UNCERTAIN) {
    // Margin is zero for every score inside the band, so printing the number says nothing.
    facts.push("no margin, the score is inside the band");
  } else {
    facts.push(`band margin ${data.confidence.toFixed(0)}%`);
  }

  if (data.spread_exceeds_limit) {
    const overruled = data.overridden_by === FACE_PATHWAY;
    facts.push(`checks ${overruled ? "overruled" : "disagree"} ${oddsGap(data.logit_spread)}`);
  }
  facts.push(data.calibrated ? "calibrated" : "uncalibrated");
  return facts.join(" · ");
}

/* Two significant figures, because the payload rounds the spread to three decimals and exp is steep
   here: a spread of 7.7836 is 2401x in odds and the rounded 7.784 prints as 2402x. Nothing about
   these detectors supports a fourth digit, and quoting one invites someone to reconcile it. */
function oddsGap(spread) {
  const odds = Math.exp(spread || 0);
  if (odds >= 10000) return "by over 10000x in odds";
  const rounded = odds >= 100 ? Number(odds.toPrecision(2)) : Math.round(odds);
  return `by about ${rounded}x in odds`;
}

function prettyName(name) {
  return String(name).replace(/_/g, " ");
}

/* The same three bands the verdict uses. A hard 0.5 cut was used for the face boxes and the signal
   table before, which painted a reading of 0.55 the same red as one of 1.00 while the headline said
   uncertain, so the colour and the words contradicted each other. */
function bandOf(p) {
  if (p >= thresholds.ai_min) return "flagged";
  if (p <= thresholds.authentic_max) return "authentic";
  return UNCERTAIN;
}

function renderScale(data) {
  const low = thresholds.authentic_max;
  const high = thresholds.ai_min;

  dom.scaleBand.style.left = `${low * 100}%`;
  dom.scaleBand.style.width = `${(high - low) * 100}%`;

  dom.scaleTicks.innerHTML = "";
  const cappedTicks = [];
  
  // Sort by score to detect collisions between labels
  const sortedSignals = [...data.signals].sort((a, b) => a.p_ai - b.p_ai);
  let lastLabelPos = -999;

  sortedSignals.forEach((signal) => {
    const tick = document.createElement("div");
    const classes = ["sig-tick"];
    if (signal.clamped) classes.push("is-capped");
    if (signal.counted === false) classes.push("is-uncounted");
    
    const pos = signal.p_ai * 100;
    
    // Only apply staggering logic if the label is actually visible
    if (signal.counted !== false) {
      // If this label is too close to the last standard-height label, drop it down
      if (pos - lastLabelPos < 15) {
        classes.push("alt-label");
        lastLabelPos = -999; // The next label is safe to sit at the standard height
      } else {
        lastLabelPos = pos;
      }
    }

    tick.className = classes.join(" ");
    tick.style.left = `${pos}%`;
    tick.dataset.label = shortName(signal.name);
    tick.title = [
      `${signal.name}: ${signal.p_ai.toFixed(3)}`,
      signal.clamped ? "capped before fusing, so the needle does not follow it this far" : "",
      signal.counted === false ? "did not contribute to the score" : "",
    ]
      .filter(Boolean)
      .join(", ");
    if (signal.clamped) cappedTicks.push(shortName(signal.name));
    dom.scaleTicks.appendChild(tick);
  });

  requestAnimationFrame(() => {
    dom.needle.style.left = `${data.score_ai * 100}%`;
  });
  dom.needleValue.textContent = data.score_ai.toFixed(2);
  dom.needleValue.classList.toggle("at-start", data.score_ai < 0.08);
  dom.needleValue.classList.toggle("at-end", data.score_ai > 0.92);

  const cappedNote = cappedTicks.length
    ? ` Dashed ticks (${cappedTicks.join(", ")}) were capped before fusing, which is why the needle does not follow them to the edge.`
    : "";

  /* Provenance and the face pathway both arrive here as overrides and are not the same claim. One
     read metadata, the other is a model estimate that was allowed to decide, and telling the reader
     a detector "found explicit evidence" would overstate it. */
  if (data.overridden_by === FACE_PATHWAY) {
    dom.scaleCaption.textContent =
      `Position is the face pathway's own reading. It landed outside the bands, so it took the ` +
      `verdict by itself and nothing was averaged. The other ticks are where those checks read, ` +
      `and they were overruled rather than combined.${cappedNote}`;
  } else if (data.overridden_by) {
    dom.scaleCaption.textContent = `Position set directly by ${data.overridden_by}, which found explicit evidence rather than an estimate.`;
  } else if (data.escalated_by) {
    dom.scaleCaption.textContent = `The needle is the ensemble average, which sits in the authentic band. ${data.escalated_by} disagrees strongly enough that the verdict is held at uncertain anyway.${cappedNote}`;
  } else {
    dom.scaleCaption.textContent = `Ticks are the individual checks as they read them. Wide spread between them means the models disagree.${cappedNote}`;
  }
}

function shortName(name) {
  return name.replace(/_/g, " ").replace("pathway", "").trim().slice(0, 14);
}

function renderSignals(data) {
  dom.signalRows.innerHTML = "";

  if (!data.signals.length) {
    dom.signalRows.innerHTML = `<tr><td colspan="3">No check produced a usable result.</td></tr>`;
    dom.signalNote.textContent = "";
    return;
  }

  let excluded = 0;
  data.signals.forEach((signal) => {
    const row = document.createElement("tr");
    const band = bandOf(signal.p_ai);
    const capped = signal.clamped
      ? `<span class="sig-capped" title="Capped before fusing. This model is not calibrated, so a near certain reading is not treated as one.">capped</span>`
      : "";
    const uncounted = signal.counted === false
      ? `<span class="sig-uncounted" title="This reading did not contribute to the score, either because its weight is zero or because the fitted calibration does not include it. It can still hold the verdict at uncertain.">not counted</span>`
      : "";
    if (signal.counted === false) excluded += 1;
    row.innerHTML = `
      <td>
        <span class="sig-name">${escapeHtml(signal.name)}</span>
        ${signal.detail ? `<span class="sig-detail">${escapeHtml(signal.detail)}</span>` : ""}
      </td>
      <td><span class="sig-read" data-band="${band}">${signal.p_ai.toFixed(3)}</span>${capped}${uncounted}</td>
      <td>${signal.counted === false ? "none" : signal.weight.toFixed(2)}</td>
    `;
    dom.signalRows.appendChild(row);
  });

  const base = data.calibrated
    ? "Reads are probability estimates from a fitted calibration, and the weight column shows the fitted coefficient's prior, not the coefficient itself."
    : "Reads are relative, not probabilities. Calibration has not been fitted yet.";
  dom.signalNote.textContent = excluded
    ? `${base} ${excluded} reading did not contribute to the score.`
    : base;
}


function renderNotes(notes, errors) {
  dom.notes.innerHTML = "";
  (notes || []).forEach((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    dom.notes.appendChild(item);
  });
  (errors || []).forEach((entry) => {
    const item = document.createElement("li");
    item.textContent = `${entry.detector} failed: ${entry.error}`;
    dom.notes.appendChild(item);
  });
  dom.notesPanel.hidden = dom.notes.children.length === 0;
}


function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

loadStatus();
