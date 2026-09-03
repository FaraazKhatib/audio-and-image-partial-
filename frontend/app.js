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
  mediaModeTabs: el("mediaModeTabs"),
  drop: el("drop"),
  dropTitle: el("dropTitle"),
  dropHint: el("dropHint"),
  file: el("file"),
  canvas: el("canvas"),
  preview: el("preview"),
  audioPreview: el("audioPreview"),
  heat: el("heat"),
  boxes: el("boxes"),
  imageMeta: el("imageMeta"),
  overlayNote: el("overlayNote"),
  toggleHeat: el("toggleHeat"),
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
  provPanel: el("provPanel"),
  evidence: el("evidence"),
  notesPanel: el("notesPanel"),
  notes: el("notes"),
  raw: el("raw"),
  status: el("status"),
  statusText: el("statusText"),
  footStatus: el("footStatus"),
  introHeading: el("introHeading"),
  introBody: el("introBody"),
  dropIconImage: el("dropIconImage"),
  dropIconAudio: el("dropIconAudio"),
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

function setMediaMode(mode) {
  if (currentMode === mode) return;
  currentMode = mode;
  clearAll();

  // Update tabs
  dom.mediaModeTabs.querySelectorAll(".tab").forEach((tab) => {
    if (tab.dataset.mode === mode) tab.classList.add("active");
    else tab.classList.remove("active");
  });

  // Update intro text
  if (mode === "image") {
    dom.introHeading.textContent = "Is this image a photograph, or was it generated?";
    dom.introBody.textContent =
      "Upload an image and VeriTrust runs three independent checks: whole image generation cues, " +
      "face replacement cues, and any provenance metadata the file still carries. You see each " +
      "check separately, not just a single number.";
  } else {
    dom.introHeading.textContent = "Is this audio a real recording, or was it synthesised?";
    // No model count here on purpose. The number that actually reads a file is whatever resolved at
    // startup, which the models panel reports honestly, and copy claiming a fixed count goes stale
    // the moment a checkpoint is added or fails to load.
    dom.introBody.textContent =
      "Upload an audio file and VeriTrust runs an ensemble of independent voice spoofing " +
      "detectors over overlapping four second windows, so a few cloned seconds spliced into a " +
      "real recording are not averaged away. You see each check separately, not just a single " +
      "number.";
  }

  // Update drop zone icons and text
  if (mode === "image") {
    dom.file.accept = "image/jpeg,image/png,image/webp,image/bmp,image/tiff";
    dom.dropTitle.textContent = "Choose an image or drop it here";
    dom.dropHint.textContent = "JPEG, PNG, WebP, BMP, TIFF · up to 20 MB";
    dom.run.textContent = "Analyse image";
    dom.scaleLegendLeft.textContent = "0.00 photographic";
    dom.dropIconImage.hidden = false;
    dom.dropIconAudio.hidden = true;
  } else {
    dom.file.accept = "audio/wav,audio/mpeg,audio/flac,audio/ogg,audio/mp4,audio/webm,audio/aac";
    dom.dropTitle.textContent = "Choose an audio file or drop it here";
    dom.dropHint.textContent = "WAV, MP3, FLAC, M4A, OGG · up to 40 MB";
    dom.run.textContent = "Analyse audio";
    dom.scaleLegendLeft.textContent = "0.00 real voice";
    dom.dropIconImage.hidden = true;
    dom.dropIconAudio.hidden = false;
  }
}


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
