const STORAGE_KEY = "pseudo_label_review_state_v1";

const state = {
  manifest: null,
  entries: [],
  index: 0,
  decisions: {},
  overlayOpacity: 0.42,
};

const refs = {
  totalCount: document.getElementById("totalCount"),
  acceptedCount: document.getElementById("acceptedCount"),
  rejectedCount: document.getElementById("rejectedCount"),
  progressPill: document.getElementById("progressPill"),
  decisionPill: document.getElementById("decisionPill"),
  diceValue: document.getElementById("diceValue"),
  canvas: document.getElementById("previewCanvas"),
  opacitySlider: document.getElementById("opacitySlider"),
  summaryBlock: document.getElementById("summaryBlock"),
  metaId: document.getElementById("metaId"),
  metaSource: document.getElementById("metaSource"),
  metaName: document.getElementById("metaName"),
  metaSelectionScore: document.getElementById("metaSelectionScore"),
  metaMeanConfidence: document.getElementById("metaMeanConfidence"),
  metaReliableRatio: document.getElementById("metaReliableRatio"),
  metaFgReliableRatio: document.getElementById("metaFgReliableRatio"),
  metaObjectRatio: document.getElementById("metaObjectRatio"),
  rejectBtn: document.getElementById("rejectBtn"),
  clearBtn: document.getElementById("clearBtn"),
  acceptBtn: document.getElementById("acceptBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  exportJsonBtn: document.getElementById("exportJsonBtn"),
  exportCsvBtn: document.getElementById("exportCsvBtn"),
};

function formatNumber(value) {
  return Number(value).toFixed(3);
}

function loadPersistedState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return;
  }
  try {
    const parsed = JSON.parse(raw);
    state.decisions = parsed.decisions || {};
    state.index = Number.isInteger(parsed.index) ? parsed.index : 0;
    if (typeof parsed.overlayOpacity === "number") {
      state.overlayOpacity = parsed.overlayOpacity;
    }
  } catch (error) {
    console.warn("Failed to restore review state", error);
  }
}

function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      decisions: state.decisions,
      index: state.index,
      overlayOpacity: state.overlayOpacity,
    }),
  );
}

async function loadImage(src) {
  const image = new Image();
  image.decoding = "async";
  image.src = src;
  await image.decode();
  return image;
}

function drawPreview(baseImage, maskImage, boundaryImage) {
  const canvas = refs.canvas;
  const ctx = canvas.getContext("2d");
  canvas.width = baseImage.naturalWidth;
  canvas.height = baseImage.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(baseImage, 0, 0);

  const workCanvas = document.createElement("canvas");
  workCanvas.width = canvas.width;
  workCanvas.height = canvas.height;
  const workCtx = workCanvas.getContext("2d");

  workCtx.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
  const maskPixels = workCtx.getImageData(0, 0, canvas.width, canvas.height);
  for (let index = 0; index < maskPixels.data.length; index += 4) {
    const value = maskPixels.data[index];
    maskPixels.data[index] = 255;
    maskPixels.data[index + 1] = 79;
    maskPixels.data[index + 2] = 87;
    maskPixels.data[index + 3] = value > 127 ? Math.round(255 * state.overlayOpacity) : 0;
  }
  workCtx.putImageData(maskPixels, 0, 0);
  ctx.drawImage(workCanvas, 0, 0);

  if (boundaryImage) {
    workCtx.clearRect(0, 0, canvas.width, canvas.height);
    workCtx.drawImage(boundaryImage, 0, 0, canvas.width, canvas.height);
    const boundaryPixels = workCtx.getImageData(0, 0, canvas.width, canvas.height);
    for (let index = 0; index < boundaryPixels.data.length; index += 4) {
      const value = boundaryPixels.data[index];
      boundaryPixels.data[index] = 29;
      boundaryPixels.data[index + 1] = 204;
      boundaryPixels.data[index + 2] = 255;
      boundaryPixels.data[index + 3] = value > 127 ? 190 : 0;
    }
    workCtx.putImageData(boundaryPixels, 0, 0);
    ctx.drawImage(workCanvas, 0, 0);
  }
}

function currentEntry() {
  return state.entries[state.index] || null;
}

function updateCounters() {
  const accepted = Object.values(state.decisions).filter((value) => value === "accept").length;
  const rejected = Object.values(state.decisions).filter((value) => value === "reject").length;
  refs.totalCount.textContent = String(state.entries.length);
  refs.acceptedCount.textContent = String(accepted);
  refs.rejectedCount.textContent = String(rejected);
}

function updateDecisionPill(entry) {
  const decision = state.decisions[entry.item_id];
  refs.decisionPill.textContent = decision ? decision.toUpperCase() : "No decision";
  refs.decisionPill.className = "pill pill-dark";
  if (decision === "accept") {
    refs.decisionPill.classList.add("decision-accept");
  } else if (decision === "reject") {
    refs.decisionPill.classList.add("decision-reject");
  }
}

function updateMeta(entry) {
  refs.progressPill.textContent = `${state.index + 1} / ${state.entries.length}`;
  refs.diceValue.textContent = formatNumber(entry.predicted_dice);
  refs.metaId.textContent = entry.item_id;
  refs.metaSource.textContent = entry.source;
  refs.metaName.textContent = entry.original_name;
  refs.metaSelectionScore.textContent = formatNumber(entry.selection_score);
  refs.metaMeanConfidence.textContent = formatNumber(entry.avg_confidence);
  refs.metaReliableRatio.textContent = formatNumber(entry.reliable_ratio);
  refs.metaFgReliableRatio.textContent = formatNumber(entry.fg_reliable_ratio);
  refs.metaObjectRatio.textContent = formatNumber(entry.object_ratio);
  refs.summaryBlock.textContent = JSON.stringify(
    {
      sources: state.manifest.source_breakdown,
      dice_calibration: state.manifest.dice_calibration,
      predicted_dice_summary: state.manifest.predicted_dice_summary,
    },
    null,
    2,
  );
  updateDecisionPill(entry);
}

async function render() {
  const entry = currentEntry();
  if (!entry) {
    return;
  }
  updateCounters();
  updateMeta(entry);
  const [baseImage, maskImage] = await Promise.all([
    loadImage(entry.review_image_href),
    loadImage(entry.review_mask_href),
  ]);
  let boundaryImage = null;
  try {
    boundaryImage = await loadImage(entry.review_boundary_mask_href);
  } catch (error) {
    boundaryImage = null;
  }
  drawPreview(baseImage, maskImage, boundaryImage);
  persistState();
}

function move(delta) {
  const next = Math.min(Math.max(state.index + delta, 0), state.entries.length - 1);
  if (next !== state.index) {
    state.index = next;
    render();
  }
}

function setDecision(decision) {
  const entry = currentEntry();
  if (!entry) {
    return;
  }
  if (!decision) {
    delete state.decisions[entry.item_id];
  } else {
    state.decisions[entry.item_id] = decision;
  }
  updateCounters();
  updateDecisionPill(entry);
  persistState();
}

function buildExportRows() {
  return state.entries.map((entry) => ({
    item_id: entry.item_id,
    source: entry.source,
    original_name: entry.original_name,
    predicted_dice: formatNumber(entry.predicted_dice),
    selection_score: formatNumber(entry.selection_score),
    decision: state.decisions[entry.item_id] || "",
  }));
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportJson() {
  downloadBlob(
    new Blob([JSON.stringify(buildExportRows(), null, 2)], { type: "application/json" }),
    "pseudo_label_review_decisions.json",
  );
}

function exportCsv() {
  const rows = buildExportRows();
  const header = Object.keys(rows[0] || {});
  const lines = [header.join(",")];
  rows.forEach((row) => {
    lines.push(
      header
        .map((key) => {
          const value = String(row[key] ?? "").replaceAll('"', '""');
          return `"${value}"`;
        })
        .join(","),
    );
  });
  downloadBlob(new Blob([lines.join("\n")], { type: "text/csv" }), "pseudo_label_review_decisions.csv");
}

async function init() {
  loadPersistedState();
  refs.opacitySlider.value = String(Math.round(state.overlayOpacity * 100));
  const response = await fetch("./data/manifest.json");
  state.manifest = await response.json();
  state.entries = state.manifest.entries || [];
  state.index = Math.min(Math.max(state.index, 0), Math.max(state.entries.length - 1, 0));
  render();
}

refs.rejectBtn.addEventListener("click", () => setDecision("reject"));
refs.clearBtn.addEventListener("click", () => setDecision(null));
refs.acceptBtn.addEventListener("click", () => setDecision("accept"));
refs.prevBtn.addEventListener("click", () => move(-1));
refs.nextBtn.addEventListener("click", () => move(1));
refs.exportJsonBtn.addEventListener("click", exportJson);
refs.exportCsvBtn.addEventListener("click", exportCsv);
refs.opacitySlider.addEventListener("input", (event) => {
  state.overlayOpacity = Number(event.target.value) / 100;
  render();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    setDecision("reject");
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    setDecision("accept");
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    move(-1);
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    move(1);
  } else if (event.key === "Backspace") {
    event.preventDefault();
    setDecision(null);
  } else if (event.key.toLowerCase() === "e") {
    exportJson();
    exportCsv();
  }
});

init().catch((error) => {
  console.error(error);
  refs.summaryBlock.textContent = `Failed to load manifest:\n${String(error)}`;
});
