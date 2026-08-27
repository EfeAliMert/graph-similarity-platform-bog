const leftEditor = document.querySelector("#leftGraph");
const rightEditor = document.querySelector("#rightGraph");
const datasetSelect = document.querySelector("#datasetSelect");
const leftGraphSelect = document.querySelector("#leftGraphSelect");
const rightGraphSelect = document.querySelector("#rightGraphSelect");
const bestMethodSelect = document.querySelector("#bestMethodSelect");
const bestPairScope = document.querySelector("#bestPairScope");
const bestPairLimit = document.querySelector("#bestPairLimit");
const bestPairPanel = document.querySelector("#bestPairPanel");
const message = document.querySelector("#message");
const resultsTable = document.querySelector("#resultsTable");
const groundTruthPanel = document.querySelector("#groundTruthPanel");
const sampleMeta = document.querySelector("#sampleMeta");
const statsLeft = document.querySelector("#statsLeft");
const statsRight = document.querySelector("#statsRight");
const previewLargeLeft = document.querySelector("#previewLargeLeft");
const previewLargeRight = document.querySelector("#previewLargeRight");
const previewLeftStats = document.querySelector("#previewLeftStats");
const previewRightStats = document.querySelector("#previewRightStats");
const previewLeftName = document.querySelector("#previewLeftName");
const previewRightName = document.querySelector("#previewRightName");
const previewLeftLegend = document.querySelector("#previewLeftLegend");
const previewRightLegend = document.querySelector("#previewRightLegend");
const previewMeta = document.querySelector("#previewMeta");
const datasetDetails = document.querySelector("#datasetDetails");
const accuracySampleSize = document.querySelector("#accuracySampleSize");
const accuracySampleMode = document.querySelector("#accuracySampleMode");
const accuracyTopK = document.querySelector("#accuracyTopK");
const accuracySeed = document.querySelector("#accuracySeed");
const evaluateBtn = document.querySelector("#evaluateBtn");
const exportJsonBtn = document.querySelector("#exportJsonBtn");
const exportCsvBtn = document.querySelector("#exportCsvBtn");
const accuracyTable = document.querySelector("#accuracyTable");
const retrievalBudgets = document.querySelector("#retrievalBudgets");
const retrievalTopK = document.querySelector("#retrievalTopK");
const retrievalModelSelect = document.querySelector("#retrievalModelSelect");
const runRetrievalBtn = document.querySelector("#runRetrievalBtn");
const runRerankingBtn = document.querySelector("#runRerankingBtn");
const retrievalPanel = document.querySelector("#retrievalPanel");
const refreshMatrixBtn = document.querySelector("#refreshMatrixBtn");
const matrixPanel = document.querySelector("#matrixPanel");
const datasetUploadForm = document.querySelector("#datasetUploadForm");
const uploadDatasetBtn = document.querySelector("#uploadDatasetBtn");
const datasetUploadStatus = document.querySelector("#datasetUploadStatus");
const infoTooltip = document.querySelector("#infoTooltip");
const loadingOverlay = document.querySelector("#loadingOverlay");
const loadingTitle = document.querySelector("#loadingTitle");
const loadingDetail = document.querySelector("#loadingDetail");
const loadingTrack = document.querySelector("#loadingTrack");
const loadingProgressFill = document.querySelector("#loadingProgressFill");
const loadingProgressMeta = document.querySelector("#loadingProgressMeta");
const loadingStage = document.querySelector("#loadingStage");
const loadingProgressValue = document.querySelector("#loadingProgressValue");
const themeSelect = document.querySelector("#themeSelect");
const runBtn = document.querySelector("#runBtn");
const hpoModeSelect = document.querySelector("#hpoModeSelect");
const pairDataset = document.querySelector("#pairDataset");
const pairLeft = document.querySelector("#pairLeft");
const pairRight = document.querySelector("#pairRight");
let currentMeta = JSON.parse(document.querySelector("#initialMeta")?.textContent || "{}");
let currentDatasetInfo = null;
let comparisonRequestId = 0;
let bestPairRequestId = 0;
let accuracyRequestId = 0;
let retrievalRequestId = 0;
let datasetLoadRequestId = 0;
let graphChoicesRequestId = 0;
let activePairLoadKey = "";
let activePairLoadPromise = null;
let matrixRefreshTimer = null;
let lastBenchmarkPayload = null;
let loadingSequence = 0;
let fullWorkspaceInitialized = false;
let fullWorkspacePromise = null;
const loadingOperations = new Map();
const THEME_STORAGE_KEY = "graph-sim-theme";

runBtn.addEventListener("click", runComparison);
document.querySelector("#loadDatasetBtn").addEventListener("click", loadSelectedDataset);
document.querySelector("#swapBtn").addEventListener("click", swapGraphs);
document.querySelector("#formatBtn").addEventListener("click", formatEditors);
document.querySelector("#findBestBtn").addEventListener("click", findBestPair);
leftGraphSelect?.addEventListener("change", loadSelectedDataset);
rightGraphSelect?.addEventListener("change", loadSelectedDataset);
evaluateBtn?.addEventListener("click", runAccuracyCheck);
exportJsonBtn?.addEventListener("click", () => exportBenchmark("json"));
exportCsvBtn?.addEventListener("click", () => exportBenchmark("csv"));
runRetrievalBtn?.addEventListener("click", runRetrievalAblation);
runRerankingBtn?.addEventListener("click", runGnnRerankingAblation);
refreshMatrixBtn?.addEventListener("click", () =>
  runWithLoading(
    "Refreshing research status",
    "Reading checkpoint audits and benchmark artifacts...",
    loadResearchMatrix,
  ),
);
datasetUploadForm?.addEventListener("submit", uploadDataset);
themeSelect?.addEventListener("change", () => {
  setInterfaceTheme(themeSelect.value);
});
document.addEventListener("mouseover", (event) => {
  const trigger = event.target.closest?.(".info-tip");
  if (trigger) showInfoTooltip(trigger);
});
document.addEventListener("mouseout", (event) => {
  const trigger = event.target.closest?.(".info-tip");
  if (trigger && !trigger.contains(event.relatedTarget)) hideInfoTooltip();
});
document.addEventListener("focusin", (event) => {
  const trigger = event.target.closest?.(".info-tip");
  if (trigger) showInfoTooltip(trigger);
});
document.addEventListener("focusout", (event) => {
  if (event.target.closest?.(".info-tip")) hideInfoTooltip();
});
document.addEventListener("click", (event) => {
  const trigger = event.target.closest?.(".info-tip");
  if (!trigger) return;
  event.preventDefault();
  event.stopPropagation();
  showInfoTooltip(trigger);
});
window.addEventListener("scroll", hideInfoTooltip, { passive: true });
window.addEventListener("resize", hideInfoTooltip);
datasetSelect.addEventListener("change", async () => {
  const datasetId = datasetSelect.value;
  const loadingToken = beginLoading(
    "Switching dataset",
    `Loading graph catalog for ${selectedOptionLabel(datasetSelect)}...`,
  );
  comparisonRequestId++;
  bestPairRequestId++;
  accuracyRequestId++;
  retrievalRequestId++;
  lastBenchmarkPayload = null;
  syncBenchmarkExports();
  document.querySelector("#findBestBtn").disabled = false;
  try {
    const choicesLoaded = await loadGraphChoices({ preserveCurrent: false, datasetId });
    if (!choicesLoaded || datasetSelect.value !== datasetId) return;
    const pairLoaded = await loadSelectedDataset({ datasetId });
    if (!pairLoaded || datasetSelect.value !== datasetId) return;
    loadResearchMatrix();
  } finally {
    endLoading(loadingToken);
  }
});

renderPairMeta(currentMeta);
initializeInterfaceTheme();
setupSectionTabs();
initializeApp();

function initializeInterfaceTheme() {
  const activeTheme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  setInterfaceTheme(activeTheme, { persist: false });
}

function setInterfaceTheme(theme, options = {}) {
  const activeTheme = theme === "dark" ? "dark" : "light";
  if (activeTheme === "dark") {
    document.documentElement.dataset.theme = "dark";
  } else {
    delete document.documentElement.dataset.theme;
  }
  if (themeSelect) themeSelect.value = activeTheme;
  if (options.persist !== false) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, activeTheme);
    } catch (_error) {
      // Theme switching remains available even if persistence is blocked.
    }
  }
}

function setupSectionTabs() {
  const links = [...document.querySelectorAll(".section-tabs a[href^='#']")];
  if (!links.length) return;

  const setActive = (activeLink) => {
    links.forEach((link) => link.classList.toggle("is-active", link === activeLink));
  };

  links.forEach((link) => {
    link.addEventListener("click", () => setActive(link));
  });

  if (!("IntersectionObserver" in window)) return;
  const targetLinks = new Map(
    links
      .map((link) => [document.querySelector(link.getAttribute("href")), link])
      .filter(([target]) => target),
  );
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
      if (visible) setActive(targetLinks.get(visible.target));
    },
    { rootMargin: "-18% 0px -68% 0px", threshold: [0, 0.2, 0.6] },
  );
  targetLinks.forEach((_link, target) => observer.observe(target));
}

function showInfoTooltip(trigger) {
  if (!infoTooltip || !trigger?.dataset?.tip) return;
  infoTooltip.textContent = trigger.dataset.tip;
  infoTooltip.hidden = false;
  infoTooltip.style.left = "0px";
  infoTooltip.style.top = "0px";
  const triggerRect = trigger.getBoundingClientRect();
  const tooltipRect = infoTooltip.getBoundingClientRect();
  const left = clampRange(
    triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2,
    10,
    Math.max(10, window.innerWidth - tooltipRect.width - 10),
  );
  const above = triggerRect.top - tooltipRect.height - 8;
  const top = above >= 10 ? above : triggerRect.bottom + 8;
  infoTooltip.style.left = `${Math.round(left)}px`;
  infoTooltip.style.top = `${Math.round(top)}px`;
}

function hideInfoTooltip() {
  if (infoTooltip) infoTooltip.hidden = true;
}

async function initializeApp() {
  const loadingToken = beginLoading(
    "Loading platform",
    "Reading datasets and experiment records...",
  );
  try {
    await initializeResearchWorkspace();
  } finally {
    endLoading(loadingToken);
  }
}

async function initializeResearchWorkspace() {
  if (fullWorkspaceInitialized) return;
  if (fullWorkspacePromise) return fullWorkspacePromise;
  fullWorkspacePromise = (async () => {
    await loadGraphChoices({ preserveCurrent: true });
    await loadResearchMatrix();
    await runComparison({ previewOnly: true, skipPairReload: true });
    fullWorkspaceInitialized = true;
  })();
  try {
    await fullWorkspacePromise;
  } finally {
    fullWorkspacePromise = null;
  }
}

async function loadSelectedDataset(options = {}) {
  const datasetId = options?.datasetId || datasetSelect.value;
  const runModelsAfterLoad = options?.runModels === true;
  if (!datasetId) return false;
  const selectedLeft = leftGraphSelect?.value || "";
  const selectedRight = rightGraphSelect?.value || "";
  const pairLoadKey = `${datasetId}\n${selectedLeft}\n${selectedRight}`;
  if (activePairLoadPromise && activePairLoadKey === pairLoadKey) {
    return activePairLoadPromise;
  }

  const requestId = ++datasetLoadRequestId;
  const loadingToken = beginLoading(
    "Loading graph pair",
    `${selectedLeft || "Graph A"} vs ${selectedRight || "Graph B"}`,
  );
  const promise = (async () => {
    setMessage("");
    const params = new URLSearchParams();
    if (selectedLeft && selectedRight) {
      params.set("left", selectedLeft);
      params.set("right", selectedRight);
    }
    const pairPath = params.toString()
      ? `/api/datasets/${encodeURIComponent(datasetId)}/pair?${params.toString()}`
      : `/api/datasets/${encodeURIComponent(datasetId)}`;
    const response = await fetch(pairPath);
    const payload = await response.json();
    if (requestId !== datasetLoadRequestId || datasetSelect.value !== datasetId) {
      return false;
    }
    if (!response.ok) {
      setMessage(payload.error || "Dataset could not be loaded.");
      return false;
    }
    leftEditor.value = JSON.stringify(payload.left, null, 2);
    rightEditor.value = JSON.stringify(payload.right, null, 2);
    currentMeta = payload.meta || {};
    currentDatasetInfo = payload.dataset || currentDatasetInfo;
    updateGroundTruthControls();
    renderPairMeta(currentMeta);
    renderDatasetDetails(payload.dataset, payload.meta);
    await runComparison({
      previewOnly: !runModelsAfterLoad,
      skipPairReload: true,
    });
    return true;
  })();
  activePairLoadKey = pairLoadKey;
  activePairLoadPromise = promise;
  try {
    return await promise;
  } finally {
    endLoading(loadingToken);
    if (activePairLoadPromise === promise) {
      activePairLoadKey = "";
      activePairLoadPromise = null;
    }
  }
}

async function uploadDataset(event) {
  event.preventDefault();
  if (!datasetUploadForm || !uploadDatasetBtn) return;
  const formData = new FormData(datasetUploadForm);
  const archive = formData.get("archive");
  if (!(archive instanceof File) || !archive.name) {
    setUploadStatus("Select a graph archive.", "error");
    return;
  }

  uploadDatasetBtn.disabled = true;
  setUploadStatus("Validating and storing dataset...", "working");
  const loadingToken = beginLoading(
    "Uploading dataset",
    `Validating and storing ${archive.name} locally...`,
  );
  try {
    const response = await fetch("/api/datasets/upload", {
      method: "POST",
      body: formData,
    });
    let payload;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = { error: `Upload failed with HTTP ${response.status}.` };
    }
    if (!response.ok) {
      setUploadStatus(payload.error || "Dataset upload failed.", "error");
      return;
    }

    const dataset = payload.dataset;
    await refreshDatasetCatalog(dataset.id);
    await loadGraphChoices({ preserveCurrent: false });
    await loadSelectedDataset();
    datasetUploadForm.reset();
    const trainingState = dataset.ground_truth_exact
      ? "exact GED training ready for 5 models"
      : dataset.ground_truth_kind === "approximate_benchmark"
        ? "approximate GED benchmark training ready for 5 models"
        : dataset.ground_truth_kind === "unverified_ged"
          ? "user-provided unverified GED training ready for 5 models"
        : "structural proxy training ready for 5 models";
    setUploadStatus(
      `${dataset.name} stored at ${dataset.archive_path} · ${dataset.graph_count} graphs · ${trainingState}.`,
      "success",
    );
  } catch (error) {
    setUploadStatus(`Dataset upload failed: ${error.message}`, "error");
  } finally {
    endLoading(loadingToken);
    uploadDatasetBtn.disabled = false;
  }
}

async function refreshDatasetCatalog(selectedId = datasetSelect.value) {
  const response = await fetch("/api/datasets");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Dataset catalog failed.");
  datasetSelect.innerHTML = "";
  (payload.datasets || []).forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.name} | ${(dataset.tasks || []).join("/")} | train ${dataset.train_graphs} test ${dataset.test_graphs}`;
    datasetSelect.appendChild(option);
  });
  if ([...datasetSelect.options].some((option) => option.value === selectedId)) {
    datasetSelect.value = selectedId;
  }
}

function setUploadStatus(text, state = "") {
  if (!datasetUploadStatus) return;
  datasetUploadStatus.textContent = text;
  datasetUploadStatus.dataset.state = state;
}

async function loadGraphChoices({
  preserveCurrent = false,
  datasetId = datasetSelect.value,
} = {}) {
  if (!datasetId || !leftGraphSelect || !rightGraphSelect) return false;
  const requestId = ++graphChoicesRequestId;
  const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/graphs`);
  const payload = await response.json();
  if (requestId !== graphChoicesRequestId || datasetSelect.value !== datasetId) {
    return false;
  }
  if (!response.ok) {
    setMessage(payload.error || "Graphs could not be loaded.");
    return false;
  }

  const graphMembers = new Set((payload.graphs || []).map((graph) => graph.member));
  const selectedLeft = preserveCurrent && graphMembers.has(currentMeta.left_graph) ? currentMeta.left_graph : "";
  const selectedRight = preserveCurrent && graphMembers.has(currentMeta.right_graph) ? currentMeta.right_graph : "";
  renderGraphSelect(leftGraphSelect, payload.graphs || [], selectedLeft, "train");
  renderGraphSelect(rightGraphSelect, payload.graphs || [], selectedRight, "test");
  const selectedMeta = preserveCurrent
    ? currentMeta
    : { left_graph: leftGraphSelect.value, right_graph: rightGraphSelect.value };
  currentDatasetInfo = payload.dataset || currentDatasetInfo;
  updateGroundTruthControls();
  renderDatasetDetails(payload.dataset, selectedMeta);
  return true;
}

function renderGraphSelect(select, graphs, selectedMember, preferredSplit) {
  select.innerHTML = "";
  const grouped = {
    train: graphs.filter((graph) => graph.split === "train"),
    test: graphs.filter((graph) => graph.split === "test"),
    graph: graphs.filter((graph) => !["train", "test"].includes(graph.split)),
  };

  Object.entries(grouped).forEach(([split, splitGraphs]) => {
    if (!splitGraphs.length) return;
    const group = document.createElement("optgroup");
    group.label = split === "graph" ? "graphs" : split;
    splitGraphs.forEach((graph) => {
      const option = document.createElement("option");
      option.value = graph.member;
      option.textContent = graph.label;
      group.appendChild(option);
    });
    select.appendChild(group);
  });

  const fallback =
    graphs.find((graph) => graph.member === selectedMember)?.member ||
    graphs.find((graph) => graph.split === preferredSplit)?.member ||
    graphs[0]?.member ||
    "";
  select.value = fallback;
}

async function runComparison(options = {}) {
  const skipPairReload = options?.skipPairReload === true;
  const previewOnly = options?.previewOnly === true;
  const datasetId = datasetSelect.value;
  const requestId = ++comparisonRequestId;
  setMessage("");
  if (!skipPairReload && selectedPairChanged()) {
    await loadSelectedDataset({ datasetId, runModels: !previewOnly });
    return;
  }

  let left;
  let right;
  try {
    left = JSON.parse(leftEditor.value);
    right = JSON.parse(rightEditor.value);
  } catch (error) {
    setMessage(`JSON error: ${error.message}`);
    return;
  }

  const methods = previewOnly ? [] : selectedMethodIds();
  if (!previewOnly && methods.length === 0) {
    setMessage("Select at least one model.");
    return;
  }

  const hpoMode = hpoModeSelect?.value || "quick";
  const hpoModeLabel = hpoModeSelect?.selectedOptions?.[0]?.textContent?.trim() || "Quick HPO";
  const loadingToken = beginLoading(
    previewOnly ? "Loading graph preview" : `Running models · ${hpoModeLabel}`,
    previewOnly
      ? "Reading graph structure and exact target metadata..."
      : `${methods.length} model${methods.length === 1 ? "" : "s"} queued for automatic preparation and inference...`,
    previewOnly ? null : { percent: 0, stage: "Dataset check" },
  );
  if (!previewOnly) {
    runBtn.disabled = true;
    if (hpoModeSelect) hpoModeSelect.disabled = true;
  }
  try {
    const requestBody = {
      left,
      right,
      methods,
      preview_only: previewOnly,
      dataset: datasetId,
      meta: currentMeta,
      hpo_mode: hpoMode,
    };
    let payload;
    if (previewOnly) {
      const response = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      payload = await response.json();
      if (!response.ok) {
        setMessage(payload.error || "Comparison failed.");
        return;
      }
    } else {
      payload = await runAutomaticModelJob(
        requestBody,
        requestId,
        datasetId,
        loadingToken,
      );
      if (!payload) return;
    }
    if (requestId !== comparisonRequestId || datasetSelect.value !== datasetId) return;

    renderStats(statsLeft, payload.stats.left);
    renderStats(statsRight, payload.stats.right);
    drawGraph(document.querySelector("#previewLeft"), payload.graphs.left);
    drawGraph(document.querySelector("#previewRight"), payload.graphs.right);
    renderGraphVisualization(payload);
    renderGroundTruth(payload.ground_truth);
    if (previewOnly) {
      if (resultsTable) {
        resultsTable.innerHTML = '<div class="result-row"><div>Select models and click Run Models.</div></div>';
      }
    } else {
      renderResults(payload.results, payload.ground_truth);
    }
    if (!datasetDetails.textContent.trim()) {
      loadDatasetInfo(datasetSelect.value);
    }
  } catch (error) {
    setMessage(`Model run failed: ${error.message}`);
  } finally {
    endLoading(loadingToken);
    if (!previewOnly) {
      runBtn.disabled = false;
      if (hpoModeSelect) hpoModeSelect.disabled = false;
    }
  }
}

async function runAutomaticModelJob(requestBody, requestId, datasetId, loadingToken) {
  const startResponse = await fetch("/api/model-runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });
  const started = await startResponse.json();
  if (!startResponse.ok) {
    throw new Error(started.error || "Automatic model run could not start.");
  }
  const jobId = started.job?.id;
  if (!jobId) throw new Error("The server did not return a model-run job id.");

  while (requestId === comparisonRequestId && datasetSelect.value === datasetId) {
    const response = await fetch(`/api/model-runs/${encodeURIComponent(jobId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Model-run status could not be read.");
    const job = payload.job || {};
    const progress = job.progress || {};
    updateLoading(
      loadingToken,
      progress.title || "Running models",
      progress.detail || "Preparing the selected models...",
      {
        percent: Number(progress.percent || 0),
        stage: progress.stage ? humanizeStage(progress.stage) : "Preparing",
      },
    );
    if (job.status === "completed") return job.result;
    if (["failed", "interrupted"].includes(job.status)) {
      throw new Error(job.error || "The automatic model run did not complete.");
    }
    await delay(1000);
  }
  return null;
}

function humanizeStage(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function findBestPair() {
  if (!datasetSelect.value) return;
  const datasetId = datasetSelect.value;
  const requestId = ++bestPairRequestId;
  comparisonRequestId++;
  const button = document.querySelector("#findBestBtn");
  button.disabled = true;
  setMessage("Searching best pair...");
  if (bestPairPanel) bestPairPanel.innerHTML = "";
  const loadingToken = beginLoading(
    "Finding the most similar pair",
    `${selectedOptionLabel(bestMethodSelect)} · ${selectedOptionLabel(bestPairScope)} · scanning up to ${Number(bestPairLimit?.value || 8)} candidates`,
  );

  try {
    const method = selectedBestPairMethod();
    if (!method) {
      setMessage("Select a runnable model for best-pair search.");
      return;
    }
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/best-pair`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        methods: [method],
        max_pairs: Number(bestPairLimit?.value || 8),
        scope: bestPairScope?.value || "train-test",
      }),
    });
    const payload = await response.json();
    if (requestId !== bestPairRequestId || datasetSelect.value !== datasetId) return;
    if (!response.ok) {
      renderBestPair(payload.search);
      setMessage(payload.error || "Best pair search failed.");
      return;
    }

    leftEditor.value = JSON.stringify(payload.left, null, 2);
    rightEditor.value = JSON.stringify(payload.right, null, 2);
    currentMeta = payload.meta || {};
    if (leftGraphSelect) leftGraphSelect.value = currentMeta.left_graph || leftGraphSelect.value;
    if (rightGraphSelect) rightGraphSelect.value = currentMeta.right_graph || rightGraphSelect.value;
    renderPairMeta(currentMeta);
    renderDatasetDetails(payload.dataset, payload.meta);
    renderStats(statsLeft, payload.stats.left);
    renderStats(statsRight, payload.stats.right);
    drawGraph(document.querySelector("#previewLeft"), payload.graphs.left);
    drawGraph(document.querySelector("#previewRight"), payload.graphs.right);
    renderGraphVisualization(payload);
    renderBestPair(payload.search);
    if (payload.search?.winner?.results) {
      renderGroundTruth(payload.ground_truth || null);
      renderResults(payload.search.winner.results, payload.ground_truth || null);
    }
    setMessage("");
  } finally {
    endLoading(loadingToken);
    button.disabled = false;
  }
}

function selectedPairChanged() {
  if (!leftGraphSelect?.value || !rightGraphSelect?.value) return false;
  return (
    currentMeta.dataset_id !== datasetSelect.value ||
    currentMeta.left_graph !== leftGraphSelect.value ||
    currentMeta.right_graph !== rightGraphSelect.value
  );
}

function selectedMethodIds() {
  return [...document.querySelectorAll("input[name='method']:checked")].map((input) => input.value);
}

function renderStats(target, stats) {
  target.textContent = `nodes ${stats.nodes} | edges ${stats.edges} | density ${stats.density} | components ${stats.components}`;
}

function renderGraphVisualization(payload) {
  if (!payload?.graphs || !payload?.stats) return;
  drawGraph(previewLargeLeft, payload.graphs.left, { large: true });
  drawGraph(previewLargeRight, payload.graphs.right, { large: true });
  renderPreviewStats(previewLeftStats, payload.stats.left);
  renderPreviewStats(previewRightStats, payload.stats.right);
  renderLegend(previewLeftLegend, payload.graphs.left);
  renderLegend(previewRightLegend, payload.graphs.right);
  if (previewLeftName) previewLeftName.textContent = currentMeta.left_graph || "manual input";
  if (previewRightName) previewRightName.textContent = currentMeta.right_graph || "manual input";
  if (previewMeta) previewMeta.textContent = `${currentMeta.left_graph || "manual input"} vs ${currentMeta.right_graph || "manual input"}`;
}

function renderPreviewStats(target, stats) {
  if (!target || !stats) return;
  target.innerHTML = `
    <span><b>${stats.nodes}</b> nodes</span>
    <span><b>${stats.edges}</b> edges</span>
    <span><b>${stats.avg_degree}</b> avg deg</span>
    <span><b>${stats.triangles}</b> triangles</span>
  `;
}

function renderLegend(target, graph) {
  if (!target) return;
  const counts = new Map();
  (graph.nodes || []).forEach((node) => counts.set(String(node.label), (counts.get(String(node.label)) || 0) + 1));
  const rows = [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 8)
    .map(
      ([label, count]) => `
        <span><i style="--swatch: ${colorForLabel(label)}"></i>${escapeHtml(label)} <b>${count}</b></span>
      `
    );
  target.innerHTML = rows.length ? rows.join("") : `<span><i style="--swatch: #97a3b6"></i>empty <b>0</b></span>`;
}

function renderPairMeta(meta) {
  if (!meta) return;
  const dataset = meta.dataset || meta.source || "Manual pair";
  const left = meta.left_graph || "manual input";
  const right = meta.right_graph || "manual input";
  if (pairDataset) pairDataset.textContent = dataset;
  if (pairLeft) pairLeft.textContent = left;
  if (pairRight) pairRight.textContent = right;
  if (sampleMeta) sampleMeta.textContent = `${dataset} | ${left} vs ${right}`;
}

function renderGroundTruth(groundTruth) {
  if (!groundTruthPanel) return;
  if (!groundTruth) {
    groundTruthPanel.innerHTML = "";
    return;
  }
  const referenceLabel = groundTruthReferenceLabel(groundTruth);
  groundTruthPanel.innerHTML = `
    <div>
      <span>${escapeHtml(referenceLabel)} ${escapeHtml(groundTruth.task)}</span>
      <b>${formatNumber(groundTruth.distance)}</b>
    </div>
    <div>
      <span>${escapeHtml(referenceLabel)} similarity</span>
      <b>${(groundTruth.similarity * 100).toFixed(1)}%</b>
    </div>
    <div>
      <span>Normalized ${escapeHtml(groundTruth.task)}</span>
      <b>${formatNumber(groundTruth.normalized_distance)}</b>
    </div>
    <div>
      <span>Source</span>
      <code>${escapeHtml(groundTruth.source)}</code>
    </div>
  `;
}

function groundTruthReferenceLabel(groundTruth) {
  if (groundTruth?.reference_kind === "approximate_benchmark") return "Approximate benchmark";
  if (groundTruth?.reference_kind === "unverified_ged") return "User-provided unverified";
  if (groundTruth?.exact === true || groundTruth?.reference_kind === "exact") return "Exact";
  return "Structural proxy";
}

function renderResults(results, groundTruth = null) {
  if (!results.length) {
    resultsTable.innerHTML = `<div class="result-row"><div>No models returned.</div></div>`;
    return;
  }
  const closestId = closestModelId(results, groundTruth);
  resultsTable.innerHTML = results
    .map((result) => {
      const hasNativeScore = typeof result.model_score === "number";
      const nativePercent = hasNativeScore
        ? Math.round(result.model_score * 1000) / 10
        : null;
      const hasComparableScore =
        typeof result.comparable_similarity === "number";
      const comparablePercent = hasComparableScore
        ? Math.round(result.comparable_similarity * 1000) / 10
        : null;
      const rawRegression = result.adapter_metrics?.raw_score;
      const transformation = result.score_transformation || {};
      const rawModelOutput = transformation.raw_model_output;
      const predictedNormalizedGed = transformation.predicted_normalized_ged;
      const predictedGed = transformation.predicted_ged;
      const calibrationApplied = result.adapter_metrics?.calibration_applied === true;
      const calibrationAuditMse = result.adapter_metrics?.calibration_audit_mse;
      const calibrationAuditRawMse = result.adapter_metrics?.calibration_audit_mse_raw;
      const hyperparameters = result.adapter_metrics?.hyperparameters;
      const hpo = result.adapter_metrics?.hpo;
      const hpoMetadataStatus = result.adapter_metrics?.hpo_metadata_status;
      const hyperparameterLine = hyperparameters && typeof hyperparameters === "object"
        ? Object.entries(hyperparameters).map(([key, value]) => `${key}=${value}`).join(" · ")
        : "";
      const missing = (result.missing_requirements || []).join(", ");
      const missingFiles = (result.missing_files || []).join(", ");
      const checkpoints = (result.checkpoints || []).length;
      const runnableDatasets = (result.runnable_datasets || []).join(", ") || "none";
      const datasetMatch = result.dataset_supported === false ? "no" : "yes";
      const datasetRunnable = result.dataset_runnable === true ? "yes" : result.dataset_runnable === false ? "no" : "unknown";
      const latency = typeof result.latency_ms === "number" ? `${result.latency_ms} ms` : "n/a";
      const errorLine = groundTruthErrorLine(result, groundTruth);
      const splitMetadata = result.adapter_metrics?.pair_split;
      const pairOverlap =
        splitMetadata?.pair_overlap_count ?? splitMetadata?.pair_overlap;
      const validationLine =
        pairOverlap === 0
          ? `verified zero pair overlap · ${escapeHtml(splitMetadata.strategy || "deterministic holdout")}`
          : "not recorded in this checkpoint; retraining required for research claims";
      const checkpointType = result.official_pretrained
        ? "Author-released pretrained checkpoint"
        : "Locally trained checkpoint (not official pretrained weights)";
      const inputVerification =
        result.input_binding === "Direct graph payload"
          ? "direct JSON payload"
          : result.input_matches_dataset_pair === true
            ? "dataset files verified against editors"
            : result.input_matches_dataset_pair === false
              ? "editor content does not match dataset files"
              : "not available";
      return `
        <article class="result-row result-row--model">
          <div class="result-title">
            <strong>${escapeHtml(result.name)}</strong>
            ${result.id === closestId ? `<small class="best-model-badge">Closest to ${escapeHtml(groundTruthReferenceLabel(groundTruth))} GED</small>` : ""}
            <span>${escapeHtml(result.paper)} · ${escapeHtml(result.family)} · end-to-end ${escapeHtml(latency)}</span>
          </div>
          <div class="score" style="--accent: ${result.accent}">
            ${
              hasComparableScore
                ? `<span class="score-label">Comparable similarity</span><b>${comparablePercent.toFixed(1)}%</b><small class="score-status">${escapeHtml(statusText(result))}</small><div class="score-bar"><span style="--width: ${comparablePercent}%"></span></div>`
                : result.status === "executed"
                  ? `<span class="score-label">Comparable similarity</span><b class="status-badge">Unavailable</b><small class="score-status">Inference completed; no valid GED inversion</small>`
                : `<b class="status-badge status-badge--${escapeHtml(result.status)}">${escapeHtml(result.status_label || "Not run")}</b>`
            }
          </div>
          <div class="result-detail">
            <div class="score-transform-grid" aria-label="Original and transformed model outputs">
              <div>
                <span>Raw model output</span>
                <b>${typeof rawModelOutput === "number" ? formatNumber(rawModelOutput) : "n/a"}</b>
                <small>${calibrationApplied ? "Before validation calibration" : "Direct model output"}</small>
              </div>
              <div>
                <span>Native target score</span>
                <b>${hasNativeScore ? formatNumber(result.model_score) : "n/a"}</b>
                <small>${escapeHtml(result.score_semantics || "not recorded")}</small>
              </div>
              <div>
                <span>Predicted normalized GED</span>
                <b>${typeof predictedNormalizedGed === "number" ? formatNumber(predictedNormalizedGed) : "n/a"}</b>
                <small>Model-specific inversion</small>
              </div>
              <div>
                <span>Predicted GED</span>
                <b>${typeof predictedGed === "number" ? formatNumber(predictedGed) : "n/a"}</b>
                <small>Distance scale</small>
              </div>
              <div>
                <span>Canonical similarity</span>
                <b>${hasComparableScore ? `${comparablePercent.toFixed(1)}%` : "n/a"}</b>
                <small>exp(-GED / average size)</small>
              </div>
            </div>
            ${escapeHtml(result.detail)}
            ${hasNativeScore ? `<br><span>${calibrationApplied ? "Validation-calibrated model output" : "Native model output"}: ${nativePercent.toFixed(1)}% · ${escapeHtml(result.score_semantics || "semantics not recorded")}</span>` : ""}
            ${typeof rawRegression === "number" && calibrationApplied ? `<br><span>Raw regression output before calibration: ${formatNumber(rawRegression)} · position: ${escapeHtml(result.adapter_metrics?.calibration_position || "not recorded")}</span>` : ""}
            ${typeof calibrationAuditMse === "number" && typeof calibrationAuditRawMse === "number" ? `<br><span>Calibration audit MSE: ${formatNumber(calibrationAuditRawMse)} raw → ${formatNumber(calibrationAuditMse)} calibrated · ${escapeHtml(result.adapter_metrics?.calibration_audit_pairs ?? "unknown")} held-out validation pairs · test graphs used: no</span>` : ""}
            ${hasComparableScore ? `<br><span>Comparable similarity: ${comparablePercent.toFixed(1)}% using exp(-predicted GED / average graph size)</span>` : result.status === "executed" ? `<br><span>Comparable similarity is unavailable because this output cannot be converted to a non-negative GED.</span>` : ""}
            <br><span>Implementation: ${escapeHtml(result.implementation_origin || "not recorded")}</span>
            <br><span>Architecture: <code>${escapeHtml(result.architecture_class || "not recorded")}</code></span>
            ${result.runtime_architecture_class ? `<br><span>Runtime class loaded: <code>${escapeHtml(result.runtime_architecture_class)}</code></span>` : ""}
            <br><span>Code note: ${escapeHtml(result.implementation_note || "")}</span>
            <br><span>Checkpoint: ${escapeHtml(checkpointType)}</span>
            <br><span>Checkpoint note: ${escapeHtml(result.checkpoint_note || "")}</span>
            <br><span>Training seed: ${escapeHtml(result.adapter_metrics?.seed ?? "not recorded")}</span>
            ${hyperparameterLine ? `<br><span>Hyperparameters: <code>${escapeHtml(hyperparameterLine)}</code></span>` : ""}
            ${hpo?.study_id ? `<br><span>HPO: ${escapeHtml(hpo.completed_trials ?? "n/a")} trials · validation MSE ${formatMetric(hpo.validation_mse)} · test used for selection: ${hpo.test_set_used_for_selection ? "yes" : "no"}</span>` : ""}
            ${hpoMetadataStatus && !["not_recorded", "verified_checkpoint"].includes(hpoMetadataStatus) ? `<br><span>HPO metadata: stale or unverifiable sidecar ignored for this active checkpoint</span>` : ""}
            <br><span>Validation split: ${validationLine}</span>
            ${result.selected_checkpoint ? `<br><span>Selected checkpoint: <code>${escapeHtml(result.selected_checkpoint)}</code></span>` : ""}
            <br><span>Score semantics: ${escapeHtml(result.score_semantics || "not recorded")}</span>
            <br><span>Input binding: ${escapeHtml(result.input_binding || "not recorded")} · ${escapeHtml(inputVerification)}</span>
            <br><code>${escapeHtml(result.local_path)}</code>
            <br><span>Python: <code>${escapeHtml(result.python || "not configured")}</code></span>
            <br><span>Environment: ${escapeHtml(result.environment || "")}</span>
            <br><span>Datasets: ${escapeHtml((result.supported_datasets || []).join(", ") || "not registered")}</span>
            <br><span>Dataset match: ${datasetMatch}</span>
            <br><span>Runnable here: ${datasetRunnable}</span>
            <br><span>Runnable datasets: ${escapeHtml(runnableDatasets)}</span>
            <br><span>Command: ${escapeHtml(result.command)}</span>
            ${errorLine}
            ${(result.setup || []).length ? `<br><span>Setup: ${(result.setup || []).map((line) => `<code>${escapeHtml(line)}</code>`).join(" ")}</span>` : ""}
            ${result.missing_runtime ? `<br><span>Runtime: missing</span>` : ""}
            ${missing ? `<br><span>Missing: ${escapeHtml(missing)}</span>` : ""}
            ${missingFiles ? `<br><span>Missing files: ${escapeHtml(missingFiles)}</span>` : ""}
            <br><span>Checkpoints: ${checkpoints}</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function closestModelId(results, groundTruth) {
  if (!groundTruth) return null;
  const candidates = results
    .map((result) => ({
      id: result.id,
      error:
        result.status === "executed" && typeof result.adapter_metrics?.predicted_ged === "number"
          ? Math.abs(result.adapter_metrics.predicted_ged - groundTruth.distance)
          : Number.POSITIVE_INFINITY,
    }))
    .filter((candidate) => Number.isFinite(candidate.error));
  candidates.sort((left, right) => left.error - right.error);
  return candidates[0]?.id || null;
}

function groundTruthErrorLine(result, groundTruth) {
  if (!groundTruth || result.status !== "executed") return "";
  const predictedGed = result.adapter_metrics?.predicted_ged;
  if (typeof predictedGed !== "number") return "";
  const gedError = Math.abs(predictedGed - groundTruth.distance);
  const similarityError =
    typeof result.canonical_similarity === "number"
      ? Math.abs(result.canonical_similarity - groundTruth.similarity)
      : null;
  return `<br><span>${escapeHtml(groundTruthReferenceLabel(groundTruth))} check: predicted GED ${formatNumber(predictedGed)} | error ${formatNumber(gedError)}${
    similarityError === null ? "" : ` | similarity error ${formatNumber(similarityError)}`
  }</span>`;
}

async function runAccuracyCheck() {
  if (!datasetSelect.value || !accuracyTable) return;
  if (!datasetHasGedGroundTruth()) {
    const text = "This dataset has no registered GED benchmark; only structural-proxy comparison is available.";
    setMessage(text);
    accuracyTable.innerHTML = `<div class="accuracy-empty">${escapeHtml(text)}</div>`;
    return;
  }
  const methods = selectedMethodIds();
  if (methods.length === 0) {
    setMessage("Select at least one model.");
    return;
  }
  const datasetId = datasetSelect.value;
  const requestId = ++accuracyRequestId;
  evaluateBtn.disabled = true;
  setMessage("Checking accuracy against the registered GED benchmark...");
  accuracyTable.innerHTML = `<div class="accuracy-empty">Checking selected models...</div>`;
  const loadingToken = beginLoading(
    "Running GED benchmark",
    `${methods.length} model${methods.length === 1 ? "" : "s"} · ${Number(accuracySampleSize?.value || 12)} graph pairs · seed ${Number(accuracySeed?.value || 379)}`,
  );
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        methods,
        sample_size: Number(accuracySampleSize?.value || 12),
        scope: bestPairScope?.value || "train-test",
        sample_mode: accuracySampleMode?.value || "stratified",
        top_k: Number(accuracyTopK?.value || 5),
        seed: Number(accuracySeed?.value || 379),
      }),
    });
    const payload = await response.json();
    if (requestId !== accuracyRequestId || datasetSelect.value !== datasetId) return;
    if (!response.ok) {
      setMessage(payload.error || "Accuracy check failed.");
      accuracyTable.innerHTML = `<div class="accuracy-empty">${escapeHtml(payload.error || "Accuracy check failed.")}</div>`;
      return;
    }
    lastBenchmarkPayload = payload;
    syncBenchmarkExports();
    renderAccuracy(payload);
    setMessage("");
  } finally {
    endLoading(loadingToken);
    updateGroundTruthControls();
  }
}

function renderAccuracy(payload) {
  if (!accuracyTable) return;
  const rows = payload.models || [];
  if (!rows.length) {
    accuracyTable.innerHTML = `<div class="accuracy-empty">No model was evaluated.</div>`;
    return;
  }
  const protocol = payload.protocol || {};
  const referenceTitle = protocol.reference_exact === false
    ? "Approximate GED benchmark"
    : "Exact GED benchmark";
  const pairLabel = `${escapeHtml(payload.dataset_id)} | ${payload.sample_size}/${payload.candidate_pairs} pairs | ${escapeHtml(payload.scope)} | ${escapeHtml(protocol.sampling || "stratified")} | seed ${escapeHtml(protocol.seed)}`;
  accuracyTable.innerHTML = `
    <div class="accuracy-summary">
      <b>${escapeHtml(referenceTitle)} · run ${escapeHtml(payload.run_id || "unsaved")}</b>
      <span>${pairLabel}</span>
      <span>Artifact: <code>${escapeHtml(payload.artifact_path || "not persisted")}</code> · elapsed ${formatMetric(payload.latency_ms)} ms</span>
      <span class="benchmark-warning">${escapeHtml(protocol.checkpoint_warning || "")}</span>
      ${protocol.approximate_reference_note ? `<span class="benchmark-warning">${escapeHtml(protocol.approximate_reference_note)}</span>` : ""}
    </div>
    ${rows.map(renderAccuracyRow).join("")}
  `;
}

function renderAccuracyRow(row) {
  const evaluated = row.status === "evaluated";
  const gedMae = evaluated ? formatMetric(row.mae_ged) : "n/a";
  const statusLabel = evaluated
    ? "Evaluated"
    : row.status === "not_evaluable"
      ? "Executed, no valid GED prediction"
      : row.samples?.find((sample) => sample.status_label)?.status_label || "Not executed";
  const firstDetail = row.samples?.find((sample) => sample.detail)?.detail || "";
  const samples = (row.samples || [])
    .slice(0, 4)
    .map((sample) => {
      const reference = `${formatNumber(sample.exact_similarity)} sim / GED ${formatNumber(sample.exact_ged)}`;
      const referenceLabel = sample.reference_exact === false ? "approx. benchmark" : "exact";
      const predicted =
        typeof sample.predicted_similarity === "number"
          ? `${formatNumber(sample.predicted_similarity)} canonical sim / GED ${formatNumber(sample.predicted_ged)}`
          : escapeHtml(sample.status_label || sample.status || "not executed");
      const rawScore =
        typeof sample.model_score === "number" &&
        typeof sample.predicted_similarity === "number" &&
        Math.abs(sample.model_score - sample.predicted_similarity) > 1e-6
          ? ` · raw model score ${formatNumber(sample.model_score)}`
          : "";
      return `
        <div class="accuracy-sample">
          <code>${escapeHtml(sample.left_graph)}</code>
          <span>${referenceLabel} ${reference}</span>
          <span>model ${predicted}${rawScore}</span>
          <code>${escapeHtml(sample.right_graph)}</code>
        </div>
      `;
    })
    .join("");
  return `
    <article class="accuracy-row">
      <div class="result-title">
        <strong>${escapeHtml(row.name)}</strong>
        <span>${escapeHtml(row.paper)} · ${row.evaluated_samples}/${row.attempted_samples} samples evaluated · ${row.executed_samples} forward passes executed</span>
      </div>
      <div class="accuracy-score">
        <span class="score-label">GED MAE</span>
        <b>${gedMae}</b>
        <small>${escapeHtml(statusLabel)}</small>
      </div>
      <div class="quality-metrics">
        <div><b>${formatMetric(row.mse_similarity_x1e3)}</b><span>Similarity MSE x1e3</span></div>
        <div><b>${formatMetric(row.rmse_ged)}</b><span>RMSE GED</span></div>
        <div><b>${formatMetric(row.mae_normalized_ged)}</b><span>MAE normalized GED</span></div>
        <div><b>${formatMetric(row.spearman_ged)}</b><span>Spearman GED</span></div>
        <div><b>${formatMetric(row.kendall_ged)}</b><span>Kendall tau-b</span></div>
        <div><b>${formatMetric(row.precision_at_k)}</b><span>Precision@${row.top_k || "k"}</span></div>
        <div><b>${formatMetric(row.precision_at_10)}</b><span>Precision@10</span></div>
        <div><b>${formatMetric(row.ndcg_at_k)}</b><span>NDCG@${row.top_k || "k"}</span></div>
        <div><b>${formatMetric(row.mae_similarity)}</b><span>MAE similarity</span></div>
        <div><b>${formatMetric(row.latency_p50_ms)}</b><span>P50 adapter ms</span></div>
        <div><b>${formatMetric(row.latency_p95_ms)}</b><span>P95 adapter ms</span></div>
        <div><b>${formatMetric(row.throughput_pairs_per_second)}</b><span>Pairs / second</span></div>
        <div><b>${formatMetric(row.peak_rss_mb)}</b><span>Peak RSS MB</span></div>
        <div><b>${formatMetric(row.projected_samples)}</b><span>Projected inputs</span></div>
      </div>
      <div class="accuracy-detail">
        ${row.mae_ged_ci95 ? `<p>GED MAE 95% bootstrap CI: ${formatMetric(row.mae_ged_ci95[0])}–${formatMetric(row.mae_ged_ci95[1])}</p>` : ""}
        ${row.projected_samples > 0 ? `<p class="benchmark-warning">${row.projected_samples}/${row.evaluated_samples} pairs exceeded this checkpoint's input cap; SEGMN used its recorded deterministic projection.</p>` : ""}
        ${renderSizeGeneralization(row.size_generalization)}
        ${firstDetail ? `<p>${escapeHtml(firstDetail)}</p>` : ""}
        <div class="accuracy-samples">${samples}</div>
      </div>
    </article>
  `;
}

function renderSizeGeneralization(rows) {
  if (!Array.isArray(rows) || !rows.length) return "";
  return `
    <div class="size-generalization">
      ${rows
        .map(
          (row) => `
            <span>
              <b>${escapeHtml(row.bucket)}</b>
              n=${row.samples}, size ${formatMetric(row.average_size_min)}-${formatMetric(row.average_size_max)},
              GED MAE ${formatMetric(row.mae_ged)}, MSE x1e3 ${formatMetric(row.mse_similarity_x1e3)}
            </span>
          `,
        )
        .join("")}
    </div>
  `;
}

function syncBenchmarkExports() {
  const disabled = !lastBenchmarkPayload;
  if (exportJsonBtn) exportJsonBtn.disabled = disabled;
  if (exportCsvBtn) exportCsvBtn.disabled = disabled;
}

function exportBenchmark(format) {
  if (!lastBenchmarkPayload) return;
  const runId = lastBenchmarkPayload.run_id || "graph-benchmark";
  if (format === "json") {
    downloadText(
      `${runId}.json`,
      JSON.stringify(lastBenchmarkPayload, null, 2),
      "application/json",
    );
    return;
  }
  const columns = [
    "run_id",
    "dataset_id",
    "model_id",
    "model_name",
    "sample_size",
    "status",
    "mae_ged",
    "rmse_ged",
    "mae_normalized_ged",
    "mae_similarity",
    "mse_similarity_x1e3",
    "rmse_similarity",
    "spearman_ged",
    "kendall_ged",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_rss_mb",
  ];
  const rows = (lastBenchmarkPayload.models || []).map((model) => [
    runId,
    lastBenchmarkPayload.dataset_id,
    model.id,
    model.name,
    lastBenchmarkPayload.sample_size,
    model.status,
    model.mae_ged,
    model.rmse_ged,
    model.mae_normalized_ged,
    model.mae_similarity,
    model.mse_similarity_x1e3,
    model.rmse_similarity,
    model.spearman_ged,
    model.kendall_ged,
    model.precision_at_k,
    model.recall_at_k,
    model.ndcg_at_k,
    model.latency_p50_ms,
    model.latency_p95_ms,
    model.peak_rss_mb,
  ]);
  const csv = [columns, ...rows]
    .map((row) => row.map(csvCell).join(","))
    .join("\n");
  downloadText(`${runId}.csv`, csv, "text/csv");
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadText(filename, content, contentType) {
  const url = URL.createObjectURL(new Blob([content], { type: `${contentType};charset=utf-8` }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function runRetrievalAblation() {
  if (!datasetSelect.value || !retrievalPanel) return;
  if (!datasetHasGedGroundTruth()) {
    retrievalPanel.innerHTML = `<div class="accuracy-empty">A registered GED benchmark is required for retrieval ablation.</div>`;
    return;
  }
  const budgets = String(retrievalBudgets?.value || "")
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
  if (!budgets.length) {
    setMessage("Enter at least one positive candidate budget.");
    return;
  }
  const datasetId = datasetSelect.value;
  const requestId = ++retrievalRequestId;
  runRetrievalBtn.disabled = true;
  retrievalPanel.innerHTML = `<div class="accuracy-empty">Running retrieval ablation...</div>`;
  const loadingToken = beginLoading(
    "Running structural prefilter",
    `Evaluating candidate budgets ${budgets.join(", ")} against the GED benchmark...`,
  );
  try {
    const response = await fetch(
      `/api/datasets/${encodeURIComponent(datasetId)}/retrieval-ablation`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          budgets,
          top_k: Number(retrievalTopK?.value || 10),
          scope: bestPairScope?.value || "train-test",
        }),
      },
    );
    const payload = await response.json();
    if (requestId !== retrievalRequestId || datasetSelect.value !== datasetId) return;
    if (!response.ok) {
      retrievalPanel.innerHTML = `<div class="accuracy-empty">${escapeHtml(payload.error || "Retrieval ablation failed.")}</div>`;
      return;
    }
    renderRetrievalAblation(payload);
    setMessage("");
  } finally {
    endLoading(loadingToken);
    runRetrievalBtn.disabled = false;
  }
}

async function runGnnRerankingAblation() {
  if (
    !datasetSelect.value ||
    !retrievalPanel ||
    !retrievalModelSelect?.value
  ) {
    return;
  }
  if (!datasetHasGedGroundTruth()) {
    retrievalPanel.innerHTML = `<div class="accuracy-empty">A registered GED benchmark is required for GNN reranking evaluation.</div>`;
    return;
  }
  const budgets = parseRetrievalBudgets();
  if (!budgets.length) {
    setMessage("Enter at least one positive candidate budget.");
    return;
  }
  const datasetId = datasetSelect.value;
  const requestId = ++retrievalRequestId;
  runRerankingBtn.disabled = true;
  retrievalPanel.innerHTML = `<div class="accuracy-empty">Starting real GNN reranking job...</div>`;
  const loadingToken = beginLoading(
    "Starting GNN reranking",
    `${selectedOptionLabel(retrievalModelSelect)} · candidate budgets ${budgets.join(", ")}`,
  );
  try {
    const response = await fetch(
      `/api/datasets/${encodeURIComponent(datasetId)}/reranking-ablation`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method_id: retrievalModelSelect.value,
          budgets,
          top_k: Number(retrievalTopK?.value || 10),
          scope: bestPairScope?.value || "train-test",
        }),
      },
    );
    const payload = await response.json();
    if (requestId !== retrievalRequestId || datasetSelect.value !== datasetId) return;
    if (!response.ok) {
      retrievalPanel.innerHTML = `<div class="accuracy-empty">${escapeHtml(payload.error || "GNN reranking could not start.")}</div>`;
      return;
    }
    await pollRerankingJob(payload.job.id, requestId, datasetId, loadingToken);
  } finally {
    endLoading(loadingToken);
    runRerankingBtn.disabled = !datasetHasGedGroundTruth();
  }
}

function parseRetrievalBudgets() {
  return String(retrievalBudgets?.value || "")
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
}

async function pollRerankingJob(jobId, requestId, datasetId, loadingToken) {
  while (requestId === retrievalRequestId && datasetSelect.value === datasetId) {
    const response = await fetch(`/api/reranking-jobs/${encodeURIComponent(jobId)}`);
    const payload = await response.json();
    if (!response.ok) {
      retrievalPanel.innerHTML = `<div class="accuracy-empty">${escapeHtml(payload.error || "Reranking job failed.")}</div>`;
      return;
    }
    const job = payload.job || {};
    updateLoading(
      loadingToken,
      "Scoring candidates with real GNNs",
      `${job.model_name || selectedOptionLabel(retrievalModelSelect)} · job ${job.id || jobId} · ${job.status || "running"}`,
    );
    if (job.status === "completed") {
      renderGnnReranking(job.result);
      return;
    }
    if (job.status === "failed") {
      retrievalPanel.innerHTML = `<div class="accuracy-empty">${escapeHtml(job.error || "Reranking job failed.")}</div>`;
      return;
    }
    retrievalPanel.innerHTML = `
      <div class="accuracy-empty">
        ${escapeHtml(job.model_name)} is scoring up to ${Math.max(...(job.budgets || [0]))} candidates.
        Job <code>${escapeHtml(job.id)}</code> is ${escapeHtml(job.status)}.
      </div>
    `;
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
}

function renderGnnReranking(payload) {
  const rows = (payload.budgets || [])
    .map(
      (row) => `
        <div class="retrieval-row retrieval-row--gnn">
          <b>${row.budget}</b>
          <span>${formatMetric(row.candidate_recall_at_k)}</span>
          <span>${formatMetric(row.reranked_recall_at_k)}</span>
          <span>${formatMetric(row.reranked_precision_at_k)}</span>
          <span>${formatMetric(row.reranked_ndcg_at_k)}</span>
          <span>${formatMetric(row.model_selected_ged_regret)}</span>
          <span>${formatMetric(row.latency_total_ms)} ms</span>
        </div>
      `,
    )
    .join("");
  retrievalPanel.innerHTML = `
    <div class="accuracy-summary">
      <b>${escapeHtml(payload.model_name)} real GNN reranking · ${escapeHtml(payload.dataset_id)}</b>
      <span>${payload.maximum_scored_candidates}/${payload.total_pairs} candidates scored once · top-${payload.top_k}</span>
      <span>Artifact: <code>${escapeHtml(payload.artifact_path)}</code> · elapsed ${formatMetric(payload.latency_ms)} ms</span>
    </div>
    <div class="retrieval-table">
      <div class="retrieval-row retrieval-row--head retrieval-row--gnn">
        <b>Budget</b><span>Candidate R@k</span><span>Reranked R@k</span><span>P@k</span><span>NDCG@k</span><span>GED regret</span><span>Model time</span>
      </div>
      ${rows}
    </div>
  `;
}

function renderRetrievalAblation(payload) {
  const rows = (payload.budgets || [])
    .map(
      (row) => `
        <div class="retrieval-row">
          <b>${row.budget}</b>
          <span>${formatMetric(row.recall_at_k)}</span>
          <span>${formatMetric(row.precision)}</span>
          <span>${row.exact_best_recalled ? "yes" : "no"}</span>
          <span>${formatMetric(row.best_ged_in_candidates)}</span>
          <span>${formatMetric(row.best_ged_regret)}</span>
          <span>${formatMetric(row.reduction_percent)}%</span>
        </div>
      `,
    )
    .join("");
  retrievalPanel.innerHTML = `
    <div class="accuracy-summary">
      <b>${escapeHtml(payload.dataset_id)} · ${payload.total_pairs} GED-reference pairs · top-${payload.top_k}</b>
      <span>Exact best: <code>${escapeHtml(payload.exact_best_pair?.left_graph)}</code> vs <code>${escapeHtml(payload.exact_best_pair?.right_graph)}</code> · GED ${formatMetric(payload.exact_best_pair?.exact_ged)}</span>
      <span>Artifact: <code>${escapeHtml(payload.artifact_path)}</code> · elapsed ${formatMetric(payload.latency_ms)} ms</span>
    </div>
    <div class="retrieval-table">
      <div class="retrieval-row retrieval-row--head">
        <b>Budget</b><span>Recall@k</span><span>Precision</span><span>Best found</span><span>Best GED</span><span>Regret</span><span>Reduction</span>
      </div>
      ${rows}
    </div>
  `;
}

async function loadResearchMatrix() {
  if (!matrixPanel) return;
  if (refreshMatrixBtn) refreshMatrixBtn.disabled = true;
  try {
    const response = await fetch("/api/research-summary");
    const payload = await response.json();
    if (!response.ok) {
      matrixPanel.innerHTML = `<div class="accuracy-empty">${escapeHtml(payload.error || "Research matrix failed.")}</div>`;
      return;
    }
    renderResearchMatrix(
      payload.summary,
      payload.matrix_status,
      payload.checkpoint_audit,
    );
  } finally {
    if (refreshMatrixBtn) refreshMatrixBtn.disabled = false;
  }
}

function renderResearchMatrix(summary, status, checkpointAudit) {
  const statusMarkup = status && !status.finished
    ? `
      <div class="matrix-progress">
        <b>Run ${escapeHtml(status.run_id)} in progress</b>
        <span>${status.completed}/${status.total_expected} completed · ${status.failed} failed</span>
        <progress value="${status.completed}" max="${Math.max(status.total_expected, 1)}"></progress>
        ${
          status.current
            ? `<code>${escapeHtml(status.current.model_id)} · ${escapeHtml(status.current.dataset_id)} · seed ${escapeHtml(status.current.seed)}</code>`
            : ""
        }
      </div>
    `
    : "";
  scheduleMatrixRefresh(Boolean(status && !status.finished));
  const auditMarkup = checkpointAudit
    ? `
      <div class="checkpoint-audit ${checkpointAudit.complete ? "checkpoint-audit--complete" : ""}">
        <b>Checkpoint protocol ${checkpointAudit.verified}/${checkpointAudit.total}</b>
        <span>${checkpointAudit.complete ? "All active checkpoint protocols verified; accuracy is reported separately" : `${checkpointAudit.unverified?.length || 0} checkpoints require protocol review`}</span>
        <code>${escapeHtml(checkpointAudit.artifact_path)}</code>
      </div>
    `
    : "";
  if (!summary) {
    matrixPanel.innerHTML = `${statusMarkup}${auditMarkup}<div class="accuracy-empty">No executed research matrix summary is available.</div>`;
    return;
  }
  const rows = (summary.rows || [])
    .map((row) => {
      const mse = row.metrics?.mse_similarity_x1e3 || {};
      const rho = row.metrics?.spearman_ged || {};
      const latency = row.metrics?.latency_p50_ms || {};
      const memory = row.metrics?.peak_rss_mb || {};
      return `
        <div class="matrix-row">
          <strong>${escapeHtml(row.dataset_id)} · ${escapeHtml(row.model_name)}</strong>
          <span>${row.evaluated_seeds?.length || 0}/${row.expected_seeds?.length || 0}</span>
          <span>${meanStdMetric(mse)}</span>
          <span>${meanStdMetric(rho)}</span>
          <span>${meanStdMetric(latency)}</span>
          <span>${meanStdMetric(memory)}</span>
          <span>${row.pair_split_verified ? "protocol verified" : "protocol unverified"}</span>
        </div>
      `;
    })
    .join("");
  matrixPanel.innerHTML = `
    ${statusMarkup}
    ${auditMarkup}
    <div class="accuracy-summary">
      <b>Run ${escapeHtml(summary.run_id)} · ${summary.matrix_complete ? "complete" : "incomplete"}</b>
      <span>Expected seeds: ${(summary.expected_seeds || []).map(escapeHtml).join(", ")}</span>
      <span>Manifest: <code>${escapeHtml(summary.manifest_path)}</code></span>
    </div>
    <div class="matrix-table">
      <div class="matrix-row matrix-row--head">
        <strong>Dataset · model</strong><span>Seeds</span><span>MSE x1e3</span><span>Spearman</span><span>P50 ms</span><span>Peak MB</span><span>Split</span>
      </div>
      ${rows}
    </div>
  `;
}

function scheduleMatrixRefresh(active) {
  if (active && !matrixRefreshTimer) {
    matrixRefreshTimer = window.setInterval(loadResearchMatrix, 5000);
  } else if (!active && matrixRefreshTimer) {
    window.clearInterval(matrixRefreshTimer);
    matrixRefreshTimer = null;
  }
}

function meanStdMetric(metric) {
  if (!metric || typeof metric.mean !== "number") return "n/a";
  return `${formatMetric(metric.mean)} +/- ${formatMetric(metric.std)}`;
}

function statusText(result) {
  if (result.status === "executed") return "Checkpoint inference completed";
  return result.status_label || "Not run";
}

function renderBestPair(search) {
  if (!bestPairPanel) return;
  if (!search || !search.winner) {
    bestPairPanel.innerHTML = "";
    return;
  }
  const winner = search.winner;
  const percent = typeof winner.score === "number" ? `${(winner.score * 100).toFixed(1)}%` : "n/a";
  const method = (search.method_ids || [])
    .map((id) => {
      if (id === "exact-ged") return "GED Benchmark";
      if (id === "structure-search") return "Structure Search";
      return id;
    })
    .join(", ");
  const candidateRows = (search.candidates || [])
    .slice(0, 5)
    .map(
      (candidate) => `
        <div class="best-pair-candidate">
          <code>${escapeHtml(candidate.left_graph)}</code>
          <span>${
            candidate.exact_ged !== undefined
              ? `GED ${escapeHtml(candidate.exact_ged)}`
              : typeof candidate.score === "number"
                ? `${(candidate.score * 100).toFixed(1)}%`
                : "n/a"
          }</span>
          <code>${escapeHtml(candidate.right_graph)}</code>
        </div>
      `
    )
    .join("");
  bestPairPanel.innerHTML = `
    <div class="best-pair-summary">
      <div>
        <span>${escapeHtml(search.selection_label || "Best Pair")}</span>
        <b>${winner.exact_ged !== undefined ? `GED ${escapeHtml(winner.exact_ged)}` : escapeHtml(percent)}</b>
      </div>
      <div>
        <span>Graph A</span>
        <code>${escapeHtml(winner.left_graph)}</code>
      </div>
      <div>
        <span>Graph B</span>
        <code>${escapeHtml(winner.right_graph)}</code>
      </div>
      <div>
        <span>Model</span>
        <code>${escapeHtml(method)}</code>
      </div>
      <div>
        <span>Scored</span>
        <b>${search.scored_pairs}/${search.total_pairs}</b>
      </div>
    </div>
    ${
      search.exhaustive === false
        ? `<div class="best-pair-note">Top structural candidates were scored by the selected checkpoint.</div>`
        : ""
    }
    ${candidateRows ? `<div class="best-pair-candidates">${candidateRows}</div>` : ""}
  `;
}

async function loadDatasetInfo(datasetId) {
  if (!datasetId) return;
  const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}`);
  const payload = await response.json();
  if (response.ok) {
    currentDatasetInfo = payload.dataset || currentDatasetInfo;
    updateGroundTruthControls();
    renderDatasetDetails(payload.dataset, currentMeta.dataset ? currentMeta : payload.meta);
  }
}

function renderDatasetDetails(dataset, meta) {
  if (!datasetDetails || !dataset) return;
  currentDatasetInfo = dataset;
  updateGroundTruthControls();
  const gt = dataset.ground_truth_paths || [];
  const hasGed = datasetHasGedGroundTruth(dataset);
  const selectedLeft = meta?.left_graph || leftGraphSelect?.value || "";
  const selectedRight = meta?.right_graph || rightGraphSelect?.value || "";
  datasetDetails.innerHTML = `
    <div class="dataset-metrics">
      <div><b>${escapeHtml(dataset.name)}</b><span>${escapeHtml(dataset.domain)}</span></div>
      <div><b>${dataset.graph_count}</b><span>graphs</span></div>
      <div><b>${dataset.train_graphs}</b><span>train</span></div>
      <div><b>${dataset.test_graphs}</b><span>test</span></div>
      <div><b>${escapeHtml((dataset.tasks || []).join("/"))}</b><span>tasks</span></div>
    </div>
    <div class="dataset-paths">
      <span>Archive <code>${escapeHtml(dataset.archive_path)}</code></span>
      ${
        selectedLeft && selectedRight
          ? `<span>Selected <code>${escapeHtml(selectedLeft)}</code> and <code>${escapeHtml(selectedRight)}</code></span>`
          : ""
      }
      <span>${hasGed ? `${dataset.ground_truth_exact ? "Exact" : "Approximate"} GED benchmark ready` : "GED benchmark unavailable for this dataset"}</span>
      ${
        dataset.uploaded
          ? `<span>Local upload · ${
              dataset.training_ready
                ? dataset.ground_truth_kind === "approximate_benchmark"
                  ? "5-model training with approximate GED benchmark"
                  : dataset.ground_truth_kind === "unverified_ged"
                    ? "5-model training with user-provided unverified GED"
                  : dataset.ground_truth_exact === false
                    ? "5-model training with registered structural proxy"
                    : "5-model training with exact GED"
                : "5-model training with structural GED proxy"
            }</span>`
          : ""
      }
      <span>Ground truth ${gt.length ? gt.map((path) => `<code>${escapeHtml(path)}</code>`).join(" ") : "not registered"}</span>
    </div>
  `;
}

function selectedBestPairMethod() {
  if (!bestMethodSelect) return selectedMethodIds()[0] || "simgnn";
  if (bestMethodSelect.value === "exact-ged" && !datasetHasGedGroundTruth()) {
    const fallback = [...bestMethodSelect.options].find((option) => option.value !== "exact-ged" && !option.disabled);
    if (fallback) bestMethodSelect.value = fallback.value;
  }
  return bestMethodSelect.value || selectedMethodIds()[0] || "simgnn";
}

function datasetHasGedGroundTruth(dataset = currentDatasetInfo) {
  if (!dataset) return true;
  if (dataset.ground_truth_benchmark === false) return false;
  if (!["exact", "approximate_benchmark"].includes(dataset.ground_truth_kind)) return false;
  const paths = dataset.ground_truth_paths || [];
  return paths.some((path) => {
    const value = String(path).toLowerCase().replaceAll("\\", "/");
    return value.includes("_ged_") || value.endsWith("/ged.json");
  });
}

function updateGroundTruthControls() {
  const hasGed = datasetHasGedGroundTruth();
  const exactOption = [...(bestMethodSelect?.options || [])].find((option) => option.value === "exact-ged");
  if (exactOption) {
    exactOption.disabled = !hasGed;
    exactOption.textContent = hasGed
      ? currentDatasetInfo?.ground_truth_exact
        ? "Exact GED"
        : "Approximate GED benchmark"
      : "GED benchmark (unavailable)";
  }
  if (!hasGed && bestMethodSelect?.value === "exact-ged") {
    const fallback = [...bestMethodSelect.options].find((option) => option.value !== "exact-ged" && !option.disabled);
    if (fallback) bestMethodSelect.value = fallback.value;
  }
  if (evaluateBtn) {
    evaluateBtn.disabled = !hasGed;
    evaluateBtn.title = hasGed
      ? "Compare selected models against the registered GED benchmark"
      : "This dataset has no registered GED benchmark";
  }
  if (runRetrievalBtn) {
    runRetrievalBtn.disabled = !hasGed;
    runRetrievalBtn.title = hasGed
      ? "Measure structural prefilter candidate recall"
      : "This dataset has no registered GED benchmark";
  }
  if (runRerankingBtn) {
    runRerankingBtn.disabled = !hasGed;
    runRerankingBtn.title = hasGed
      ? "Run checkpoint-backed GNN reranking against the GED benchmark"
      : "This dataset has no registered GED benchmark";
  }
  if (!hasGed && accuracyTable) {
    accuracyTable.innerHTML = `<div class="accuracy-empty">GED accuracy is unavailable for this dataset.</div>`;
    lastBenchmarkPayload = null;
    syncBenchmarkExports();
  }
  if (!hasGed && retrievalPanel) {
    retrievalPanel.innerHTML = `<div class="accuracy-empty">Retrieval ablation requires a registered GED benchmark.</div>`;
  }
}

function drawGraph(svg, graph, options = {}) {
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const viewBox = svg.viewBox?.baseVal;
  const width = viewBox?.width || 260;
  const height = viewBox?.height || 200;
  const positions = graphLayout(nodes, edges, width, height);
  const degrees = new Map(nodes.map((node) => [node.id, 0]));
  edges.forEach((edge) => {
    degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
    degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
  });
  const maxDegree = Math.max(1, ...degrees.values());

  edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const line = svgElement("line", {
      x1: source.x,
      y1: source.y,
      x2: target.x,
      y2: target.y,
    });
    line.setAttribute("class", "graph-edge");
    svg.appendChild(line);
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    if (!position) return;
    const color = colorForLabel(node.label);
    const baseRadius = options.large ? 5.8 : 4.5;
    const degreeBoost = (degrees.get(node.id) || 0) / maxDegree;
    const radius = Math.max(options.large ? 4 : 3, baseRadius + degreeBoost * (options.large ? 4.2 : 2.2));
    const halo = svgElement("circle", { cx: position.x, cy: position.y, r: radius + 3, fill: color, opacity: 0.12 });
    halo.setAttribute("class", "graph-node-halo");
    svg.appendChild(halo);
    const circle = svgElement("circle", { cx: position.x, cy: position.y, r: radius, fill: color });
    circle.setAttribute("class", "graph-node");
    const title = svgElement("title", {});
    title.textContent = `node ${node.id} | label ${node.label} | degree ${degrees.get(node.id) || 0}`;
    circle.appendChild(title);
    svg.appendChild(circle);
    if (nodes.length <= (options.large ? 48 : 24)) {
      const label = svgElement("text", { x: position.x, y: position.y + 3, "text-anchor": "middle" });
      label.setAttribute("class", "graph-label");
      label.textContent = String(node.label).slice(0, 2);
      svg.appendChild(label);
    }
  });

  if (nodes.length === 0) {
    const empty = svgElement("text", { x: width / 2, y: height / 2, "text-anchor": "middle" });
    empty.setAttribute("class", "graph-empty");
    empty.textContent = "empty graph";
    svg.appendChild(empty);
  }
}

function graphLayout(nodes, edges, width, height) {
  const margin = Math.min(width, height) * 0.13;
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = Math.max(20, width / 2 - margin);
  const radiusY = Math.max(20, height / 2 - margin);
  const positions = new Map();
  const nodeIds = nodes.map((node) => node.id);
  if (nodeIds.length === 1) {
    positions.set(nodeIds[0], { x: centerX, y: centerY });
    return positions;
  }

  nodeIds.forEach((id, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(nodeIds.length, 1) - Math.PI / 2;
    const jitter = 1 + (stableUnit(`${id}:r`) - 0.5) * 0.18;
    positions.set(id, {
      x: centerX + Math.cos(angle) * radiusX * jitter,
      y: centerY + Math.sin(angle) * radiusY * jitter,
    });
  });

  const edgePairs = edges
    .map((edge) => [edge.source, edge.target])
    .filter(([source, target]) => positions.has(source) && positions.has(target));
  const iterations = Math.min(90, 28 + nodeIds.length);
  const ideal = Math.sqrt((width * height) / Math.max(nodeIds.length, 1)) * 0.75;

  for (let step = 0; step < iterations; step += 1) {
    const forces = new Map(nodeIds.map((id) => [id, { x: 0, y: 0 }]));
    for (let i = 0; i < nodeIds.length; i += 1) {
      for (let j = i + 1; j < nodeIds.length; j += 1) {
        const leftId = nodeIds[i];
        const rightId = nodeIds[j];
        const left = positions.get(leftId);
        const right = positions.get(rightId);
        let dx = left.x - right.x;
        let dy = left.y - right.y;
        const distance = Math.max(Math.hypot(dx, dy), 0.01);
        const force = (ideal * ideal) / distance;
        dx /= distance;
        dy /= distance;
        forces.get(leftId).x += dx * force;
        forces.get(leftId).y += dy * force;
        forces.get(rightId).x -= dx * force;
        forces.get(rightId).y -= dy * force;
      }
    }

    edgePairs.forEach(([sourceId, targetId]) => {
      const source = positions.get(sourceId);
      const target = positions.get(targetId);
      let dx = source.x - target.x;
      let dy = source.y - target.y;
      const distance = Math.max(Math.hypot(dx, dy), 0.01);
      const force = (distance * distance) / ideal;
      dx /= distance;
      dy /= distance;
      forces.get(sourceId).x -= dx * force;
      forces.get(sourceId).y -= dy * force;
      forces.get(targetId).x += dx * force;
      forces.get(targetId).y += dy * force;
    });

    const cooling = 0.032 * (1 - step / iterations);
    nodeIds.forEach((id) => {
      const point = positions.get(id);
      const force = forces.get(id);
      point.x = clampRange(point.x + force.x * cooling, margin, width - margin);
      point.y = clampRange(point.y + force.y * cooling, margin, height - margin);
    });
  }
  return positions;
}

function stableUnit(value) {
  let hash = 2166136261;
  for (const char of String(value)) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 10000) / 10000;
}

function clampRange(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function swapGraphs() {
  const left = leftEditor.value;
  const leftMember = currentMeta.left_graph;
  const rightMember = currentMeta.right_graph;
  leftEditor.value = rightEditor.value;
  rightEditor.value = left;
  if (leftMember && rightMember) {
    if (leftGraphSelect) leftGraphSelect.value = rightMember;
    if (rightGraphSelect) rightGraphSelect.value = leftMember;
    currentMeta = {
      ...currentMeta,
      left_graph: rightMember,
      right_graph: leftMember,
    };
    renderPairMeta(currentMeta);
  }
  runComparison({ previewOnly: true, skipPairReload: true });
}

function formatEditors() {
  try {
    leftEditor.value = JSON.stringify(JSON.parse(leftEditor.value), null, 2);
    rightEditor.value = JSON.stringify(JSON.parse(rightEditor.value), null, 2);
    setMessage("");
  } catch (error) {
    setMessage(`JSON error: ${error.message}`);
  }
}

function svgElement(tag, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function colorForLabel(label) {
  const colors = ["#2563eb", "#0f766e", "#b45309", "#be185d", "#4d7c0f", "#7c3aed", "#c2410c", "#0891b2"];
  let hash = 0;
  for (const char of String(label)) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return colors[hash % colors.length];
}

function beginLoading(title, detail, progress = null) {
  const token = ++loadingSequence;
  loadingOperations.set(token, { title, detail, progress });
  renderLoadingState();
  return token;
}

function updateLoading(token, title, detail, progress = null) {
  if (!loadingOperations.has(token)) return;
  loadingOperations.set(token, { title, detail, progress });
  renderLoadingState();
}

function endLoading(token) {
  loadingOperations.delete(token);
  renderLoadingState();
}

function renderLoadingState() {
  if (!loadingOverlay) return;
  const operations = [...loadingOperations.values()];
  const current = operations[operations.length - 1];
  if (!current) {
    loadingOverlay.hidden = true;
    loadingTrack?.classList.remove("is-determinate");
    if (loadingProgressMeta) loadingProgressMeta.hidden = true;
    document.body.classList.remove("is-loading");
    document.body.removeAttribute("aria-busy");
    return;
  }
  if (loadingTitle) loadingTitle.textContent = current.title;
  if (loadingDetail) loadingDetail.textContent = current.detail;
  const determinate = current.progress && Number.isFinite(Number(current.progress.percent));
  loadingTrack?.classList.toggle("is-determinate", Boolean(determinate));
  if (loadingProgressMeta) loadingProgressMeta.hidden = !determinate;
  if (determinate) {
    const percent = Math.min(100, Math.max(0, Number(current.progress.percent)));
    if (loadingProgressFill) loadingProgressFill.style.width = `${percent}%`;
    if (loadingProgressValue) loadingProgressValue.textContent = `${Math.round(percent)}%`;
    if (loadingStage) loadingStage.textContent = current.progress.stage || "Preparing";
  } else if (loadingProgressFill) {
    loadingProgressFill.style.removeProperty("width");
  }
  loadingOverlay.hidden = false;
  document.body.classList.add("is-loading");
  document.body.setAttribute("aria-busy", "true");
}

async function runWithLoading(title, detail, action) {
  const token = beginLoading(title, detail);
  try {
    return await action();
  } finally {
    endLoading(token);
  }
}

function selectedOptionLabel(select) {
  return select?.selectedOptions?.[0]?.textContent?.trim() || select?.value || "selection";
}

function setMessage(text) {
  message.textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMetric(value) {
  return typeof value === "number" ? formatNumber(value) : "n/a";
}

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  const displayValue = Object.is(value, -0) ? 0 : value;
  if (Math.abs(displayValue) >= 100) return displayValue.toFixed(1);
  if (Math.abs(displayValue) >= 10) return displayValue.toFixed(2);
  return displayValue.toFixed(4);
}
