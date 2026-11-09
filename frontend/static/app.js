/* Final runtime integration. All model/evidence text is rendered as plain text. */
const $ = id => document.getElementById(id);
const labels = {
  clean: "Clean", retrieval_failure: "Retrieval failure", grounding_failure: "Grounding failure",
  confidence_failure: "Confidence failure", coverage_failure: "Coverage failure"
};
const PRESETS = {
  president: "Which college president of Notre Dame is credited with preventing more confrontations between students and the KKK?",
  publication: "In what year did the student paper Common Sense begin publication at Notre Dame?"
};
const storageKey = "self_healing_rag_endpoint";
const sameServer = /^https?:$/.test(location.protocol) ? location.origin : "http://localhost:8000";
let API_BASE_URL = sameServer;
try { API_BASE_URL = localStorage.getItem(storageKey) || sameServer; } catch { /* Storage may be disabled. */ }
let busy = false;
let lastResult = null;
let serverThresholds = {retrieval_similarity:.45, entailment_score:.5, mean_entropy:2, coverage_score:.45};

function icons() { window.lucide?.createIcons(); }
function text(id, value) { $(id).textContent = value; }
function status(message, error = false) {
  text("query-status", message);
  $("query-status").dataset.error = String(error);
}
function toggleModal() {
  const dialog = $("endpoint-modal");
  if (dialog.open) dialog.close();
  else { $("api-url-input").value = API_BASE_URL; dialog.showModal(); $("api-url-input").focus(); }
}
function setEndpoint(value) {
  API_BASE_URL = value;
  try { localStorage.setItem(storageKey, value); } catch { /* In-memory choice still works. */ }
  $("endpoint-modal").close();
  checkHealth();
}
function useCurrentServer() { setEndpoint(sameServer); }
function saveGateway() {
  try {
    const url = new URL($("api-url-input").value.trim());
    if (!/^https?:$/.test(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error();
    setEndpoint(url.href.replace(/\/+$/, ""));
  } catch {
    $("api-url-input").setCustomValidity("Enter a complete http:// or https:// backend URL.");
    $("api-url-input").reportValidity();
  }
}
$("api-url-input").addEventListener("input", () => $("api-url-input").setCustomValidity(""));

async function request(path, options = {}, timeout = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {...options, signal:controller.signal});
    let body;
    try { body = await response.json(); } catch { throw new Error("This URL did not return the RAG API. Check Gateway settings."); }
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail :
        response.status === 422 ? "Enter a question between 1 and 2,000 characters." :
        response.status === 503 ? "The API is reachable, but the models are not ready. Run the Colab loading cell." : `API request failed (${response.status}).`;
      throw new Error(detail);
    }
    return body;
  } finally { clearTimeout(timer); }
}

async function checkHealth() {
  const endpoint = API_BASE_URL;
  text("gateway-label", "Checking connection");
  try {
    const health = await request("/health");
    if (endpoint !== API_BASE_URL) return;
    if (health.ready !== true) throw new Error("Pipeline is not ready.");
    text("gateway-label", "Backend ready");
    $("gateway-label").parentElement.style.cssText = "background:#ecfdf5;color:#047857";
    const meta = await request("/metadata");
    if (endpoint !== API_BASE_URL) return;
    serverThresholds = meta.thresholds;
    if (meta.version !== "4.2") status("This backend uses a different runtime version. Launch the updated notebook.", true);
  } catch (error) {
    if (endpoint !== API_BASE_URL) return;
    text("gateway-label", "Backend unavailable");
    $("gateway-label").parentElement.style.cssText = "background:#fffbeb;color:#92400e";
    if (!busy) status(`${error.message} Open Gateway to choose the Colab URL.`, true);
  }
}

function applyPreset(key) { if (!busy) { $("query-input").value = PRESETS[key] || ""; $("query-input").focus(); } }
function clearConsole() {
  if (busy) return;
  $("query-input").value = "";
  text("initial-answer-view", "Awaiting query execution...");
  text("healed-answer-view", "Awaiting query execution...");
  text("diagnosis-label", "STANDBY"); text("recovery-action-label", "None");
  text("verification-tag", "Awaiting assessment");
  $("verification-tag").className = "resolution-tag";
  for (const id of ["badge-ret-box", "badge-ent-box", "badge-tok-box", "badge-cov-box"]) updateBadgeView(id, "STANDBY", null);
  $("diagnostic-details").hidden = true;
  lastResult = null;
  status("Enter a question to run the final pipeline.");
}
function loadRow(query) {
  if (busy) return;
  $("query-input").value = query;
  $("console").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth"});
  runLiveAudit();
}
function updateBadgeView(id, value, pass) {
  $(id).className = `telemetry-badge-${pass === true ? "pass" : pass === false ? "fail" : "skip"} mono`;
  text(id, value);
}
function signalText(key, signals, thresholds) {
  if (key === "entailment_score" && signals.is_refusal) return "N/A (refusal)";
  const value = signals[key];
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A (unavailable)";
  const pass = key === "mean_entropy" ? value < thresholds[key] : value >= thresholds[key];
  return `${value.toFixed(4)} ${key === "mean_entropy" ? "nats " : ""}${pass ? "PASS" : "FAIL"}`;
}
function outcomeMessage(data) {
  if (data.response_outcome === "supported") return "Supported by signal checks";
  if (data.response_outcome === "refusal") return "Refusal — no factual answer claimed";
  if (data.response_outcome !== "unverified") return data.response_outcome;
  const failed = [], missing = [];
  const names = {retrieval_similarity:"retrieval", entailment_score:"grounding", mean_entropy:"token uncertainty", coverage_score:"knowledge coverage"};
  for (const [key,name] of Object.entries(names)) {
    const value = data.post_signals?.[key], threshold = data.thresholds?.[key];
    if (typeof value !== "number" || !Number.isFinite(value) || typeof threshold !== "number") missing.push(name);
    else if (key === "mean_entropy" ? value >= threshold : value < threshold) failed.push(name);
  }
  const reasons = [];
  if (failed.length) reasons.push(`${failed.join(", ")} ${failed.length === 1 ? "check" : "checks"} failed`);
  if (missing.length) reasons.push(`${missing.join(", ")} unavailable`);
  return `Unverified — ${reasons.join("; ") || "verification incomplete"}`;
}
function row(parent, values) {
  const tr = document.createElement("tr");
  for (const value of values) {
    const td = document.createElement("td"); td.textContent = value; tr.append(td);
  }
  parent.append(tr); return tr;
}
function definition(parent, term, value) {
  const wrap = document.createElement("div");
  const dt = document.createElement("dt"); dt.textContent = term;
  const dd = document.createElement("dd"); dd.textContent = value;
  wrap.append(dt, dd); parent.append(wrap);
}
function renderDetails(data) {
  const trace = data.recovery_trace || {};
  text("run-summary", `BERT selected ${labels[data.predicted_failure.toLowerCase()] || data.predicted_failure}. Runtime ${data.version}; completed in ${data.elapsed_seconds.toFixed(2)} seconds.`);
  $("bert-probabilities").replaceChildren();
  for (const [key, value] of Object.entries(data.class_probabilities || {})) definition($("bert-probabilities"), labels[key] || key, `${(value*100).toFixed(2)}%`);
  $("recovery-decision").replaceChildren();
  definition($("recovery-decision"), "Action", data.action_taken);
  definition($("recovery-decision"), "Query rewrite", trace.rewritten_query || "Not requested for this route");
  definition($("recovery-decision"), "Selected response", trace.candidate_used ? "Recovery candidate" : trace.route === "clean" ? "Initial response" : data.response_outcome === "refusal" ? "Refusal" : "Initial response retained");
  definition($("recovery-decision"), "Reason", trace.reason || data.verification_status);
  $("signal-comparison").replaceChildren();
  for (const [key, name] of [["retrieval_similarity","Retrieval similarity"],["entailment_score","Grounding (NLI)"],["mean_entropy","Token entropy"],["coverage_score","Index proximity"]]) {
    row($("signal-comparison"), [name, signalText(key,data.pre_signals,data.thresholds), signalText(key,data.post_signals,data.thresholds)]);
  }
  $("evidence-list").replaceChildren();
  for (const evidence of data.final_contexts || []) {
    const li = document.createElement("li"); li.textContent = evidence; $("evidence-list").append(li);
  }
  if (!data.final_contexts?.length) { const li = document.createElement("li"); li.textContent = "No evidence passages returned."; $("evidence-list").append(li); }
  text("recovery-trace", JSON.stringify(trace,null,2));
  $("diagnostic-details").hidden = false;
}

async function runLiveAudit() {
  if (busy) return;
  const query = $("query-input").value.trim();
  if (!query || query.length > 2000) { status("Enter a question between 1 and 2,000 characters.", true); return; }
  busy = true; lastResult = null;
  $("run-btn").disabled = true;
  text("run-btn", "Running…");
  $("diagnostic-details").hidden = true;
  text("initial-answer-view", "Retrieving local evidence and generating the initial answer…");
  text("healed-answer-view", "BERT diagnosis, evidence recovery and response assessment are running…");
  text("verification-tag", "Assessment in progress"); $("verification-tag").className = "resolution-tag";
  text("diagnosis-label", "RUNNING"); text("recovery-action-label", "Pending diagnosis");
  for (const id of ["badge-ret-box","badge-ent-box","badge-tok-box","badge-cov-box"]) updateBadgeView(id,"RUNNING",null);
  status("Running the final local pipeline. Keep the Colab runtime connected.");
  try {
    const data = await request("/query", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query})},180000);
    if (!data.pre_signals || !data.post_signals || !data.response_outcome || !data.thresholds) throw new Error("This backend is outdated. Launch the updated deployment notebook.");
    lastResult = data;
    text("initial-answer-view", data.initial_answer); text("healed-answer-view", data.healed_answer);
    text("diagnosis-label", `BERT · ${labels[data.predicted_failure.toLowerCase()] || data.predicted_failure}`);
    text("recovery-action-label", data.action_taken);
    const outcomeText = outcomeMessage(data);
    text("verification-tag", outcomeText);
    $("verification-tag").className = `resolution-tag ${data.response_outcome === "supported" ? "tag-healed" : data.response_outcome === "refusal" ? "tag-refusal" : "tag-unverified"}`;
    serverThresholds = data.thresholds;
    for (const [id,key] of [["badge-ret-box","retrieval_similarity"],["badge-ent-box","entailment_score"],["badge-tok-box","mean_entropy"],["badge-cov-box","coverage_score"]]) {
      const value = signalText(key,data.pre_signals,serverThresholds);
      updateBadgeView(id,value,value.startsWith("N/A") ? null : value.endsWith("PASS"));
    }
    renderDetails(data);
    const changed = data.initial_answer.trim() !== data.healed_answer.trim();
    const selection = data.response_outcome === "refusal" ? "The system returned a refusal." :
      changed && data.recovery_trace?.candidate_used ? "Recovery generated an updated answer." :
      !changed ? "The initial answer was retained." : "The final response changed.";
    status(`Complete. ${selection} ${outcomeText}.`);
  } catch (error) {
    const message = error.name === "AbortError" ? "The request timed out. Check the Colab runtime, then retry." : error.message || "Connection failed. Check the Gateway URL and retry.";
    status(message,true);
    text("initial-answer-view", "No completed result was returned.");
    text("healed-answer-view", "Check the connection or runtime, then try again.");
    text("verification-tag", "Run incomplete");
    text("diagnosis-label", "NOT AVAILABLE"); text("recovery-action-label", "Not available");
    for (const id of ["badge-ret-box","badge-ent-box","badge-tok-box","badge-cov-box"]) updateBadgeView(id,"NOT AVAILABLE",null);
  } finally {
    busy = false; $("run-btn").disabled = false; text("run-btn","Audit & Self-Heal");
  }
}

function renderBenchmark(data) {
  const percent = value => `${(value*100).toFixed(2)}%`;
  text("metric-clean",percent(data.clean_accuracy.fraction)); text("count-clean",`${data.clean_accuracy.correct} / ${data.clean_accuracy.samples} clean cases`);
  text("metric-answered",percent(data.answered_accuracy.fraction)); text("count-answered",`${data.answered_accuracy.correct} / ${data.answered_accuracy.answered} answered QA cases`);
  text("metric-refusal",percent(data.refusal_compliance.fraction)); text("count-refusal",`${data.refusal_compliance.refusals} / ${data.refusal_compliance.designated_challenges} designated challenges`);
  text("metric-rate",percent(data.answer_rate.fraction)); text("count-rate",`${data.answer_rate.answered} / ${data.answer_rate.samples} QA cases answered; ${data.qa_refusals} refused`);
  text("aggregate-metrics",`Across all 120 reference-scored QA cases, exact match increased from ${percent(data.initial_exact_match.fraction)} to ${percent(data.full_qa_exact_match.fraction)} (+${data.extra_exact_answers} exact answers). Final token F1: ${percent(data.token_f1)}. Completed ${data.regression_rows} cases with ${data.execution_errors} execution errors.`);
  text("comparison-note",data.comparison_note);
  for (const item of data.comparison) row($("comparison-rows"),[item.name,percent(item.exact_match),percent(item.token_f1),`${item.answered_qa} / 120`,`${item.coverage_refusals} / 18`]);
}

function renderLiveExamples(data) {
  $("benchmark-rows").replaceChildren();
  text("benchmark-note", data.note);
  for (const example of data.examples.filter(item => item.initial_answer.trim() !== item.final_answer.trim() && item.outcome !== "refusal")) {
    const tr = row($("benchmark-rows"),[example.query,example.initial_answer,example.final_answer,example.outcome]);
    const td = document.createElement("td"); const button = document.createElement("button");
    button.className = "btn btn-secondary"; button.textContent = "Run live question";
    button.setAttribute("aria-label",`Run live recovery example: ${example.query}`);
    button.addEventListener("click",()=>loadRow(example.query)); td.append(button); tr.append(td);
  }
}

$("download-trace").addEventListener("click",()=>{
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult,null,2)],{type:"application/json"}));
  const link = document.createElement("a"); link.href=url; link.download="rag-query-trace.json"; link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$("query-input").addEventListener("keydown",event=>{ if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runLiveAudit(); });
icons();
if (window.RAG_BENCHMARK) renderBenchmark(window.RAG_BENCHMARK);
else status("The benchmark data file is missing. Restore static/benchmark-data.js.",true);
if (window.RAG_LIVE_EXAMPLES) renderLiveExamples(window.RAG_LIVE_EXAMPLES);
else text("benchmark-note", "Recent live examples are unavailable. Restore static/live-examples.js.");
checkHealth();
