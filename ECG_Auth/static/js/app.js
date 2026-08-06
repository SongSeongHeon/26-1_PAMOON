// 기본 상태
const state = {
  selectedFile: null,
  uploaded: false,
  extracted: false,
  templateRegistered: false,
  verified: false,
  waveformData: null,
  templateWaveformData: null,
  verifyWaveformData: null,
  animationTimer: null,
  latestPollingTimer: null,
  latestPollingBusy: false,
  latestResultSignature: null,
  lastReport: null,
  lastTemplate: null,
  lastInference: null,
  lastHrv: null,
  lastMode: null,
  currentSubject: null,
  currentFileMeta: null,
  processedResultSignatures: new Set(),
  recordSignatures: new Set(),
};

const $ = (id) => document.getElementById(id);

// DOM
const rawInput = $("rawInput");
const modalBackdrop = $("modalBackdrop");
const modalCloseBtn = $("modalCloseBtn");
const qualityInfoBtn = $("qualityInfoBtn");
const subjectOpenBtn = $("subjectOpenBtn");
const currentSubjectText = $("currentSubjectText");

// 초기 실행
bindEvents();
resetTrustStatus();
clearAuthResult();
loadCurrentSubject();
startLatestResultPolling();

// 이벤트 연결
function bindEvents() {
  if (rawInput) {
    rawInput.addEventListener("change", handleRawInputChange);
  }

  if (subjectOpenBtn) {
    subjectOpenBtn.addEventListener("click", openSubjectModal);
  }

  if ($("authLogBtn")) {
    $("authLogBtn").addEventListener("click", openAuthLogModal);
  }

  if ($("exportBtn")) {
    $("exportBtn").addEventListener("click", saveCurrentReportPdf);
  }

  if ($("resetBtn")) {
    $("resetBtn").addEventListener("click", resetAll);
  }

  if (qualityInfoBtn) {
    qualityInfoBtn.addEventListener("click", openQualityInfoModal);
  }

  if (modalCloseBtn) {
    modalCloseBtn.addEventListener("click", closeModal);
  }

  if (modalBackdrop) {
    modalBackdrop.addEventListener("click", (event) => {
      if (event.target === modalBackdrop) {
        event.preventDefault();
        event.stopPropagation();
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModal();
    }
  });
}

// 측정자 설정 모달 열기
function openSubjectModal() {
  const subject = state.currentSubject || {
    subject_id: "",
    birth_date: "",
    name: "",
  };

  openModal(
    "측정자 설정",
    `
    <div class="subject-modal-card">
      <div class="subject-modal-summary">
        <span>측정에 사용할 대상을 설정합니다.</span>
        <strong>사용자 ID는 S001 형식으로 입력합니다.</strong>
      </div>

      <div class="subject-modal-form">
        <label class="subject-modal-field">
          <span>사용자 ID</span>
          <input
            id="modalSubjectIdInput"
            type="text"
            value="${escapeHtml(subject.subject_id || "")}"
            placeholder="S001"
            autocomplete="off"
          />
        </label>

        <label class="subject-modal-field">
          <span>생년월일</span>
          <input
            id="modalSubjectBirthInput"
            type="text"
            value="${escapeHtml(subject.birth_date || "")}"
            placeholder="YYYYMMDD"
            inputmode="numeric"
            autocomplete="off"
          />
        </label>
      </div>

      <div class="subject-modal-actions">
        <button class="subject-modal-btn cancel" id="subjectCancelBtn" type="button">
          취소
        </button>

        <button class="subject-modal-btn save" id="subjectSaveBtn" type="button">
          저장
        </button>
      </div>
    </div>
    `,
  );

  const modalBox = document.querySelector(".modal-box");
  if (modalBox) {
    modalBox.classList.add("subject-modal-box");
  }

  const saveBtn = $("subjectSaveBtn");
  const cancelBtn = $("subjectCancelBtn");
  const birthInput = $("modalSubjectBirthInput");

  if (birthInput) {
    birthInput.addEventListener("input", (event) => {
      event.target.value = normalizeBirthDateInput(event.target.value);
    });
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", setCurrentSubjectFromModal);
  }

  if (cancelBtn) {
    cancelBtn.addEventListener("click", closeModal);
  }
}

// 측정자 설정 저장
async function setCurrentSubjectFromModal() {
  const subjectId = normalizeSubjectId($("modalSubjectIdInput")?.value);
  const birthDate = normalizeBirthDateInput($("modalSubjectBirthInput")?.value);

  if (!isValidSubjectId(subjectId)) {
    toast("사용자 ID는 S001 형식으로 입력하세요.");
    return;
  }

  if (!birthDate || birthDate.length !== 8) {
    toast("생년월일은 YYYYMMDD 형식으로 입력하세요.");
    return;
  }

  const payload = {
    subject_id: subjectId,
    birth_date: birthDate,
    name: subjectId,
  };

  const saveBtn = $("subjectSaveBtn");

  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.textContent = "저장 중...";
  }

  try {
    const response = await fetch("/api/subject/set", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await parseJsonResponse(response);

    if (!response.ok || !data.success) {
      throw new Error(data.message || "측정자 설정에 실패했습니다.");
    }

    state.currentSubject = data.current_subject || payload;
    updateCurrentSubjectUI(state.currentSubject);
    closeModal();

    toast("측정자가 설정되었습니다.");
  } catch (error) {
    console.error("[SUBJECT SET ERROR]", error);
    toast(error.message || "측정자 설정 중 오류가 발생했습니다.");
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = "저장";
    }
  }
}

// 현재 측정자 조회
async function loadCurrentSubject() {
  try {
    const response = await fetch("/api/subject/current", {
      method: "GET",
      cache: "no-store",
    });

    const data = await parseJsonResponse(response);

    if (!response.ok || !data.success || !data.has_subject) {
      state.currentSubject = null;
      updateCurrentSubjectUI(null);
      return;
    }

    state.currentSubject = data.current_subject;
    updateCurrentSubjectUI(state.currentSubject);
  } catch (error) {
    console.warn("[SUBJECT CURRENT ERROR]", error.message);
    updateCurrentSubjectUI(null);
  }
}

// 현재 측정자 표시 갱신
function updateCurrentSubjectUI(subject) {
  if (!currentSubjectText) return;

  if (!subject) {
    currentSubjectText.textContent = "측정자 미설정";
    currentSubjectText.classList.remove("active");
    return;
  }

  const subjectId = subject.subject_id || "-";
  const birthDate = subject.birth_date || "";

  currentSubjectText.textContent = birthDate
    ? `현재 대상: ${subjectId} / ${birthDate}`
    : `현재 대상: ${subjectId}`;

  currentSubjectText.classList.add("active");
}

// 최신 측정 결과 자동 확인
function startLatestResultPolling() {
  stopLatestResultPolling();

  fetchLatestResultFromServer();

  state.latestPollingTimer = setInterval(() => {
    fetchLatestResultFromServer();
  }, 3000);
}

function stopLatestResultPolling() {
  if (!state.latestPollingTimer) return;

  clearInterval(state.latestPollingTimer);
  state.latestPollingTimer = null;
}

async function fetchLatestResultFromServer() {
  if (state.latestPollingBusy) {
    return;
  }

  state.latestPollingBusy = true;

  try {
    const response = await fetch("/api/latest-result", {
      method: "GET",
      cache: "no-store",
    });

    const data = await parseJsonResponse(response);

    if (!response.ok || !data.success || !data.has_result || !data.result) {
      return;
    }

    const signature = buildLatestResultSignature(data);

    if (
      signature === state.latestResultSignature ||
      hasProcessedResult(signature)
    ) {
      return;
    }

    state.latestResultSignature = signature;
    rememberProcessedResult(signature);

    applyProcessedEcgResult(data.result, {
      silent: true,
      source: "latest",
    });
  } catch (error) {
    console.warn("[LATEST RESULT POLLING ERROR]", error.message);
  } finally {
    state.latestPollingBusy = false;
  }
}

function buildLatestResultSignature(data = {}) {
  return buildResultSignature(data.result || {});
}

function buildManualResultSignature(data = {}) {
  return buildResultSignature(data);
}

function buildResultSignature(data = {}) {
  const report = data.report || {};
  const inference = data.inference || {};
  const template = data.template || {};
  const mode = data.mode || "";

  const subjectId =
    report.subject_id ||
    inference.subject_id ||
    data.subject_id ||
    state.currentFileMeta?.subject_id ||
    "";

  const filename = normalizeResultFilename(
    report.filename ||
      data.filename ||
      state.currentFileMeta?.filename ||
      state.selectedFile?.name ||
      "",
  );

  return [
    mode,
    subjectId,
    filename,
    report.measured_at || "",
    inference.cosine_similarity ?? "",
    inference.matching_score ?? "",
    inference.decision || "",
    template.status || "",
  ].join("|");
}

function normalizeResultFilename(filename = "") {
  return String(filename || "")
    .replace(/^manual_[^_]+_\d{8}_\d{6}_[a-f0-9]+_/i, "")
    .replace(/^watch_[^_]+_\d{8}_\d{6}_[a-f0-9]+_/i, "")
    .trim();
}

function hasProcessedResult(signature) {
  return Boolean(signature && state.processedResultSignatures.has(signature));
}

function rememberProcessedResult(signature) {
  if (!signature) return;

  state.processedResultSignatures.add(signature);

  if (state.processedResultSignatures.size > 100) {
    const first = state.processedResultSignatures.values().next().value;
    state.processedResultSignatures.delete(first);
  }
}

// ECG JSON 선택 처리
async function handleRawInputChange(event) {
  const file = event.target.files[0];

  if (!file) {
    state.selectedFile = null;
    state.currentFileMeta = null;
    setText("fileName", "선택된 파일 없음");
    return;
  }

  const isJson =
    file.type === "application/json" ||
    file.name.toLowerCase().endsWith(".json");

  if (!isJson) {
    state.selectedFile = null;
    state.currentFileMeta = null;

    if (rawInput) rawInput.value = "";

    setText("fileName", "선택된 파일 없음");
    toast("JSON 파일만 선택할 수 있습니다.");
    return;
  }

  state.selectedFile = file;
  setText("fileName", file.name);

  try {
    const metadata = await readRawJsonMetadata(file);
    state.currentFileMeta = metadata;

    updateCurrentSubjectUI({
      subject_id: metadata.subject_id,
      birth_date: state.currentSubject?.birth_date || "",
      name: metadata.subject_name,
    });
  } catch (error) {
    state.currentFileMeta = null;

    if (rawInput) rawInput.value = "";

    setText("fileName", "선택된 파일 없음");
    toast(error.message);
    return;
  }

  await processSelectedRawJson(file);
}

// ECG JSON 메타데이터 읽기
async function readRawJsonMetadata(file) {
  const text = await file.text();
  let data = {};

  try {
    data = JSON.parse(text);
  } catch (error) {
    throw new Error("JSON 파일 형식이 올바르지 않습니다.");
  }

  const filename = file.name || data.filename || "";
  const subjectId = resolveSubjectIdFromJson(data, filename);
  const sessionId = resolveSessionIdFromJson(data, filename);
  const subjectName = normalizeSubjectNameFromJson(data, subjectId);

  return {
    subject_id: subjectId,
    session_id: sessionId,
    subject_name: subjectName,
    filename,
    measured_at: data.measured_at || "",
    sampling_rate: data.sampling_rate || "",
    duration_seconds: data.duration_seconds || "",
    raw: data,
  };
}

// JSON/파일명에서 사용자 ID 결정
function resolveSubjectIdFromJson(data = {}, filename = "") {
  const candidates = [
    data.subject_id,
    data.user_id,
    data.patient_id,
    data.name,
  ];

  for (const candidate of candidates) {
    const normalized = normalizeSubjectId(candidate);

    if (isValidSubjectId(normalized)) {
      return normalized;
    }
  }

  const fromFilename = extractSubjectIdFromFilename(filename);

  if (fromFilename) {
    return fromFilename;
  }

  throw new Error(
    "subject_id를 확인할 수 없습니다. 파일명을 S001_T01_... 형식으로 바꾸거나 JSON 내부 subject_id를 S001 형식으로 수정하세요.",
  );
}

// JSON/파일명에서 세션 ID 결정
function resolveSessionIdFromJson(data = {}, filename = "") {
  const candidates = [data.session_id, data.trial_id, data.record_id];

  for (const candidate of candidates) {
    const normalized = normalizeSessionId(candidate);

    if (normalized) {
      return normalized;
    }
  }

  const match = String(filename || "").match(/^S\d{3}_(T\d{2})_/i);

  if (match) {
    return match[1].toUpperCase();
  }

  return "";
}

// 사용자 표시명 결정
function normalizeSubjectNameFromJson(data = {}, fallbackSubjectId = "") {
  const candidates = [
    data.subject_name,
    data.real_name,
    data.display_name,
    data.name,
  ];

  for (const candidate of candidates) {
    const value = String(candidate || "").trim();

    if (
      value &&
      value !== "USER-01" &&
      value !== "ECG 데이터" &&
      value.toLowerCase() !== "ecg data"
    ) {
      return value;
    }
  }

  return fallbackSubjectId || "ECG 데이터";
}

// 파일명에서 S001 추출
function extractSubjectIdFromFilename(filename = "") {
  const match = String(filename || "").match(/^(S\d{3})_/i);
  return match ? match[1].toUpperCase() : "";
}

// 사용자 ID 정규화
function normalizeSubjectId(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replace(/\s+/g, "");
}

// 사용자 ID 유효성 검사
function isValidSubjectId(value) {
  return /^S\d{3}$/.test(String(value || "")) && value !== "USER-01";
}

// 세션 ID 정규화
function normalizeSessionId(value) {
  const text = String(value || "")
    .trim()
    .toUpperCase();

  if (/^T\d{2}$/.test(text)) {
    return text;
  }

  if (/^\d+$/.test(text)) {
    return `T${text.padStart(2, "0")}`;
  }

  return "";
}

// ECG 처리 FormData 생성
function buildRawEcgFormData(file, metadata = {}) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("subject_id", metadata.subject_id || "");

  if (metadata.session_id) {
    formData.append("session_id", metadata.session_id);
  }

  if (metadata.subject_name) {
    formData.append("subject_name", metadata.subject_name);
  }

  if (metadata.filename) {
    formData.append("filename", metadata.filename);
  }

  return formData;
}

// ECG JSON 수동 처리
async function processSelectedRawJson(file) {
  setLoading(true, "ECG 데이터 처리 중...");
  setStatus("ECG 데이터 처리 중");
  setText("decisionText", "ECG 데이터 처리 중");
  setText("decisionSubText", "ECG 데이터를 분석하고 있습니다.");

  let metadata = state.currentFileMeta;

  try {
    if (!metadata || metadata.filename !== file.name) {
      metadata = await readRawJsonMetadata(file);
      state.currentFileMeta = metadata;
    }

    const formData = buildRawEcgFormData(file, metadata);

    console.log("[RAW ECG METADATA]", {
      subject_id: metadata.subject_id,
      session_id: metadata.session_id,
      subject_name: metadata.subject_name,
      filename: metadata.filename,
    });

    const response = await fetch("/api/raw-ecg/process", {
      method: "POST",
      body: formData,
    });

    const data = await parseJsonResponse(response);

    console.log("[API /api/raw-ecg/process]", data);

    if (!response.ok || !data.success) {
      throw new Error(data.message || "ECG 데이터 처리에 실패했습니다.");
    }

    const signature = buildManualResultSignature(data);
    state.latestResultSignature = signature;
    rememberProcessedResult(signature);

    applyProcessedEcgResult(data, {
      silent: false,
      source: "manual",
    });
  } catch (error) {
    console.error("[ECG JSON PROCESS ERROR]", error);

    const userMessage = normalizeUserErrorMessage(
      error.message || "ECG 데이터 처리 중 오류가 발생했습니다.",
    );

    toast(userMessage);
    setStatus("처리 실패");
    setText("decisionText", "처리 실패");
    setText("decisionSubText", userMessage);
  } finally {
    setLoading(false);

    if (rawInput) {
      rawInput.value = "";
    }
  }
}

// ECG 처리 결과 적용
function applyProcessedEcgResult(data, options = {}) {
  const silent = options.silent === true;
  const source = options.source || "manual";

  state.uploaded = true;
  state.extracted = true;
  state.waveformData = data.waveform || null;
  state.lastReport = data.report || state.lastReport;
  state.lastTemplate = data.template || state.lastTemplate;
  state.lastInference = data.inference || null;
  state.lastHrv = data.hrv || null;
  state.lastMode = data.mode || null;

  if (data.report?.subject_id) {
    state.currentSubject = {
      ...(state.currentSubject || {}),
      subject_id: data.report.subject_id,
      name:
        state.currentFileMeta?.subject_name ||
        data.report.subject_name ||
        data.report.name ||
        data.report.subject_id,
    };
    updateCurrentSubjectUI(state.currentSubject);
  }

  if (source === "watch" || source === "latest") {
    setText("fileName", data.report?.filename || "워치 ECG 데이터 수신");
  }

  updateReportInfo(data.report || {});
  updateTrustStatus(data.hrv);

  markStep("stepUpload", "done");
  markStep("stepExtract", "done");

  if (data.mode === "register") {
    handleRegisterMode(data, { silent, source });
    return;
  }

  if (data.mode === "verify") {
    handleVerifyMode(data, { silent, source });
    return;
  }

  applyRawEcgResponse(data);

  if (!silent) {
    toast("ECG 데이터 처리가 완료되었습니다.");
  }
}

// ECG JSON 응답 적용
function applyRawEcgResponse(data) {
  const report = data.report || {};
  const waveform = data.waveform || {};
  const rawSummary = data.raw_summary || {};

  state.uploaded = true;
  state.extracted = true;
  state.templateRegistered = false;
  state.verified = false;

  state.lastMode = "raw_ecg";
  state.lastReport = report;
  state.waveformData = waveform;
  state.templateWaveformData = waveform;
  state.verifyWaveformData = null;
  state.lastTemplate = null;
  state.lastInference = null;
  state.lastHrv = null;

  setText("reportName", getDisplaySubjectName(report));
  setText("measuredTime", report.measured_at || "-");
  setText("heartRate", report.heart_rate ? `${report.heart_rate} bpm` : "-");
  setText(
    "deviceName",
    report.device || rawSummary.device || "Samsung Galaxy Watch 6",
  );
  setText(
    "samplingRate",
    report.sampling_rate || `${rawSummary.sampling_rate || 500} Hz`,
  );
  setText("leadType", normalizeLeadText(report.lead || "Lead I ECG"));

  setText("modeValue", "ECG 데이터 입력");
  setStatus("ECG 데이터 처리 완료");
  setText("decisionText", "ECG 데이터 입력 확인");
  setText(
    "decisionSubText",
    `샘플 ${rawSummary.raw_sample_count || "-"}개, ${rawSummary.sampling_rate || "-"} Hz ECG 데이터가 로드되었습니다.`,
  );

  markStep("stepUpload", "done");
  markStep("stepExtract", "done");
  resetStep("stepRegister");
  resetStep("stepVerify");

  updateRawSummaryToQualityArea(rawSummary);
  renderWaveform(waveform);
}

// ECG JSON 요약을 신호 품질 영역에 표시
function updateRawSummaryToQualityArea(rawSummary = {}) {
  const sampleCount =
    rawSummary.raw_sample_count || rawSummary.sample_count || "-";
  const samplingRate = rawSummary.sampling_rate || "-";
  const duration = rawSummary.duration_seconds || "-";
  const qualityStatus = rawSummary.quality_status || "확인";
  const qualityReason = rawSummary.quality_reason || "-";
  const completeness = formatRatioPercent(rawSummary.collection_completeness);
  const missing = rawSummary.missing_sample_estimate ?? "-";
  const flatTail = rawSummary.flat_tail_detected ? "감지됨" : "없음";
  const sanityText = formatSanityCheckSummary(rawSummary.sanity_check);

  setText("trustBadge", translateCollectionQualityStatus(qualityStatus));
  setText("trustLevel", "ECG 데이터 입력");
  setText(
    "trustMessage",
    `샘플 ${sampleCount}개, ${samplingRate} Hz, ${duration}초입니다. 완성도 ${completeness}, 누락 추정 ${missing}개, 후반부 flatline ${flatTail}, 사유 ${qualityReason}${sanityText ? `. ${sanityText}` : ""}.`,
  );

  setText("hrvHeartRate", "-");
  setText("hrvQualityLabel", translateCollectionQualityStatus(qualityStatus));
  setText("hrvValidBeat", "-");
  setText("hrvStabilityScore", "-");

  setText("hrvPdfHeartRate", "-");
  setText("hrvEstimatedHeartRate", "-");
  setText("hrvSdnn", "-");
  setText("hrvRmssd", "-");

  const trustCard = $("trustCard");
  const trustBadge = $("trustBadge");
  const statusClass = getTrustStatusClass(qualityStatus);

  if (trustCard) {
    trustCard.classList.remove("stable", "caution", "warning", "unavailable");
    trustCard.classList.add(statusClass);
  }

  if (trustBadge) {
    trustBadge.classList.remove(
      "waiting",
      "stable",
      "caution",
      "warning",
      "unavailable",
    );
    trustBadge.classList.add(statusClass);
  }

  console.log("[ECG DATA SUMMARY]", rawSummary);
}

function formatSanityCheckSummary(sanity = {}) {
  if (!sanity || typeof sanity !== "object") {
    return "";
  }

  const status = String(sanity.status || "").toLowerCase();
  const warnings = Array.isArray(sanity.warnings) ? sanity.warnings : [];
  const correctedFields = Array.isArray(sanity.corrected_fields)
    ? sanity.corrected_fields
    : [];

  if (!status || (status === "ok" && warnings.length === 0)) {
    return "";
  }

  const parts = [];

  if (status === "corrected") {
    parts.push("서버에서 메타데이터 불일치를 자동 보정했습니다");
  } else if (status !== "unavailable") {
    parts.push(`sanity=${status}`);
  }

  if (correctedFields.length > 0) {
    parts.push(`보정 필드 ${correctedFields.join(", ")}`);
  }

  if (warnings.length > 0) {
    parts.push(`경고 ${warnings.join(", ")}`);
  }

  return parts.join(" / ");
}

// 등록 모드 처리
function handleRegisterMode(data, options = {}) {
  const report = data.report || {};
  const template = data.template || {};
  const subjectId =
    report.subject_id ||
    state.currentFileMeta?.subject_id ||
    state.currentSubject?.subject_id ||
    "";
  const userName = getDisplaySubjectName(report);
  const silent = options.silent === true;
  const source = options.source || "manual";

  state.templateRegistered = true;
  state.verified = false;
  state.lastTemplate = template;
  state.lastInference = null;
  state.lastHrv = data.hrv || state.lastHrv;
  state.lastMode = "register";

  state.templateWaveformData = data.template_waveform || data.waveform || null;
  state.verifyWaveformData = null;

  renderWaveformComparison(state.templateWaveformData, null, "template");

  setTemplateStatus("registered", {
    main: "템플릿 등록 완료",
    desc: subjectId
      ? `${subjectId} 기준 템플릿으로 저장되었습니다.`
      : "현재 ECG 데이터가 기준 템플릿으로 저장되었습니다.",
  });

  markStep("stepRegister", "done");
  resetStep("stepVerify");

  clearAuthResult();
  setText("modeValue", "템플릿 등록");
  setText(
    "decisionSubText",
    subjectId
      ? `${subjectId} ECG 데이터가 기준 템플릿으로 저장되었습니다.`
      : "첫 번째 ECG 데이터가 기준 템플릿으로 저장되었습니다.",
  );

  updateTrustStatus(data.hrv);
  setStatus("템플릿 등록 완료");

  console.log("[TEMPLATE DEBUG]", {
    template,
    segmentation: template.segmentation,
    collection_quality: template.collection_quality,
    waveform_quality: template.waveform_quality,
  });

  if (!silent) {
    toast(`${userName} ECG 템플릿이 저장되었습니다.`);
  } else if (source === "watch" || source === "latest") {
    toast("워치 ECG 데이터가 템플릿으로 등록되었습니다.");
  }
}

// 인증 모드 처리
function handleVerifyMode(data, options = {}) {
  const report = data.report || {};
  const inference = data.inference || {};
  const subjectId =
    report.subject_id ||
    inference.subject_id ||
    state.currentFileMeta?.subject_id ||
    state.currentSubject?.subject_id ||
    "";
  const userName = getDisplaySubjectName(report);
  const silent = options.silent === true;
  const source = options.source || "manual";

  state.templateRegistered = true;
  state.verified = true;
  state.lastTemplate = data.template || state.lastTemplate;
  state.lastInference = inference;
  state.lastHrv = data.hrv || state.lastHrv;
  state.lastMode = "verify";

  state.templateWaveformData =
    data.template_waveform || state.templateWaveformData || null;
  state.verifyWaveformData = data.waveform || null;

  renderWaveformComparison(
    state.templateWaveformData,
    state.verifyWaveformData,
    "verify",
  );

  setText("modeValue", "인증 비교");

  setTemplateStatus("verify", {
    main: "인증 비교 완료",
    desc: subjectId
      ? `${subjectId} 기준 템플릿과 현재 ECG 데이터를 비교했습니다.`
      : "기준 템플릿과 현재 ECG 데이터를 비교했습니다.",
  });

  markStep("stepRegister", "done");
  markStep("stepVerify", "done");

  updateAuthResult(inference);
  updateTrustStatus(data.hrv);
  addRecord(inference, data.auth_log || {}, report);

  setStatus("인증 처리 완료");

  console.log("[INFERENCE DEBUG]", {
    inference,
    decision_reason: inference.decision_reason,
    segmentation: inference.segmentation,
    similarity_detail: inference.similarity_detail,
    collection_quality: inference.collection_quality,
    waveform_quality: inference.waveform_quality,
  });

  if (!silent) {
    toast(`${userName} ECG 인증 비교가 완료되었습니다.`);
  } else if (source === "watch" || source === "latest") {
    toast("워치 ECG 데이터 인증 결과가 반영되었습니다.");
  }
}

// 표시 이름 결정
function getDisplaySubjectName(report = {}) {
  const subjectId =
    report.subject_id || state.currentFileMeta?.subject_id || "";
  const candidate =
    state.currentFileMeta?.subject_name ||
    report.subject_name ||
    report.name ||
    subjectId;

  const displayName = normalizeDisplayName(candidate);

  if (
    subjectId &&
    displayName &&
    displayName !== subjectId &&
    displayName !== "ECG 데이터"
  ) {
    return `${displayName} (${subjectId})`;
  }

  return displayName || subjectId || "ECG 데이터";
}

// 보고서 정보 갱신
function updateReportInfo(report = {}) {
  const displayName = getDisplaySubjectName(report);
  const leadText = normalizeLeadText(report.lead || "Lead I ECG");

  setText("reportName", displayName);
  setText("measuredTime", report.measured_at || "-");
  setText("heartRate", report.heart_rate ? `${report.heart_rate} bpm` : "-");
  setText("deviceName", report.device || "-");
  setText("samplingRate", report.sampling_rate || "-");
  setText("leadType", leadText);
}

function isComparableInferenceResult(result = {}) {
  const reason = String(result.decision_reason || "").toLowerCase();
  const score = Number(result.matching_score);
  const similarity = Number(result.cosine_similarity);
  const embeddingCount = Number(result.embedding_count);

  if (
    reason.includes("quality gate") ||
    reason.includes("inference error") ||
    reason.includes("not enough") ||
    reason.includes("failed to extract") ||
    reason.includes("r_peak") ||
    reason.includes("r-peak")
  ) {
    return false;
  }

  if (!Number.isFinite(score) || !Number.isFinite(similarity)) {
    return false;
  }

  if (Number.isFinite(embeddingCount) && embeddingCount <= 0) {
    return false;
  }

  return true;
}

// 인증 결과 갱신
function updateAuthResult(result = {}) {
  const comparable = isComparableInferenceResult(result);

  const score = toFiniteNumber(result.matching_score, 0);
  const similarity = toFiniteNumber(result.cosine_similarity, 0);
  const threshold = toFiniteNumber(result.threshold, 0.85);
  const decision = result.decision || "-";
  const decisionLabel = getDecisionLabel(decision);
  const normalizedDecision = String(decision).toLowerCase();

  setText("thresholdValue", threshold.toFixed(3));
  setText("decisionValue", decisionLabel);
  setText("decisionText", decisionLabel);

  if (!comparable) {
    setText("matchingScore", "--");
    setText("similarityValue", "-");
    setText(
      "decisionSubText",
      "ECG 신호에서 인증 비교에 사용할 수 있는 유효 박동을 충분히 추출하지 못했습니다.",
    );
    renderScoreGauge(0, "unavailable");

    const decisionText = $("decisionText");
    if (decisionText) {
      decisionText.classList.remove("success", "fail");
      decisionText.classList.add("fail");
    }

    return;
  }

  setText("matchingScore", score.toFixed(1));
  setText("similarityValue", similarity.toFixed(3));

  if (normalizedDecision === "authenticated") {
    setText(
      "decisionSubText",
      "기준 템플릿과 현재 ECG의 유사도가 기준값 이상입니다.",
    );
  } else if (normalizedDecision === "rejected") {
    setText(
      "decisionSubText",
      "기준 템플릿과 현재 ECG의 유사도가 기준값보다 낮습니다.",
    );
  } else {
    setText("decisionSubText", "현재 ECG와 기준 템플릿을 비교했습니다.");
  }

  renderScoreGauge(score, normalizedDecision);

  const decisionText = $("decisionText");
  if (decisionText) {
    decisionText.classList.remove("success", "fail");

    if (normalizedDecision === "authenticated") {
      decisionText.classList.add("success");
    } else if (normalizedDecision === "rejected") {
      decisionText.classList.add("fail");
    }
  }
}

// 디버깅 요약 포맷
function formatEmbeddingSummary(payload = {}) {
  const count = toFiniteNumber(payload.embedding_count, NaN);
  const dim = toFiniteNumber(payload.embedding_dim, NaN);

  if (Number.isFinite(count) && Number.isFinite(dim)) {
    return `템플릿 beat embedding ${count}개 / ${dim}D`;
  }

  if (Number.isFinite(count)) {
    return `템플릿 beat embedding ${count}개`;
  }

  if (Number.isFinite(dim)) {
    return `${dim}D embedding`;
  }

  return "";
}

function formatSegmentationSummary(segmentation = {}) {
  if (!segmentation || typeof segmentation !== "object") {
    return "";
  }

  const rPeaks = segmentation.r_peak_count;
  const segments = segmentation.segment_count;
  const reason = segmentation.reason;

  const parts = [];

  if (rPeaks !== undefined && rPeaks !== null) {
    parts.push(`R-peak ${rPeaks}개`);
  }

  if (segments !== undefined && segments !== null) {
    parts.push(`사용 beat ${segments}개`);
  }

  if (reason && reason !== "ok") {
    parts.push(`segment reason=${reason}`);
  }

  return parts.join(", ");
}

function formatSegmentComparisonSummary(inference = {}) {
  const queryCount = inference.embedding_count;
  const templateCount = inference.template_embedding_count;
  const segmentationText = formatSegmentationSummary(inference.segmentation);

  const parts = [];

  if (templateCount !== undefined && templateCount !== null) {
    parts.push(`템플릿 beat ${templateCount}개`);
  }

  if (queryCount !== undefined && queryCount !== null) {
    parts.push(`인증 beat ${queryCount}개`);
  }

  if (segmentationText) {
    parts.push(segmentationText);
  }

  return parts.join(", ");
}

function formatSimilarityDetail(detail = {}) {
  if (!detail || typeof detail !== "object") {
    return "";
  }

  const median = detail.top_score_median;
  const mean = detail.top_score_mean;
  const count = detail.top_score_count;
  const total = detail.score_count;

  const parts = [];

  if (median !== undefined && median !== null) {
    parts.push(`top median ${Number(median).toFixed(3)}`);
  }

  if (mean !== undefined && mean !== null) {
    parts.push(`top mean ${Number(mean).toFixed(3)}`);
  }

  if (count !== undefined && total !== undefined) {
    parts.push(`top ${count}/${total}`);
  }

  return parts.join(", ");
}

function formatDecisionReason(reason = "") {
  const text = String(reason || "").trim();

  if (!text) {
    return "";
  }

  const translated = {
    "Authenticated.": "판정 사유: 기준값 이상",
    "Rejected by similarity threshold.": "판정 사유: 유사도 기준값 미만",
    "Rejected by waveform quality gate.": "판정 사유: 파형 품질 기준 미달",
  };

  return translated[text] || `판정 사유: ${text}`;
}

function formatCollectionQualitySummary(quality = {}) {
  if (!quality || typeof quality !== "object") {
    return "";
  }

  const status = quality.quality_status;
  const reason = quality.quality_reason;
  const completeness = quality.collection_completeness;
  const missing = quality.missing_sample_estimate;
  const flatTail = quality.flat_tail_detected;
  const sanityText = formatSanityCheckSummary(quality.sanity_check);

  const parts = [];

  if (status) {
    parts.push(`수집 품질 ${status}`);
  }

  if (reason) {
    parts.push(`사유 ${reason}`);
  }

  if (completeness !== undefined && completeness !== null) {
    parts.push(`완성도 ${formatRatioPercent(completeness)}`);
  }

  if (missing !== undefined && missing !== null) {
    parts.push(`누락 ${missing}개`);
  }

  if (flatTail === true) {
    parts.push("후반부 flatline 감지");
  }

  if (sanityText) {
    parts.push(sanityText);
  }

  return parts.join(", ");
}

// 점수 게이지 렌더링
function renderScoreGauge(score = 0, decision = "") {
  const progress = $("scoreGaugeProgress");
  const value = $("matchingScore");

  if (!progress || !value) return;

  const safeScore = Math.max(0, Math.min(100, Number(score) || 0));
  const radius = 78;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - safeScore / 100);

  progress.style.strokeDasharray = `${circumference}`;

  if (decision === "unavailable") {
    progress.style.opacity = "0";
    progress.style.strokeDashoffset = `${circumference}`;
    value.textContent = "--";
    return;
  }

  progress.style.opacity = "1";
  progress.style.strokeDashoffset = `${circumference}`;

  if (decision === "rejected") {
    progress.style.stroke = "var(--red)";
  } else {
    progress.style.stroke = "url(#scoreGradient)";
  }

  requestAnimationFrame(() => {
    progress.style.strokeDashoffset = `${offset}`;
  });

  value.textContent = safeScore.toFixed(1);
}

// ECG 신호 품질 상태 갱신
function updateTrustStatus(hrv = null) {
  if (!hrv) {
    resetTrustStatus();
    return;
  }

  const status = String(hrv.quality_status || "unavailable").toLowerCase();
  const qualityLabel = hrv.quality_label || "-";
  const trustLevel = hrv.trust_level || "-";
  const message =
    hrv.quality_message || "ECG 신호 품질 상태를 확인할 수 없습니다.";

  setText("trustBadge", qualityLabel);
  setText("trustLevel", trustLevel);
  setText("trustMessage", message);

  setText("hrvHeartRate", formatBpm(hrv.heart_rate_bpm));
  setText("hrvQualityLabel", qualityLabel);
  setText("hrvValidBeat", formatBeatCount(hrv.valid_rr_count));
  setText("hrvStabilityScore", formatScore(hrv.rr_stability_score));

  setText("hrvPdfHeartRate", "-");
  setText("hrvEstimatedHeartRate", formatBpm(hrv.ecg_estimated_heart_rate_bpm));
  setText("hrvSdnn", formatMs(hrv.sdnn_ms));
  setText("hrvRmssd", formatMs(hrv.rmssd_ms));

  const trustCard = $("trustCard");
  const trustBadge = $("trustBadge");

  if (trustCard) {
    trustCard.classList.remove("stable", "caution", "warning", "unavailable");
    trustCard.classList.add(getTrustStatusClass(status));
  }

  if (trustBadge) {
    trustBadge.classList.remove(
      "waiting",
      "stable",
      "caution",
      "warning",
      "unavailable",
    );
    trustBadge.classList.add(getTrustStatusClass(status));
  }
}

// ECG 신호 품질 상태 초기화
function resetTrustStatus() {
  setText("trustBadge", "대기 중");
  setText("trustLevel", "-");
  setText("trustMessage", "ECG 데이터를 수신하면 신호 품질 상태가 표시됩니다.");

  setText("hrvHeartRate", "-");
  setText("hrvQualityLabel", "-");
  setText("hrvValidBeat", "-");
  setText("hrvStabilityScore", "-");

  setText("hrvPdfHeartRate", "-");
  setText("hrvEstimatedHeartRate", "-");
  setText("hrvSdnn", "-");
  setText("hrvRmssd", "-");

  const trustCard = $("trustCard");
  const trustBadge = $("trustBadge");

  if (trustCard) {
    trustCard.classList.remove("stable", "caution", "warning", "unavailable");
  }

  if (trustBadge) {
    trustBadge.classList.remove("stable", "caution", "warning", "unavailable");
    trustBadge.classList.add("waiting");
  }
}

// ECG 신호 품질 설명 모달
function openQualityInfoModal() {
  openModal(
    "ECG 신호 품질",
    `
    <div class="quality-info-content">
      <div class="quality-info-section">
        <strong>리듬 안정성</strong>
        <p>박동 간격이 일정한지 확인합니다. 불안정하면 인증 결과 해석에 주의가 필요합니다.</p>
      </div>

      <div class="quality-info-section">
        <strong>유효 박동 수</strong>
        <p>분석에 사용 가능한 박동 수입니다. 너무 적으면 품질 판단이 불안정할 수 있습니다.</p>
      </div>

      <div class="quality-info-section">
        <strong>RR 안정도</strong>
        <p>R-peak 간격의 안정성을 나타냅니다. 값이 낮으면 잡음이나 측정 오류 가능성이 있습니다.</p>
      </div>

      <div class="quality-info-section">
        <strong>SDNN</strong>
        <p>RR interval의 표준편차입니다. 박동 간격의 전체 변동성을 나타냅니다.</p>
      </div>

      <div class="quality-info-section">
        <strong>RMSSD</strong>
        <p>연속된 RR interval 변화량입니다. 짧은 구간의 심박 변동을 확인합니다.</p>
      </div>

      <div class="quality-info-note">
        <p>이 항목은 진단 지표가 아니라 인증 결과 해석을 위한 보조 지표입니다.</p>
      </div>
    </div>
  `,
  );
}

// 인증 기록 모달
async function openAuthLogModal() {
  if (!state.lastReport) {
    toast("조회할 사용자가 없습니다. ECG 데이터를 먼저 수신하세요.");
    return;
  }

  setLoading(true, "인증 기록 불러오는 중...");

  try {
    const response = await fetch("/api/auth-logs", {
      method: "GET",
    });

    const data = await parseJsonResponse(response);

    if (!response.ok || !data.success) {
      throw new Error(data.message || "인증 기록을 불러오지 못했습니다.");
    }

    const userName = normalizeDisplayName(
      data.user?.name || data.user?.subject_id,
    );
    const birthDate = data.user?.birth_date || "-";
    const userKey = data.user?.user_key || data.user?.subject_id || "-";
    const logs = Array.isArray(data.logs) ? data.logs : [];

    if (logs.length === 0) {
      openModal(
        "인증 기록",
        `
        <div class="modal-info-list">
          ${modalInfoRow("사용자", userName)}
          ${modalInfoRow("생년월일", birthDate)}
          ${modalInfoRow("사용자 키", userKey)}
        </div>

        <div class="modal-note">
          저장된 인증 기록이 없습니다.
        </div>
      `,
      );

      return;
    }

    const rows = logs.map(buildAuthLogRow).join("");

    openModal(
      "인증 기록",
      `
      <div class="modal-info-list log-user-summary">
        ${modalInfoRow("사용자", userName)}
        ${modalInfoRow("생년월일", birthDate)}
        ${modalInfoRow("사용자 키", userKey)}
      </div>

      <div class="log-table-wrap">
        <table class="log-table">
          <thead>
            <tr>
              <th>처리 시간</th>
              <th>구분</th>
              <th>파일명</th>
              <th>유사도</th>
              <th>점수</th>
              <th>신호 품질</th>
              <th>결과</th>
            </tr>
          </thead>
          <tbody>
            ${rows}
          </tbody>
        </table>
      </div>

      <div class="modal-note">
        인증 결과와 함께 당시 ECG 신호 품질 등급을 표시합니다.
      </div>
    `,
    );
  } catch (error) {
    console.error(error);
    toast(error.message);
  } finally {
    setLoading(false);
  }
}

// PDF 리포트 저장
function saveCurrentReportPdf() {
  if (
    !state.lastReport &&
    !state.lastTemplate &&
    !state.lastInference &&
    !state.lastHrv
  ) {
    toast("저장할 결과가 없습니다. ECG 데이터를 먼저 수신하세요.");
    return;
  }

  if (!window.jspdf || !window.jspdf.jsPDF) {
    toast("PDF 저장 라이브러리를 불러오지 못했습니다.");
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF("p", "mm", "a4");

  const report = state.lastReport || {};
  const template = state.lastTemplate || {};
  const inference = state.lastInference || {};
  const hrv = state.lastHrv || {};

  const modeLabel = getModeLabel(state.lastMode);
  const decisionLabel = getDecisionLabel(inference.decision || "-");
  const qualityLabel =
    hrv.quality_label ||
    report.quality_status ||
    report.raw_quality_label ||
    "-";

  let y = 18;

  doc.setFont("Cafe24Simplehae", "normal");
  doc.setFontSize(18);
  doc.text("ECG 인증 결과 리포트", 14, y);

  y += 8;
  doc.setFont("Cafe24Simplehae", "normal");
  doc.setFontSize(10);
  doc.text(`저장 시간: ${getNowText()}`, 14, y);

  y += 12;
  drawPdfSectionTitle(doc, "사용자 및 ECG 데이터 정보", y);
  y += 8;
  y = drawPdfRows(doc, y, [
    [
      "사용자 ID",
      report.subject_id || state.currentFileMeta?.subject_id || "-",
    ],
    ["이름", getDisplaySubjectName(report)],
    ["측정 시간", report.measured_at || "-"],
    ["평균 심박수", report.heart_rate ? `${report.heart_rate} bpm` : "-"],
    ["측정 기기", report.device || "-"],
    ["샘플링 주파수", report.sampling_rate || "-"],
    ["리드", normalizeLeadText(report.lead || "Lead I ECG")],
    ["파일명", report.filename || state.selectedFile?.name || "-"],
  ]);

  y += 6;
  drawPdfSectionTitle(doc, "인증 결과", y);
  y += 8;
  y = drawPdfRows(doc, y, [
    ["처리 모드", modeLabel],
    ["모델", template.model || inference.model || "Plain CNN1D"],
    [
      "특징 벡터",
      `${template.embedding_dim || inference.embedding_dim || 256}D ECG Embedding`,
    ],
    [
      "템플릿 beat 수",
      template.embedding_count || inference.template_embedding_count || "-",
    ],
    ["인증 beat 수", inference.embedding_count || "-"],
    ["유사도", formatPdfNumber(inference.cosine_similarity, 3)],
    ["일치 점수", formatPdfPercent(inference.matching_score)],
    [
      "기준값",
      formatPdfNumber(inference.threshold ?? template.threshold ?? 0.85, 3),
    ],
    ["판정 결과", decisionLabel],
    ["판정 사유", inference.decision_reason || "-"],
    ["분할 정보", formatSegmentComparisonSummary(inference) || "-"],
    ["상위 유사도", formatSimilarityDetail(inference.similarity_detail) || "-"],
  ]);

  y += 6;
  drawPdfSectionTitle(doc, "ECG 신호 품질 분석", y);
  y += 8;
  y = drawPdfRows(doc, y, [
    ["신호 품질 등급", qualityLabel],
    ["리듬 안정성", hrv.trust_level || "-"],
    ["품질 판정 근거", hrv.quality_message || report.quality_reason || "-"],
    [
      "수집 품질",
      formatCollectionQualitySummary(inference.collection_quality || report) ||
        "-",
    ],
    ["추정 심박수", formatNullableBpm(hrv.heart_rate_bpm)],
    ["ECG 추정 심박수", formatNullableBpm(hrv.ecg_estimated_heart_rate_bpm)],
    ["유효 박동 수", formatNullableCount(hrv.valid_rr_count)],
    ["RR 안정도", formatPdfNumber(hrv.rr_stability_score, 3)],
    ["SDNN", formatNullableMs(hrv.sdnn_ms)],
    ["RMSSD", formatNullableMs(hrv.rmssd_ms)],
  ]);

  y += 8;
  doc.setFont("Cafe24Simplehae", "normal");
  doc.setFontSize(9);
  doc.text(
    "참고: ECG 신호 품질은 인증 결과 해석을 위한 보조 지표입니다.",
    14,
    Math.min(y, 285),
  );

  const userNameForFile = sanitizeFileName(
    report.subject_id || normalizeDisplayName(report.name) || "ECG데이터",
  );
  const fileName = `ECG_인증_리포트_${userNameForFile}_${getFileTimestamp()}.pdf`;

  doc.save(fileName);
  toast("PDF 인증 리포트가 저장되었습니다.");
}

// 서버 초기화
async function resetAll() {
  try {
    await fetch("/api/reset", {
      method: "POST",
    });
  } catch (error) {
    console.warn("Server reset failed:", error);
  }

  resetUI();
}

// 인증 결과 기록 추가
function addRecord(result = {}, authLog = {}, report = {}) {
  const list = $("recordList");
  if (!list) return;

  const signature = buildRecordSignature(result, authLog, report);

  if (signature && state.recordSignatures.has(signature)) {
    return;
  }

  if (signature) {
    state.recordSignatures.add(signature);

    if (state.recordSignatures.size > 100) {
      const first = state.recordSignatures.values().next().value;
      state.recordSignatures.delete(first);
    }
  }

  const empty = list.querySelector(".empty");
  if (empty) {
    empty.remove();
  }

  const decision = result.decision || authLog.decision || "-";
  const decisionLabel = getDecisionLabel(decision);
  const comparable = isComparableInferenceResult(result);
  const score = toFiniteNumber(
    result.matching_score ?? authLog.matching_score,
    0,
  );
  const scoreText = comparable ? `${score.toFixed(1)}%` : "비교 불가";
  const createdAt = formatRecordTime(authLog.created_at) || getNowText();
  const item = document.createElement("div");

  item.className = "record-item";
  item.title = [
    formatDecisionReason(result.decision_reason),
    formatSegmentComparisonSummary(result),
    formatSimilarityDetail(result.similarity_detail),
  ]
    .filter(Boolean)
    .join(" / ");

  item.innerHTML = `
    <span class="record-dot ${getDecisionClass(decision)}"></span>
    <div>
      <strong>${escapeHtml(decisionLabel)}</strong><br />
      <span>${escapeHtml(createdAt)}</span>
    </div>
    <div class="record-score">${escapeHtml(scoreText)}</div>
  `;

  list.prepend(item);
}

function buildRecordSignature(result = {}, authLog = {}, report = {}) {
  if (authLog.id !== null && authLog.id !== undefined && authLog.id !== "") {
    return `log:${authLog.id}`;
  }

  return [
    report.subject_id ||
      result.subject_id ||
      state.currentFileMeta?.subject_id ||
      "",
    normalizeResultFilename(
      report.filename || state.currentFileMeta?.filename || "",
    ),
    report.measured_at || "",
    result.cosine_similarity ?? "",
    result.matching_score ?? "",
    result.decision || authLog.decision || "",
  ].join("|");
}

function formatRecordTime(value) {
  if (!value) return "";

  const text = String(value);

  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(text)) {
    return text.slice(0, 16).replace("T", " ");
  }

  if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}/.test(text)) {
    return text.slice(0, 16);
  }

  return text;
}

// 인증 결과 기록 초기화
function resetRecordList() {
  const list = $("recordList");
  if (!list) return;

  list.innerHTML = `
    <div class="record-item empty">
      최근 인증 없음
    </div>
  `;
}

// ECG 파형 렌더링
function renderWaveform(waveform) {
  renderWaveformComparison(waveform, null, "template");
}

// ECG 등록/인증 비교 파형 렌더링
function renderWaveformComparison(
  templateWaveform,
  verifyWaveform,
  mode = "template",
) {
  clearWaveform();

  const hasTemplate =
    templateWaveform &&
    Array.isArray(templateWaveform.amplitude) &&
    templateWaveform.amplitude.length > 0;

  const hasVerify =
    verifyWaveform &&
    Array.isArray(verifyWaveform.amplitude) &&
    verifyWaveform.amplitude.length > 0;

  if (!hasTemplate && !hasVerify) {
    return;
  }

  const empty = $("ecgEmpty");
  if (empty) {
    empty.style.display = "none";
  }

  const totalPoints = 900;

  if (mode === "verify" && hasTemplate && hasVerify) {
    const templateSignal = resampleArray(
      preprocessDisplayWaveform(templateWaveform.amplitude),
      totalPoints,
    );

    const verifySignal = resampleArray(
      preprocessDisplayWaveform(verifyWaveform.amplitude),
      totalPoints,
    );

    const diffRange = findMaxDifferenceWindow(templateSignal, verifySignal, {
      totalSeconds: 30,
      windowSeconds: 3,
    });

    showDifferenceWindow(diffRange, totalPoints);

    drawWaveformPath("ecgTrailPath", templateSignal, totalPoints, {
      stroke: "rgba(255, 107, 53, 0.38)",
      strokeWidth: 2.0,
    });

    startWaveformAnimation(verifyWaveform, {
      pathId: "ecgPath",
      stroke: "#1aa6c9",
      strokeWidth: 2.3,
      totalPoints,
    });

    return;
  }

  hideDifferenceWindow();

  if (hasTemplate) {
    startWaveformAnimation(templateWaveform, {
      pathId: "ecgPath",
      stroke: "var(--orange)",
      strokeWidth: 2.3,
      totalPoints,
    });

    return;
  }

  if (hasVerify) {
    startWaveformAnimation(verifyWaveform, {
      pathId: "ecgPath",
      stroke: "#1aa6c9",
      strokeWidth: 2.3,
      totalPoints,
    });
  }
}

// ECG 파형 초기화
function clearWaveform() {
  stopWaveformAnimation();
  hideDifferenceWindow();

  const ecgPath = $("ecgPath");
  const trailPath = $("ecgTrailPath");
  const empty = $("ecgEmpty");

  if (ecgPath) {
    ecgPath.setAttribute("d", "");
    ecgPath.removeAttribute("style");
  }

  if (trailPath) {
    trailPath.setAttribute("d", "");
    trailPath.removeAttribute("style");
  }

  if (empty) {
    empty.style.display = "grid";
  }
}

// ECG 애니메이션
function startWaveformAnimation(waveform, options = {}) {
  stopWaveformAnimation();

  if (!waveform || !Array.isArray(waveform.amplitude)) return;

  const source = preprocessDisplayWaveform(waveform.amplitude);
  if (source.length === 0) return;

  const totalPoints = options.totalPoints || 900;
  const displaySource = resampleArray(source, totalPoints);

  const pathId = options.pathId || "ecgPath";
  const stroke = options.stroke || "var(--orange)";
  const strokeWidth = options.strokeWidth || 2.2;

  let cursor = 1;
  let pauseCount = 0;

  const step = 2;
  const interval = 45;
  const pauseFrames = 35;

  state.animationTimer = setInterval(() => {
    if (cursor <= totalPoints) {
      const segment = displaySource.slice(0, cursor);

      drawWaveformPath(pathId, segment, totalPoints, {
        stroke,
        strokeWidth,
      });

      cursor += step;
      return;
    }

    if (pauseCount < pauseFrames) {
      pauseCount += 1;
      return;
    }

    cursor = 1;
    pauseCount = 0;

    const pathElement = $(pathId);
    if (pathElement) {
      pathElement.setAttribute("d", "");
    }
  }, interval);
}

function stopWaveformAnimation() {
  if (!state.animationTimer) return;

  clearInterval(state.animationTimer);
  state.animationTimer = null;
}

function drawWaveformPath(pathId, segment, totalPoints = 900, options = {}) {
  const pathElement = $(pathId);

  if (!pathElement) return;

  if (!segment || segment.length === 0) {
    pathElement.setAttribute("d", "");
    return;
  }

  const width = 1000;
  const height = 320;
  const midY = height / 2;
  const scaleY = 78;
  const stepX = width / (totalPoints - 1);

  let path = "";

  segment.forEach((value, index) => {
    const x = index * stepX;
    const y = midY - value * scaleY;

    path +=
      index === 0
        ? `M ${x.toFixed(2)} ${y.toFixed(2)}`
        : ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  });

  pathElement.setAttribute("d", path);
  pathElement.style.fill = "none";
  pathElement.style.stroke = options.stroke || "var(--orange)";
  pathElement.style.strokeWidth = String(options.strokeWidth || 2.1);
  pathElement.style.strokeLinejoin = "round";
  pathElement.style.strokeLinecap = "round";
}

function findMaxDifferenceWindow(
  templateSignal = [],
  verifySignal = [],
  options = {},
) {
  const totalSeconds = options.totalSeconds || 30;
  const windowSeconds = options.windowSeconds || 3;

  const length = Math.min(templateSignal.length, verifySignal.length);

  if (length < 2) {
    return null;
  }

  const pointsPerSecond = length / totalSeconds;
  const windowSize = Math.max(10, Math.round(pointsPerSecond * windowSeconds));

  if (length <= windowSize) {
    return {
      startIndex: 0,
      endIndex: length - 1,
      startSecond: 0,
      endSecond: totalSeconds,
      meanDifference: calculateMeanAbsoluteDifference(
        templateSignal,
        verifySignal,
        0,
        length,
      ),
    };
  }

  let bestStart = 0;
  let bestScore = -1;

  for (let start = 0; start <= length - windowSize; start += 1) {
    const end = start + windowSize;
    const score = calculateMeanAbsoluteDifference(
      templateSignal,
      verifySignal,
      start,
      end,
    );

    if (score > bestScore) {
      bestScore = score;
      bestStart = start;
    }
  }

  const bestEnd = bestStart + windowSize;
  const startSecond = (bestStart / length) * totalSeconds;
  const endSecond = (bestEnd / length) * totalSeconds;

  return {
    startIndex: bestStart,
    endIndex: bestEnd,
    startSecond,
    endSecond,
    meanDifference: bestScore,
  };
}

function calculateMeanAbsoluteDifference(
  templateSignal,
  verifySignal,
  start,
  end,
) {
  let sum = 0;
  let count = 0;

  for (let index = start; index < end; index += 1) {
    const templateValue = Number(templateSignal[index]);
    const verifyValue = Number(verifySignal[index]);

    if (!Number.isFinite(templateValue) || !Number.isFinite(verifyValue)) {
      continue;
    }

    sum += Math.abs(templateValue - verifyValue);
    count += 1;
  }

  return count > 0 ? sum / count : 0;
}

function showDifferenceWindow(diffRange, totalPoints = 900) {
  const diffWindow = $("ecgDiffWindow");
  const diffLabel = $("ecgDiffLabel");

  if (!diffRange || !diffWindow || !diffLabel) {
    return;
  }

  const startRatio = diffRange.startIndex / totalPoints;
  const endRatio = diffRange.endIndex / totalPoints;
  const widthRatio = Math.max(0.02, endRatio - startRatio);

  diffWindow.style.display = "block";
  diffWindow.style.left = `${startRatio * 100}%`;
  diffWindow.style.width = `${widthRatio * 100}%`;

  diffLabel.style.display = "inline-flex";
  diffLabel.textContent = `파형 차이 최대 구간: ${diffRange.startSecond.toFixed(1)}s ~ ${diffRange.endSecond.toFixed(1)}s`;
}

function hideDifferenceWindow() {
  const diffWindow = $("ecgDiffWindow");
  const diffLabel = $("ecgDiffLabel");

  if (diffWindow) {
    diffWindow.style.display = "none";
    diffWindow.style.left = "0";
    diffWindow.style.width = "0";
  }

  if (diffLabel) {
    diffLabel.style.display = "none";
    diffLabel.textContent = "파형 차이 최대 구간: -";
  }
}

function preprocessDisplayWaveform(points = []) {
  const values = points.map(Number).filter(Number.isFinite);

  if (values.length === 0) {
    return [];
  }

  const sorted = [...values].sort((a, b) => a - b);
  const pick = (ratio) => {
    const index = Math.min(
      sorted.length - 1,
      Math.max(0, Math.round((sorted.length - 1) * ratio)),
    );
    return sorted[index];
  };

  const median = pick(0.5);
  const p01 = pick(0.01);
  const p99 = pick(0.99);
  const absScale = Math.max(
    Math.abs(p99 - median),
    Math.abs(median - p01),
    1e-6,
  );

  return values.map((value) => {
    const normalized = (value - median) / absScale;
    return Math.max(-1.25, Math.min(1.25, normalized));
  });
}

function resampleArray(values = [], targetLength) {
  if (values.length === 0) return [];
  if (values.length === targetLength) return values;

  const result = [];
  const oldLength = values.length;

  for (let i = 0; i < targetLength; i += 1) {
    const position = (i * (oldLength - 1)) / (targetLength - 1);
    const left = Math.floor(position);
    const right = Math.min(left + 1, oldLength - 1);
    const ratio = position - left;

    result.push(values[left] * (1 - ratio) + values[right] * ratio);
  }

  return result;
}

// 화면 초기화
function resetUI() {
  resetState();

  if (rawInput) rawInput.value = "";
  setText("fileName", "선택된 파일 없음");

  updateReportInfo({
    name: "-",
    measured_at: "-",
    heart_rate: null,
    device: "-",
    sampling_rate: "-",
    lead: "Lead I ECG",
  });

  clearWaveform();
  clearAuthResult();
  resetTrustStatus();
  resetTemplateStatus();
  resetRecordList();
  resetSteps();

  state.currentSubject = null;
  updateCurrentSubjectUI(null);

  setStatus("Ready");
  toast("등록 템플릿과 화면 상태가 초기화되었습니다.");
}

function resetState() {
  state.selectedFile = null;
  state.uploaded = false;
  state.extracted = false;
  state.templateRegistered = false;
  state.verified = false;
  state.waveformData = null;
  state.templateWaveformData = null;
  state.verifyWaveformData = null;
  state.lastReport = null;
  state.lastTemplate = null;
  state.lastInference = null;
  state.lastHrv = null;
  state.lastMode = null;
  state.latestResultSignature = null;
  state.currentSubject = null;
  state.currentFileMeta = null;
  state.processedResultSignatures = new Set();
  state.recordSignatures = new Set();
}

function clearAuthResult() {
  setText("matchingScore", "--");
  setText("similarityValue", "-");
  setText("thresholdValue", "0.850");
  setText("decisionValue", "-");
  setText("decisionText", "대기 중");
  setText("decisionSubText", "ECG 데이터를 수신하면 결과가 표시됩니다.");
  setText("modeValue", "-");

  const decisionText = $("decisionText");
  if (decisionText) {
    decisionText.classList.remove("success", "fail");
  }

  renderScoreGauge(0, "");
}

function resetTemplateStatus() {
  setText("templateStatus", "등록된 템플릿 없음");
  setText("templateMain", "등록된 템플릿 없음");
  setText("templateDesc", "첫 번째 ECG 데이터가 기준 템플릿으로 저장됩니다.");

  const templateStatus = $("templateStatus");
  if (templateStatus) {
    templateStatus.classList.remove("registered");
  }

  const templatePanel = document.querySelector(".template-status");
  if (templatePanel) {
    templatePanel.classList.remove("active-template", "verify-mode");
  }

  const templateDot = $("templateDot");
  if (templateDot) {
    templateDot.classList.remove("on");
  }

  const templateMain = $("templateMain");
  if (templateMain) {
    templateMain.classList.remove("success-text", "info-text");
  }
}

// 템플릿 상태 표시
function setTemplateStatus(mode, content) {
  setText("templateStatus", "등록 템플릿 활성화");

  const templateStatus = $("templateStatus");
  const templatePanel = document.querySelector(".template-status");
  const templateDot = $("templateDot");
  const templateMain = $("templateMain");

  if (templateStatus) {
    templateStatus.classList.add("registered");
  }

  if (templateDot) {
    templateDot.classList.add("on");
  }

  if (templatePanel) {
    templatePanel.classList.remove("active-template", "verify-mode");
    templatePanel.classList.add(
      mode === "verify" ? "verify-mode" : "active-template",
    );
  }

  if (templateMain) {
    templateMain.textContent = content.main;
    templateMain.classList.remove("success-text", "info-text");
    templateMain.classList.add(
      mode === "verify" ? "info-text" : "success-text",
    );
  }

  setText("templateDesc", content.desc);
}

// 단계 표시
function markStep(id, status) {
  const step = $(id);
  if (!step) return;

  step.classList.remove("active", "done");

  if (status === "done") {
    step.classList.add("done");
  }

  if (status === "active") {
    step.classList.add("active");
  }
}

function resetStep(id) {
  const step = $(id);
  if (!step) return;

  step.classList.remove("active", "done");
}

function resetSteps() {
  ["stepUpload", "stepExtract", "stepRegister", "stepVerify"].forEach(
    resetStep,
  );
  markStep("stepUpload", "active");
}

// 로딩 표시
function setLoading(isLoading, message = "Processing...") {
  document.querySelectorAll(".action-btn").forEach((button) => {
    button.disabled = isLoading;
  });

  if (isLoading) {
    setStatus(message);
    document.body.classList.add("loading");
  } else {
    document.body.classList.remove("loading");
  }
}

// 상태 표시
function setStatus(text) {
  setText("systemStatus", text);
}

// 토스트
function toast(message) {
  const toastEl = $("toast");
  if (!toastEl) return;

  toastEl.textContent = message;
  toastEl.classList.add("show");

  setTimeout(() => {
    toastEl.classList.remove("show");
  }, 8000);
}

// 모달
function openModal(title, html) {
  const backdrop = $("modalBackdrop");
  const modalTitle = $("modalTitle");
  const modalBody = $("modalBody");

  if (!backdrop || !modalTitle || !modalBody) {
    toast("모달 영역을 찾을 수 없습니다.");
    return;
  }

  modalTitle.textContent = title;
  modalBody.innerHTML = html;
  backdrop.classList.add("show");
}

function closeModal() {
  if (!modalBackdrop) return;

  const modalBox = document.querySelector(".modal-box");
  if (modalBox) {
    modalBox.classList.remove("subject-modal-box");
  }

  modalBackdrop.classList.remove("show");
}

function modalInfoRow(label, value) {
  return `
    <div class="modal-info-row">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function buildAuthLogRow(log) {
  const modeText = log.mode === "register" ? "등록" : "인증";
  const decisionText = log.decision || "-";
  const decisionLabel = getDecisionLabel(decisionText);
  const similarityText =
    log.cosine_similarity !== null && log.cosine_similarity !== undefined
      ? Number(log.cosine_similarity).toFixed(3)
      : "-";
  const scoreText =
    log.matching_score !== null && log.matching_score !== undefined
      ? `${Number(log.matching_score).toFixed(1)}%`
      : "-";
  const qualityText = extractLogQualityLabel(log);
  const decisionClass = getDecisionClass(decisionText);
  const qualityClass = getQualityClassFromLabel(qualityText);

  return `
    <tr>
      <td>${escapeHtml(log.created_at || "-")}</td>
      <td>${escapeHtml(modeText)}</td>
      <td class="filename-cell">${escapeHtml(log.filename || "-")}</td>
      <td>${escapeHtml(similarityText)}</td>
      <td>${escapeHtml(scoreText)}</td>
      <td><span class="log-quality ${qualityClass}">${escapeHtml(qualityText)}</span></td>
      <td><span class="log-decision ${decisionClass}">${escapeHtml(decisionLabel)}</span></td>
    </tr>
  `;
}

// PDF 유틸
function drawPdfSectionTitle(doc, title, y) {
  doc.setFont("Cafe24Simplehae", "normal");
  doc.setFontSize(12);
  doc.text(title, 14, y);
  doc.setDrawColor(40, 40, 40);
  doc.line(14, y + 2, 196, y + 2);
}

function drawPdfRows(doc, startY, rows) {
  let y = startY;

  doc.setFontSize(10);

  rows.forEach(([label, value]) => {
    const safeValue = String(value ?? "-");

    doc.setFont("Cafe24Simplehae", "normal");
    doc.text(String(label), 14, y);

    doc.setFont("Cafe24Simplehae", "normal");
    const splitValue = doc.splitTextToSize(safeValue, 120);
    doc.text(splitValue, 70, y);

    y += Math.max(7, splitValue.length * 5);
  });

  return y;
}

// 포맷 유틸
function formatBpm(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} bpm` : "-";
}

function formatMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} ms` : "-";
}

function formatBeatCount(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number)}개` : "-";
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "-";
}

function formatPdfNumber(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}

function formatPdfPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "-";
}

function formatNullableBpm(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} bpm` : "-";
}

function formatNullableMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} ms` : "-";
}

function formatNullableCount(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number)}` : "-";
}

function formatRatioPercent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "-";
  }

  return `${(number * 100).toFixed(1)}%`;
}

function translateCollectionQualityStatus(status) {
  const normalized = String(status || "").toLowerCase();

  if (normalized === "stable") return "양호";
  if (normalized === "caution") return "주의";
  if (normalized === "warning") return "낮음";
  if (normalized === "unavailable") return "분석 불가";

  return status || "-";
}

function getTrustStatusClass(status) {
  const normalized = String(status || "").toLowerCase();

  if (normalized === "stable") return "stable";
  if (normalized === "caution") return "caution";
  if (normalized === "warning") return "warning";
  if (normalized === "unavailable") return "unavailable";

  return "unavailable";
}

function getQualityClassFromLabel(label) {
  const text = String(label || "");

  if (text.includes("양호") || text.includes("높음") || text.includes("확인")) {
    return "stable";
  }

  if (text.includes("ECG 데이터")) {
    return "stable";
  }

  if (text.includes("주의") || text.includes("중간")) {
    return "caution";
  }

  if (
    text.includes("낮음") ||
    text.includes("위험") ||
    text.includes("불안정") ||
    text.includes("재측정") ||
    text.includes("추출 품질 낮음")
  ) {
    return "warning";
  }

  return "neutral";
}

function extractLogQualityLabel(log = {}) {
  return (
    log.quality_label ||
    log.signal_quality ||
    log.hrv_quality_label ||
    log.hrv?.quality_label ||
    log.hrv?.trust_level ||
    "-"
  );
}

function getModeLabel(mode) {
  const normalized = String(mode || "").toLowerCase();

  if (normalized === "register") return "템플릿 등록";
  if (normalized === "verify") return "인증 비교";
  if (normalized === "raw_ecg") return "ECG 데이터 입력";

  return "-";
}

// 기본 유틸
function setText(id, value) {
  const element = $(id);
  if (!element) return;

  element.textContent = value;
}

function normalizeName(value) {
  if (value === null || value === undefined) {
    return "";
  }

  return String(value).replace(/\s+/g, "").trim();
}

function normalizeDisplayName(value) {
  const name = normalizeName(value);
  const lowerName = name.toLowerCase();

  if (
    !name ||
    name === "-" ||
    lowerName === "ecgjsonsample" ||
    lowerName === "ecgjson" ||
    lowerName === "rawsample" ||
    lowerName === "rawecgsample"
  ) {
    return "ECG 데이터";
  }

  return name;
}

function normalizeLeadText(value) {
  return String(value)
    .replace("Lead I-like ECG", "Lead I ECG")
    .replace("Lead I-like", "Lead I ECG")
    .replace("lead I ECG", "Lead I ECG");
}

function normalizeBirthDateInput(value) {
  return String(value || "")
    .replace(/\D/g, "")
    .slice(0, 8);
}

function toFiniteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function getNowText() {
  const now = new Date();

  return (
    `${now.getFullYear()}.` +
    `${String(now.getMonth() + 1).padStart(2, "0")}.` +
    `${String(now.getDate()).padStart(2, "0")} ` +
    `${String(now.getHours()).padStart(2, "0")}:` +
    `${String(now.getMinutes()).padStart(2, "0")}`
  );
}

function getFileTimestamp() {
  const now = new Date();

  return (
    `${now.getFullYear()}` +
    `${String(now.getMonth() + 1).padStart(2, "0")}` +
    `${String(now.getDate()).padStart(2, "0")}_` +
    `${String(now.getHours()).padStart(2, "0")}` +
    `${String(now.getMinutes()).padStart(2, "0")}` +
    `${String(now.getSeconds()).padStart(2, "0")}`
  );
}

function sanitizeFileName(value) {
  return String(value || "")
    .replace(/[\\/:*?"<>|]/g, "")
    .replace(/\s+/g, "_")
    .trim();
}

function getDecisionLabel(decision) {
  const normalized = String(decision).toLowerCase();

  if (normalized === "authenticated") {
    return "인증 성공";
  }

  if (normalized === "rejected") {
    return "인증 실패";
  }

  if (normalized === "registered") {
    return "등록 완료";
  }

  if (normalized === "verified") {
    return "인증 처리 완료";
  }

  return decision || "-";
}

function getDecisionClass(decision) {
  const normalized = String(decision).toLowerCase();

  if (normalized === "authenticated") {
    return "success";
  }

  if (normalized === "rejected") {
    return "fail";
  }

  return "neutral";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeUserErrorMessage(message = "") {
  const text = String(message || "");

  if (
    text.includes("템플릿 등록 조건을 만족하지 못했습니다") ||
    text.includes("등록 템플릿으로 사용할 수 없는 ECG") ||
    text.includes("flat_tail_detected") ||
    text.includes("registration_eligible")
  ) {
    return "등록 템플릿으로 사용할 수 없는 ECG입니다.";
  }

  if (
    text.includes("Rejected by collection quality gate") ||
    text.includes("verification_eligible")
  ) {
    return "인증 비교에 사용할 수 없는 ECG입니다.";
  }

  if (
    text.includes("subject_id를 확인할 수 없습니다") ||
    text.includes("S001_T01")
  ) {
    return "사용자 라벨을 확인할 수 없습니다.";
  }

  if (
    text.includes("JSON") ||
    text.includes("서버 응답을 JSON으로 해석하지 못했습니다")
  ) {
    return "ECG JSON 파일을 처리할 수 없습니다.";
  }

  return text || "ECG 데이터 처리 중 오류가 발생했습니다.";
}

async function parseJsonResponse(response) {
  try {
    return await response.json();
  } catch (error) {
    throw new Error("서버 응답을 JSON으로 해석하지 못했습니다.");
  }
}
