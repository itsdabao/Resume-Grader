document.addEventListener("DOMContentLoaded", () => {
  const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/query";

  const btnAsk = document.getElementById("btnAsk");
  const btnClear = document.getElementById("btnClear");
  const inputQuestion = document.getElementById("userQuestion");
  const inputTenant = document.getElementById("tenantId");
  const inputBranch = document.getElementById("branchId");
  const inputSession = document.getElementById("sessionId");
  const inputUserId = document.getElementById("userId");

  const answerContainer = document.getElementById("answerContainer");
  const sourcesContainer = document.getElementById("sourcesContainer");
  const statusLabel = document.getElementById("statusLabel");
  const metaLine = document.getElementById("metaLine");
  const btnUp = document.getElementById("btnUp");
  const btnDown = document.getElementById("btnDown");
  const feedbackStatus = document.getElementById("feedbackStatus");

  let socket = null;
  let isStreaming = false;
  let lastTraceId = null;
  let lastTenant = null;

  async function applyBranchToggle() {
    try {
      const resp = await fetch("/public/config", { cache: "no-store" });
      if (!resp.ok) return;
      const cfg = await resp.json();
      const enabled = !!(cfg && cfg.enable_branch_filter);
      if (inputBranch) {
        inputBranch.disabled = !enabled;
        if (!enabled) inputBranch.value = "";
        const label = document.querySelector('label[for="branchId"]');
        if (label) label.textContent = enabled ? "Branch ID (tuỳ chọn)" : "Branch ID (đang tắt)";
      }
    } catch {}
  }

  function safeUUID() {
    try {
      if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
    } catch (e) {}
    return "sess_" + Math.random().toString(16).slice(2) + "_" + Date.now().toString(16);
  }

  function getOrCreateSessionId(tenantId) {
    const key = "novarag.session_id." + String(tenantId || "default");
    const existing = localStorage.getItem(key);
    if (existing) return existing;
    const sid = `${tenantId || "tenant"}:web:${safeUUID().slice(0, 8)}`;
    localStorage.setItem(key, sid);
    return sid;
  }

  function setMeta({ traceId, route }) {
    lastTraceId = traceId || null;
    const parts = [];
    if (route) parts.push(`<span class="demo-meta-pill">route: ${String(route)}</span>`);
    if (traceId) {
      parts.push(`<span class="demo-meta-pill">trace: ${String(traceId).slice(0, 12)}…</span>`);
      parts.push(`<button class="demo-copy" type="button" data-copy="${String(traceId)}">Copy trace</button>`);
    }
    metaLine.innerHTML = parts.join(" ");
    const copyBtn = metaLine.querySelector(".demo-copy");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        const t = copyBtn.getAttribute("data-copy");
        try {
          await navigator.clipboard.writeText(t || "");
          copyBtn.textContent = "Copied";
          setTimeout(() => (copyBtn.textContent = "Copy trace"), 900);
        } catch (e) {
          copyBtn.textContent = "Copy failed";
          setTimeout(() => (copyBtn.textContent = "Copy trace"), 900);
        }
      });
    }
  }

  function resetFeedbackUI() {
    if (feedbackStatus) feedbackStatus.textContent = "";
    if (btnUp) btnUp.disabled = true;
    if (btnDown) btnDown.disabled = true;
  }

  function enableFeedbackUI() {
    if (btnUp) btnUp.disabled = !lastTraceId;
    if (btnDown) btnDown.disabled = !lastTraceId;
  }

  function setUIState(state) {
    if (!statusLabel || !btnAsk) return;
    statusLabel.className = "demo-panel__status-label";

    switch (state) {
      case "idle":
        btnAsk.disabled = false;
        btnAsk.textContent = "Gửi";
        statusLabel.textContent = "Sẵn sàng";
        statusLabel.classList.add("status-idle");
        isStreaming = false;
        break;
      case "streaming":
        btnAsk.disabled = true;
        btnAsk.textContent = "Đang xử lý…";
        statusLabel.textContent = "Đang trả lời…";
        statusLabel.classList.add("status-streaming");
        isStreaming = true;
        break;
      case "complete":
        btnAsk.disabled = false;
        btnAsk.textContent = "Gửi câu khác";
        statusLabel.textContent = "Hoàn tất";
        statusLabel.classList.add("status-complete");
        isStreaming = false;
        enableFeedbackUI();
        break;
      case "error":
        btnAsk.disabled = false;
        btnAsk.textContent = "Thử lại";
        statusLabel.textContent = "Lỗi";
        statusLabel.classList.add("status-error");
        isStreaming = false;
        break;
      default:
        break;
    }
  }

  function resetOutput() {
    if (answerContainer) {
      answerContainer.textContent = "";
      const p = document.createElement("p");
      p.className = "demo-output__placeholder";
      p.textContent = "Câu trả lời sẽ xuất hiện tại đây…";
      answerContainer.appendChild(p);
    }
    if (sourcesContainer) sourcesContainer.innerHTML = "";
    if (metaLine) metaLine.textContent = "";
    lastTraceId = null;
    resetFeedbackUI();
  }

  function handleError(msg) {
    if (answerContainer) {
      answerContainer.innerHTML = `<span style="color: #fed7aa;">⚠️ ${msg}</span>`;
    }
    setUIState("error");
    try {
      if (socket) socket.close();
    } catch {}
  }

  function handleServerMessage(message) {
    if (!answerContainer || !sourcesContainer) return;

    if (message.type === "meta") {
      sourcesContainer.innerHTML = "";
      if (Array.isArray(message.sources)) {
        message.sources.forEach((s) => {
          const badge = document.createElement("span");
          badge.className = "demo-source-badge";
          badge.textContent = String(s);
          sourcesContainer.appendChild(badge);
        });
      }
      lastTenant = inputTenant ? inputTenant.value.trim() : null;
      setMeta({ traceId: message.trace_id, route: message.route });
      return;
    }

    if (message.type === "chunk") {
      const placeholder = answerContainer.querySelector(".demo-output__placeholder");
      if (placeholder) placeholder.remove();
      answerContainer.textContent += message.text || "";
      answerContainer.scrollTop = answerContainer.scrollHeight;
      return;
    }

    if (message.type === "end") {
      setUIState("complete");
      try {
        if (socket) socket.close();
      } catch {}
      return;
    }

    if (message.type === "error") {
      handleError(message.message || message.text || "Đã xảy ra lỗi không xác định.");
    }
  }

  async function submitFeedback(rating) {
    if (!lastTraceId) return;
    if (feedbackStatus) feedbackStatus.textContent = "Đang gửi…";
    if (btnUp) btnUp.disabled = true;
    if (btnDown) btnDown.disabled = true;
    try {
      const resp = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trace_id: lastTraceId,
          tenant_id: lastTenant,
          rating: rating,
          comment: null,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      if (feedbackStatus) feedbackStatus.textContent = "Đã ghi nhận. Cảm ơn bạn!";
    } catch (e) {
      if (feedbackStatus) feedbackStatus.textContent = "Gửi feedback thất bại.";
    }
  }

  if (btnUp) btnUp.addEventListener("click", () => submitFeedback(1));
  if (btnDown) btnDown.addEventListener("click", () => submitFeedback(-1));

  if (btnClear) {
    btnClear.addEventListener("click", () => {
      resetOutput();
      if (inputQuestion) inputQuestion.value = "";
      if (inputQuestion) inputQuestion.focus();
    });
  }

  if (btnAsk) {
    btnAsk.addEventListener("click", (e) => {
      e.preventDefault();
      const question = (inputQuestion?.value || "").trim();
      const tenantId = (inputTenant?.value || "").trim() || null;
      const branchId = (inputBranch?.value || "").trim() || null;
      const sessionId = (inputSession?.value || "").trim() || null;
      const userId = (inputUserId?.value || "").trim() || null;

      if (!question) {
        alert("Vui lòng nhập câu hỏi.");
        return;
      }

      resetOutput();
      lastTenant = tenantId;
      setUIState("streaming");

      try {
        socket = new WebSocket(WS_URL);
        socket.onopen = () => {
          const sid = sessionId || getOrCreateSessionId(tenantId || "tenant");
          if (inputSession && !sessionId) inputSession.value = sid;
          socket.send(
            JSON.stringify({
              question: question,
              tenant_id: tenantId,
              branch_id: branchId,
              session_id: sid,
              user_id: userId,
              history: [],
            })
          );
        };
        socket.onmessage = (event) => {
          try {
            handleServerMessage(JSON.parse(event.data));
          } catch (err) {
            console.error("JSON parse error:", err);
          }
        };
        socket.onerror = (err) => {
          console.error("WebSocket error:", err);
          handleError("Lỗi kết nối tới server.");
        };
        socket.onclose = () => {
          if (isStreaming) setUIState("idle");
        };
      } catch (err) {
        handleError("Không thể khởi tạo kết nối WebSocket.");
      }
    });
  }

  setUIState("idle");
  resetFeedbackUI();
  applyBranchToggle();
});
