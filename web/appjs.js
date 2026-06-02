document.addEventListener("DOMContentLoaded", () => {
  // Mark page loaded for hero animations
  document.body.classList.add("page-loaded");

  // --- 1. WebSocket demo logic ---
  const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/query";

  const btnAsk = document.getElementById("btnAsk");
  const inputQuestion = document.getElementById("userQuestion");
  const inputTenant = document.getElementById("tenantId");
  const inputBranch = document.getElementById("branchId");
  const inputSession = document.getElementById("sessionId");
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
    if (!metaLine) return;
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
        btnAsk.textContent = "Hỏi trợ lý";
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
        btnAsk.textContent = "Hỏi câu khác";
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

  function handleError(msg) {
    if (answerContainer) {
      answerContainer.innerHTML = `<span style="color: #fed7aa;">⚠️ ${msg}</span>`;
    }
    setUIState("error");
    if (socket) socket.close();
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
      // Remove placeholder on first chunk
      const placeholder = answerContainer.querySelector(".demo-output__placeholder");
      if (placeholder) {
        placeholder.remove();
      }
      answerContainer.textContent += message.text || "";
      answerContainer.scrollTop = answerContainer.scrollHeight;
      return;
    }

    if (message.type === "end") {
      setUIState("complete");
      if (socket) socket.close();
      return;
    }

    if (message.type === "error") {
      handleError(message.text || "Đã xảy ra lỗi không xác định.");
    }
  }

  if (btnAsk) {
    btnAsk.addEventListener("click", (e) => {
      e.preventDefault();
      if (!inputQuestion || !inputTenant) return;

      const question = inputQuestion.value.trim();
      const tenantId = inputTenant.value.trim() || null;
      const branchId = inputBranch ? inputBranch.value.trim() || null : null;
      const sessionId = inputSession ? inputSession.value.trim() || null : null;

      if (!question) {
        alert("Vui lòng nhập câu hỏi để trải nghiệm demo.");
        return;
      }

      if (answerContainer) {
        answerContainer.textContent = "";
      }
      if (sourcesContainer) {
        sourcesContainer.innerHTML = "";
      }
      if (metaLine) metaLine.textContent = "";
      lastTraceId = null;
      lastTenant = tenantId;
      resetFeedbackUI();
      setUIState("streaming");

      try {
        socket = new WebSocket(WS_URL);

        socket.onopen = () => {
          const sid = sessionId || getOrCreateSessionId(tenantId || "tenant");
          if (inputSession && !sessionId) inputSession.value = sid;
          const payload = {
            question: question,
            tenant_id: tenantId,
            branch_id: branchId,
            session_id: sid,
            history: [],
          };
          socket.send(JSON.stringify(payload));
        };

        socket.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
          } catch (err) {
            console.error("JSON parse error:", err);
          }
        };

        socket.onerror = (err) => {
          console.error("WebSocket error:", err);
          handleError("Lỗi kết nối tới server demo.");
        };

        socket.onclose = () => {
          if (isStreaming) {
            setUIState("idle");
          }
        };
      } catch (err) {
        handleError("Không thể khởi tạo kết nối WebSocket.");
      }
    });
  }

  async function submitFeedback(rating) {
    if (!lastTraceId) return;
    if (feedbackStatus) feedbackStatus.textContent = "Đang gửi…";
    if (btnUp) btnUp.disabled = true;
    if (btnDown) btnDown.disabled = true;
    try {
      const resp = await fetch("/admin/api/feedback", {
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

  applyBranchToggle();

  // --- 2. Smooth scroll for nav links ---
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      const targetId = link.getAttribute("href");
      if (!targetId || targetId === "#") return;
      const target = document.querySelector(targetId);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  // --- 3. Scroll-based reveal animations ---
  const revealEls = document.querySelectorAll(".js-reveal");
  if ("IntersectionObserver" in window && revealEls.length > 0) {
    const revealObserver = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.18 }
    );
    revealEls.forEach((el) => revealObserver.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("is-visible"));
  }
});
