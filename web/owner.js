function qs(sel) {
  return document.querySelector(sel);
}

function fmtTs(ts) {
  const x = Number(ts || 0) * 1000;
  if (!x) return "";
  return new Date(x).toLocaleString();
}

function maskPhone(s) {
  const raw = String(s || "").replace(/\s+/g, "");
  if (raw.length < 6) return raw;
  return raw.slice(0, 2) + "****" + raw.slice(-3);
}

function buildQuery(params) {
  const qp = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    const vv = String(v).trim();
    if (!vv) return;
    qp.set(k, vv);
  });
  const s = qp.toString();
  return s ? "?" + s : "";
}

async function apiGet(path) {
  const resp = await fetch(path, { credentials: "include" });
  if (resp.status === 401) throw Object.assign(new Error("unauthorized"), { status: 401 });
  if (!resp.ok) {
    let msg = "";
    try {
      const j = await resp.json();
      msg = j && j.detail ? String(j.detail) : JSON.stringify(j);
    } catch {
      msg = await resp.text();
    }
    throw new Error(msg || `HTTP ${resp.status}`);
  }
  return await resp.json();
}

async function apiPost(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (resp.status === 401) throw Object.assign(new Error("unauthorized"), { status: 401 });
  if (!resp.ok) {
    let msg = "";
    try {
      const j = await resp.json();
      msg = j && j.detail ? String(j.detail) : JSON.stringify(j);
    } catch {
      msg = await resp.text();
    }
    throw new Error(msg || `HTTP ${resp.status}`);
  }
  return await resp.json();
}

function setTab(name) {
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("tab--active", b.dataset.tab === name);
  });
  ["metrics", "logs", "handoffs"].forEach((t) => {
    const el = qs("#tab_" + t);
    el.style.display = t === name ? "" : "none";
  });
}

function showLogin(errMsg) {
  qs("#loginCard").style.display = "";
  qs("#appCard").style.display = "none";
  qs("#btnLogout").style.display = "none";
  qs("#loginErr").textContent = errMsg || "";
}

function showApp() {
  qs("#loginCard").style.display = "none";
  qs("#appCard").style.display = "";
  qs("#btnLogout").style.display = "";
  qs("#loginErr").textContent = "";
}

async function loadTenants() {
  const sel = qs("#fTenant");
  sel.innerHTML = '<option value="">(all)</option>';
  const data = await apiGet("/owner/api/tenants");
  (data.tenants || []).forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  });
}

async function loadMetrics() {
  const params = {
    tenant_id: qs("#fTenant").value,
    since: qs("#fSince").value,
    until: qs("#fUntil").value,
  };
  const data = await apiGet("/owner/api/metrics" + buildQuery(params));
  const box = qs("#metricsBox");
  box.innerHTML = "";
  const items = [
    ["total_requests", data.total_requests],
    ["error_requests", data.error_requests],
    ["p95_ms", data.p95_ms],
    ["satisfaction_rate", data.satisfaction_rate],
    ["handoff_count", data.handoff_count],
    ["handoff_rate", data.handoff_rate],
    ["avg_time_ms", data.avg_time_ms],
    ["feedback_total", data.feedback_total],
  ];
  items.forEach(([k, v]) => {
    const div = document.createElement("div");
    div.className = "kpi__item";
    div.innerHTML = `<div class="kpi__label">${k}</div><div class="kpi__value">${v ?? "-"}</div>`;
    box.appendChild(div);
  });
}

async function loadLogs() {
  const params = {
    tenant_id: qs("#fTenant").value,
    since: qs("#fSince").value,
    until: qs("#fUntil").value,
    route: qs("#fRoute").value,
    status: qs("#fStatus").value,
    q: qs("#fQ").value,
    limit: 100,
    offset: 0,
  };
  const data = await apiGet("/owner/api/logs" + buildQuery(params));
  const body = qs("#logsBody");
  body.innerHTML = "";
  (data.rows || []).forEach((r) => {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    tr.innerHTML = `
      <td>${fmtTs(r.ts)}</td>
      <td>${r.tenant_id ?? ""}</td>
      <td>${r.route ?? ""}</td>
      <td>${r.status ?? ""}</td>
      <td>${r.latency_ms ? Number(r.latency_ms).toFixed(1) : ""}</td>
      <td>${r.sources_count ?? 0}</td>
      <td title="${(r.question_preview || "").replace(/"/g, "&quot;")}">${r.question_preview ?? ""}</td>
    `;
    tr.addEventListener("click", async () => {
      try {
        const detail = await apiGet("/owner/api/logs/" + encodeURIComponent(r.trace_id));
        qs("#traceDetail").textContent = JSON.stringify(detail, null, 2);
      } catch (e) {
        qs("#traceDetail").textContent = String(e);
      }
    });
    body.appendChild(tr);
  });
}

async function loadHandoffs() {
  const params = {
    tenant_id: qs("#fTenant").value,
    since: qs("#fSince").value,
    until: qs("#fUntil").value,
    status: qs("#fHStatus").value,
    limit: 200,
    offset: 0,
  };
  const data = await apiGet("/owner/api/handoffs" + buildQuery(params));
  const body = qs("#handoffsBody");
  body.innerHTML = "";
  (data.rows || []).forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${fmtTs(r.ts)}</td>
      <td>${r.tenant_id ?? ""}</td>
      <td>${maskPhone(r.phone)}</td>
      <td>${r.status ?? ""}</td>
      <td>${(r.message || "").slice(0, 280)}</td>
    `;
    body.appendChild(tr);
  });
}

async function refreshActive() {
  const active = document.querySelector(".tab--active")?.dataset?.tab || "metrics";
  qs("#appErr").textContent = "";
  if (active === "metrics") return loadMetrics();
  if (active === "logs") return loadLogs();
  if (active === "handoffs") return loadHandoffs();
}

async function init() {
  try {
    showApp();
    await loadTenants();
    await loadMetrics();
  } catch (e) {
    if (e && e.status === 401) return showLogin();
    showLogin(String(e));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab").forEach((b) => {
    b.addEventListener("click", async () => {
      setTab(b.dataset.tab);
      await refreshActive();
    });
  });

  qs("#btnRefresh").addEventListener("click", async () => {
    try {
      await refreshActive();
    } catch (e) {
      qs("#appErr").textContent = String(e);
    }
  });

  qs("#btnLogin").addEventListener("click", async () => {
    const u = qs("#loginUser").value;
    const p = qs("#loginPass").value;
    try {
      await apiPost("/owner/auth/login", { username: u, password: p });
      await init();
    } catch (e) {
      qs("#loginErr").textContent = e && e.status === 401 ? "Sai tài khoản/mật khẩu" : String(e);
    }
  });

  qs("#btnLogout").addEventListener("click", async () => {
    try {
      await apiPost("/owner/auth/logout", {});
    } finally {
      showLogin();
    }
  });

  ["#fTenant", "#fSince", "#fUntil", "#fRoute", "#fStatus", "#fQ", "#fHStatus"].forEach((id) => {
    const el = qs(id);
    if (!el) return;
    el.addEventListener("change", async () => {
      try {
        await refreshActive();
      } catch (e) {
        qs("#appErr").textContent = String(e);
      }
    });
  });

  setTab("metrics");
  init();
});
