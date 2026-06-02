function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString("vi-VN", { hour12: false });
}

function fmtMs(v) {
  if (v === null || v === undefined) return "—";
  return Math.round(Number(v)) + " ms";
}

function pct(v) {
  if (v === null || v === undefined) return "—";
  return Math.round(Number(v) * 1000) / 10 + "%";
}

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

function qs(id) {
  return document.getElementById(id);
}

function buildQuery(params) {
  const out = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v === null || v === undefined) return;
    const s = String(v).trim();
    if (!s) return;
    out.set(k, s);
  });
  const q = out.toString();
  return q ? "?" + q : "";
}

function setCard(id, value, hintId, hint) {
  const el = qs(id);
  if (el) el.textContent = value;
  if (hintId) {
    const h = qs(hintId);
    if (h) h.textContent = hint || "";
  }
}

async function loadTenants() {
  const sel = qs("tenantSelect");
  if (!sel) return;
  sel.innerHTML = "";
  const optAll = document.createElement("option");
  optAll.value = "";
  optAll.textContent = "all tenants";
  sel.appendChild(optAll);

  const data = await getJson("/admin/api/tenants");
  (data.tenants || []).forEach((t) => {
    const o = document.createElement("option");
    o.value = t;
    o.textContent = t;
    sel.appendChild(o);
  });
}

function getFilters() {
  return {
    tenant_id: qs("tenantSelect")?.value || "",
    since: qs("sinceInput")?.value || "",
    until: qs("untilInput")?.value || "",
  };
}

async function refreshMetrics() {
  const f = getFilters();
  const data = await getJson("/admin/api/metrics" + buildQuery(f));
  setCard("mAvg", data.avg_time_ms !== null ? fmtMs(data.avg_time_ms) : "—");
  setCard("mP95", data.p95_ms !== null ? fmtMs(data.p95_ms) : "—");
  setCard("mSat", data.satisfaction_rate !== null ? pct(data.satisfaction_rate) : "—", "mSatHint", `feedback: ${data.feedback_total ?? 0}`);
  setCard("mHandoff", data.handoff_rate !== null ? pct(data.handoff_rate) : "—", "mHandoffHint", `tickets: ${data.handoff_count ?? 0}`);
}

function renderLogs(rows) {
  const tb = qs("logsTable")?.querySelector("tbody");
  if (!tb) return;
  tb.innerHTML = "";
  (rows || []).forEach((r) => {
    const tr = document.createElement("tr");
    const status = String(r.status || "");
    const badgeCls = status === "ERROR" ? "admin-badge admin-badge--err" : "admin-badge admin-badge--ok";
    tr.innerHTML = `
      <td>${fmtTime(r.ts)}</td>
      <td>${r.route ? `<span class="admin-badge">${r.route}</span>` : "—"}</td>
      <td><span class="${badgeCls}">${status || "—"}</span></td>
      <td>${r.latency_ms !== null ? fmtMs(r.latency_ms) : "—"}</td>
      <td>${r.sources_count ?? 0}</td>
      <td title="${(r.question || "").replace(/"/g, "&quot;")}">${(r.question || "").slice(0, 140)}</td>
    `;
    tb.appendChild(tr);
  });
}

async function loadLogs() {
  const f = getFilters();
  const params = {
    ...f,
    q: qs("qSearch")?.value || "",
    route: qs("routeFilter")?.value || "",
    status: qs("statusFilter")?.value || "",
    limit: 120,
    offset: 0,
  };
  const data = await getJson("/admin/api/logs" + buildQuery(params));
  renderLogs(data.rows || []);
}

function renderHandoffs(rows) {
  const tb = qs("handoffTable")?.querySelector("tbody");
  if (!tb) return;
  tb.innerHTML = "";
  (rows || []).forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${fmtTime(r.ts)}</td>
      <td><span class="admin-badge">${r.status || "—"}</span></td>
      <td>${r.phone || "—"}</td>
      <td>${(r.message || "").slice(0, 220)}</td>
    `;
    tb.appendChild(tr);
  });
}

async function loadHandoffs() {
  const f = getFilters();
  const params = {
    ...f,
    status: qs("handoffStatus")?.value || "",
    limit: 120,
    offset: 0,
  };
  const data = await getJson("/admin/api/handoffs" + buildQuery(params));
  renderHandoffs(data.rows || []);
}

function todayISO() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

let autoTimer = null;
function setAuto(on) {
  const btn = qs("btnAuto");
  if (!btn) return;
  btn.dataset.auto = on ? "on" : "off";
  btn.textContent = `Auto: ${on ? "on" : "off"}`;
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
  if (on) {
    autoTimer = setInterval(async () => {
      try {
        await refreshMetrics();
        await loadLogs();
        await loadHandoffs();
      } catch (e) {
        console.warn(e);
      }
    }, 8000);
  }
}

async function initialLoad() {
  await loadTenants();
  if (qs("sinceInput")) qs("sinceInput").value = todayISO();
  if (qs("untilInput")) qs("untilInput").value = todayISO();
  await refreshMetrics();
  await loadLogs();
  await loadHandoffs();
}

document.addEventListener("DOMContentLoaded", () => {
  qs("btnRefresh")?.addEventListener("click", async () => {
    await refreshMetrics();
    await loadLogs();
    await loadHandoffs();
  });
  qs("btnLoadLogs")?.addEventListener("click", loadLogs);
  qs("btnLoadHandoffs")?.addEventListener("click", loadHandoffs);
  qs("btnAuto")?.addEventListener("click", () => {
    const on = qs("btnAuto")?.dataset.auto !== "on";
    setAuto(on);
  });

  initialLoad().catch((e) => {
    console.error(e);
    alert("Không load được dashboard. Kiểm tra server đang chạy và DATABASE_URL.");
  });
});

