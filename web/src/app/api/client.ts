type QueryContext = {
  tenant_id?: string | null;
  branch_id?: string | null;
  session_id?: string | null;
  user_id?: string | null;
};

export type PublicConfig = {
  enable_branch_filter: boolean;
  public_query_enabled: boolean;
};

export type QueryRequest = {
  question: string;
  tenant_id?: string | null;
  branch_id?: string | null;
  history?: Array<{ role: string; content: string }>;
  session_id?: string | null;
  user_id?: string | null;
};

export type QueryResponse = {
  answer: string;
  sources: string[];
  trace_id?: string;
  time_ms?: number;
  route?: string;
};

export type FeedbackRequest = {
  trace_id: string;
  tenant_id?: string | null;
  rating: 1 | -1;
  comment?: string | null;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").trim();
const API_KEY = (import.meta.env.VITE_PUBLIC_QUERY_API_KEY || "").trim();
const DEFAULT_TENANT_ID = (import.meta.env.VITE_TENANT_ID || "").trim();
const DEFAULT_BRANCH_ID = (import.meta.env.VITE_BRANCH_ID || "").trim();
const DEFAULT_USER_ID = (import.meta.env.VITE_USER_ID || "").trim();

const SESSION_KEY = "cv-review.session";

function buildUrl(path: string) {
  if (!API_BASE) return path;
  return `${API_BASE.replace(/\/$/, "")}${path}`;
}

function getOrCreateSessionId() {
  if (typeof window === "undefined") return null;
  try {
    const existing = window.localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
    const sid = `cv:${Math.random().toString(16).slice(2)}:${Date.now().toString(16)}`;
    window.localStorage.setItem(SESSION_KEY, sid);
    return sid;
  } catch {
    return null;
  }
}

export function hasApiKey() {
  return Boolean(API_KEY);
}

export function getApiKeyValue() {
  return API_KEY || null;
}

export function buildApiUrl(path: string) {
  return buildUrl(path);
}

export function getQueryContext(): QueryContext {
  return {
    tenant_id: DEFAULT_TENANT_ID || null,
    branch_id: DEFAULT_BRANCH_ID || null,
    user_id: DEFAULT_USER_ID || null,
    session_id: getOrCreateSessionId(),
  };
}

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers || undefined);
  if (API_KEY) headers.set("x-api-key", API_KEY);
  if (options?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(buildUrl(path), {
    ...options,
    headers,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }

  return response.json() as Promise<T>;
}

export async function fetchPublicConfig(): Promise<PublicConfig | null> {
  try {
    return await fetchJson<PublicConfig>("/public/config", { cache: "no-store" });
  } catch {
    return null;
  }
}

export async function queryBackend(payload: QueryRequest): Promise<QueryResponse> {
  return fetchJson<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function sendFeedback(payload: FeedbackRequest): Promise<{ ok: boolean; id?: string }> {
  return fetchJson("/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
