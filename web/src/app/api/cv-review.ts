import type { Candidate } from "../components/leaderboard-card";
import type { CVData, AnnotatedSegment } from "../components/cv-preview";
import type { Commentary } from "../components/commentary-panel";
import type { JobDescription } from "../components/job-description-card";
import { buildApiUrl, getApiKeyValue, getQueryContext, queryBackend } from "./client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type BackendAnnotation = {
  text: string;
  kind: "bonus" | "penalty";
  tip: string;
};

export type BackendReviewResult = {
  reasoning?: string;
  sources?: string[];
  traceId?: string;
  route?: string;
  timeMs?: number;
};

export type StructuredReviewResult = {
  score: number;
  strengths: string[];
  weaknesses: string[];
  reasoning: string;
  annotations: BackendAnnotation[];
};

// ---------------------------------------------------------------------------
// Env
// ---------------------------------------------------------------------------

const REVIEW_ENDPOINT = (import.meta.env.VITE_CV_REVIEW_ENDPOINT || "").trim();
const UPLOAD_ENDPOINT = (import.meta.env.VITE_CV_UPLOAD_ENDPOINT || "").trim();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function segmentsToText(segs: Array<{ text: string }>) {
  return segs.map((s) => s.text).join("");
}

function cvToText(cv: CVData) {
  const summary = segmentsToText(cv.summary || []);
  const skills = segmentsToText(cv.skills || []);
  const experience = (cv.experience || [])
    .map((exp) => {
      const bullets = exp.bullets
        .map((b) => segmentsToText(b))
        .filter(Boolean)
        .map((b) => `- ${b}`)
        .join("\n");
      return `${exp.role} @ ${exp.company} (${exp.period})\n${bullets}`;
    })
    .join("\n\n");
  const education = (cv.education || [])
    .map((ed) => `${ed.degree} - ${ed.school} (${ed.year})`)
    .join("\n");

  return [
    summary ? `Summary:\n${summary}` : "",
    experience ? `Experience:\n${experience}` : "",
    skills ? `Skills:\n${skills}` : "",
    education ? `Education:\n${education}` : "",
  ]
    .filter(Boolean)
    .join("\n\n");
}

function buildReviewPrompt(candidate: Candidate, cv: CVData | null, jd?: JobDescription) {
  const jdText = (jd?.text || "").trim();
  const cvText = cv ? cvToText(cv) : "";

  return [
    "You are an AI CV reviewer.",
    "Return a short evaluation paragraph for the candidate fit.",
    "Focus on strengths, risks, and hiring recommendation.",
    "",
    `Candidate: ${candidate.name} (${candidate.role})`,
    `Internal score: ${candidate.score}/100`,
    jdText ? `Job description:\n${jdText}` : "",
    cvText ? `CV detail:\n${cvText}` : "",
  ]
    .filter(Boolean)
    .join("\n");
}

/**
 * Convert backend annotations into AnnotatedSegment[] by splitting the
 * plain text of a CV section around the annotated phrases.
 */
export function applyAnnotations(
  plainText: string,
  annotations: BackendAnnotation[],
): AnnotatedSegment[] {
  if (!plainText || !annotations.length) {
    return [{ text: plainText }];
  }

  // Find all annotation matches with their positions
  type Match = { start: number; end: number; ann: BackendAnnotation };
  const matches: Match[] = [];
  for (const ann of annotations) {
    const idx = plainText.indexOf(ann.text);
    if (idx >= 0) {
      matches.push({ start: idx, end: idx + ann.text.length, ann });
    }
  }

  if (matches.length === 0) {
    return [{ text: plainText }];
  }

  // Sort by position and remove overlaps (keep first match)
  matches.sort((a, b) => a.start - b.start);
  const filtered: Match[] = [matches[0]];
  for (let i = 1; i < matches.length; i++) {
    if (matches[i].start >= filtered[filtered.length - 1].end) {
      filtered.push(matches[i]);
    }
  }

  // Build segments
  const segments: AnnotatedSegment[] = [];
  let cursor = 0;
  for (const m of filtered) {
    if (m.start > cursor) {
      segments.push({ text: plainText.slice(cursor, m.start) });
    }
    segments.push({
      text: plainText.slice(m.start, m.end),
      annotation: { kind: m.ann.kind, tip: m.ann.tip },
    });
    cursor = m.end;
  }
  if (cursor < plainText.length) {
    segments.push({ text: plainText.slice(cursor) });
  }
  return segments;
}

// ---------------------------------------------------------------------------
// Structured review (calls /cv/review)
// ---------------------------------------------------------------------------

export async function requestStructuredReview(
  cvText: string,
  jobDescription: string,
  candidateId?: string,
): Promise<StructuredReviewResult | null> {
  if (!REVIEW_ENDPOINT) return null;

  const headers = new Headers({ "Content-Type": "application/json" });
  const apiKey = getApiKeyValue();
  if (apiKey) headers.set("x-api-key", apiKey);

  const payload = {
    candidate_id: candidateId,
    cv_text: cvText,
    job_description: jobDescription || undefined,
  };

  const response = await fetch(buildApiUrl(REVIEW_ENDPOINT), {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const data = await response.json();
  return {
    score: data.score ?? 0,
    strengths: data.strengths ?? [],
    weaknesses: data.weaknesses ?? [],
    reasoning: data.reasoning ?? "",
    annotations: data.annotations ?? [],
  };
}

// ---------------------------------------------------------------------------
// Legacy reasoning endpoint (fallback to /query RAG)
// ---------------------------------------------------------------------------

export async function requestBackendReasoning(
  candidate: Candidate,
  cv: CVData | null,
  jobDescription: JobDescription,
): Promise<BackendReviewResult | null> {
  // Try structured review first when REVIEW_ENDPOINT is set
  if (REVIEW_ENDPOINT && cv) {
    const cvText = cvToText(cv);
    try {
      const structured = await requestStructuredReview(
        cvText,
        jobDescription.text,
        candidate.id,
      );
      if (structured) {
        return {
          reasoning: structured.reasoning,
          sources: [],
        };
      }
    } catch (e) {
      console.warn("Structured review failed, falling back to RAG:", e);
    }
  }

  // Fallback: use generic /query endpoint
  const prompt = buildReviewPrompt(candidate, cv, jobDescription);
  const context = getQueryContext();
  const result = await queryBackend({
    question: prompt,
    tenant_id: context.tenant_id ?? undefined,
    branch_id: context.branch_id ?? undefined,
    session_id: context.session_id ?? undefined,
    user_id: context.user_id ?? undefined,
    history: [],
  });

  return {
    reasoning: result.answer,
    sources: result.sources,
    traceId: result.trace_id,
    route: result.route,
    timeMs: result.time_ms,
  };
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

export async function uploadResumes(files: File[], jobDescription?: string): Promise<Candidate[] | null> {
  if (!UPLOAD_ENDPOINT) return null;
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  if (jobDescription) {
    form.append("job_description", jobDescription);
  }

  const headers = new Headers();
  const apiKey = getApiKeyValue();
  if (apiKey) headers.set("x-api-key", apiKey);

  const response = await fetch(buildApiUrl(UPLOAD_ENDPOINT), {
    method: "POST",
    headers,
    body: form,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const data = await response.json();
  if (Array.isArray(data)) return data as Candidate[];
  if (Array.isArray(data?.candidates)) return data.candidates as Candidate[];
  return null;
}
