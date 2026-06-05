import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Moon, Sun, Brain } from "lucide-react";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { ChatPanel } from "./components/chat-panel";
import { LeaderboardCard, type Candidate } from "./components/leaderboard-card";
import { CommentaryPanel, type BackendMeta } from "./components/commentary-panel";
import type { Commentary } from "./components/commentary-panel";
import { CVPreview, type CVData } from "./components/cv-preview";
import { JobDescriptionCard, type JobDescription } from "./components/job-description-card";
import { fetchPublicConfig, hasApiKey, type PublicConfig } from "./api/client";
import {
  requestStructuredReview,
  requestBackendReasoning,
  uploadResumes,
  applyAnnotations,
  type StructuredReviewResult,
} from "./api/cv-review";
import {
  SAMPLE_CANDIDATES,
  COMMENTARIES,
  CV_DATA,
  buildFallbackCV,
} from "./data";

// ---------------------------------------------------------------------------
// Stores for backend-provided review data (keyed by candidate id)
// ---------------------------------------------------------------------------

type ReviewCache = Record<
  string,
  { commentary: Commentary; cv: CVData | null; raw: StructuredReviewResult }
>;

export default function App() {
  const [dark, setDark] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>(SAMPLE_CANDIDATES);
  const [selectedId, setSelectedId] = useState<string | null>("1");
  const [jobDescription, setJobDescription] = useState<JobDescription>({
    text: "",
  });
  const [publicConfig, setPublicConfig] = useState<PublicConfig | null>(null);
  const [backendStatus, setBackendStatus] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const [backendMeta, setBackendMeta] = useState<BackendMeta | null>(null);
  const [backendReasoning, setBackendReasoning] = useState<string | null>(null);

  // In-memory stores for structured reviews from backend
  const [reviewCache, setReviewCache] = useState<ReviewCache>({});
  // Track uploaded CV texts so we can send them to /cv/review
  const uploadedTexts = useRef<Record<string, string>>({});

  useEffect(() => {
    const root = document.documentElement;
    if (dark) root.classList.add("dark");
    else root.classList.remove("dark");
  }, [dark]);

  useEffect(() => {
    let active = true;
    fetchPublicConfig().then((cfg) => {
      if (active) setPublicConfig(cfg);
    });
    return () => {
      active = false;
    };
  }, []);

  const selected = useMemo(
    () => candidates.find((c) => c.id === selectedId) ?? null,
    [candidates, selectedId],
  );

  const hasJobDescription = jobDescription.text.trim().length > 0;

  // ---------------------------------------------------------------------------
  // Commentary & CV: prefer backend review cache, then static data, then fallback
  // ---------------------------------------------------------------------------

  const commentary = useMemo(() => {
    if (!selected) return null;

    // 1) Check backend review cache
    const cached = reviewCache[selected.id];
    if (cached) return cached.commentary;

    // 2) Check static mock data
    const staticComm = COMMENTARIES[selected.id];
    if (staticComm) return staticComm;

    // 3) Fallback
    return {
      score: selected.score,
      strengths: [
        "Relevant experience for the target role",
        "Clear structure and readable layout",
      ],
      weaknesses: [
        "Some bullets lack quantified outcomes",
        "Skills section could be more focused",
      ],
      reasoning: `${selected.name} scores ${selected.score}/100 based on relevance, clarity, and signal density. The model weighs measurable achievements, stack alignment, and tenure stability. See annotated CV for specific bonuses and penalties contributing to the final score.`,
    };
  }, [selected, reviewCache]);

  const cv = useMemo(() => {
    if (!selected) return null;

    // 1) Check backend review cache (has annotated CV)
    const cached = reviewCache[selected.id];
    if (cached?.cv) return cached.cv;

    // 2) Check static data
    const staticCV = CV_DATA[selected.id];
    if (staticCV) return staticCV;

    // 3) Fallback
    return buildFallbackCV(selected);
  }, [selected, reviewCache]);

  const commentaryWithJD = commentary
    ? {
        ...commentary,
        reasoning: backendReasoning
          ? backendReasoning
          : hasJobDescription
            ? `${commentary.reasoning} JD match layer is active: the ranking is now framed against the supplied job description, prioritizing must-have skills, seniority, responsibilities, and measurable outcomes.`
            : commentary.reasoning,
      }
    : null;

  // ---------------------------------------------------------------------------
  // Backend review effect: fetch structured review for selected candidate
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let active = true;
    const canQuery = Boolean(publicConfig?.public_query_enabled) || hasApiKey();

    setBackendReasoning(null);
    setBackendMeta(null);
    setBackendStatus("idle");

    if (!selected || !canQuery) {
      return () => {
        active = false;
      };
    }

    // Already have cached review for this candidate?
    if (reviewCache[selected.id]) {
      return () => {
        active = false;
      };
    }

    // Need CV text to call /cv/review
    const cvText = uploadedTexts.current[selected.id] || "";

    const timer = window.setTimeout(() => {
      setBackendStatus("loading");

      // If we have uploaded text → use structured review
      if (cvText) {
        requestStructuredReview(cvText, jobDescription.text, selected.id)
          .then((result) => {
            if (!active || !result) {
              setBackendStatus("idle");
              return;
            }
            // Build annotated CVData from plain text + annotations
            const annotatedCV = buildCVFromReview(selected, cvText, result);
            const comm: Commentary = {
              score: result.score,
              strengths: result.strengths,
              weaknesses: result.weaknesses,
              reasoning: result.reasoning,
            };
            setReviewCache((prev) => ({
              ...prev,
              [selected.id]: { commentary: comm, cv: annotatedCV, raw: result },
            }));
            setBackendReasoning(result.reasoning);
            setBackendStatus("ready");
          })
          .catch(() => {
            if (!active) return;
            setBackendStatus("error");
          });
      } else {
        // Fallback: legacy /query RAG reasoning
        requestBackendReasoning(selected, cv, jobDescription)
          .then((result) => {
            if (!active) return;
            if (!result || !result.reasoning) {
              setBackendStatus("idle");
              return;
            }
            setBackendReasoning(result.reasoning);
            setBackendMeta({
              sources: result.sources,
              traceId: result.traceId,
              route: result.route,
              timeMs: result.timeMs,
            });
            setBackendStatus("ready");
          })
          .catch(() => {
            if (!active) return;
            setBackendStatus("error");
          });
      }
    }, 700);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [
    selected,
    cv,
    jobDescription.text,
    publicConfig?.public_query_enabled,
    reviewCache,
  ]);

  // ---------------------------------------------------------------------------
  // Upload handler
  // ---------------------------------------------------------------------------

  const handleUpload = useCallback(
    async (files: File[]) => {
      // Must have JD
      if (!jobDescription.text.trim()) {
        alert("Vui lòng nhập Job Description trước khi chấm điểm CV!");
        return;
      }

      try {
        const backendCandidates = await uploadResumes(files, jobDescription.text);
        if (backendCandidates && backendCandidates.length > 0) {
          const combined = backendCandidates.sort((a, b) => b.score - a.score);

          // Store extracted text from backend
          const fileTexts: Record<string, string> = {};
          for (const cand of combined) {
            if (cand.cv_text) {
              fileTexts[cand.id] = cand.cv_text;
            }
          }

          uploadedTexts.current = { ...uploadedTexts.current, ...fileTexts };
          setCandidates(combined);
          setSelectedId(combined[0]?.id ?? null);
          setReviewCache({}); // Clear cache on new upload
          return;
        } else {
           alert("Upload returns no candidates or VITE env is missing.");
        }
      } catch (err: any) {
        console.error("Upload error:", err);
        alert("Upload Failed! " + err.message + "\nAre you sure you restarted 'npm run dev' after the .env was added?");
        return;
      }
    },
    [jobDescription],
  );

  function handleReset() {
    setCandidates([]);
    setSelectedId(null);
    setReviewCache({});
    uploadedTexts.current = {};
  }

  return (
    <div className="size-full min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-foreground">
      <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-background/80 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="size-9 rounded-lg bg-emerald-500 flex items-center justify-center">
            <Brain className="size-5 text-white" />
          </div>
          <div>
            <h2 className="leading-tight">CV Reviewer</h2>
            <p className="text-xs text-muted-foreground leading-tight">
              AI-powered ranking & insights
            </p>
          </div>
          <Badge
            variant="outline"
            className="ml-3 rounded-full border-emerald-500/40 text-emerald-700 dark:text-emerald-400 bg-emerald-500/10"
          >
            Beta
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground hidden md:inline">
            {candidates.length} candidates analyzed · {hasJobDescription ? "JD active" : "No JD"}
          </span>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setDark((d) => !d)}
            aria-label="Toggle dark mode"
          >
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-5 gap-4 p-4 min-h-0">
        <div className="lg:col-span-2 flex flex-col gap-4 min-h-0">
          <JobDescriptionCard value={jobDescription} onChange={setJobDescription} />
          <div className="h-[38%] min-h-[300px]">
            <LeaderboardCard
              candidates={candidates}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onUpload={handleUpload}
              onReset={handleReset}
            />
          </div>
          <div className="flex-1 min-h-0 flex">
            <CommentaryPanel
              commentary={commentaryWithJD}
              candidateName={selected?.name}
              backendStatus={backendStatus}
              backendMeta={backendMeta}
            />
          </div>
        </div>

        <div className="lg:col-span-3 flex min-h-0">
          <Tabs defaultValue="preview" className="flex-1 flex flex-col min-h-0 border rounded-lg bg-card text-card-foreground shadow-sm overflow-hidden">
            <div className="flex items-center justify-between px-4 border-b bg-muted/50">
              <TabsList className="h-12 bg-transparent gap-4">
                <TabsTrigger 
                  value="preview" 
                  className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 h-12"
                >
                  CV Preview
                </TabsTrigger>
                <TabsTrigger 
                  value="chat" 
                  className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none px-0 h-12"
                >
                  Ask AI (RAG)
                </TabsTrigger>
              </TabsList>
            </div>
            
            <TabsContent value="preview" className="flex-1 min-h-0 m-0 overflow-hidden flex flex-col">
              <CVPreview cv={cv} />
            </TabsContent>
            <TabsContent value="chat" className="flex-1 min-h-0 m-0 overflow-hidden flex flex-col">
              <ChatPanel />
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper: build CVData from plain text + backend annotations
// ---------------------------------------------------------------------------

function buildCVFromReview(
  candidate: Candidate,
  cvText: string,
  review: StructuredReviewResult,
): CVData {
  const name = candidate.name;
  const title = candidate.role;

  // Apply annotations to the entire text block, preserving original \n newlines
  const fullTextSegs = applyAnnotations(cvText, review.annotations);

  return {
    name,
    title,
    email: "",
    location: "",
    full_text: fullTextSegs,
    // Provide empty fallbacks for legacy structure
    summary: [],
    experience: [],
    skills: [],
    education: [],
  };
}
