import { Sparkles, TrendingUp, TrendingDown } from "lucide-react";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Textarea } from "./ui/textarea";
import { Separator } from "./ui/separator";
import { cn } from "./ui/utils";

export type Commentary = {
  score: number;
  strengths: string[];
  weaknesses: string[];
  reasoning: string;
};

export type BackendMeta = {
  sources?: string[];
  traceId?: string;
  route?: string;
  timeMs?: number;
};

function scoreColor(score: number) {
  if (score >= 90) return "bg-emerald-500 text-white";
  if (score >= 70) return "bg-amber-500 text-white";
  return "bg-rose-500 text-white";
}

export function CommentaryPanel({
  commentary,
  candidateName,
  backendStatus = "idle",
  backendMeta,
}: {
  commentary: Commentary | null;
  candidateName?: string;
  backendStatus?: "idle" | "loading" | "ready" | "error";
  backendMeta?: BackendMeta | null;
}) {
  if (!commentary) {
    return (
      <Card className="flex-1 flex items-center justify-center p-6 text-center">
        <div className="text-muted-foreground text-sm">
          <Sparkles className="size-5 mx-auto mb-2 opacity-50" />
          Upload CVs to see AI commentary
        </div>
      </Card>
    );
  }

  return (
    <Card className="flex-1 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-emerald-500" />
          <h3>AI Commentary & Reasoning</h3>
        </div>
        <Badge
          className={cn(
            "rounded-full px-3 h-7 text-sm",
            scoreColor(commentary.score),
          )}
        >
          {commentary.score}/100
        </Badge>
      </div>

      <div className="flex-1 overflow-auto px-5 py-4 space-y-4">
        {candidateName && (
          <p className="text-sm text-muted-foreground">
            Evaluating <span className="text-foreground">{candidateName}</span>
          </p>
        )}
        {backendStatus === "loading" && (
          <p className="text-xs text-muted-foreground">
            Fetching backend reasoning...
          </p>
        )}
        {backendStatus === "error" && (
          <p className="text-xs text-rose-500">
            Backend unavailable. Showing local evaluation.
          </p>
        )}
        {backendStatus === "ready" && backendMeta && (
          <p className="text-xs text-muted-foreground">
            Backend response
            {backendMeta.route ? ` · route: ${backendMeta.route}` : ""}
            {typeof backendMeta.timeMs === "number" ? ` · ${backendMeta.timeMs}ms` : ""}
            {backendMeta.sources?.length ? ` · ${backendMeta.sources.length} sources` : ""}
          </p>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="flex items-center gap-1.5 mb-2 text-emerald-600 dark:text-emerald-400">
              <TrendingUp className="size-3.5" />
              <span className="text-xs uppercase tracking-wide">Strengths</span>
            </div>
            <ul className="space-y-1.5">
              {commentary.strengths.map((s, i) => (
                <li key={i} className="text-sm flex gap-2">
                  <span className="text-emerald-500 mt-1.5 size-1.5 rounded-full bg-emerald-500 shrink-0" />
                  <span>{s}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="flex items-center gap-1.5 mb-2 text-rose-600 dark:text-rose-400">
              <TrendingDown className="size-3.5" />
              <span className="text-xs uppercase tracking-wide">Weaknesses</span>
            </div>
            <ul className="space-y-1.5">
              {commentary.weaknesses.map((s, i) => (
                <li key={i} className="text-sm flex gap-2">
                  <span className="text-rose-500 mt-1.5 size-1.5 rounded-full bg-rose-500 shrink-0" />
                  <span>{s}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <Separator />

        <div>
          <label className="text-xs uppercase tracking-wide text-muted-foreground mb-2 block">
            Deep AI Reasoning
          </label>
          <Textarea
            value={commentary.reasoning}
            readOnly
            className="min-h-[140px] resize-none bg-muted/40 text-sm leading-relaxed"
          />
        </div>
      </div>
    </Card>
  );
}
