import { Info, FileText, Download } from "lucide-react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { cn } from "./ui/utils";

export type Annotation = {
  kind: "bonus" | "penalty";
  tip: string;
};

export type AnnotatedSegment = {
  text: string;
  annotation?: Annotation;
};

export type CVData = {
  name: string;
  title: string;
  email: string;
  location: string;
  summary: AnnotatedSegment[];
  experience: {
    role: string;
    company: string;
    period: string;
    bullets: AnnotatedSegment[][];
  }[];
  skills: AnnotatedSegment[];
  education: { school: string; degree: string; year: string }[];
  full_text?: AnnotatedSegment[];
};

function Annotated({ seg }: { seg: AnnotatedSegment }) {
  if (!seg.annotation) return <span>{seg.text}</span>;
  const isBonus = seg.annotation.kind === "bonus";
  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              "relative inline cursor-help rounded px-0.5 mx-[1px]",
              isBonus
                ? "bg-emerald-500/20 text-emerald-900 dark:text-emerald-200 underline decoration-emerald-500 decoration-2 underline-offset-2"
                : "bg-rose-500/20 text-rose-900 dark:text-rose-200 line-through decoration-rose-500 decoration-2",
            )}
          >
            {seg.text}
            <Info
              className={cn(
                "inline size-3 ml-1 -mt-0.5 align-middle",
                isBonus ? "text-emerald-600" : "text-rose-600",
              )}
            />
          </span>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          className={cn(
            "max-w-xs",
            isBonus ? "bg-emerald-600" : "bg-rose-600",
          )}
        >
          <div className="flex items-start gap-1.5">
            <Badge
              variant="secondary"
              className="bg-white/20 text-white border-0 rounded-full text-[10px] px-1.5"
            >
              {isBonus ? "+ Bonus" : "− Penalty"}
            </Badge>
            <p className="text-xs leading-snug">{seg.annotation.tip}</p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function Segments({ segs }: { segs: AnnotatedSegment[] }) {
  return (
    <>
      {segs.map((s, i) => (
        <Annotated key={i} seg={s} />
      ))}
    </>
  );
}

export function CVPreview({ cv }: { cv: CVData | null }) {
  if (!cv) {
    return (
      <Card className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <FileText className="size-10 mx-auto mb-3 opacity-40" />
          <p>Select a candidate to preview their CV</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="flex-1 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-6 py-4 border-b border-border">
        <div className="flex items-center gap-2">
          <FileText className="size-4 text-muted-foreground" />
          <h3>CV Preview</h3>
          <Badge variant="outline" className="ml-2 rounded-full text-xs">
            Annotated
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-3 text-xs text-muted-foreground mr-2">
            <span className="flex items-center gap-1">
              <span className="size-2 rounded-full bg-emerald-500" /> Bonus
            </span>
            <span className="flex items-center gap-1">
              <span className="size-2 rounded-full bg-rose-500" /> Penalty
            </span>
          </div>
          <Button variant="outline" size="sm">
            <Download className="size-3.5 mr-1" /> Export
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-auto bg-slate-100 dark:bg-slate-900/40 p-6">
        <div className="mx-auto max-w-2xl bg-white dark:bg-slate-950 shadow-sm rounded-lg p-10 text-slate-900 dark:text-slate-100 leading-relaxed">
          <header className="border-b border-slate-200 dark:border-slate-800 pb-5 mb-5">
            <h1 className="mb-1">{cv.name}</h1>
            <p className="text-emerald-600 dark:text-emerald-400">{cv.title}</p>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              {cv.email} {cv.email && cv.location && "·"} {cv.location}
            </p>
          </header>

          {cv.full_text ? (
            <div className="whitespace-pre-wrap text-sm leading-relaxed">
              <Segments segs={cv.full_text} />
            </div>
          ) : (
            <>
              <section className="mb-6">
                <h4 className="uppercase tracking-wide text-xs text-slate-500 mb-2">
                  Summary
                </h4>
                <p className="text-sm">
                  <Segments segs={cv.summary} />
                </p>
              </section>

              <section className="mb-6">
                <h4 className="uppercase tracking-wide text-xs text-slate-500 mb-3">
                  Experience
                </h4>
                <div className="space-y-5">
                  {cv.experience.map((e, i) => (
                    <div key={i}>
                      <div className="flex items-baseline justify-between">
                        <p>
                          {e.role} ·{" "}
                          <span className="text-slate-600 dark:text-slate-400">
                            {e.company}
                          </span>
                        </p>
                        <span className="text-xs text-slate-500">{e.period}</span>
                      </div>
                      <ul className="mt-2 space-y-1.5 text-sm list-disc pl-5">
                        {e.bullets.map((b, j) => (
                          <li key={j}>
                            <Segments segs={b} />
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </section>

              <section className="mb-6">
                <h4 className="uppercase tracking-wide text-xs text-slate-500 mb-2">
                  Skills
                </h4>
                <p className="text-sm">
                  <Segments segs={cv.skills} />
                </p>
              </section>

              <section>
                <h4 className="uppercase tracking-wide text-xs text-slate-500 mb-2">
                  Education
                </h4>
                <div className="space-y-1.5 text-sm">
                  {cv.education.map((ed, i) => (
                    <div key={i} className="flex justify-between">
                      <span>
                        {ed.degree} — {ed.school}
                      </span>
                      <span className="text-slate-500">{ed.year}</span>
                    </div>
                  ))}
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </Card>
  );
}
