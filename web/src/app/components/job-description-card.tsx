import { useState } from "react";
import { BriefcaseBusiness, Sparkles, X, ClipboardPaste } from "lucide-react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Textarea } from "./ui/textarea";

export type JobDescription = {
  text: string;
  fileName?: string;
};

type Props = {
  value: JobDescription;
  onChange: (value: JobDescription) => void;
};

const SAMPLE_JD = `AI Engineer Intern — Agentic RAG Systems

We are looking for a motivated AI Engineering Intern to join our team working on production-grade RAG (Retrieval-Augmented Generation) and Agentic AI systems.

Requirements:
- Strong Python skills (async, FastAPI)
- Experience with vector databases (Qdrant, FAISS, Pinecone)
- Understanding of LLM APIs (OpenAI, Gemini, Groq)
- Knowledge of RAG architecture and prompt engineering
- Familiarity with LlamaIndex or LangChain
- Good English communication (IELTS 6.5+)

Nice to have:
- Experience with Docker, PostgreSQL
- Understanding of evaluation frameworks (RAGAS, DeepEval)
- Contributions to open-source AI projects
- Experience with multimodal retrieval systems`;

export function JobDescriptionCard({ value, onChange }: Props) {
  const hasJD = value.text.trim().length > 0;

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text.trim()) {
        onChange({ text });
      }
    } catch {
      // Clipboard API may not be available
    }
  };

  return (
    <Card className="overflow-hidden border-emerald-500/20 bg-background/95 shadow-sm">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-gradient-to-r from-emerald-500/10 via-transparent to-slate-500/5">
        <div className="flex items-center gap-2 min-w-0">
          <div className="size-8 rounded-lg bg-emerald-500/15 flex items-center justify-center shrink-0">
            <BriefcaseBusiness className="size-4 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div className="min-w-0">
            <h3 className="text-base leading-tight">Job Description</h3>
            <p className="text-xs text-muted-foreground truncate">
              Paste JD để AI so khớp CV theo yêu cầu công việc
            </p>
          </div>
        </div>
        {hasJD ? (
          <Badge className="rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/15">
            <Sparkles className="size-3 mr-1" /> Active
          </Badge>
        ) : null}
      </div>

      <div className="p-4 space-y-3">
        <Textarea
          value={value.text}
          onChange={(event) => onChange({ ...value, text: event.target.value })}
          placeholder="Paste JD / yêu cầu công việc vào đây: responsibilities, must-have skills, seniority, domain..."
          className="min-h-[120px] max-h-[180px] bg-slate-50/80 dark:bg-slate-900/60 resize-y"
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">
            {value.text.trim().length} characters
          </span>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePaste}
            >
              <ClipboardPaste className="size-3.5 mr-1" /> Paste
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onChange({ text: SAMPLE_JD })}
            >
              Sample JD
            </Button>
            {hasJD ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => onChange({ text: "" })}
              >
                <X className="size-3.5 mr-1" /> Clear
              </Button>
            ) : null}
          </div>
        </div>
      </div>
    </Card>
  );
}
