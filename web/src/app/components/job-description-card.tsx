import { useCallback, useState } from "react";
import { BriefcaseBusiness, FileUp, Sparkles, X } from "lucide-react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Textarea } from "./ui/textarea";
import { cn } from "./ui/utils";

export type JobDescription = {
  text: string;
  fileName?: string;
};

type Props = {
  value: JobDescription;
  onChange: (value: JobDescription) => void;
};

const SAMPLE_JD = `Senior Frontend Engineer\n\nWe need a React + TypeScript specialist to lead design-system delivery, improve Core Web Vitals, mentor engineers, and partner with product/design. Must have Vite, Tailwind, accessibility, testing, and measurable performance wins.`;

export function JobDescriptionCard({ value, onChange }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const hasJD = value.text.trim().length > 0 || Boolean(value.fileName);

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      const readable = file.type.startsWith("text/") || /\.(txt|md)$/i.test(file.name);
      const text = readable ? await file.text() : value.text;
      onChange({ text, fileName: file.name });
    },
    [onChange, value.text],
  );

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
              Add JD để AI so khớp CV theo yêu cầu công việc
            </p>
          </div>
        </div>
        {hasJD ? (
          <Badge className="rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/15">
            <Sparkles className="size-3 mr-1" /> Active
          </Badge>
        ) : null}
      </div>

      <div className="grid gap-3 p-4 lg:grid-cols-[1fr_170px]">
        <div className="space-y-2">
          <Textarea
            value={value.text}
            onChange={(event) => onChange({ ...value, text: event.target.value })}
            placeholder="Paste JD / yêu cầu công việc: responsibilities, must-have skills, seniority, domain..."
            className="min-h-[112px] max-h-[150px] bg-slate-50/80 dark:bg-slate-900/60"
          />
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>{value.text.trim().length} characters</span>
            {value.fileName ? (
              <Badge variant="secondary" className="rounded-full max-w-[260px] truncate">
                {value.fileName}
              </Badge>
            ) : null}
          </div>
        </div>

        <label
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            handleFile(event.dataTransfer.files[0]);
          }}
          className={cn(
            "flex min-h-[112px] cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-3 text-center transition-colors",
            dragOver
              ? "border-emerald-500 bg-emerald-500/10"
              : "border-border bg-muted/25 hover:border-emerald-400/70 hover:bg-emerald-500/5",
          )}
        >
          <input
            type="file"
            accept=".pdf,.doc,.docx,.txt,.md"
            className="hidden"
            onChange={(event) => handleFile(event.target.files?.[0])}
          />
          <FileUp className="mb-2 size-5 text-emerald-600 dark:text-emerald-400" />
          <span className="text-sm">Upload JD file</span>
          <span className="mt-1 text-xs text-muted-foreground">PDF, DOCX, TXT</span>
        </label>

        <div className="flex flex-wrap gap-2 lg:col-span-2">
          <Button type="button" variant="outline" size="sm" onClick={() => onChange({ text: SAMPLE_JD })}>
            Use sample JD
          </Button>
          {hasJD ? (
            <Button type="button" variant="ghost" size="sm" onClick={() => onChange({ text: "" })}>
              <X className="size-3.5 mr-1" /> Clear JD
            </Button>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
