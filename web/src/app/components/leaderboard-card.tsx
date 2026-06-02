import { useState, useCallback } from "react";
import { Upload, FileText, Trophy, X, ArrowUpDown } from "lucide-react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { cn } from "./ui/utils";

export type Candidate = {
  id: string;
  name: string;
  role: string;
  score: number;
  cv_text?: string;
};

type Props = {
  candidates: Candidate[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (files: File[]) => void;
  onReset: () => void;
};

function scoreColor(score: number) {
  if (score >= 90)
    return "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30";
  if (score >= 70)
    return "bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30";
  return "bg-rose-500/15 text-rose-700 dark:text-rose-400 border-rose-500/30";
}

export function LeaderboardCard({
  candidates,
  selectedId,
  onSelect,
  onUpload,
  onReset,
}: Props) {
  const [dragOver, setDragOver] = useState(false);
  const hasData = candidates.length > 0;

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onUpload(Array.from(files));
    },
    [onUpload],
  );

  return (
    <Card className="flex flex-col overflow-hidden h-full">
      <div className="flex items-center justify-between px-5 py-4 border-b border-border">
        <div className="flex items-center gap-2">
          {hasData ? (
            <Trophy className="size-4 text-emerald-500" />
          ) : (
            <Upload className="size-4 text-muted-foreground" />
          )}
          <h3>{hasData ? "Candidate Leaderboard" : "Upload Resumes"}</h3>
        </div>
        {hasData && (
          <Button variant="ghost" size="sm" onClick={onReset}>
            <X className="size-3.5 mr-1" /> Clear
          </Button>
        )}
      </div>

      {!hasData ? (
        <div className="flex-1 p-5">
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFiles(e.dataTransfer.files);
            }}
            className={cn(
              "flex flex-col items-center justify-center h-full min-h-[260px] rounded-lg border-2 border-dashed cursor-pointer transition-colors text-center px-6",
              dragOver
                ? "border-emerald-500 bg-emerald-500/5"
                : "border-border hover:border-emerald-400/60 hover:bg-muted/40",
            )}
          >
            <input
              type="file"
              multiple
              accept=".pdf,.doc,.docx"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <div className="size-12 rounded-full bg-emerald-500/10 flex items-center justify-center mb-4">
              <Upload className="size-5 text-emerald-600 dark:text-emerald-400" />
            </div>
            <p className="mb-1">Drop CVs here, or click to browse</p>
            <p className="text-sm text-muted-foreground mb-4">
              Supports PDF, DOC, DOCX — multiple files
            </p>
            <Badge variant="secondary" className="rounded-full">
              <FileText className="size-3 mr-1" /> Batch analysis enabled
            </Badge>
          </label>
        </div>
      ) : (
        <div className="flex-1 overflow-auto">
          <div className="grid grid-cols-[40px_1fr_auto] items-center gap-3 px-5 py-2.5 text-xs uppercase tracking-wide text-muted-foreground border-b border-border bg-muted/30">
            <span>Rank</span>
            <span className="flex items-center gap-1">
              Candidate <ArrowUpDown className="size-3" />
            </span>
            <span>Score</span>
          </div>
          <ul>
            {candidates.map((c, i) => {
              const active = c.id === selectedId;
              return (
                <li key={c.id}>
                  <button
                    onClick={() => onSelect(c.id)}
                    className={cn(
                      "w-full grid grid-cols-[40px_1fr_auto] items-center gap-3 px-5 py-3 text-left border-b border-border transition-colors",
                      active
                        ? "bg-emerald-500/10"
                        : "hover:bg-muted/50",
                    )}
                  >
                    <span
                      className={cn(
                        "size-7 rounded-full flex items-center justify-center text-sm",
                        i === 0
                          ? "bg-emerald-500 text-white"
                          : i === 1
                            ? "bg-slate-400 text-white"
                            : i === 2
                              ? "bg-amber-600 text-white"
                              : "bg-muted text-muted-foreground",
                      )}
                    >
                      {i + 1}
                    </span>
                    <div className="min-w-0">
                      <p className="truncate">{c.name}</p>
                      <p className="text-xs text-muted-foreground truncate">
                        {c.role}
                      </p>
                    </div>
                    <Badge
                      variant="outline"
                      className={cn("rounded-full", scoreColor(c.score))}
                    >
                      {c.score}
                    </Badge>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </Card>
  );
}
