import { useState, useRef, useEffect } from "react";
import { Send, User, Bot, Loader2 } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Card } from "./ui/card";
import { cn } from "./ui/utils";
import { Badge } from "./ui/badge";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  latency?: number;
};

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Xin chào! Tôi có thể giúp gì cho bạn trong việc tìm kiếm ứng viên?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5m timeout

      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userMessage,
          tenant_id: null,
          history: messages
            .filter((m) => m.role !== "assistant" || m.content !== "Xin chào! Tôi có thể giúp gì cho bạn trong việc tìm kiếm ứng viên?")
            .map((m) => ({ role: m.role, content: m.content })),
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        if (response.status === 503) {
           throw new Error("Backend đang tải Model hoặc index, vui lòng thử lại sau ít phút.");
        }
        throw new Error("Lỗi khi kết nối đến RAG Backend");
      }

      const data = await response.json();
      
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "Không tìm thấy câu trả lời phù hợp.",
          sources: data.sources,
          latency: data.time_ms,
        },
      ]);
    } catch (error: any) {
      console.error("Chat error:", error);
      
      let errorMsg = "Đã có lỗi xảy ra khi xử lý câu hỏi của bạn. Vui lòng thử lại.";
      if (error.name === "AbortError") {
         errorMsg = "Kết nối đến RAG Backend quá thời gian (Timeout). Server có thể đang tải Model hoặc quá tải. Vui lòng đợi 1-2 phút rồi thử lại.";
      } else if (error.message) {
         errorMsg = error.message;
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: errorMsg,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="flex flex-col h-full overflow-hidden border-0 shadow-none rounded-none bg-transparent">
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="flex flex-col gap-4 pb-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={cn(
                "flex w-full gap-3 text-sm",
                msg.role === "user" ? "flex-row-reverse" : "flex-row"
              )}
            >
              <div
                className={cn(
                  "flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow-sm",
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground"
                )}
              >
                {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div
                className={cn(
                  "flex flex-col gap-2 max-w-[85%]",
                  msg.role === "user" ? "items-end" : "items-start"
                )}
              >
                <div
                  className={cn(
                    "rounded-xl px-4 py-3 whitespace-pre-wrap leading-relaxed shadow-sm",
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground rounded-tr-sm"
                      : "bg-white dark:bg-slate-900 border rounded-tl-sm text-foreground"
                  )}
                >
                  {msg.content}
                </div>
                
                {msg.sources && msg.sources.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    <span className="text-[10px] text-muted-foreground uppercase font-semibold mr-1 self-center">
                      Sources ({msg.sources.length}):
                    </span>
                    {msg.sources.slice(0, 3).map((src, i) => (
                      <Badge key={i} variant="outline" className="text-[10px] h-5 px-1.5 font-normal bg-white/50 dark:bg-black/20">
                        {src.split('/').pop()?.slice(0, 20) || 'doc'}...
                      </Badge>
                    ))}
                    {msg.sources.length > 3 && (
                      <Badge variant="outline" className="text-[10px] h-5 px-1.5 font-normal">
                        +{msg.sources.length - 3} more
                      </Badge>
                    )}
                  </div>
                )}
                {msg.latency && (
                  <span className="text-[10px] text-muted-foreground">
                    ⚡ {msg.latency} ms
                  </span>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex w-full gap-3 text-sm flex-row">
              <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow-sm bg-muted text-muted-foreground">
                <Bot size={16} />
              </div>
              <div className="rounded-xl px-4 py-3 bg-white dark:bg-slate-900 border rounded-tl-sm flex items-center shadow-sm">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Đang suy nghĩ...</span>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
      <div className="p-4 border-t bg-slate-50/50 dark:bg-slate-950/50">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex gap-2"
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Hỏi về ứng viên (VD: Liệt kê top 3 ứng viên Java)..."
            disabled={isLoading}
            className="flex-1 bg-white dark:bg-slate-900 shadow-sm"
          />
          <Button type="submit" disabled={isLoading || !input.trim()} className="shadow-sm">
            <Send className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </Button>
        </form>
      </div>
    </Card>
  );
}
