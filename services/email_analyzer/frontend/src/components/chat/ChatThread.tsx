import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { DraftResult, ProjectAnalysis } from "../../analysisTypes";
import AnalysisCard from "./AnalysisCard";
import type { HomeMessage } from "./messageTypes";

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1" aria-hidden>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="animate-typing-dot h-1.5 w-1.5 rounded-full bg-ai-primary"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </span>
  );
}

export type ChatSuggestion = { key: string; label: string; onClick: () => void };

export default function ChatThread({
  messages,
  busy,
  onSend,
  placeholder = "Votre message… (Entrée pour envoyer, Maj+Entrée pour une ligne)",
  toolsSlot,
  onWidenDays,
  onSelectThread,
  onRequestDraft,
  suggestions,
}: {
  messages: HomeMessage[];
  busy: boolean;
  onSend: (text: string) => void;
  placeholder?: string;
  toolsSlot?: React.ReactNode;
  onWidenDays?: (days: number) => void;
  onSelectThread?: (subject: string) => void;
  onRequestDraft?: (projectName: string, analysis: ProjectAnalysis) => Promise<DraftResult>;
  suggestions?: ChatSuggestion[];
}) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  function submit() {
    const trimmed = input.trim();
    if (!trimmed || busy) return;
    onSend(trimmed);
    setInput("");
  }

  const showThread = messages.length > 0;

  return (
    <div
      className={`flex flex-col rounded-lg border border-border-default bg-surface shadow-sm ${
        showThread ? "h-[min(70vh,560px)]" : ""
      }`}
    >
      {showThread ? (
        <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
          {messages.map((m) => {
            if (m.kind === "analysis") {
              return (
                <div key={m.id} className="flex justify-start">
                  <AnalysisCard
                    projectName={m.projectName}
                    analysis={m.analysis}
                    onRequestDraft={onRequestDraft ? () => onRequestDraft(m.projectName, m.analysis) : undefined}
                    onSelectThread={m.allowThreadDrilldown ? onSelectThread : undefined}
                  />
                </div>
              );
            }

            if (m.kind === "status") {
              return (
                <div key={m.id} className="space-y-2">
                  <p className="flex items-center justify-center gap-2 text-center text-xs text-text-muted">
                    <TypingDots />
                    {m.content}
                  </p>
                  {m.progress && m.progress.total > 0 ? (
                    <div className="mx-auto h-2 w-full max-w-[92%] overflow-hidden rounded-full bg-bg-tertiary">
                      <div
                        className="h-full rounded-full bg-ai-primary transition-all duration-500 ease-out"
                        style={{ width: `${Math.min(100, (m.progress.processed / m.progress.total) * 100)}%` }}
                      />
                    </div>
                  ) : null}
                </div>
              );
            }

            if (m.kind === "empty-report") {
              return (
                <div key={m.id} className="flex justify-start">
                  <div className="max-w-[92%] rounded-lg border border-border-default bg-surface p-4 shadow-sm">
                    <p className="text-sm font-semibold text-text-primary">Aucun email trouvé</p>
                    <p className="mt-1.5 text-sm text-text-secondary">
                      {m.info.message || "La recherche IMAP n'a retourné aucun message sur la période choisie."}
                    </p>
                    {m.info.imapFolder || m.info.daysBack ? (
                      <p className="mt-1.5 text-xs text-text-muted">
                        {m.info.imapFolder ? <>Dossier IMAP : {m.info.imapFolder}. </> : null}
                        {m.info.daysBack ? (
                          <>
                            Fenêtre IMAP : {m.info.daysBack} jour{m.info.daysBack > 1 ? "s" : ""}.
                          </>
                        ) : null}
                      </p>
                    ) : null}
                    {m.info.widenDayOptions.length > 0 && onWidenDays ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {m.info.widenDayOptions.map((n) => (
                          <button
                            key={n}
                            type="button"
                            onClick={() => onWidenDays(n)}
                            disabled={busy}
                            className="rounded-full border border-border-default px-3 py-1.5 text-xs font-medium text-text-secondary transition hover:bg-bg-tertiary disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            Chercher sur {n} jours
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              );
            }

            if (m.kind === "error") {
              return (
                <div key={m.id} role="alert" className="flex justify-start">
                  <div className="max-w-[92%] rounded-2xl border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
                    {m.content}
                  </div>
                </div>
              );
            }

            return (
              <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[92%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-accent-primary text-white"
                      : "border border-border-default bg-bg-tertiary text-text-primary"
                  }`}
                >
                  {m.role === "assistant" ? (
                    <div className="markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <pre className="whitespace-pre-wrap font-sans">{m.content}</pre>
                  )}
                </div>
              </div>
            );
          })}

          {busy ? (
            <p className="flex items-center justify-center gap-2 text-center text-xs text-text-muted">
              <TypingDots />
              En cours…
            </p>
          ) : null}

          <div ref={bottomRef} />
        </div>
      ) : null}

      <div className={`p-3 ${showThread ? "border-t border-border-subtle" : ""}`}>
        {suggestions && suggestions.length > 0 ? (
          <div className="mb-2 flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <button
                key={s.key}
                type="button"
                onClick={s.onClick}
                disabled={busy}
                className="max-w-[260px] truncate rounded-full border border-border-default bg-bg-tertiary/60 px-3 py-1.5 text-xs text-text-secondary transition hover:bg-bg-tertiary hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-50"
                title={s.label}
              >
                {s.label}
              </button>
            ))}
          </div>
        ) : null}
        <div className="flex gap-2">
          {toolsSlot}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={4}
            placeholder={placeholder}
            disabled={busy}
            className="min-h-[110px] flex-1 resize-y rounded-xl border border-border-default bg-bg-tertiary/60 px-4 py-3 text-base text-text-primary outline-none ring-accent-primary placeholder:text-text-muted focus:border-accent-primary focus:bg-bg-tertiary focus:ring-2 disabled:opacity-50"
          />
          <button
            type="button"
            onClick={submit}
            disabled={busy || !input.trim()}
            className="self-end rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            Envoyer
          </button>
        </div>
      </div>
    </div>
  );
}
