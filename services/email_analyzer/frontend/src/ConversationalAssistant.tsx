import React, { useEffect, useRef, useState } from "react";
import { apiFetch } from "./apiClient";

export type ChatTurn = { role: "user" | "assistant"; content: string };

type AssistantProvider = "openai" | "gemini" | "none";

type Props = {
  projectName: string;
  /** Aligné sur l’analyse : fenêtre IMAP ou période prédéfinie. */
  imapPeriod: string | null;
  imapDays: number;
  sessionKey: string;
  provider: AssistantProvider;
  openaiModel: string;
  geminiModel: string;
  messages: ChatTurn[];
  setMessages: React.Dispatch<React.SetStateAction<ChatTurn[]>>;
  parseResponseJson: (res: Response) => Promise<unknown>;
  parseApiError: (data: unknown) => string;
};

const INITIAL_USER_PROMPT =
  "Bonjour. Démarre la session : présente-toi brièvement, résume ce que tu retiens des derniers emails de ce projet, puis pose 2 questions ciblées pour affiner le résumé selon mes besoins (rôle, temps, priorité).";

export function ConversationalAssistant({
  projectName,
  imapPeriod,
  imapDays,
  sessionKey,
  provider,
  openaiModel,
  geminiModel,
  messages,
  setMessages,
  parseResponseJson,
  parseApiError,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const bootstrapDoneRef = useRef(false);

  useEffect(() => {
    bootstrapDoneRef.current = false;
  }, [sessionKey]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    if (provider === "none" || messages.length > 0 || bootstrapDoneRef.current) return;
    bootstrapDoneRef.current = true;

    let cancelled = false;
    async function bootstrap() {
      setBusy(true);
      setError(null);
      const userMsg: ChatTurn = { role: "user", content: INITIAL_USER_PROMPT };
      try {
        const body = {
          project_name: projectName,
          period: imapPeriod,
          days: imapDays,
          messages: [userMsg],
          assistant_provider: provider,
          openai_model: openaiModel,
          gemini_model: geminiModel,
        };
        const res = await apiFetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await parseResponseJson(res);
        if (!res.ok) throw new Error(parseApiError(data));
        const msg = (data as { message?: string }).message;
        if (!msg) throw new Error("Réponse vide");
        if (!cancelled) {
          setMessages([userMsg, { role: "assistant", content: msg }]);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Erreur");
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [
    sessionKey,
    provider,
    messages.length,
    projectName,
    openaiModel,
    geminiModel,
    parseResponseJson,
    parseApiError,
    setMessages,
    imapPeriod,
    imapDays,
  ]);

  async function sendUserMessage(content: string) {
    const trimmed = content.trim();
    if (!trimmed || provider === "none" || busy) return;

    const userMsg: ChatTurn = { role: "user", content: trimmed };
    const nextThread = [...messages, userMsg];
    const rollback = messages;
    setMessages(nextThread);
    setInput("");
    setBusy(true);
    setError(null);

    try {
      const body = {
        project_name: projectName,
        period: imapPeriod,
        days: imapDays,
        messages: nextThread,
        assistant_provider: provider,
        openai_model: openaiModel,
        gemini_model: geminiModel,
      };
      const res = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await parseResponseJson(res);
      if (!res.ok) throw new Error(parseApiError(data));
      const msg = (data as { message?: string }).message;
      if (!msg) throw new Error("Réponse vide");
      setMessages([...nextThread, { role: "assistant", content: msg }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
      setMessages(rollback);
    } finally {
      setBusy(false);
    }
  }

  if (provider === "none") {
    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50/90 px-4 py-3 text-sm text-amber-950">
        <p className="font-medium">Assistant conversationnel indisponible</p>
        <p className="mt-1 text-amber-900/90">
          Choisissez <strong>OpenAI</strong> ou <strong>Gemini</strong> dans le formulaire de gauche,
          puis relancez l&apos;analyse pour activer le chat.
        </p>
      </div>
    );
  }

  return (
    <div className="flex min-h-[min(60vh,480px)] flex-col rounded-2xl border border-stone-200/80 bg-white shadow-sm shadow-stone-200/50">
      <div className="border-b border-stone-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-900">Assistant projet</h2>
        <p className="mt-0.5 text-xs text-slate-500">
          Synthèse orientée impact — posez des précisions, l&apos;assistant adapte le résumé.
        </p>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && busy ? (
          <p className="text-center text-sm text-slate-500">Connexion à l&apos;assistant…</p>
        ) : null}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[92%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-slate-800 text-white"
                  : "border border-stone-200 bg-stone-50 text-slate-800"
              }`}
            >
              <pre className="whitespace-pre-wrap font-sans">{m.content}</pre>
            </div>
          </div>
        ))}

        {busy && messages.length > 0 ? (
          <p className="text-center text-xs text-slate-400">Réponse en cours…</p>
        ) : null}

        {error ? (
          <div
            role="alert"
            className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-900"
          >
            {error}
          </div>
        ) : null}

        <div ref={bottomRef} />
      </div>

      <div className="border-t border-stone-100 p-3">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void sendUserMessage(input);
              }
            }}
            rows={2}
            placeholder="Votre message… (Entrée pour envoyer, Maj+Entrée pour une ligne)"
            disabled={busy}
            className="min-h-[44px] flex-1 resize-y rounded-xl border border-stone-200 bg-stone-50/50 px-3 py-2 text-sm outline-none ring-slate-300 placeholder:text-slate-400 focus:border-slate-300 focus:bg-white focus:ring-2 disabled:opacity-50"
          />
          <button
            type="button"
            onClick={() => void sendUserMessage(input)}
            disabled={busy || !input.trim()}
            className="self-end rounded-xl bg-slate-800 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Envoyer
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConversationalAssistant;
