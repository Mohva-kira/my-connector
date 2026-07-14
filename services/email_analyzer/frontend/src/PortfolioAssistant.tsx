import React, { useEffect, useRef, useState } from "react";
import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

type ChatTurn = { role: "user" | "assistant"; content: string };

const BOOTSTRAP_GREETING: ChatTurn = {
  role: "assistant",
  content:
    "Bonjour ! Je suis votre assistante. Posez-moi une question sur vos projets — " +
    "décisions en attente, risques, échéances — je réponds à partir de ce que j'ai analysé.",
};

// Messages courts pendant l'attente d'une réponse (cf. context/ui-context.md :
// retour vivant sans bruit visuel, même discipline que ConversationalAssistant).
const STATUS_MESSAGES = ["Je consulte vos projets…", "Je prépare une réponse…"];

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

// Assistant permanent, portefeuille (tous projets) — contrairement à
// ConversationalAssistant.tsx (scoped à un seul projet/une analyse, historique
// vidé à chaque nouvelle analyse), celui-ci charge/persiste son historique via
// /api/assistant/messages et /api/assistant/chat (email_analyzer/db/models.py
// ::AssistantMessage), donc survit aux rechargements de page.
export default function PortfolioAssistant({
  initialQuestion,
}: {
  // Question déjà tapée ailleurs (barre de recherche de la sidebar, voir
  // components/AppShell.tsx) — envoyée automatiquement une fois l'historique
  // chargé, sans que l'utilisateur ait à la retaper dans le champ de chat.
  initialQuestion?: string;
} = {}) {
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusIndex, setStatusIndex] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await apiFetch("/api/assistant/messages");
        const data = await parseResponseJson(res);
        if (!res.ok) throw new Error(parseApiError(data));
        const rows = data as ChatTurn[];
        if (!cancelled) setMessages(rows.length > 0 ? rows : [BOOTSTRAP_GREETING]);
      } catch {
        if (!cancelled) setMessages([BOOTSTRAP_GREETING]);
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (loaded && initialQuestion) void sendMessage(initialQuestion);
    // Ne dépend que de `loaded` : initialQuestion ne doit être envoyée
    // qu'une fois, même si le composant re-render ensuite.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    if (!busy) {
      setStatusIndex(0);
      return;
    }
    const id = setInterval(() => setStatusIndex((i) => (i + 1) % STATUS_MESSAGES.length), 1400);
    return () => clearInterval(id);
  }, [busy]);

  async function sendMessage(content: string) {
    const trimmed = content.trim();
    if (!trimmed || busy) return;
    const userMsg: ChatTurn = { role: "user", content: trimmed };
    const rollback = messages;
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const res = await apiFetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Pas de assistant_provider : le serveur choisit le fournisseur dont
        // la clé API est configurée (voir api/routers/assistant.py::_select_provider).
        body: JSON.stringify({ message: trimmed }),
      });
      const data = await parseResponseJson(res);
      if (!res.ok) throw new Error(parseApiError(data));
      const reply = (data as { message?: string }).message;
      if (!reply) throw new Error("Réponse vide");
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
      setMessages(rollback);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-[420px] flex-col rounded-2xl border border-border-default bg-surface shadow-sm lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)]">
      <div className="border-b border-border-subtle px-4 py-3">
        <h2 className="text-sm font-semibold text-text-primary">Assistant</h2>
        <p className="mt-0.5 text-xs text-text-muted">
          Toujours disponible — posez une question sur l&apos;ensemble de vos projets.
        </p>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {!loaded ? (
          <p className="flex items-center justify-center gap-2 text-center text-sm text-text-muted">
            <TypingDots /> Chargement de la conversation…
          </p>
        ) : null}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[92%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-accent-primary text-white"
                  : "border border-border-default bg-bg-tertiary text-text-primary"
              }`}
            >
              <pre className="whitespace-pre-wrap font-sans">{m.content}</pre>
            </div>
          </div>
        ))}
        {busy ? (
          <p className="flex items-center justify-center gap-2 text-center text-xs text-text-muted">
            <TypingDots /> {STATUS_MESSAGES[statusIndex]}
          </p>
        ) : null}
        {error ? (
          <div
            role="alert"
            className="rounded-xl border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger"
          >
            {error}
          </div>
        ) : null}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border-subtle p-3">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void sendMessage(input);
              }
            }}
            rows={2}
            placeholder="Posez une question sur vos projets… (Entrée pour envoyer)"
            disabled={busy}
            className="min-h-[44px] flex-1 resize-y rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2 text-sm text-text-primary outline-none ring-accent-primary placeholder:text-text-muted focus:border-accent-primary focus:bg-bg-tertiary focus:ring-2 disabled:opacity-50"
          />
          <button
            type="button"
            onClick={() => void sendMessage(input)}
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
