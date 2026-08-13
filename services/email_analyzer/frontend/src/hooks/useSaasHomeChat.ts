import { useEffect, useRef, useState } from "react";
import {
  createConversation,
  fetchConversations,
  fetchMessages,
  streamChatMessage,
  type ChatTurn,
  type Conversation,
} from "../assistantApi";
import type { HomeMessage } from "../components/chat/messageTypes";

function toHomeMessage(turn: ChatTurn): HomeMessage {
  return { id: crypto.randomUUID(), role: turn.role, kind: "text", content: turn.content };
}

// Libellés affichés pendant qu'un tool tourne (voir onToolCall ci-dessous) —
// le flux SSE porte déjà le nom de l'outil (assistantApi.ts::ChatStreamEvent),
// jusqu'ici jeté au profit d'un texte générique unique "Recherche en cours…".
// Un outil absent de cette table (nouvel outil ajouté côté backend sans mise
// à jour ici) retombe sur ce même texte générique, jamais une erreur.
const TOOL_LABELS: Record<string, string> = {
  search_projects: "Recherche parmi vos projets…",
  search_memory: "Consultation de vos souvenirs…",
  find_contact: "Recherche d'un contact…",
  list_meetings: "Consultation de l'agenda…",
  list_tickets: "Vérification des tickets…",
  list_objectives: "Vérification de vos objectifs…",
  summarize_project: "Relecture du résumé du projet…",
  search_emails: "Consultation des emails…",
  search_documents: "Recherche dans vos documents…",
  draft_reply: "Préparation d'un brouillon de réponse…",
};

function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? "Recherche en cours…";
}

// Câblage backend de l'accueil chat-first en mode SaaS : porte le
// chargement/envoi de PortfolioAssistant.tsx (/api/assistant/conversations,
// persistant via les tables conversations/assistant_messages), avec
// plusieurs conversations possibles (bouton "nouvelle conversation" +
// panneau d'historique), contrairement au chat legacy qui repart de zéro
// à chaque analyse. `enabled=false` (mode legacy) n'effectue aucun appel
// réseau : ces routes répondent 503 sans SaaS
// (api/routers/assistant.py::_require_saas), et les deux hooks de mode
// sont toujours instanciés côté ChatHome (les hooks React ne peuvent pas
// être conditionnels).
export function useSaasHomeChat(enabled: boolean, initialQuestion?: string) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<HomeMessage[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const initialQuestionSentRef = useRef(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    void (async () => {
      try {
        let list = await fetchConversations();
        if (list.length === 0) {
          list = [await createConversation()];
        }
        if (cancelled) return;
        setConversations(list);
        const active = list[0];
        setActiveConversationId(active.id);
        const rows = await fetchMessages(active.id);
        if (!cancelled) setMessages(rows.map(toHomeMessage));
      } catch {
        if (!cancelled) setMessages([]);
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  async function send(text: string) {
    const trimmed = text.trim();
    if (!enabled || !trimmed || busy || !activeConversationId) return;
    const conversationId = activeConversationId;
    const userMsg = toHomeMessage({ role: "user", content: trimmed });
    setMessages((prev) => [...prev, userMsg]);
    setBusy(true);

    // Placeholder assistant créé au premier token reçu (pas immédiatement,
    // pour ne pas afficher une bulle vide redondante avec l'indicateur
    // "En cours…" déjà affiché tant que `busy` est vrai) puis rempli de
    // façon incrémentale — vrai streaming token par token, plus de faux
    // effet machine-à-écrire côté ChatThread.
    let assistantId: string | null = null;
    let assistantContent = "";
    let statusId: string | null = null;
    let pendingTools = 0;

    function ensureAssistantMessage(): string {
      if (assistantId) return assistantId;
      const id = crypto.randomUUID();
      assistantId = id;
      setMessages((prev) => [...prev, { id, role: "assistant", kind: "text", content: "" }]);
      return id;
    }

    try {
      await streamChatMessage(conversationId, trimmed, {
        onToken: (delta) => {
          const id = ensureAssistantMessage();
          assistantContent += delta;
          const content = assistantContent;
          setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, content } : m)));
        },
        onToolCall: (name, status) => {
          if (status === "start") {
            pendingTools += 1;
            const label = toolLabel(name);
            if (!statusId) {
              const id = crypto.randomUUID();
              statusId = id;
              setMessages((prev) => [
                ...prev,
                { id, role: "assistant", kind: "status", content: label },
              ]);
            } else {
              const id = statusId;
              setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, content: label } : m)));
            }
          } else {
            pendingTools = Math.max(0, pendingTools - 1);
            if (pendingTools === 0 && statusId) {
              const id = statusId;
              statusId = null;
              setMessages((prev) => prev.filter((m) => m.id !== id));
            }
          }
        },
        onDone: () => {
          setConversations((prev) =>
            prev
              .map((c) =>
                c.id === conversationId ? { ...c, updated_at: new Date().toISOString() } : c,
              )
              .sort((a, b) => b.updated_at.localeCompare(a.updated_at)),
          );
        },
        onError: (detail) => {
          setMessages((prev) => [
            ...prev.filter((m) => m.id !== userMsg.id && m.id !== assistantId && m.id !== statusId),
            { id: crypto.randomUUID(), role: "assistant", kind: "error", content: detail },
          ]);
        },
      });
    } catch (e) {
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== userMsg.id && m.id !== assistantId && m.id !== statusId),
        {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "error",
          content: e instanceof Error ? e.message : "Erreur",
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function newConversation() {
    if (!enabled) return;
    const conversation = await createConversation();
    setConversations((prev) => [conversation, ...prev]);
    setActiveConversationId(conversation.id);
    setMessages([]);
  }

  async function switchConversation(id: string) {
    if (!enabled || id === activeConversationId) return;
    setActiveConversationId(id);
    setLoaded(false);
    try {
      const rows = await fetchMessages(id);
      setMessages(rows.map(toHomeMessage));
    } finally {
      setLoaded(true);
    }
  }

  useEffect(() => {
    if (loaded && initialQuestion && !initialQuestionSentRef.current) {
      initialQuestionSentRef.current = true;
      void send(initialQuestion);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded]);

  return {
    messages,
    busy: busy || !loaded,
    loaded,
    send,
    conversations,
    activeConversationId,
    newConversation,
    switchConversation,
  };
}
