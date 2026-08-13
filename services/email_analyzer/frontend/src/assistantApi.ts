import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de ConversationOut/AssistantMessageOut/ChatResponse
// (api/routers/assistant.py) — multi-conversation pour l'assistant
// portefeuille (bouton "nouvelle conversation" + panneau d'historique).
export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ChatTurn = { role: "user" | "assistant"; content: string };

export async function fetchConversations(): Promise<Conversation[]> {
  const res = await apiFetch("/api/assistant/conversations");
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Conversation[];
}

export async function createConversation(): Promise<Conversation> {
  const res = await apiFetch("/api/assistant/conversations", { method: "POST" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Conversation;
}

export async function fetchMessages(conversationId: string): Promise<ChatTurn[]> {
  const res = await apiFetch(`/api/assistant/conversations/${conversationId}/messages`);
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as ChatTurn[];
}

// Événements du flux SSE émis par POST .../chat (voir api/sse.py,
// ConversationService.stream_message) — un "tool_call" (status "start")
// précède toujours le "tool_result" (status "done"|"error") correspondant.
export type ChatStreamEvent =
  | { type: "token"; delta: string }
  | { type: "tool_call"; name: string; status: "start" }
  | { type: "tool_result"; name: string; status: "done" | "error" }
  | { type: "done"; message: string }
  | { type: "error"; erreur: string };

export type ChatStreamHandlers = {
  onToken: (delta: string) => void;
  onToolCall: (name: string, status: "start" | "done" | "error") => void;
  onDone: (fullMessage: string) => void;
  onError: (detail: string) => void;
};

function dispatchSseBlock(block: string, handlers: ChatStreamHandlers): void {
  const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
  if (!dataLine) return;
  let event: ChatStreamEvent;
  try {
    event = JSON.parse(dataLine.slice("data:".length).trim()) as ChatStreamEvent;
  } catch {
    return;
  }
  switch (event.type) {
    case "token":
      handlers.onToken(event.delta);
      break;
    case "tool_call":
      handlers.onToolCall(event.name, "start");
      break;
    case "tool_result":
      handlers.onToolCall(event.name, event.status);
      break;
    case "done":
      handlers.onDone(event.message);
      break;
    case "error":
      handlers.onError(event.erreur);
      break;
  }
}

/**
 * Remplace l'ancien `sendChatMessage` (JSON unique, plus de streaming) :
 * la route retourne désormais un flux `text/event-stream`. Pas d'EventSource
 * (ne supporte ni corps POST ni en-tête Authorization) — lecture manuelle du
 * ReadableStream, `apiFetch` fournit déjà l'en-tête Authorization.
 */
export async function streamChatMessage(
  conversationId: string,
  message: string,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const res = await apiFetch(`/api/assistant/conversations/${conversationId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Pas de assistant_provider : le serveur choisit le fournisseur dont la
    // clé API est configurée (api/routers/assistant.py::_select_provider).
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    const data = await parseResponseJson(res);
    handlers.onError(parseApiError(data));
    return;
  }
  if (!res.body) {
    handlers.onError("Réponse vide (streaming non supporté par ce navigateur).");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      dispatchSseBlock(buffer.slice(0, boundary), handlers);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim()) dispatchSseBlock(buffer, handlers);
}
