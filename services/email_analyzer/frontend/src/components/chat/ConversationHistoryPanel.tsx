import { formatSince } from "../../hooks/useBrief";
import type { Conversation } from "../../assistantApi";

// Panneau glissant depuis la droite listant les conversations de
// l'assistant portefeuille (bouton "Historique" dans ChatHome, mode SaaS
// uniquement) — même convention d'overlay que DiscoverProjectsModal.tsx
// (fixed inset-0 + backdrop), mais ancré à droite car c'est une liste et
// non un formulaire.
export default function ConversationHistoryPanel({
  conversations,
  activeId,
  onSelect,
  onClose,
}: {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/40"
      role="dialog"
      aria-modal="true"
      aria-label="Historique des conversations"
      onClick={onClose}
    >
      <div
        className="flex h-full w-full max-w-sm flex-col gap-2 overflow-y-auto bg-surface p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text-primary">Conversations</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Fermer"
            className="rounded-lg px-2 py-1 text-text-muted hover:bg-bg-tertiary hover:text-text-primary"
          >
            ✕
          </button>
        </div>

        {conversations.length === 0 ? (
          <p className="text-sm text-text-muted">Aucune conversation.</p>
        ) : (
          conversations.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => onSelect(c.id)}
              className={`rounded-lg px-3 py-2 text-left transition ${
                c.id === activeId
                  ? "bg-accent-soft text-accent-primary"
                  : "text-text-secondary hover:bg-bg-tertiary"
              }`}
            >
              <p className="truncate text-sm font-medium">{c.title}</p>
              <p className="text-xs text-text-muted">{formatSince(c.updated_at)}</p>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
