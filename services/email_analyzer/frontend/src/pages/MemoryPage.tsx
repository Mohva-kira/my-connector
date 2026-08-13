import { Navigate } from "react-router-dom";
import { getAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import { useSaasSession } from "../SaasPanels";
import { useMemoryAndKnowledge } from "../hooks/useMemoryAndKnowledge";
import type { Contact } from "../knowledgeApi";

const CATEGORY_LABELS: Record<string, string> = {
  preference: "Préférence",
  fact: "Fait",
  context: "Contexte",
};

function groupContactsByOrganization(contacts: Contact[]) {
  const groups = new Map<string, Contact[]>();
  for (const contact of contacts) {
    const key = contact.organization_id ?? "__none__";
    const list = groups.get(key) ?? [];
    list.push(contact);
    groups.set(key, list);
  }
  return groups;
}

// Page "Mémoire & Contacts" — première surface frontend pour Memory (Phase 5)
// et Knowledge (Phase 6), qui n'avaient jusqu'ici aucune UI (confirmé : aucune
// référence à /api/memory ou /api/knowledge dans frontend/src avant cette
// unité). Ces deux piliers influencent déjà le chat portefeuille en silence
// depuis Unit 48 (ContextService) — cette page les rend simplement visibles,
// en lecture seule sauf pour la suppression d'un souvenir (DELETE /api/memory/{id}
// existe déjà côté backend, jamais appelé depuis le frontend jusqu'ici).
export default function MemoryPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);
  const { memories, contacts, organizations, loading, error, removeMemory } =
    useMemoryAndKnowledge(saasEnabled);

  if (!saasEnabled) {
    return <Navigate to="/" replace />;
  }

  if (!getAccessToken()) {
    return <Navigate to="/login" replace />;
  }

  if (!me) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-bg-primary px-4">
        <p className="text-sm text-text-muted">Chargement du compte…</p>
      </div>
    );
  }

  const orgNameById = new Map(organizations.map((o) => [o.id, o.name]));
  const contactGroups = groupContactsByOrganization(contacts);

  return (
    <AppShell me={me}>
      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">
          Mémoire &amp; Contacts
        </h1>
        <p className="mt-1 text-sm text-text-muted">
          Ce que l'assistant a retenu de vous, et les contacts détectés dans vos projets.
        </p>
      </header>

      {error ? <p className="mb-4 text-sm text-danger">{error}</p> : null}
      {loading ? <p className="text-sm text-text-muted">Chargement…</p> : null}

      {!loading ? (
        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-lg border border-border-default bg-surface p-4">
            <h2 className="mb-3 text-sm font-semibold text-text-primary">
              Souvenirs ({memories.length})
            </h2>
            {memories.length === 0 ? (
              <p className="text-sm text-text-muted">
                Aucun souvenir extrait pour l'instant.
              </p>
            ) : (
              <ul className="flex flex-col gap-2">
                {memories.map((memory) => (
                  <li
                    key={memory.id}
                    className="flex items-start justify-between gap-3 rounded-lg border border-border-default bg-bg-tertiary/40 p-3"
                  >
                    <div>
                      <span className="mb-1 inline-block rounded-full bg-accent-soft px-1.5 py-0.5 text-[11px] font-semibold text-accent-primary">
                        {CATEGORY_LABELS[memory.category] ?? memory.category}
                      </span>
                      <p className="text-sm text-text-primary">{memory.content}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => void removeMemory(memory.id)}
                      className="shrink-0 rounded-lg px-2 py-1 text-xs text-text-secondary hover:bg-bg-tertiary hover:text-danger"
                    >
                      Supprimer
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-lg border border-border-default bg-surface p-4">
            <h2 className="mb-3 text-sm font-semibold text-text-primary">
              Contacts ({contacts.length})
            </h2>
            {contacts.length === 0 ? (
              <p className="text-sm text-text-muted">Aucun contact détecté pour l'instant.</p>
            ) : (
              <ul className="flex flex-col gap-4">
                {Array.from(contactGroups.entries()).map(([orgId, group]) => (
                  <li key={orgId}>
                    <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted">
                      {orgId === "__none__" ? "Sans organisation" : orgNameById.get(orgId) ?? "—"}
                    </p>
                    <ul className="flex flex-col gap-1.5">
                      {group.map((contact) => (
                        <li
                          key={contact.id}
                          className="rounded-lg border border-border-default bg-bg-tertiary/40 px-3 py-2 text-sm text-text-primary"
                        >
                          {contact.name}
                        </li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      ) : null}
    </AppShell>
  );
}
