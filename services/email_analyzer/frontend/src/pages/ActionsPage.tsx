import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import { fetchActions, updateActionStatus, type ActionOut } from "../actionsApi";
import { getAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import { useSaasSession } from "../SaasPanels";

function isToday(iso: string): boolean {
  return new Date(iso).toDateString() === new Date().toDateString();
}

function isThisWeek(iso: string): boolean {
  const diffDays = (new Date(iso).getTime() - Date.now()) / 86_400_000;
  return diffDays >= 0 && diffDays <= 7;
}

function groupActions(actions: ActionOut[]): {
  today: ActionOut[];
  week: ActionOut[];
  later: ActionOut[];
} {
  const today: ActionOut[] = [];
  const week: ActionOut[] = [];
  const later: ActionOut[] = [];
  for (const a of actions) {
    if (a.deadline && isToday(a.deadline)) today.push(a);
    else if (a.deadline && isThisWeek(a.deadline)) week.push(a);
    else later.push(a);
  }
  return { today, week, later };
}

function formatDeadline(iso: string): string {
  try {
    return new Intl.DateTimeFormat("fr-FR", { day: "2-digit", month: "short" }).format(new Date(iso));
  } catch {
    return iso;
  }
}

function ActionRow({
  action,
  onDone,
  onDismiss,
  onOpen,
}: {
  action: ActionOut;
  onDone: () => void;
  onDismiss: () => void;
  onOpen: () => void;
}) {
  return (
    <li className="flex items-start gap-3 rounded-xl border border-border-subtle bg-surface px-3 py-2.5 shadow-sm transition hover:border-accent-primary/30">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDone();
        }}
        aria-label="Marquer comme fait"
        title="Marquer comme fait"
        className="mt-0.5 h-4 w-4 flex-none rounded border border-border-default transition hover:border-success hover:bg-success/10"
      />
      <button
        type="button"
        onClick={onOpen}
        className="min-w-0 flex-1 text-left"
        aria-haspopup="dialog"
      >
        <p className="text-sm text-text-primary">{action.description}</p>
        <p className="mt-0.5 text-xs text-text-muted">
          {action.project_name}
          {action.deadline ? ` · ${formatDeadline(action.deadline)}` : ""}
        </p>
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDismiss();
        }}
        className="flex-none text-xs text-text-muted transition hover:text-text-secondary"
      >
        Ignorer
      </button>
    </li>
  );
}

function ActionGroup({
  title,
  actions,
  onDone,
  onDismiss,
  onOpen,
}: {
  title: string;
  actions: ActionOut[];
  onDone: (id: string) => void;
  onDismiss: (id: string) => void;
  onOpen: (action: ActionOut) => void;
}) {
  if (actions.length === 0) return null;
  return (
    <section>
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">{title}</h2>
      <ul className="space-y-2">
        {actions.map((a) => (
          <ActionRow
            key={a.id}
            action={a}
            onDone={() => onDone(a.id)}
            onDismiss={() => onDismiss(a.id)}
            onOpen={() => onOpen(a)}
          />
        ))}
      </ul>
    </section>
  );
}

export function DetailBlock({ label, content, fallback }: { label: string; content: string | null; fallback: string }) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">{label}</h3>
      <p className={`mt-1 text-sm leading-relaxed ${content ? "text-text-primary" : "text-text-muted italic"}`}>
        {content || fallback}
      </p>
    </div>
  );
}

function ActionDetailModal({
  action,
  onClose,
  onDone,
  onDismiss,
}: {
  action: ActionOut;
  onClose: () => void;
  onDone: () => void;
  onDismiss: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="action-detail-title"
      onClick={onClose}
    >
      <div
        className="animate-modal-in w-full max-w-lg rounded-2xl bg-surface p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="action-detail-title" className="text-lg font-semibold text-text-primary">
          {action.description}
        </h2>
        <p className="mt-1 text-xs text-text-muted">
          {action.project_name}
          {action.deadline ? ` · Échéance : ${formatDeadline(action.deadline)}` : ""}
        </p>

        <div className="mt-5 space-y-4 border-t border-border-subtle pt-4">
          <DetailBlock
            label="Pourquoi cette action"
            content={action.rationale}
            fallback="Aucun contexte détaillé n'a pu être extrait pour cette action."
          />
          <DetailBlock
            label="Personnes concernées"
            content={action.stakeholder}
            fallback="Non précisé."
          />
          <DetailBlock
            label="Pour éviter que ça se reproduise"
            content={action.advice}
            fallback="Pas de conseil disponible pour cette action."
          />
        </div>

        <div className="mt-6 flex items-center justify-end gap-3 border-t border-border-subtle pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl px-4 py-2 text-sm text-text-secondary transition hover:text-text-primary"
          >
            Fermer
          </button>
          <button
            type="button"
            onClick={() => {
              onDismiss();
              onClose();
            }}
            className="rounded-xl border border-border-default px-4 py-2 text-sm text-text-secondary transition hover:border-text-secondary"
          >
            Ignorer
          </button>
          <button
            type="button"
            onClick={() => {
              onDone();
              onClose();
            }}
            className="rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover"
          >
            Marquer comme fait
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ActionsPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);
  const [actions, setActions] = useState<ActionOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAction, setSelectedAction] = useState<ActionOut | null>(null);

  useEffect(() => {
    if (!me) return;
    let cancelled = false;
    void (async () => {
      setLoading(true);
      setError(null);
      try {
        const rows = await fetchActions("pending");
        if (!cancelled) setActions(rows);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Erreur");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [me]);

  async function handleStatus(id: string, status: "completed" | "dismissed") {
    const rollback = actions;
    setActions((prev) => prev.filter((a) => a.id !== id));
    try {
      await updateActionStatus(id, status);
    } catch (e) {
      setActions(rollback);
      setError(e instanceof Error ? e.message : "Erreur");
    }
  }

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

  const { today, week, later } = groupActions(actions);

  return (
    <AppShell me={me}>
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Actions</h1>
        <p className="mt-1 text-sm text-text-muted">
          Les actions que je vous recommande, tous projets confondus.
        </p>
      </header>

      {error ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          {error}
        </div>
      ) : null}

      {loading ? (
        <p className="text-sm text-text-muted">Chargement…</p>
      ) : actions.length === 0 ? (
        <p className="text-sm text-text-muted">Aucune action en attente — tout est à jour.</p>
      ) : (
        <div className="max-w-2xl space-y-6">
          <ActionGroup
            title="Aujourd'hui"
            actions={today}
            onDone={(id) => void handleStatus(id, "completed")}
            onDismiss={(id) => void handleStatus(id, "dismissed")}
            onOpen={setSelectedAction}
          />
          <ActionGroup
            title="Cette semaine"
            actions={week}
            onDone={(id) => void handleStatus(id, "completed")}
            onDismiss={(id) => void handleStatus(id, "dismissed")}
            onOpen={setSelectedAction}
          />
          <ActionGroup
            title="Plus tard"
            actions={later}
            onDone={(id) => void handleStatus(id, "completed")}
            onDismiss={(id) => void handleStatus(id, "dismissed")}
            onOpen={setSelectedAction}
          />
        </div>
      )}

      {selectedAction ? (
        <ActionDetailModal
          action={selectedAction}
          onClose={() => setSelectedAction(null)}
          onDone={() => void handleStatus(selectedAction.id, "completed")}
          onDismiss={() => void handleStatus(selectedAction.id, "dismissed")}
        />
      ) : null}
    </AppShell>
  );
}
