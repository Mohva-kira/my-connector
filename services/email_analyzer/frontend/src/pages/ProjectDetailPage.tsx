import { useEffect, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { apiFetch, getAccessToken } from "../apiClient";
import { parseApiError, parseResponseJson } from "../apiUtils";
import { updateActionStatus } from "../actionsApi";
import AppShell from "../components/AppShell";
import {
  DecisionsAndRisks,
  SENTIMENT_CLASS,
  SENTIMENT_LABEL,
  formatTimestamp,
  type ProjectListItem,
} from "../ProjectHub";
import { useSaasSession } from "../SaasPanels";
import { DetailBlock } from "./ActionsPage";

// Miroir de SuggestedActionOut/EmailOut (api/routers/projects.py,
// ProjectDetailOut) — cette page est le seul endroit qui consomme
// GET /api/projects/{id}, pas besoin d'un module API séparé (même
// convention que les appels apiFetch inline de ProjectHub.tsx).
type SuggestedActionDetail = {
  id: string;
  description: string;
  deadline: string | null;
  status: string;
  created_at: string;
  rationale: string | null;
  stakeholder: string | null;
  advice: string | null;
};

type ProjectEmailSummary = {
  id: string;
  subject: string | null;
  received_at: string | null;
  recipient_status: string;
  importance_score: number | null;
  tags: string[] | null;
  classification_score: number | null;
};

type ProjectDetail = ProjectListItem & {
  suggested_actions: SuggestedActionDetail[];
  top_emails: ProjectEmailSummary[];
};

async function fetchProjectDetail(id: string): Promise<ProjectDetail> {
  const res = await apiFetch(`/api/projects/${id}`);
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as ProjectDetail;
}

const ACTION_STATUS_LABEL: Record<string, string> = {
  pending: "En attente",
  completed: "Fait",
  dismissed: "Ignorée",
};

const RECIPIENT_LABEL: Record<string, string> = {
  direct_to: "Destinataire direct",
  cc: "En copie",
};

function formatDeadline(iso: string | null): string {
  if (!iso) return "Sans échéance";
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

function ActionDetailRow({
  action,
  onDone,
  onDismiss,
}: {
  action: SuggestedActionDetail;
  onDone: () => void;
  onDismiss: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-xl border border-border-subtle bg-surface p-3 shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start justify-between gap-3 text-left"
      >
        <div className="min-w-0">
          <p className="text-sm text-text-primary">{action.description}</p>
          <p className="mt-0.5 text-xs text-text-muted">
            {ACTION_STATUS_LABEL[action.status] ?? action.status} · {formatDeadline(action.deadline)}
          </p>
        </div>
        <span className="shrink-0 text-xs text-text-muted">{open ? "▾" : "▸"}</span>
      </button>

      {open ? (
        <div className="mt-3 space-y-3 border-t border-border-subtle pt-3">
          <DetailBlock
            label="Pourquoi cette action"
            content={action.rationale}
            fallback="Aucun contexte détaillé n'a pu être extrait pour cette action."
          />
          <DetailBlock label="Personnes concernées" content={action.stakeholder} fallback="Non précisé." />
          <DetailBlock
            label="Pour éviter que ça se reproduise"
            content={action.advice}
            fallback="Pas de conseil disponible pour cette action."
          />
          {action.status === "pending" ? (
            <div className="flex gap-2 pt-1">
              <button
                type="button"
                onClick={onDismiss}
                className="rounded-lg border border-border-default px-3 py-1.5 text-xs text-text-secondary transition hover:border-text-secondary"
              >
                Ignorer
              </button>
              <button
                type="button"
                onClick={onDone}
                className="rounded-lg bg-accent-primary px-3 py-1.5 text-xs font-medium text-white transition hover:bg-accent-hover"
              >
                Marquer comme fait
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

function EmailRow({ email }: { email: ProjectEmailSummary }) {
  return (
    <li className="flex items-start justify-between gap-3 rounded-xl border border-border-subtle bg-surface px-3 py-2.5">
      <div className="min-w-0">
        <p className="truncate text-sm text-text-primary">{email.subject ?? "(Sans objet)"}</p>
        <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-text-muted">
          <span>{formatTimestamp(email.received_at)}</span>
          <span>·</span>
          <span>{RECIPIENT_LABEL[email.recipient_status] ?? email.recipient_status}</span>
          {(email.tags ?? []).map((tag) => (
            <span
              key={tag}
              className="rounded-full bg-bg-tertiary px-2 py-0.5 text-[11px] font-medium text-text-secondary"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
      {email.importance_score !== null ? (
        <span className="shrink-0 rounded-full bg-ai-glow px-2 py-0.5 text-xs font-semibold text-ai-primary">
          {email.importance_score}
        </span>
      ) : null}
    </li>
  );
}

export default function ProjectDetailPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { projectId } = useParams<{ projectId: string }>();
  const { me } = useSaasSession(saasEnabled, sessionTick);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!me || !projectId) return;
    let cancelled = false;
    void (async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchProjectDetail(projectId);
        if (!cancelled) setProject(data);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Erreur");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [me, projectId]);

  async function handleActionStatus(actionId: string, status: "completed" | "dismissed") {
    if (!project) return;
    const rollback = project.suggested_actions;
    setProject({
      ...project,
      suggested_actions: project.suggested_actions.map((a) =>
        a.id === actionId ? { ...a, status } : a,
      ),
    });
    try {
      await updateActionStatus(actionId, status);
    } catch (e) {
      setProject({ ...project, suggested_actions: rollback });
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

  const hasDecisionsOrRisks =
    !!project?.structured_content &&
    ((project.structured_content.decisions?.length ?? 0) > 0 ||
      (project.structured_content.risques?.length ?? 0) > 0);

  return (
    <AppShell me={me}>
      <Link to="/projects" className="mb-4 inline-block text-sm text-text-muted hover:text-text-secondary">
        ← Retour aux projets
      </Link>

      {error ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          {error}
        </div>
      ) : null}

      {loading ? <p className="text-sm text-text-muted">Chargement…</p> : null}

      {!loading && project ? (
        <div className="max-w-3xl space-y-8">
          <header>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <h1 className="text-2xl font-semibold tracking-tight text-text-primary">{project.name}</h1>
              {project.sentiment ? (
                <span
                  className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold ${SENTIMENT_CLASS[project.sentiment]}`}
                >
                  {SENTIMENT_LABEL[project.sentiment]}
                </span>
              ) : (
                <span className="shrink-0 rounded-full bg-bg-tertiary px-2.5 py-1 text-xs font-medium text-text-muted">
                  Pas encore analysé
                </span>
              )}
            </div>
            <p className="mt-1 text-xs text-text-muted">
              {formatTimestamp(project.last_processed_email_timestamp)}
            </p>
          </header>

          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">Résumé</h2>
            <p className="text-sm leading-relaxed text-text-secondary">
              {project.summary_content ??
                "Aucun résumé pour l'instant — lancez un premier rafraîchissement depuis le Hub."}
            </p>
          </section>

          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
              Décisions & risques
            </h2>
            {hasDecisionsOrRisks ? (
              <DecisionsAndRisks structured={project.structured_content} defaultOpen />
            ) : (
              <p className="text-sm text-text-muted">Aucune décision ni risque identifié pour l'instant.</p>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
              Actions suggérées
              {project.suggested_actions.length > 0 ? ` (${project.suggested_actions.length})` : ""}
            </h2>
            {project.suggested_actions.length === 0 ? (
              <p className="text-sm text-text-muted">Aucune action suggérée pour ce projet.</p>
            ) : (
              <ul className="space-y-2">
                {project.suggested_actions.map((a) => (
                  <ActionDetailRow
                    key={a.id}
                    action={a}
                    onDone={() => void handleActionStatus(a.id, "completed")}
                    onDismiss={() => void handleActionStatus(a.id, "dismissed")}
                  />
                ))}
              </ul>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
              Emails les plus importants
            </h2>
            {project.top_emails.length === 0 ? (
              <p className="text-sm text-text-muted">Aucun email notable pour l'instant.</p>
            ) : (
              <ul className="space-y-2">
                {project.top_emails.map((e) => (
                  <EmailRow key={e.id} email={e} />
                ))}
              </ul>
            )}
          </section>
        </div>
      ) : null}
    </AppShell>
  );
}
