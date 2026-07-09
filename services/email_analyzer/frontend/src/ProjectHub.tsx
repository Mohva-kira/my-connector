import React, { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Polling du job de rafraîchissement Fast-Track : même endpoint/cadence que
// l'analyse pleine (HomePage.tsx) — /api/analyze/{job_id} sert les deux types
// de jobs (voir jobs.py, générique par job_id).
const REFRESH_POLL_INTERVAL_MS = 2500;
const REFRESH_POLL_TIMEOUT_MS = 2 * 60 * 1000;

type Sentiment = "on_track" | "under_tension" | "awaiting_feedback";

export type ProjectListItem = {
  id: string;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  summary_content: string | null;
  sentiment: Sentiment | null;
  last_processed_email_timestamp: string | null;
  pending_actions_count: number;
};

const SENTIMENT_LABEL: Record<Sentiment, string> = {
  on_track: "Sur la bonne voie",
  under_tension: "Sous tension",
  awaiting_feedback: "En attente de retour",
};

// Les 3 sentiments réellement produits par le backend (ProjectSentiment,
// db/models.py) correspondent déjà exactement à 3 des 5 tokens de la
// "Project Health Scale" (ui-context.md) : on_track=#22C55E=success,
// awaiting_feedback=#F59E0B=warning, under_tension=#EF4444=danger (valeurs
// identiques). Pas de nouveau token ajouté pour les 2 niveaux intermédiaires
// (Needs Attention / Delayed) : le backend ne les distingue pas aujourd'hui.
const SENTIMENT_CLASS: Record<Sentiment, string> = {
  on_track: "bg-success/10 text-success",
  under_tension: "bg-danger/10 text-danger",
  awaiting_feedback: "bg-warning/10 text-warning",
};

function formatTimestamp(iso: string | null): string {
  if (!iso) return "Jamais analysé";
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

async function pollRefreshJob(jobId: string): Promise<Record<string, unknown>> {
  const deadline = Date.now() + REFRESH_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, REFRESH_POLL_INTERVAL_MS));
    const res = await apiFetch(`/api/analyze/${jobId}`, {
      method: "GET",
      signal: AbortSignal.timeout(30_000),
    });
    const data = (await parseResponseJson(res)) as Record<string, unknown>;
    if (!res.ok) throw new Error(parseApiError(data));
    const status = data.status as string;
    if (status === "done") return (data.result ?? {}) as Record<string, unknown>;
    if (status === "error") throw new Error((data.error as string) || "Échec du rafraîchissement.");
  }
  throw new Error("Le rafraîchissement prend trop de temps. Réessayez plus tard.");
}

function ProjectCard({
  project,
  onRefreshed,
}: {
  project: ProjectListItem;
  onRefreshed: (id: string, patch: Partial<ProjectListItem>) => void;
}) {
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  async function handleRefresh() {
    setRefreshError(null);
    setRefreshing(true);
    try {
      const startRes = await apiFetch(`/api/projects/${project.id}/refresh`, { method: "POST" });
      const startData = (await parseResponseJson(startRes)) as Record<string, unknown>;
      if (!startRes.ok) throw new Error(parseApiError(startData));
      const jobId = startData.job_id as string | undefined;
      if (!jobId) throw new Error("Réponse inattendue du serveur (job_id manquant).");
      const result = await pollRefreshJob(jobId);
      onRefreshed(project.id, {
        sentiment: (result.sentiment as Sentiment | null) ?? project.sentiment,
        summary_content: (result.content as string | null) ?? project.summary_content,
        last_processed_email_timestamp:
          (result.last_processed_email_timestamp as string | null) ??
          project.last_processed_email_timestamp,
      });
    } catch (err) {
      setRefreshError(err instanceof Error ? err.message : "Erreur inconnue");
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <article
      className={`flex flex-col gap-3 rounded-2xl border p-5 shadow-sm transition ${
        refreshing
          ? "animate-fasttrack-pulse border-fasttrack-active bg-fasttrack-bg"
          : "border-border-default bg-surface"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-lg font-semibold text-text-primary">{project.name}</h3>
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

      <p className="min-h-[2.5rem] text-sm text-text-secondary">
        {project.summary_content ?? "Aucun résumé pour l'instant — lancez un premier rafraîchissement."}
      </p>

      <div className="flex items-center justify-between text-xs text-text-muted">
        <span>{formatTimestamp(project.last_processed_email_timestamp)}</span>
        {project.pending_actions_count > 0 ? (
          <span className="rounded-full bg-ai-glow px-2 py-0.5 font-semibold text-ai-primary">
            {project.pending_actions_count} action{project.pending_actions_count > 1 ? "s" : ""} en attente
          </span>
        ) : null}
      </div>

      {refreshError ? <p className="text-xs text-danger">{refreshError}</p> : null}

      <button
        type="button"
        onClick={() => void handleRefresh()}
        disabled={refreshing}
        className="mt-1 self-start rounded-full border border-info/40 px-3 py-1.5 text-xs font-medium text-info transition hover:bg-info/10 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {refreshing ? "Actualisation…" : "Actualiser (Fast-Track)"}
      </button>
    </article>
  );
}

function CreateProjectForm({
  onCreated,
  onCancel,
}: {
  onCreated: (project: ProjectListItem) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setError(null);
    setBusy(true);
    try {
      const res = await apiFetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });
      const data = await parseResponseJson(res);
      if (!res.ok) throw new Error(parseApiError(data));
      onCreated(data as ProjectListItem);
      setName("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      onSubmit={(e) => void handleSubmit(e)}
      className="mb-6 flex flex-wrap items-end gap-3 rounded-2xl border border-border-default bg-surface p-4"
    >
      <label className="flex-1 text-sm text-text-secondary">
        Nom du projet
        <input
          autoFocus
          className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-2 text-sm text-text-primary"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Ex. Refonte site client X"
          required
        />
      </label>
      <button
        type="submit"
        disabled={busy}
        className="rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
      >
        Créer
      </button>
      <button
        type="button"
        onClick={onCancel}
        className="rounded-xl px-4 py-2 text-sm text-text-secondary"
      >
        Annuler
      </button>
      {error ? <p className="w-full text-sm text-danger">{error}</p> : null}
    </form>
  );
}

export default function ProjectHub() {
  const [projects, setProjects] = useState<ProjectListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);

  const loadProjects = useCallback(async () => {
    setError(null);
    try {
      const res = await apiFetch("/api/projects");
      const data = await parseResponseJson(res);
      if (!res.ok) throw new Error(parseApiError(data));
      setProjects(data as ProjectListItem[]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
    }
  }, []);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  function patchProject(id: string, patch: Partial<ProjectListItem>) {
    setProjects((prev) => (prev ? prev.map((p) => (p.id === id ? { ...p, ...patch } : p)) : prev));
  }

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-text-muted">
          Vos projets, leur santé IA et les actions encore en attente. Rafraîchissez un projet
          sans attendre la synchronisation planifiée.
        </p>
        {!showCreateForm ? (
          <button
            type="button"
            onClick={() => setShowCreateForm(true)}
            className="shrink-0 rounded-full bg-accent-primary px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover"
          >
            + Nouveau projet
          </button>
        ) : null}
      </div>

      {showCreateForm ? (
        <CreateProjectForm
          onCancel={() => setShowCreateForm(false)}
          onCreated={(project) => {
            setProjects((prev) => (prev ? [project, ...prev] : [project]));
            setShowCreateForm(false);
          }}
        />
      ) : null}

      {error ? (
        <div role="alert" className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          {error}
        </div>
      ) : null}

      {projects === null && !error ? (
        <p className="text-sm text-text-muted">Chargement des projets…</p>
      ) : null}

      {projects !== null && projects.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border-default p-8 text-center text-sm text-text-muted">
          Aucun projet pour l&apos;instant. Créez-en un pour commencer à suivre son activité.
        </div>
      ) : null}

      {projects !== null && projects.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {projects.map((p) => (
            <ProjectCard key={p.id} project={p} onRefreshed={patchProject} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
