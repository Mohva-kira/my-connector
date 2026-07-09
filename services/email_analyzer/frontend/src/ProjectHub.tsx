import React, { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Polling du job de rafraîchissement Fast-Track : même endpoint/cadence que
// l'analyse pleine (HomePage.tsx) — /api/analyze/{job_id} sert les deux types
// de jobs (voir jobs.py, générique par job_id).
const REFRESH_POLL_INTERVAL_MS = 2500;
const REFRESH_POLL_TIMEOUT_MS = 2 * 60 * 1000;

type Sentiment = "on_track" | "under_tension" | "awaiting_feedback";

// Signaux de classification multi-critères (email_analyzer/classification.py) —
// toutes les clés sont optionnelles côté backend (ProjectRules.from_dict tolère
// un JSON incomplet ou absent).
export type RulesMatrix = {
  keywords?: string[];
  sender_domains?: string[];
  sender_emails?: string[];
  client_names?: string[];
  company_names?: string[];
  reference_numbers?: string[];
};

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
  rules_matrix: RulesMatrix | null;
};

// Un champ par signal de classification (email_analyzer/classification.py,
// ProjectRules) — pilote à la fois le formulaire de création et l'éditeur.
const RULES_MATRIX_FIELDS: Array<{ key: keyof RulesMatrix; label: string; placeholder: string }> = [
  { key: "keywords", label: "Mots-clés", placeholder: "Ex. refonte, API paiement" },
  { key: "sender_domains", label: "Domaines expéditeur", placeholder: "Ex. client.com" },
  { key: "sender_emails", label: "Adresses expéditeur", placeholder: "Ex. contact@client.com" },
  { key: "client_names", label: "Noms de clients", placeholder: "Ex. Société X" },
  { key: "company_names", label: "Noms de sociétés", placeholder: "Ex. Alias Corp" },
  { key: "reference_numbers", label: "Références internes", placeholder: "Ex. PRJ-1234" },
];

function parseCsv(value: string): string[] {
  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function rulesMatrixToCsv(rules: RulesMatrix | null | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  for (const { key } of RULES_MATRIX_FIELDS) {
    out[key] = (rules?.[key] ?? []).join(", ");
  }
  return out;
}

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

function RulesMatrixEditor({
  project,
  onUpdated,
}: {
  project: ProjectListItem;
  onUpdated: (id: string, patch: Partial<ProjectListItem>) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [values, setValues] = useState<Record<string, string>>(() => rulesMatrixToCsv(project.rules_matrix));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSave() {
    setError(null);
    setBusy(true);
    try {
      const rules_matrix: RulesMatrix = {};
      for (const { key } of RULES_MATRIX_FIELDS) {
        const parsed = parseCsv(values[key] ?? "");
        if (parsed.length > 0) rules_matrix[key] = parsed;
      }
      const res = await apiFetch(`/api/projects/${project.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rules_matrix }),
      });
      const data = (await parseResponseJson(res)) as Record<string, unknown>;
      if (!res.ok) throw new Error(parseApiError(data));
      onUpdated(project.id, { rules_matrix: (data.rules_matrix as RulesMatrix | null) ?? rules_matrix });
      setExpanded(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
    } finally {
      setBusy(false);
    }
  }

  if (!expanded) {
    return (
      <button
        type="button"
        onClick={() => setExpanded(true)}
        className="self-start text-xs font-medium text-text-muted underline decoration-dotted hover:text-text-secondary"
      >
        Règles de classification
      </button>
    );
  }

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-border-default bg-bg-tertiary/40 p-3">
      {RULES_MATRIX_FIELDS.map(({ key, label, placeholder }) => (
        <label key={key} className="text-xs text-text-secondary">
          {label}
          <input
            className="mt-1 w-full rounded-lg border border-border-default bg-surface px-2 py-1.5 text-xs text-text-primary"
            value={values[key] ?? ""}
            onChange={(e) => setValues((prev) => ({ ...prev, [key]: e.target.value }))}
            placeholder={placeholder}
          />
        </label>
      ))}
      {error ? <p className="text-xs text-danger">{error}</p> : null}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => void handleSave()}
          disabled={busy}
          className="rounded-lg bg-accent-primary px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy ? "Enregistrement…" : "Enregistrer"}
        </button>
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="rounded-lg px-3 py-1.5 text-xs text-text-secondary"
        >
          Annuler
        </button>
      </div>
    </div>
  );
}

function ProjectCard({
  project,
  onRefreshed,
  onUpdated,
}: {
  project: ProjectListItem;
  onRefreshed: (id: string, patch: Partial<ProjectListItem>) => void;
  onUpdated: (id: string, patch: Partial<ProjectListItem>) => void;
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

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void handleRefresh()}
          disabled={refreshing}
          className="mt-1 self-start rounded-full border border-info/40 px-3 py-1.5 text-xs font-medium text-info transition hover:bg-info/10 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {refreshing ? "Actualisation…" : "Actualiser (Fast-Track)"}
        </button>
      </div>

      <RulesMatrixEditor project={project} onUpdated={onUpdated} />
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
  const [showRules, setShowRules] = useState(false);
  const [rulesValues, setRulesValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setError(null);
    setBusy(true);
    try {
      const rules_matrix: RulesMatrix = {};
      for (const { key } of RULES_MATRIX_FIELDS) {
        const parsed = parseCsv(rulesValues[key] ?? "");
        if (parsed.length > 0) rules_matrix[key] = parsed;
      }
      const res = await apiFetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          ...(Object.keys(rules_matrix).length > 0 ? { rules_matrix } : {}),
        }),
      });
      const data = await parseResponseJson(res);
      if (!res.ok) throw new Error(parseApiError(data));
      onCreated(data as ProjectListItem);
      setName("");
      setRulesValues({});
      setShowRules(false);
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

      {!showRules ? (
        <button
          type="button"
          onClick={() => setShowRules(true)}
          className="w-full text-left text-xs font-medium text-text-muted underline decoration-dotted hover:text-text-secondary"
        >
          Règles de classification (optionnel)
        </button>
      ) : (
        <div className="grid w-full gap-2 sm:grid-cols-2">
          {RULES_MATRIX_FIELDS.map(({ key, label, placeholder }) => (
            <label key={key} className="text-xs text-text-secondary">
              {label}
              <input
                className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-1.5 text-xs text-text-primary"
                value={rulesValues[key] ?? ""}
                onChange={(e) => setRulesValues((prev) => ({ ...prev, [key]: e.target.value }))}
                placeholder={placeholder}
              />
            </label>
          ))}
        </div>
      )}

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
            <ProjectCard key={p.id} project={p} onRefreshed={patchProject} onUpdated={patchProject} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
