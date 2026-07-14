import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { apiFetch, takePendingAutoSync } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Préréglages de fenêtre pour "chercher sur plus de jours" (voir HomePage.tsx,
// même liste) — proposés quand un premier sync ne trouve rien.
const WIDEN_WINDOW_DAYS = [60, 90, 120, 240] as const;

// Polling du job de rafraîchissement Fast-Track : même endpoint/cadence que
// l'analyse pleine (HomePage.tsx) — /api/analyze/{job_id} sert les deux types
// de jobs (voir jobs.py, générique par job_id).
const REFRESH_POLL_INTERVAL_MS = 2500;
const REFRESH_POLL_TIMEOUT_MS = 2 * 60 * 1000;

export type Sentiment = "on_track" | "under_tension" | "awaiting_feedback";

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

// Miroir partiel de llm.ProjectSummaryLLM (email_analyzer/llm.py) — seuls les
// champs affichés dans la carte projet sont typés ici.
export type StructuredContent = {
  decisions?: Array<{ description: string }>;
  risques?: Array<{ description: string; niveau: string | null }>;
};

export type ProjectListItem = {
  id: string;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  summary_content: string | null;
  sentiment: Sentiment | null;
  structured_content: StructuredContent | null;
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

export const SENTIMENT_LABEL: Record<Sentiment, string> = {
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
export const SENTIMENT_CLASS: Record<Sentiment, string> = {
  on_track: "bg-success/10 text-success",
  under_tension: "bg-danger/10 text-danger",
  awaiting_feedback: "bg-warning/10 text-warning",
};

export function formatTimestamp(iso: string | null): string {
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

export type StatusFilter = "active" | "archived" | "all";
export type SentimentFilterToken = Sentiment | "none";

// "none" est une convention purement frontend pour représenter sentiment ===
// null (jamais analysé) — le backend ne connaît que les 3 valeurs Sentiment.
export const SENTIMENT_FILTER_OPTIONS: Array<{ token: SentimentFilterToken; label: string }> = [
  { token: "on_track", label: SENTIMENT_LABEL.on_track },
  { token: "under_tension", label: SENTIMENT_LABEL.under_tension },
  { token: "awaiting_feedback", label: SENTIMENT_LABEL.awaiting_feedback },
  { token: "none", label: "Pas encore analysé" },
];

export function filterProjects(
  projects: ProjectListItem[],
  filters: { statusFilter: StatusFilter; sentimentFilter: Set<string>; search: string },
): ProjectListItem[] {
  const q = filters.search.trim().toLowerCase();
  return projects.filter((p) => {
    if (filters.statusFilter !== "all" && p.status !== filters.statusFilter) return false;
    if (filters.sentimentFilter.size > 0) {
      const token = p.sentiment ?? "none";
      if (!filters.sentimentFilter.has(token)) return false;
    }
    if (q && !p.name.toLowerCase().includes(q)) return false;
    return true;
  });
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

export const RISK_LEVEL_CLASS: Record<string, string> = {
  critique: "bg-danger/10 text-danger",
  modere: "bg-warning/10 text-warning",
  faible: "bg-success/10 text-success",
};

// Décisions/risques structurés (llm.ProjectSummaryLLM, additif à
// summary_content) — repliés par défaut pour garder la carte "dossier" dense,
// dépliables au clic plutôt qu'une page de détail séparée (F5 : évolution
// visuelle légère de ProjectHub, pas une nouvelle vue).
export function DecisionsAndRisks({
  structured,
  defaultOpen = false,
}: {
  structured: StructuredContent | null;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const decisions = structured?.decisions ?? [];
  const risques = structured?.risques ?? [];
  if (decisions.length === 0 && risques.length === 0) return null;

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 text-xs font-medium text-text-secondary hover:text-text-primary"
      >
        {open ? "▾" : "▸"} Décisions & risques
        {decisions.length > 0 ? (
          <span className="rounded-full bg-info/10 px-1.5 py-0.5 text-[11px] font-semibold text-info">
            {decisions.length}
          </span>
        ) : null}
        {risques.length > 0 ? (
          <span className="rounded-full bg-danger/10 px-1.5 py-0.5 text-[11px] font-semibold text-danger">
            {risques.length}
          </span>
        ) : null}
      </button>
      {open ? (
        <div className="mt-2 space-y-2">
          {risques.map((r, i) => (
            <div key={`risk-${i}`} className="flex items-start gap-2 text-xs">
              <span
                className={`mt-0.5 shrink-0 rounded-full px-1.5 py-0.5 font-semibold ${
                  RISK_LEVEL_CLASS[(r.niveau ?? "").toLowerCase()] ?? "bg-bg-tertiary text-text-muted"
                }`}
              >
                Risque
              </span>
              <span className="text-text-secondary">{r.description}</span>
            </div>
          ))}
          {decisions.map((d, i) => (
            <div key={`decision-${i}`} className="flex items-start gap-2 text-xs">
              <span className="mt-0.5 shrink-0 rounded-full bg-info/10 px-1.5 py-0.5 font-semibold text-info">
                Décision
              </span>
              <span className="text-text-secondary">{d.description}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// Presets "mois" pour le bouton "Analyser" — traduits en jours pour
// réutiliser tel quel le paramètre backend existant `force_days` (borne
// 1-365, voir api/main.py::refresh_project). Pas d'abstraction "mois" côté
// backend : la conversion se fait uniquement ici.
const ANALYZE_RANGE_PRESETS: Array<{ label: string; days: number }> = [
  { label: "1 mois", days: 30 },
  { label: "3 mois", days: 90 },
  { label: "6 mois", days: 180 },
  { label: "1 an", days: 365 },
];

function AnalyzeRangeModal({
  onConfirm,
  onClose,
  busy,
}: {
  onConfirm: (days: number) => void;
  onClose: () => void;
  busy: boolean;
}) {
  const [selectedDays, setSelectedDays] = useState<number>(ANALYZE_RANGE_PRESETS[0].days);
  const [customMode, setCustomMode] = useState(false);
  const [customValue, setCustomValue] = useState("");

  const effectiveDays = customMode ? Number(customValue) : selectedDays;
  const isValid = Number.isInteger(effectiveDays) && effectiveDays >= 1 && effectiveDays <= 365;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="analyze-modal-title"
    >
      <div className="w-full max-w-lg rounded-2xl bg-surface p-6 shadow-xl">
        <h3 id="analyze-modal-title" className="text-lg font-semibold text-text-primary">
          Analyser le projet
        </h3>
        <p className="mt-1 text-sm text-text-muted">
          Choisissez la période à analyser. Ceci relance une analyse complète sur la fenêtre
          sélectionnée, sans se limiter aux emails reçus depuis le dernier rafraîchissement.
        </p>

        <div className="mt-4 flex flex-wrap gap-2">
          {ANALYZE_RANGE_PRESETS.map((preset) => {
            const active = !customMode && selectedDays === preset.days;
            return (
              <button
                key={preset.label}
                type="button"
                onClick={() => {
                  setCustomMode(false);
                  setSelectedDays(preset.days);
                }}
                className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
                  active
                    ? "bg-accent-primary text-white"
                    : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
                }`}
              >
                {preset.label}
              </button>
            );
          })}
          <button
            type="button"
            onClick={() => setCustomMode(true)}
            className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
              customMode
                ? "bg-accent-primary text-white"
                : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
            }`}
          >
            Autre
          </button>
        </div>

        {customMode ? (
          <label className="mt-3 block text-sm text-text-secondary">
            Nombre de jours (1 à 365)
            <input
              type="number"
              min={1}
              max={365}
              autoFocus
              value={customValue}
              onChange={(e) => setCustomValue(e.target.value)}
              className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-2 text-sm text-text-primary"
              placeholder="Ex. 45"
            />
          </label>
        ) : null}

        {customMode && customValue && !isValid ? (
          <p className="mt-2 text-xs text-danger">Choisissez une valeur entre 1 et 365 jours.</p>
        ) : null}

        <div className="mt-6 flex gap-2">
          <button
            type="button"
            onClick={() => isValid && onConfirm(effectiveDays)}
            disabled={!isValid || busy}
            className="rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {busy ? "Analyse en cours…" : "Lancer l'analyse"}
          </button>
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded-xl px-4 py-2 text-sm text-text-secondary disabled:opacity-50"
          >
            Annuler
          </button>
        </div>
      </div>
    </div>
  );
}

function ProjectCard({
  project,
  onRefreshed,
  onUpdated,
  autoSyncJobId,
  onAutoSyncSettled,
}: {
  project: ProjectListItem;
  onRefreshed: (id: string, patch: Partial<ProjectListItem>) => void;
  onUpdated: (id: string, patch: Partial<ProjectListItem>) => void;
  autoSyncJobId?: string;
  onAutoSyncSettled?: (projectId: string) => void;
}) {
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [tagsPreview, setTagsPreview] = useState<{ tag: string; count: number }[]>([]);
  const [showWidenPrompt, setShowWidenPrompt] = useState(false);
  const [showAnalyzeModal, setShowAnalyzeModal] = useState(false);

  function applyRefreshResult(result: Record<string, unknown>, wasFirstSync: boolean) {
    onRefreshed(project.id, {
      sentiment: (result.sentiment as Sentiment | null) ?? project.sentiment,
      summary_content: (result.content as string | null) ?? project.summary_content,
      structured_content:
        (result.structured_content as StructuredContent | null) ?? project.structured_content,
      last_processed_email_timestamp:
        (result.last_processed_email_timestamp as string | null) ??
        project.last_processed_email_timestamp,
    });
    setTagsPreview((result.tags_reference as { tag: string; count: number }[] | undefined) ?? []);
    // Un delta à 0 sur un projet déjà synchronisé est un état normal (pas de
    // nouvel email depuis le dernier passage) — la proposition d'élargir la
    // fenêtre ne s'affiche que pour un tout premier sync resté bredouille.
    setShowWidenPrompt(wasFirstSync && (result.new_emails as number | undefined) === 0);
  }

  async function handleRefresh(forceDays?: number): Promise<boolean> {
    const wasFirstSync = !project.last_processed_email_timestamp;
    setRefreshError(null);
    setRefreshing(true);
    try {
      const qs = forceDays ? `?force_days=${forceDays}` : "";
      const startRes = await apiFetch(`/api/projects/${project.id}/refresh${qs}`, { method: "POST" });
      const startData = (await parseResponseJson(startRes)) as Record<string, unknown>;
      if (!startRes.ok) throw new Error(parseApiError(startData));
      const jobId = startData.job_id as string | undefined;
      if (!jobId) throw new Error("Réponse inattendue du serveur (job_id manquant).");
      const result = await pollRefreshJob(jobId);
      applyRefreshResult(result, wasFirstSync);
      return true;
    } catch (err) {
      setRefreshError(err instanceof Error ? err.message : "Erreur inconnue");
      return false;
    } finally {
      setRefreshing(false);
    }
  }

  async function handleAnalyzeConfirm(days: number) {
    const success = await handleRefresh(days);
    if (success) setShowAnalyzeModal(false);
  }

  // Auto-sync déclenché au login (saas_logic.trigger_login_auto_sync) : le job
  // tourne déjà côté serveur, cette carte n'a qu'à en suivre la progression
  // (pas de nouveau POST) — même état visuel `refreshing` que le clic manuel.
  useEffect(() => {
    if (!autoSyncJobId) return;
    const wasFirstSync = !project.last_processed_email_timestamp;
    let cancelled = false;
    setRefreshError(null);
    setRefreshing(true);
    pollRefreshJob(autoSyncJobId)
      .then((result) => {
        if (cancelled) return;
        applyRefreshResult(result, wasFirstSync);
      })
      .catch((err) => {
        if (cancelled) return;
        setRefreshError(err instanceof Error ? err.message : "Erreur inconnue");
      })
      .finally(() => {
        if (cancelled) return;
        setRefreshing(false);
        onAutoSyncSettled?.(project.id);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoSyncJobId]);

  return (
    <article
      className={`flex flex-col gap-3 rounded-2xl border p-5 shadow-sm transition ${
        refreshing
          ? "animate-fasttrack-pulse border-fasttrack-active bg-fasttrack-bg"
          : "border-border-default bg-surface"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-lg font-semibold text-text-primary">
          <Link to={`/projects/${project.id}`} className="hover:text-accent-primary hover:underline">
            {project.name}
          </Link>
        </h3>
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

      {tagsPreview.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {tagsPreview.slice(0, 3).map((t) => (
            <span
              key={t.tag}
              className="rounded-full bg-bg-tertiary px-2 py-0.5 text-[11px] font-medium text-text-secondary"
            >
              {t.tag} ×{t.count}
            </span>
          ))}
        </div>
      ) : null}

      <DecisionsAndRisks structured={project.structured_content} />

      {refreshError ? <p className="text-xs text-danger">{refreshError}</p> : null}

      {showWidenPrompt ? (
        <div className="rounded-xl border border-dashed border-border-default p-3">
          <p className="text-xs text-text-muted">
            Aucun email trouvé sur la fenêtre initiale. Chercher plus loin ?
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {WIDEN_WINDOW_DAYS.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => void handleRefresh(n)}
                disabled={refreshing}
                className="rounded-full border border-border-default px-2.5 py-1 text-xs font-medium text-text-secondary transition hover:bg-bg-tertiary disabled:cursor-not-allowed disabled:opacity-60"
              >
                {n}j
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void handleRefresh()}
          disabled={refreshing}
          className="mt-1 self-start rounded-full border border-info/40 px-3 py-1.5 text-xs font-medium text-info transition hover:bg-info/10 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {refreshing ? "Actualisation…" : "Actualiser (Fast-Track)"}
        </button>
        <button
          type="button"
          onClick={() => setShowAnalyzeModal(true)}
          disabled={refreshing}
          className="mt-1 self-start rounded-full border border-accent-primary/40 px-3 py-1.5 text-xs font-medium text-accent-primary transition hover:bg-accent-primary/10 disabled:cursor-not-allowed disabled:opacity-60"
        >
          Analyser
        </button>
      </div>

      <RulesMatrixEditor project={project} onUpdated={onUpdated} />

      {showAnalyzeModal ? (
        <AnalyzeRangeModal
          busy={refreshing}
          onConfirm={(days) => void handleAnalyzeConfirm(days)}
          onClose={() => setShowAnalyzeModal(false)}
        />
      ) : null}
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

const STATUS_FILTER_OPTIONS: Array<{ value: StatusFilter; label: string }> = [
  { value: "active", label: "Actifs" },
  { value: "archived", label: "Archivés" },
  { value: "all", label: "Tous" },
];

function ProjectFilterPanel({
  statusFilter,
  sentimentFilter,
  search,
  onStatusChange,
  onSentimentToggle,
  onSearchChange,
  onReset,
  activeCount,
}: {
  statusFilter: StatusFilter;
  sentimentFilter: Set<string>;
  search: string;
  onStatusChange: (value: StatusFilter) => void;
  onSentimentToggle: (token: SentimentFilterToken) => void;
  onSearchChange: (value: string) => void;
  onReset: () => void;
  activeCount: number;
}) {
  const [expanded, setExpanded] = useState(false);

  if (!expanded) {
    return (
      <button
        type="button"
        onClick={() => setExpanded(true)}
        className="flex shrink-0 items-center gap-2 self-start rounded-full border border-border-default bg-surface px-3 py-1.5 text-xs font-medium text-text-secondary hover:bg-bg-tertiary"
      >
        Filtres
        {activeCount > 0 ? (
          <span className="rounded-full bg-accent-primary px-1.5 py-0.5 text-[11px] font-semibold text-white">
            {activeCount}
          </span>
        ) : null}
      </button>
    );
  }

  return (
    <aside className="flex w-full shrink-0 flex-col gap-4 rounded-2xl border border-border-default bg-surface p-4 sm:w-64">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-text-primary">Filtres</h2>
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="text-xs font-medium text-text-muted hover:text-text-secondary"
        >
          Fermer
        </button>
      </div>

      <label className="text-xs text-text-secondary">
        Recherche
        <input
          type="search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Nom du projet…"
          className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary"
        />
      </label>

      <div>
        <p className="mb-1.5 text-xs font-medium text-text-secondary">Statut</p>
        <div className="flex flex-wrap gap-1.5">
          {STATUS_FILTER_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => onStatusChange(opt.value)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                statusFilter === opt.value
                  ? "bg-accent-primary text-white"
                  : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="mb-1.5 text-xs font-medium text-text-secondary">Santé</p>
        <div className="flex flex-wrap gap-1.5">
          {SENTIMENT_FILTER_OPTIONS.map((opt) => (
            <button
              key={opt.token}
              type="button"
              onClick={() => onSentimentToggle(opt.token)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                sentimentFilter.has(opt.token)
                  ? "bg-accent-primary text-white"
                  : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {activeCount > 0 ? (
        <button
          type="button"
          onClick={onReset}
          className="self-start text-xs font-medium text-text-muted underline decoration-dotted hover:text-text-secondary"
        >
          Réinitialiser les filtres
        </button>
      ) : null}
    </aside>
  );
}

// Miroir de DuplicateGroupOut/DuplicateProjectOut (api/routers/projects.py)
// — GET /api/projects/duplicates regroupe les projets actifs du tenant dont
// rules_matrix.sender_domains se chevauche (seul signal de dédup retenu).
type DuplicateProject = {
  id: string;
  name: string;
  email_count: number;
  updated_at: string;
};

type DuplicateGroup = {
  shared_domains: string[];
  projects: DuplicateProject[];
};

function DuplicatesPanel({ onMerged }: { onMerged: (targetId: string, sourceIds: string[], target: ProjectListItem) => void }) {
  const [groups, setGroups] = useState<DuplicateGroup[] | null>(null);
  const [targets, setTargets] = useState<Record<number, string>>({});
  const [merging, setMerging] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadDuplicates = useCallback(async () => {
    try {
      const res = await apiFetch("/api/projects/duplicates");
      const data = (await parseResponseJson(res)) as DuplicateGroup[] | Record<string, unknown>;
      if (!res.ok) throw new Error(parseApiError(data));
      const list = data as DuplicateGroup[];
      setGroups(list);
      // Cible par défaut : le projet du groupe avec le plus d'emails.
      setTargets(
        Object.fromEntries(
          list.map((g, i) => [
            i,
            g.projects.reduce((best, p) => (p.email_count > best.email_count ? p : best)).id,
          ]),
        ),
      );
    } catch {
      // Bandeau informatif, non bloquant : un échec de ce chargement ne doit
      // pas empêcher l'affichage normal de la liste des projets.
      setGroups(null);
    }
  }, []);

  useEffect(() => {
    void loadDuplicates();
  }, [loadDuplicates]);

  async function handleMerge(index: number, group: DuplicateGroup) {
    const targetId = targets[index];
    const sourceIds = group.projects.map((p) => p.id).filter((id) => id !== targetId);
    if (!targetId || sourceIds.length === 0) return;
    setError(null);
    setMerging(index);
    try {
      const res = await apiFetch(`/api/projects/${targetId}/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_ids: sourceIds }),
      });
      const data = (await parseResponseJson(res)) as Record<string, unknown>;
      if (!res.ok) throw new Error(parseApiError(data));
      onMerged(targetId, sourceIds, data as unknown as ProjectListItem);
      void loadDuplicates();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Échec de la fusion.");
    } finally {
      setMerging(null);
    }
  }

  if (!groups || groups.length === 0) return null;

  return (
    <div className="mb-6 rounded-2xl border border-border-default bg-surface p-4">
      <h2 className="text-sm font-semibold text-text-primary">
        {groups.length} doublon{groups.length > 1 ? "s" : ""} détecté{groups.length > 1 ? "s" : ""}
      </h2>
      <p className="mt-1 text-xs text-text-muted">
        Ces projets partagent un domaine expéditeur — choisissez celui à conserver, les autres seront
        fusionnés dedans (emails, actions, rendez-vous réassignés).
      </p>
      {error ? <p className="mt-2 text-xs text-danger">{error}</p> : null}
      <div className="mt-3 flex flex-col gap-3">
        {groups.map((group, index) => (
          <div key={group.shared_domains.join(",")} className="rounded-xl bg-bg-tertiary/40 p-3">
            <p className="text-xs text-text-secondary">
              Domaine{group.shared_domains.length > 1 ? "s" : ""} commun
              {group.shared_domains.length > 1 ? "s" : ""} : {group.shared_domains.join(", ")}
            </p>
            <div className="mt-2 flex flex-col gap-1.5">
              {group.projects.map((p) => (
                <label key={p.id} className="flex items-center gap-2 text-sm text-text-primary">
                  <input
                    type="radio"
                    name={`duplicate-target-${index}`}
                    checked={targets[index] === p.id}
                    onChange={() => setTargets((prev) => ({ ...prev, [index]: p.id }))}
                  />
                  {p.name}
                  <span className="text-xs text-text-muted">
                    ({p.email_count} email{p.email_count > 1 ? "s" : ""}, {formatTimestamp(p.updated_at)})
                  </span>
                </label>
              ))}
            </div>
            <button
              type="button"
              onClick={() => void handleMerge(index, group)}
              disabled={merging === index}
              className="mt-2 rounded-lg bg-accent-primary px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
            >
              {merging === index ? "Fusion en cours…" : "Fusionner"}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ProjectHub() {
  const [projects, setProjects] = useState<ProjectListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  // Jobs Fast-Track déclenchés automatiquement par le login (voir
  // apiClient.takePendingAutoSync) — project_id -> job_id. Consommé une seule
  // fois au montage ; chaque entrée est retirée quand sa carte a fini de
  // suivre son job (onAutoSyncSettled).
  const [autoSyncJobIds, setAutoSyncJobIds] = useState<Record<string, string>>({});

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

  useEffect(() => {
    const pending = takePendingAutoSync();
    if (pending.length === 0) return;
    setAutoSyncJobIds(Object.fromEntries(pending.map((p) => [p.project_id, p.job_id])));
  }, []);

  function patchProject(id: string, patch: Partial<ProjectListItem>) {
    setProjects((prev) => (prev ? prev.map((p) => (p.id === id ? { ...p, ...patch } : p)) : prev));
  }

  function handleMerged(targetId: string, sourceIds: string[], target: ProjectListItem) {
    setProjects((prev) =>
      prev
        ? prev
            .filter((p) => !sourceIds.includes(p.id))
            .map((p) => (p.id === targetId ? { ...p, ...target } : p))
        : prev,
    );
  }

  function settleAutoSync(projectId: string) {
    setAutoSyncJobIds((prev) => {
      if (!(projectId in prev)) return prev;
      const next = { ...prev };
      delete next[projectId];
      return next;
    });
  }

  const autoSyncCount = Object.keys(autoSyncJobIds).length;

  const statusFilter = (searchParams.get("status") as StatusFilter | null) ?? "active";
  const sentimentFilter = useMemo(
    () => new Set((searchParams.get("sentiment") ?? "").split(",").filter(Boolean)),
    [searchParams],
  );
  const search = searchParams.get("q") ?? "";
  const activeFilterCount =
    (statusFilter !== "active" ? 1 : 0) + (sentimentFilter.size > 0 ? 1 : 0) + (search.trim() ? 1 : 0);

  function updateFilter(patch: { status?: StatusFilter; sentiment?: Set<string>; q?: string }) {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (patch.status !== undefined) {
          if (patch.status === "active") next.delete("status");
          else next.set("status", patch.status);
        }
        if (patch.sentiment !== undefined) {
          if (patch.sentiment.size === 0) next.delete("sentiment");
          else next.set("sentiment", Array.from(patch.sentiment).join(","));
        }
        if (patch.q !== undefined) {
          if (!patch.q.trim()) next.delete("q");
          else next.set("q", patch.q);
        }
        return next;
      },
      { replace: true },
    );
  }

  function toggleSentiment(token: SentimentFilterToken) {
    const next = new Set(sentimentFilter);
    if (next.has(token)) next.delete(token);
    else next.add(token);
    updateFilter({ sentiment: next });
  }

  function resetFilters() {
    updateFilter({ status: "active", sentiment: new Set(), q: "" });
  }

  const filteredProjects = useMemo(
    () => (projects ? filterProjects(projects, { statusFilter, sentimentFilter, search }) : null),
    [projects, statusFilter, sentimentFilter, search],
  );

  return (
    <div>
      {autoSyncCount > 0 ? (
        <div
          role="status"
          className="mb-4 rounded-2xl border border-border-default bg-surface px-4 py-3 text-sm text-text-secondary"
        >
          Mise à jour automatique de {autoSyncCount} projet{autoSyncCount > 1 ? "s" : ""} en cours…
        </div>
      ) : null}

      <DuplicatesPanel onMerged={handleMerged} />

      <div className="flex flex-col items-start gap-4 sm:flex-row">
        <ProjectFilterPanel
          statusFilter={statusFilter}
          sentimentFilter={sentimentFilter}
          search={search}
          onStatusChange={(value) => updateFilter({ status: value })}
          onSentimentToggle={toggleSentiment}
          onSearchChange={(value) => updateFilter({ q: value })}
          onReset={resetFilters}
          activeCount={activeFilterCount}
        />

        <div className="min-w-0 flex-1">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-text-muted">
              {projects && activeFilterCount > 0
                ? `${filteredProjects?.length ?? 0} / ${projects.length} projets`
                : "Vos projets, leur santé IA et les actions encore en attente. Rafraîchissez un projet sans attendre la synchronisation planifiée."}
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

          {projects !== null && projects.length > 0 && filteredProjects?.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-border-default p-8 text-center text-sm text-text-muted">
              Aucun projet ne correspond aux filtres.{" "}
              <button
                type="button"
                onClick={resetFilters}
                className="font-medium text-accent-primary underline decoration-dotted hover:text-accent-hover"
              >
                Réinitialiser les filtres
              </button>
            </div>
          ) : null}

          {filteredProjects && filteredProjects.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {filteredProjects.map((p) => (
                <ProjectCard
                  key={p.id}
                  project={p}
                  onRefreshed={patchProject}
                  onUpdated={patchProject}
                  autoSyncJobId={autoSyncJobIds[p.id]}
                  onAutoSyncSettled={settleAutoSync}
                />
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
