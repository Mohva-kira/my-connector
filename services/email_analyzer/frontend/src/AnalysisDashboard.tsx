import React, { useMemo, useState } from "react";

/** Bloc d'analyse pour un projet (sortie de `process_latest_emails` / rapport JSON). */
export type ProjectAnalysis = {
  nb_emails?: number;
  priorité_attention?: string;
  analyse_sentiment?: { tendance?: string; méthode?: string; confiance_moyenne?: number };
  évaluation_risque?: {
    niveau_risque?: string;
    score_risque?: number;
    facteurs_risque?: string[];
    recommandation?: string;
  };
  entités_extraites?: { technologies?: string[] };
  résumé_assistant?: { texte?: string; erreur?: string; fournisseur?: string };
  emails_critiques?: Array<{ subject?: string; date?: string }>;
};

export type DraftResult = {
  template_id?: string;
  body?: string;
  checklist_tasks?: string[];
  next_meeting_hint?: string;
};

type Props = {
  projectName: string;
  analysis: ProjectAnalysis;
  initialDraft?: DraftResult | null;
  onRequestDraft?: () => Promise<DraftResult>;
};

function riskBadgeClass(niveau?: string): string {
  const n = (niveau || "").toUpperCase();
  if (n === "CRITIQUE") return "bg-priority-high/10 text-priority-high border-priority-high/30";
  if (n === "MODÉRÉ") return "bg-priority-medium/10 text-priority-medium border-priority-medium/30";
  if (n === "FAIBLE") return "bg-priority-low/10 text-priority-low border-priority-low/30";
  return "bg-bg-tertiary text-text-secondary border-border-default";
}

function buildChecklist(analysis: ProjectAnalysis): string[] {
  const risk = analysis.évaluation_risque;
  const tasks: string[] = [];
  for (const f of risk?.facteurs_risque || []) {
    if (f) tasks.push(f);
  }
  for (const em of analysis.emails_critiques?.slice(0, 3) || []) {
    if (em.subject) tasks.push(`Email critique : ${em.subject}`);
  }
  if (tasks.length === 0) {
    tasks.push("Relire le fil récent et confirmer les prochaines actions.");
  }
  return tasks.slice(0, 12);
}

export function AnalysisDashboard({
  projectName,
  analysis,
  initialDraft = null,
  onRequestDraft,
}: Props) {
  const [draft, setDraft] = useState<DraftResult | null>(initialDraft);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checklist = useMemo(() => buildChecklist(analysis), [analysis]);
  const rdvHint = draft?.next_meeting_hint || "—";
  const risk = analysis.évaluation_risque;
  const sentiment = analysis.analyse_sentiment;
  const techs = analysis.entités_extraites?.technologies?.slice(0, 8) || [];

  async function handleDraft() {
    if (!onRequestDraft) return;
    setLoading(true);
    setError(null);
    try {
      const d = await onRequestDraft();
      setDraft(d);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl rounded-2xl border border-border-default bg-surface p-6 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border-subtle pb-4">
        <div>
          <h1 className="text-lg font-semibold text-text-primary">{projectName}</h1>
          <p className="text-sm text-text-muted">
            {analysis.nb_emails ?? "?"} emails · Priorité{" "}
            <span className="font-medium text-text-secondary">
              {analysis.priorité_attention || "—"}
            </span>
          </p>
        </div>
        <span
          className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium ${riskBadgeClass(
            risk?.niveau_risque,
          )}`}
        >
          Risque : {risk?.niveau_risque || "—"} ({risk?.score_risque ?? "—"}/100)
        </span>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl bg-ai-glow p-4">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
            Sentiment
          </h2>
          <p className="mt-1 text-sm text-text-primary">
            {sentiment?.tendance || "—"}{" "}
            <span className="text-text-muted">
              ({sentiment?.méthode || "—"}
              {sentiment?.confiance_moyenne != null
                ? ` · conf. ${sentiment.confiance_moyenne}`
                : ""}
              )
            </span>
          </p>
        </div>
        <div className="rounded-xl bg-ai-glow p-4">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-text-muted">Tech</h2>
          <p className="mt-1 text-sm text-text-primary">
            {techs.length ? techs.join(", ") : "—"}
          </p>
        </div>
      </div>

      <div className="mt-6">
        <h2 className="text-sm font-semibold text-text-primary">Checklist</h2>
        <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-text-secondary">
          {checklist.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </div>

      <div className="mt-6 rounded-xl border border-dashed border-border-default p-4">
        <h2 className="text-sm font-semibold text-text-primary">Prochain RDV</h2>
        <p className="mt-1 text-sm text-text-secondary">{rdvHint}</p>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={handleDraft}
          disabled={!onRequestDraft || loading}
          className="rounded-xl bg-accent-primary px-4 py-2.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Génération…" : "Générer brouillon de réponse"}
        </button>
        {error ? <span className="text-sm text-danger">{error}</span> : null}
      </div>

      {draft?.body ? (
        <div className="mt-4 rounded-xl border border-border-default bg-bg-tertiary/50 p-4">
          <p className="text-xs font-semibold uppercase text-text-muted">
            Brouillon ({draft.template_id || "—"})
          </p>
          <pre className="mt-2 whitespace-pre-wrap font-sans text-sm text-text-primary">
            {draft.body}
          </pre>
        </div>
      ) : null}
    </div>
  );
}

export default AnalysisDashboard;
