import React, { useMemo, useState } from "react";
import { buildChecklist, riskBadgeClass, type DraftResult, type ProjectAnalysis } from "../../analysisTypes";

type Props = {
  projectName: string;
  analysis: ProjectAnalysis;
  onRequestDraft?: () => Promise<DraftResult>;
  // Fournie en mode "analyse sans filtre" : chaque fil devient un sujet
  // cliquable pour relancer une analyse ciblée sur ce sujet.
  onSelectThread?: (subject: string) => void;
};

// Carte d'analyse inline dans une bulle de chat (voir ChatThread) : même
// contrat de données que l'ancien AnalysisDashboard, mais largeur bulle et
// sections secondaires repliées par défaut (<details>) — la carte
// n'occupe plus une colonne entière, elle doit cohabiter avec les
// messages qui l'entourent.
export default function AnalysisCard({ projectName, analysis, onRequestDraft, onSelectThread }: Props) {
  const [draft, setDraft] = useState<DraftResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checklist = useMemo(() => buildChecklist(analysis), [analysis]);
  const risk = analysis.évaluation_risque;
  const sentiment = analysis.analyse_sentiment;
  const techs = analysis.entités_extraites?.technologies?.slice(0, 8) || [];
  const executiveSummary = analysis.résumé_assistant?.texte;
  const threads = analysis.threads || [];

  async function handleDraft() {
    if (!onRequestDraft) return;
    setLoading(true);
    setError(null);
    try {
      setDraft(await onRequestDraft());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-[92%] rounded-lg border border-border-default bg-surface p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border-subtle pb-3">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">{projectName}</h3>
          <p className="text-xs text-text-muted">
            {analysis.nb_emails ?? "?"} emails · Priorité{" "}
            <span className="font-medium text-text-secondary">{analysis.priorité_attention || "—"}</span>
          </p>
        </div>
        <span
          className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ${riskBadgeClass(
            risk?.niveau_risque,
          )}`}
        >
          Risque : {risk?.niveau_risque || "—"} ({risk?.score_risque ?? "—"}/100)
        </span>
      </div>

      {executiveSummary ? (
        <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-text-primary">{executiveSummary}</p>
      ) : null}

      {threads.length > 0 ? (
        <div className="mt-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
            {onSelectThread ? "Sujets détectés — cliquez pour approfondir" : "Fils de discussion"}
          </h4>
          <ul className="mt-1.5 space-y-1 text-sm text-text-secondary">
            {threads.map((t) =>
              onSelectThread ? (
                <li key={t.subject}>
                  <button
                    type="button"
                    onClick={() => onSelectThread(t.subject)}
                    className="w-full truncate rounded-lg px-1.5 py-1 text-left underline decoration-border-default underline-offset-2 transition hover:bg-bg-tertiary hover:text-text-primary"
                  >
                    {t.subject || "(sans sujet)"} — {t.count} message{t.count > 1 ? "s" : ""}
                  </button>
                </li>
              ) : (
                <li key={t.subject} className="truncate">
                  {t.subject || "(sans sujet)"} — {t.count} message{t.count > 1 ? "s" : ""}
                </li>
              ),
            )}
          </ul>
        </div>
      ) : null}

      <details className="mt-3">
        <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-text-muted">
          Sentiment, tech et checklist
        </summary>
        <div className="mt-2 grid gap-3 sm:grid-cols-2">
          <div className="rounded-xl bg-ai-glow p-3">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-text-muted">Sentiment</h4>
            <p className="mt-1 text-sm text-text-primary">
              {sentiment?.tendance || "—"}{" "}
              <span className="text-text-muted">
                ({sentiment?.méthode || "—"}
                {sentiment?.confiance_moyenne != null ? ` · conf. ${sentiment.confiance_moyenne}` : ""})
              </span>
            </p>
          </div>
          <div className="rounded-xl bg-ai-glow p-3">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-text-muted">Tech</h4>
            <p className="mt-1 text-sm text-text-primary">{techs.length ? techs.join(", ") : "—"}</p>
          </div>
        </div>
        <ul className="mt-3 list-inside list-disc space-y-1 text-sm text-text-secondary">
          {checklist.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </details>

      {onRequestDraft ? (
        <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-border-subtle pt-3">
          <button
            type="button"
            onClick={handleDraft}
            disabled={loading}
            className="rounded-xl bg-accent-primary px-3.5 py-2 text-xs font-medium text-white transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Génération…" : "Générer brouillon de réponse"}
          </button>
          {error ? <span className="text-xs text-danger">{error}</span> : null}
        </div>
      ) : null}

      {draft?.body ? (
        <div className="mt-3 rounded-xl border border-border-default bg-bg-tertiary/50 p-3">
          <p className="text-xs font-semibold uppercase text-text-muted">Brouillon ({draft.template_id || "—"})</p>
          <pre className="mt-2 whitespace-pre-wrap font-sans text-sm text-text-primary">{draft.body}</pre>
        </div>
      ) : null}
    </div>
  );
}
