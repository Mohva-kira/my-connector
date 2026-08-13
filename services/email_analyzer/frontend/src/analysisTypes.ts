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
  tags_reference?: Array<{ tag: string; count: number }>;
  threads?: Array<{ subject: string; count: number; first_date?: string | null; last_date?: string | null }>;
};

export type DraftResult = {
  template_id?: string;
  body?: string;
  checklist_tasks?: string[];
  next_meeting_hint?: string;
};

export function riskBadgeClass(niveau?: string): string {
  const n = (niveau || "").toUpperCase();
  if (n === "CRITIQUE") return "bg-priority-high/10 text-priority-high border-priority-high/30";
  if (n === "MODÉRÉ") return "bg-priority-medium/10 text-priority-medium border-priority-medium/30";
  if (n === "FAIBLE") return "bg-priority-low/10 text-priority-low border-priority-low/30";
  return "bg-bg-tertiary text-text-secondary border-border-default";
}

export function buildChecklist(analysis: ProjectAnalysis): string[] {
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
