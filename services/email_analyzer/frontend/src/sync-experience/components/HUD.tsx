import type { SyncStage } from "../types";
import { useSyncStore } from "../../store/syncStore";

const STAGE_LABELS: Record<SyncStage, string> = {
  CONNECTING: "Connexion…",
  IMPORTING: "Import des emails…",
  NORMALIZING: "Normalisation…",
  SCORING: "Notation…",
  AI_CLASSIFICATION: "Classification IA…",
  TASK_EXTRACTION: "Extraction des tâches…",
  DEADLINE_DETECTION: "Détection des échéances…",
  GENERATING_BRIEFING: "Génération du briefing…",
  COMPLETED: "Terminé",
};

/**
 * First real visual piece of the sync experience: reads the store via
 * selectors only (no EventBus here — aggregate numbers are continuous
 * render state, not discrete effects). Only renders what the backend
 * actually reports today; see bridgeAnalyzeJob.ts for which SyncStage
 * values are reachable.
 */
export function HUD(): JSX.Element {
  const stage = useSyncStore((state) => state.stage);
  const stats = useSyncStore((state) => state.stats);

  const hasTotal = stats.emailsTotal > 0;

  return (
    <div className="w-full max-w-md rounded-xl border border-border-default bg-surface p-6 text-text-primary">
      <div className="flex items-center justify-between gap-3">
        <p className="font-mono text-sm text-text-secondary">{STAGE_LABELS[stage]}</p>
        {stats.urgentFound > 0 ? (
          <span className="rounded-full bg-danger/10 px-2.5 py-1 text-xs font-semibold text-danger">
            {stats.urgentFound} urgent{stats.urgentFound > 1 ? "s" : ""}
          </span>
        ) : null}
      </div>

      {hasTotal ? (
        <>
          <p className="mt-3 text-sm font-medium">
            {stats.emailsProcessed} / {stats.emailsTotal} emails traités
          </p>
          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-bg-tertiary">
            <div
              className="h-full rounded-full bg-ai-primary transition-all duration-500 ease-out"
              style={{ width: `${Math.min(100, stats.progress)}%` }}
            />
          </div>
        </>
      ) : (
        <p className="mt-3 text-sm text-text-muted">En attente de données de progression…</p>
      )}
    </div>
  );
}
