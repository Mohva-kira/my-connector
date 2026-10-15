import { useEffect, useState } from "react";
import type { RunProgressInfo, Task } from "../api";
import { Badge, Card } from "../ui";

/** Secondes restantes avant `until` (ISO), recalculées chaque seconde ; `null` sans échéance. */
function useCountdown(until: string | null | undefined): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!until) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [until]);
  if (!until) return null;
  return Math.max(0, Math.ceil((new Date(until).getTime() - now) / 1000));
}

const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;

const STEP_TONE = { attente: "neutral", "en cours": "info", terminée: "success" } as const;
const STEP_MARK = { attente: "○", "en cours": "●", terminée: "✓" } as const;

/** Avancement d'une analyse en cours : étapes, lots de 15 et compte à rebours de la pause entre deux lots. */
export function RunProgress({ task }: { task: Task }) {
  const progress = task.meta?.progress as RunProgressInfo | undefined;
  const batch = progress?.batch ?? null;
  const remaining = useCountdown(batch?.state === "traitement" ? batch.pause_until : null);

  let batchLine = "Démarrage de la collecte…";
  if (batch) {
    if (batch.state === "traitement") {
      batchLine = remaining
        ? `Lot ${batch.batch} : enregistrement et scoring, reprise de la collecte dans ${clock(remaining)}`
        : `Lot ${batch.batch} : fin du traitement avant de reprendre la collecte…`;
    } else if (batch.state === "reprise") batchLine = `Lot ${batch.batch} traité : collecte du lot suivant…`;
    else batchLine = `Dernier lot traité (${batch.read} offres lues).`;
  } else if (progress && !progress.steps.some((s) => s.key === "collecte" && s.state === "en cours")) {
    batchLine = "";
  }

  return (
    <div className="mb-4">
      <Card title={task.title} actions={<Badge tone="info">{task.status}</Badge>}>
        {!progress ? <p className="text-sm text-text-muted">Démarrage…</p> : (
          <div className="space-y-3">
            <ol className="flex flex-wrap gap-2" aria-label="Étapes de l'analyse">
              {progress.steps.map((step) => (
                <li key={step.key}>
                  <Badge tone={STEP_TONE[step.state]}>{STEP_MARK[step.state]} {step.label}</Badge>
                </li>
              ))}
            </ol>
            {batch && (
              <div>
                <div className="flex gap-1" role="progressbar" aria-valuemin={0} aria-valuemax={batch.batches} aria-valuenow={Math.min(batch.batch, batch.batches)} aria-label="Lots traités">
                  {Array.from({ length: batch.batches }, (_, n) => {
                    const done = n + 1 < batch.batch || (n + 1 === batch.batch && batch.state !== "traitement");
                    const current = n + 1 === batch.batch && batch.state === "traitement";
                    return <span key={n} className={`h-2 flex-1 rounded-full ${done ? "bg-success" : current ? "animate-pulse bg-info" : "bg-border-default"}`} />;
                  })}
                </div>
                <p className="mt-1 text-xs text-text-muted">Lots de 15 offres, {batch.batches} au plus</p>
              </div>
            )}
            <p className="text-sm text-text-secondary" aria-live="polite">
              {batchLine}
              {progress.scored > 0 && <> · {progress.scored} offre{progress.scored > 1 ? "s" : ""} déjà scorée{progress.scored > 1 ? "s" : ""} et affichée{progress.scored > 1 ? "s" : ""} ci-dessous</>}
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}
