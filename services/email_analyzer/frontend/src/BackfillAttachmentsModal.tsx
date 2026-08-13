import { useEffect, useState } from "react";
import {
  getResumableBackfillJobId,
  pollAttachmentBackfillJob,
  startAttachmentBackfill,
  type BackfillProgress,
  type BackfillResult,
} from "./attachmentBackfillApi";

// Au-delà de ce total, on prévient que le scan peut prendre du temps mais
// continue en arrière-plan — même seuil que DiscoverProjectsModal.tsx
// (miroir de large_scan_threshold() côté backend, config.py).
const LARGE_SCAN_HINT_THRESHOLD = 1000;

const LOADING_MESSAGES = [
  "Connexion à votre messagerie…",
  "Relecture de l'historique déjà analysé…",
  "Extraction des pièces jointes…",
];

// Version simplifiée de DiscoverProjectsModal.tsx : pas d'étape de
// sélection/confirmation, le rattrapage est entièrement automatique
// (l'ingestion réutilise DocumentService.ingest_email_attachment, déjà
// idempotent) — juste démarrer, montrer la progression, afficher le résumé.
type Phase = "loading" | "message";

function summarize(result: BackfillResult): string {
  if (result._empty) {
    return result._message || "Aucun email rattaché à un projet en base.";
  }
  const created = result.attachments_created ?? 0;
  const matched = result.emails_matched ?? 0;
  const scanned = result.emails_scanned ?? 0;
  const alreadyThere = Math.max(
    0,
    (result.attachments_seen ?? 0) - created - (result.attachments_failed ?? 0),
  );
  const parts: string[] = [];
  parts.push(
    `${created} pièce${created > 1 ? "s" : ""} jointe${created > 1 ? "s" : ""} ajoutée${created > 1 ? "s" : ""}`,
  );
  parts.push(`depuis ${matched} email${matched > 1 ? "s" : ""} déjà analysé${matched > 1 ? "s" : ""}`);
  parts.push(`(${scanned} email${scanned > 1 ? "s" : ""} parcouru${scanned > 1 ? "s" : ""})`);
  let text = parts.join(" ") + ".";
  if (alreadyThere > 0) {
    text += ` ${alreadyThere} pièce${alreadyThere > 1 ? "s" : ""} jointe${alreadyThere > 1 ? "s" : ""} étai${alreadyThere > 1 ? "ent" : "t"} déjà présente${alreadyThere > 1 ? "s" : ""}.`;
  }
  return text;
}

export default function BackfillAttachmentsModal({
  onClose,
  onCompleted,
}: {
  onClose: () => void;
  onCompleted: () => void;
}) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [statusMsgIndex, setStatusMsgIndex] = useState(0);
  const [progress, setProgress] = useState<BackfillProgress | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [succeeded, setSucceeded] = useState(false);

  useEffect(() => {
    if (phase !== "loading") return;
    const id = setInterval(() => {
      setStatusMsgIndex((i) => (i + 1) % LOADING_MESSAGES.length);
    }, 2000);
    return () => clearInterval(id);
  }, [phase]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const jobId = getResumableBackfillJobId() ?? (await startAttachmentBackfill());
        const result = await pollAttachmentBackfillJob(jobId, (p) => {
          if (!cancelled) setProgress(p);
        });
        if (cancelled) return;
        if (result._error) {
          setMessage(result._error);
          setPhase("message");
          return;
        }
        setSucceeded(!result._empty);
        setMessage(summarize(result));
        setPhase("message");
      } catch (err) {
        if (cancelled) return;
        setMessage(err instanceof Error ? err.message : "Erreur inconnue pendant le rattrapage.");
        setPhase("message");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function handleClose() {
    if (succeeded) onCompleted();
    else onClose();
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="backfill-modal-title"
    >
      <div className="w-full max-w-md rounded-2xl bg-surface p-6 shadow-xl">
        {phase === "loading" && (
          <div className="flex flex-col items-center gap-4 py-8 text-center">
            <svg
              className="h-10 w-10 animate-spin text-accent-primary"
              viewBox="0 0 24 24"
              fill="none"
              aria-hidden="true"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <h3 id="backfill-modal-title" className="text-lg font-semibold text-text-primary">
              Rattrapage des pièces jointes
            </h3>
            {progress && progress.total > 0 ? (
              <div className="w-full max-w-xs">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg-tertiary">
                  <div
                    className="h-full rounded-full bg-accent-primary transition-all"
                    style={{ width: `${Math.min(100, (progress.processed / progress.total) * 100)}%` }}
                  />
                </div>
                <p className="mt-2 text-sm text-text-muted">
                  {progress.processed} / {progress.total} emails analysés
                </p>
                {progress.total > LARGE_SCAN_HINT_THRESHOLD && (
                  <p className="mt-1 text-xs text-text-muted">
                    Historique volumineux : le rattrapage peut prendre du temps. Vous pouvez fermer
                    cette fenêtre, il continue en arrière-plan — rouvrez-la pour voir la progression.
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-text-muted">{LOADING_MESSAGES[statusMsgIndex]}</p>
            )}
          </div>
        )}

        {phase === "message" && (
          <>
            <h3 id="backfill-modal-title" className="text-lg font-semibold text-text-primary">
              Rattrapage des pièces jointes
            </h3>
            <p className="mt-1 text-sm text-text-muted">{message}</p>
            <button
              type="button"
              onClick={handleClose}
              className="mt-6 rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white"
            >
              Fermer
            </button>
          </>
        )}
      </div>
    </div>
  );
}
