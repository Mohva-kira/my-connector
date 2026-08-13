import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Rattrapage rétroactif des pièces jointes (Contexte Documents, domaine #7,
// 2026-08-01) : contrairement à l'ingestion automatique du Fast-Track (emails
// nouveaux uniquement), scanne tout l'historique déjà en base — un seul
// passage IMAP complet (pas headers-only), potentiellement bien plus lourd
// qu'un scan de découverte de domaines. Budget de polling aligné sur le même
// timeout dédié côté worker arq (7200s, WorkerSettings.functions,
// analysis_tasks.py) que discoverApi.ts::DISCOVER_POLL_TIMEOUT_MS.
const BACKFILL_POLL_INTERVAL_MS = 2500;
const BACKFILL_POLL_TIMEOUT_MS = 126 * 60 * 1000;

// Même raison que DISCOVER_JOB_STORAGE_KEY (discoverApi.ts) : survivre à la
// fermeture/réouverture du modal ou à un refresh de page sans relancer un
// scan complet par-dessus celui déjà en cours côté worker.
const BACKFILL_JOB_STORAGE_KEY = "myconnector:attachment-backfill-job";

type StoredBackfillJob = { jobId: string; startedAt: number };

function readStoredBackfillJob(): StoredBackfillJob | null {
  try {
    const raw = localStorage.getItem(BACKFILL_JOB_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredBackfillJob>;
    if (!parsed.jobId || typeof parsed.startedAt !== "number") return null;
    if (Date.now() - parsed.startedAt > BACKFILL_POLL_TIMEOUT_MS) {
      clearStoredBackfillJob();
      return null;
    }
    return { jobId: parsed.jobId, startedAt: parsed.startedAt };
  } catch {
    return null;
  }
}

function writeStoredBackfillJob(jobId: string): void {
  try {
    localStorage.setItem(BACKFILL_JOB_STORAGE_KEY, JSON.stringify({ jobId, startedAt: Date.now() }));
  } catch {
    // Stockage indisponible (navigation privée, quota) : dégrade en simple
    // perte de la reprise à la réouverture, sans bloquer le scan en cours.
  }
}

export function clearStoredBackfillJob(): void {
  try {
    localStorage.removeItem(BACKFILL_JOB_STORAGE_KEY);
  } catch {
    // ignore
  }
}

// À appeler avant `startAttachmentBackfill()` : renvoie l'id d'un scan encore
// dans son budget de polling pour s'y rattacher au lieu d'en relancer un.
export function getResumableBackfillJobId(): string | null {
  return readStoredBackfillJob()?.jobId ?? null;
}

export type BackfillProgress = { processed: number; total: number };

// Miroir du dict renvoyé par _run_email_attachment_backfill_sync
// (email_analyzer/analysis_tasks.py).
export type BackfillResult = {
  _error?: string;
  _empty?: boolean;
  _message?: string;
  days_back?: number;
  oldest_email_at?: string | null;
  emails_scanned?: number;
  emails_matched?: number;
  attachments_seen?: number;
  attachments_created?: number;
  attachments_failed?: number;
};

export async function startAttachmentBackfill(): Promise<string> {
  const res = await apiFetch("/api/documents/backfill-from-emails", { method: "POST" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  const jobId = (data as { job_id: string }).job_id;
  writeStoredBackfillJob(jobId);
  return jobId;
}

export async function pollAttachmentBackfillJob(
  jobId: string,
  onTick?: (progress: BackfillProgress) => void,
): Promise<BackfillResult> {
  // Reprend le budget depuis le démarrage réel du job stocké (cas d'une
  // reprise après réouverture) plutôt que de repartir sur un budget plein à
  // chaque appel de `pollAttachmentBackfillJob`.
  const stored = readStoredBackfillJob();
  const startedAt = stored && stored.jobId === jobId ? stored.startedAt : Date.now();
  const deadline = startedAt + BACKFILL_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, BACKFILL_POLL_INTERVAL_MS));
    // Un tick isolé qui échoue (abort 30s, hoquet réseau) ne doit pas tuer
    // tout le polling tant que le budget global n'est pas dépassé — même
    // correctif que pollDomainDiscoveryJob (discoverApi.ts).
    let data: Record<string, unknown>;
    try {
      const res = await apiFetch(`/api/analyze/${jobId}`, {
        method: "GET",
        signal: AbortSignal.timeout(30_000),
      });
      data = (await parseResponseJson(res)) as Record<string, unknown>;
      if (!res.ok) throw new Error(parseApiError(data));
    } catch {
      continue;
    }
    const progress = data.progress as BackfillProgress | undefined;
    if (progress) onTick?.(progress);
    const status = data.status as string;
    if (status === "done") {
      clearStoredBackfillJob();
      return (data.result ?? {}) as BackfillResult;
    }
    if (status === "error") {
      clearStoredBackfillJob();
      throw new Error((data.error as string) || "Échec du rattrapage.");
    }
  }
  clearStoredBackfillJob();
  throw new Error("Le rattrapage prend trop de temps. Réessayez plus tard.");
}
