import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Mirroring de discoverApi.ts (découverte de domaines expéditeurs) pour la
// découverte de clés de ticket dans les sujets ([Banque_Plateforme_Numéro],
// voir email_analyzer/subject_tickets.py) — mêmes constantes de fenêtre/
// polling, même discipline de reprise via localStorage.
const DISCOVER_DAYS = 90;
const DISCOVER_POLL_INTERVAL_MS = 2500;
const DISCOVER_POLL_TIMEOUT_MS = 126 * 60 * 1000;

const DISCOVER_JOB_STORAGE_KEY = "myconnector:discover-tickets-job";

type StoredDiscoverJob = { jobId: string; startedAt: number };

function readStoredDiscoveryJob(): StoredDiscoverJob | null {
  try {
    const raw = localStorage.getItem(DISCOVER_JOB_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredDiscoverJob>;
    if (!parsed.jobId || typeof parsed.startedAt !== "number") return null;
    if (Date.now() - parsed.startedAt > DISCOVER_POLL_TIMEOUT_MS) {
      clearStoredDiscoveryJob();
      return null;
    }
    return { jobId: parsed.jobId, startedAt: parsed.startedAt };
  } catch {
    return null;
  }
}

function writeStoredDiscoveryJob(jobId: string): void {
  try {
    localStorage.setItem(DISCOVER_JOB_STORAGE_KEY, JSON.stringify({ jobId, startedAt: Date.now() }));
  } catch {
    // Stockage indisponible (navigation privée, quota) : dégrade en simple
    // perte de la reprise au réouverture, sans bloquer le scan en cours.
  }
}

export function clearStoredDiscoveryJob(): void {
  try {
    localStorage.removeItem(DISCOVER_JOB_STORAGE_KEY);
  } catch {
    // ignore
  }
}

export function getResumableTicketDiscoveryJobId(): string | null {
  return readStoredDiscoveryJob()?.jobId ?? null;
}

export type DiscoverProgress = { processed: number; total: number };

// Miroir du dict renvoyé par EmailProcessor.discover_subject_ticket_keys
// (email_analyzer/analyzer.py), annoté côté worker
// (analysis_tasks.py::_run_subject_key_discovery_sync) :
// - existing_project_id/name : déjà suivi (clé exacte connue, ou projet dont
//   le nom est quasi identique) — comportement "Déjà suivi" comme pour les
//   domaines.
// - suggested_project_id/name : projet plausible mais nécessitant une
//   confirmation explicite de l'utilisateur (pas de contrepartie côté
//   découverte de domaines — propre à cette découverte).
export type DiscoveredSubjectKey = {
  subject_key: string;
  email_count: number;
  distinct_ticket_count: number;
  sample_subjects: string[];
  latest_received_at: string | null;
  existing_project_id: string | null;
  existing_project_name: string | null;
  suggested_project_id: string | null;
  suggested_project_name: string | null;
};

export type DiscoverTicketsResult = {
  _error?: string;
  _empty?: boolean;
  _message?: string;
  subject_keys?: DiscoveredSubjectKey[];
  total_emails_scanned?: number;
  days_back?: number;
};

export async function startTicketDiscovery(): Promise<string> {
  const res = await apiFetch("/api/brief/discover-tickets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ days: DISCOVER_DAYS }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  const jobId = (data as { job_id: string }).job_id;
  writeStoredDiscoveryJob(jobId);
  return jobId;
}

export async function pollTicketDiscoveryJob(
  jobId: string,
  onTick?: (progress: DiscoverProgress) => void,
): Promise<DiscoverTicketsResult> {
  const stored = readStoredDiscoveryJob();
  const startedAt = stored && stored.jobId === jobId ? stored.startedAt : Date.now();
  const deadline = startedAt + DISCOVER_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, DISCOVER_POLL_INTERVAL_MS));
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
    const progress = data.progress as DiscoverProgress | undefined;
    if (progress) onTick?.(progress);
    const status = data.status as string;
    if (status === "done") {
      clearStoredDiscoveryJob();
      return (data.result ?? {}) as DiscoverTicketsResult;
    }
    if (status === "error") {
      clearStoredDiscoveryJob();
      throw new Error((data.error as string) || "Échec de l'analyse.");
    }
  }
  clearStoredDiscoveryJob();
  throw new Error("L'analyse prend trop de temps. Réessayez plus tard.");
}

export async function createProjectFromSubjectKey(subjectKey: string): Promise<void> {
  const res = await apiFetch("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: subjectKey,
      rules_matrix: { subject_keys: [subjectKey] },
    }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
}

// Clé déjà suivie (existing_project_id non nul) : Fast-Track de mise à jour,
// même endpoint que le bouton "Rafraîchir" du Project Hub.
export async function updateProjectFromSubjectKey(projectId: string): Promise<void> {
  const res = await apiFetch(`/api/projects/${projectId}/refresh`, { method: "POST" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
}

// Rattachement manuel choisi par l'utilisateur (suggestion confirmée, ou
// choix libre via le sélecteur "Rattacher à un projet existant") — lecture-
// fusion-écriture côté serveur sur rules_matrix.subject_keys.
export async function attachSubjectKeyToProject(projectId: string, subjectKey: string): Promise<void> {
  const res = await apiFetch(`/api/projects/${projectId}/attach-subject-key`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject_key: subjectKey }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
}

export type ProjectOption = { id: string; name: string };

// Liste minimale pour peupler le sélecteur "Rattacher à un projet existant"
// du modal — réutilise GET /api/projects (déjà utilisé par ProjectHub.tsx).
export async function listProjectOptions(): Promise<ProjectOption[]> {
  const res = await apiFetch("/api/projects", { method: "GET" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return (data as Array<{ id: string; name: string }>).map((p) => ({ id: p.id, name: p.name }));
}
