import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Fenêtre fixe du bouton "Analyser" du Brief (3 mois) — pas de sélecteur de
// plage ici, contrairement à AnalyzeRangeModal (ProjectHub.tsx) qui répond à
// un besoin différent (rafraîchir un projet existant sur une plage choisie).
const DISCOVER_DAYS = 90;

// Même cadence de polling que pollRefreshJob (ProjectHub.tsx). Budget global
// aligné sur le timeout dédié de run_domain_discovery côté worker arq (7200s,
// WorkerSettings.functions, analysis_tasks.py) plutôt que sur
// ANALYZE_POLL_TIMEOUT_MS (5 min, HomePage.tsx) : mesuré en conditions
// réelles, un scan sans filtre sur 90 jours peut porter sur plusieurs
// milliers d'emails (~5900 candidats sur une boîte active) et dépasser
// largement 5 minutes même en ne récupérant que les en-têtes.
const DISCOVER_POLL_INTERVAL_MS = 2500;
const DISCOVER_POLL_TIMEOUT_MS = 126 * 60 * 1000;

// Persiste le job en cours pour survivre à une fermeture/réouverture du modal
// (ou un refresh de page) — sans ça, un scan qui continue légitimement en
// arrière-plan côté worker devenait invisible dès que le composant démontait,
// et rouvrir le modal relançait un nouveau scan complet au lieu de se
// rattacher à celui déjà en cours.
const DISCOVER_JOB_STORAGE_KEY = "myconnector:discover-job";

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

// À appeler avant `startDomainDiscovery()` : renvoie l'id d'un scan encore
// dans son budget de polling pour s'y rattacher au lieu d'en relancer un.
export function getResumableDiscoveryJobId(): string | null {
  return readStoredDiscoveryJob()?.jobId ?? null;
}

export type DiscoverProgress = { processed: number; total: number };

// Miroir du dict renvoyé par EmailProcessor.discover_sender_domains
// (email_analyzer/analyzer.py), annoté côté worker
// (analysis_tasks.py::_run_domain_discovery_sync) avec le projet existant
// couvrant déjà ce domaine, s'il y en a un — permet de proposer une mise à
// jour plutôt qu'une création en double.
export type DiscoveredDomain = {
  domain: string;
  email_count: number;
  sender_count: number;
  sample_senders: string[];
  sample_subjects: string[];
  latest_received_at: string | null;
  existing_project_id: string | null;
  existing_project_name: string | null;
};

export type DiscoverResult = {
  _error?: string;
  _empty?: boolean;
  _message?: string;
  domains?: DiscoveredDomain[];
  total_emails_scanned?: number;
  days_back?: number;
};

export async function startDomainDiscovery(): Promise<string> {
  const res = await apiFetch("/api/brief/discover-projects", {
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

export async function pollDomainDiscoveryJob(
  jobId: string,
  onTick?: (progress: DiscoverProgress) => void,
): Promise<DiscoverResult> {
  // Reprend le budget depuis le démarrage réel du job stocké (cas d'une
  // reprise après réouverture) plutôt que de repartir sur un budget plein à
  // chaque appel de `pollDomainDiscoveryJob`.
  const stored = readStoredDiscoveryJob();
  const startedAt = stored && stored.jobId === jobId ? stored.startedAt : Date.now();
  const deadline = startedAt + DISCOVER_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, DISCOVER_POLL_INTERVAL_MS));
    // Un tick isolé qui échoue (abort 30s, hoquet réseau) ne doit pas tuer tout
    // le polling tant que le budget global n'est pas dépassé — même correctif
    // que pollAnalysis (HomePage.tsx), nécessaire ici pour la même raison sur
    // un polling tout aussi long.
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
      return (data.result ?? {}) as DiscoverResult;
    }
    if (status === "error") {
      clearStoredDiscoveryJob();
      throw new Error((data.error as string) || "Échec de l'analyse.");
    }
  }
  clearStoredDiscoveryJob();
  throw new Error("L'analyse prend trop de temps. Réessayez plus tard.");
}

export async function createProjectFromDomain(domain: string): Promise<void> {
  const res = await apiFetch("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: domain,
      rules_matrix: { sender_domains: [domain] },
    }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
}

// Domaine déjà couvert par un projet existant (`existing_project_id` non
// nul) : déclenche une mise à jour Fast-Track de ce projet plutôt qu'une
// création — même endpoint que le bouton "Rafraîchir" du Project Hub
// (POST /api/projects/{id}/refresh). Fire-and-forget : le modal se ferme
// sans attendre la fin du job, comme pour tout Fast-Track déclenché ailleurs
// dans l'app (le prochain chargement de /projects reflète le résultat).
export async function updateProjectFromDomain(projectId: string): Promise<void> {
  const res = await apiFetch(`/api/projects/${projectId}/refresh`, { method: "POST" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
}
