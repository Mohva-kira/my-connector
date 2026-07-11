const TOKEN_KEY = "email_analyzer_token";
const PENDING_AUTO_SYNC_KEY = "email_analyzer_pending_auto_sync";

export type PendingAutoSyncJob = { project_id: string; job_id: string };

/**
 * Base URL de l'API, figée au build via VITE_API_BASE.
 * - Vide (défaut) : même origine — fonctionne avec le proxy Vite en dev et
 *   quand FastAPI sert aussi le frontend build.
 * - Définie (ex. https://api.exemple.com) : frontend et API sur des origines
 *   séparées (Nginx sert dist/, uvicorn sert l'API). Nécessite CORS côté API.
 */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

/** Préfixe une URL relative commençant par "/" avec API_BASE. */
export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export function getAccessToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setAccessToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

/**
 * Jobs Fast-Track déclenchés automatiquement par le login (voir
 * api/routers/auth.py::login, TokenResponse.auto_sync_jobs) — passés via
 * sessionStorage plutôt qu'un state React levé, pour survivre à la
 * redirection /login -> / sans re-plomberie de contexte. Consommés une seule
 * fois par ProjectHub (takePendingAutoSync supprime la clé).
 */
export function setPendingAutoSync(jobs: PendingAutoSyncJob[]): void {
  try {
    if (jobs.length > 0) sessionStorage.setItem(PENDING_AUTO_SYNC_KEY, JSON.stringify(jobs));
    else sessionStorage.removeItem(PENDING_AUTO_SYNC_KEY);
  } catch {
    /* ignore */
  }
}

export function takePendingAutoSync(): PendingAutoSyncJob[] {
  try {
    const raw = sessionStorage.getItem(PENDING_AUTO_SYNC_KEY);
    sessionStorage.removeItem(PENDING_AUTO_SYNC_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as PendingAutoSyncJob[]) : [];
  } catch {
    return [];
  }
}

/** fetch avec en-tête Bearer si un jeton est stocké (mode SaaS). */
export async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? undefined);
  const t = getAccessToken();
  if (t) headers.set("Authorization", `Bearer ${t}`);
  const target = typeof input === "string" ? apiUrl(input) : input;
  return fetch(target, { ...init, headers });
}
