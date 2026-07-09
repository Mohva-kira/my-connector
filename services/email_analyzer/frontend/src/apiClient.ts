const TOKEN_KEY = "email_analyzer_token";

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

/** fetch avec en-tête Bearer si un jeton est stocké (mode SaaS). */
export async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? undefined);
  const t = getAccessToken();
  if (t) headers.set("Authorization", `Bearer ${t}`);
  const target = typeof input === "string" ? apiUrl(input) : input;
  return fetch(target, { ...init, headers });
}
