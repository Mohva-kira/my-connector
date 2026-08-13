import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de DocumentOut (api/routers/documents.py) — Contexte Documents
// (domaine #7 de la feuille de route, project-overview.md). Backend complet
// depuis le 2026-07-31 (modèle, migration, service, extraction PDF/texte,
// API) mais sans consommateur frontend avant cette unité.
export type Document = {
  id: string;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  project_id: string | null;
  has_extracted_text: boolean;
  // "upload" (dépôt manuel) ou "email" (pièce jointe ingérée automatiquement
  // par le Fast-Track, 2026-08-01) — `null` pour les documents antérieurs à
  // cette distinction, traités comme "upload".
  source: string | null;
  created_at: string;
};

export async function fetchDocuments(opts?: {
  projectId?: string;
  keyword?: string;
}): Promise<Document[]> {
  const params = new URLSearchParams();
  if (opts?.projectId) params.set("project_id", opts.projectId);
  if (opts?.keyword) params.set("keyword", opts.keyword);
  const qs = params.toString();
  const res = await apiFetch(`/api/documents${qs ? `?${qs}` : ""}`);
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Document[];
}

export async function uploadDocument(file: File, projectId?: string): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  if (projectId) form.append("project_id", projectId);
  // Pas de Content-Type manuel : le navigateur fixe le boundary multipart.
  const res = await apiFetch("/api/documents/upload", { method: "POST", body: form });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Document;
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await apiFetch(`/api/documents/${id}`, { method: "DELETE" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
}

// Téléchargement authentifié : la session vit dans un Bearer token
// (apiFetch), pas un cookie — un <a href> direct vers l'API n'enverrait pas
// le jeton. On matérialise donc un blob local pour déclencher le
// téléchargement navigateur.
export async function downloadDocument(id: string, filename: string): Promise<void> {
  const res = await apiFetch(`/api/documents/${id}/download`);
  if (!res.ok) {
    const data = await parseResponseJson(res);
    throw new Error(parseApiError(data));
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = window.document.createElement("a");
  link.href = url;
  link.download = filename;
  window.document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export type ProjectOption = { id: string; name: string };

// Liste minimale pour peupler le sélecteur "Projet" du formulaire de dépôt —
// même convention que discoverTicketsApi.ts::listProjectOptions (réutilise
// GET /api/projects, déjà utilisé par ProjectHub.tsx).
export async function listProjectOptions(): Promise<ProjectOption[]> {
  const res = await apiFetch("/api/projects", { method: "GET" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return (data as Array<{ id: string; name: string }>).map((p) => ({ id: p.id, name: p.name }));
}
