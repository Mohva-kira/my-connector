import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de ContactOut/OrganizationOut (api/routers/knowledge.py) — Knowledge
// Layer (pivot exécutif, Phase 6). Premier module frontend à consommer
// /api/knowledge/*.
export type Contact = {
  id: string;
  name: string;
  organization_id: string | null;
  created_at: string;
};

export type Organization = {
  id: string;
  name: string;
  created_at: string;
};

export async function fetchContacts(): Promise<Contact[]> {
  const res = await apiFetch("/api/knowledge/contacts");
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Contact[];
}

export async function fetchOrganizations(): Promise<Organization[]> {
  const res = await apiFetch("/api/knowledge/organizations");
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Organization[];
}

export type BulkSyncResult = {
  synced_projects: number;
  failed_projects: number;
  total_projects: number;
};

export async function syncAllKnowledge(): Promise<BulkSyncResult> {
  const res = await apiFetch("/api/knowledge/sync", { method: "POST" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as BulkSyncResult;
}
