import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de MemoryOut (api/routers/memory.py) — Memory Engine (pivot exécutif,
// Phase 5). Premier module frontend à consommer /api/memory.
export type Memory = {
  id: string;
  content: string;
  category: string;
  source: string;
  confidence: number;
  created_at: string;
};

export async function fetchMemories(): Promise<Memory[]> {
  const res = await apiFetch("/api/memory");
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Memory[];
}

export async function deleteMemory(id: string): Promise<void> {
  const res = await apiFetch(`/api/memory/${id}`, { method: "DELETE" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
}
