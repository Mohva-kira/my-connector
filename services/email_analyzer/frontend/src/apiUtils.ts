export type Health = {
  status: string;
  saas_enabled?: boolean;
  imap_configured?: boolean;
  warning?: string;
};

export function parseApiError(data: unknown): string {
  if (data && typeof data === "object" && "detail" in data) {
    const d = (data as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => JSON.stringify(x)).join(", ");
  }
  return JSON.stringify(data);
}

/** Parse corps HTTP en JSON ou lève une erreur lisible. */
export async function parseResponseJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text.trim()) {
    if (!res.ok) throw new Error(`Erreur ${res.status} (réponse vide)`);
    return {};
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(
      `Réponse non-JSON (${res.status}) : ${text.slice(0, 280)}${text.length > 280 ? "…" : ""}`,
    );
  }
}
