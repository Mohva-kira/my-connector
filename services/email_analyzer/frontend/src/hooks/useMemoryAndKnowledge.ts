import { useCallback, useEffect, useState } from "react";
import { deleteMemory, fetchMemories, type Memory } from "../memoryApi";
import { fetchContacts, fetchOrganizations, type Contact, type Organization } from "../knowledgeApi";

// Charge Memory (Phase 5) + Knowledge (Phase 6) pour la page /memory —
// mêmes deux piliers déjà injectés silencieusement dans le chat portefeuille
// depuis Unit 48 (ContextService), rendus visibles ici pour la première fois.
// `enabled=false` (mode legacy) n'appelle jamais ces endpoints (503 sans SaaS).
export function useMemoryAndKnowledge(enabled: boolean) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError(null);
    try {
      const [m, c, o] = await Promise.all([fetchMemories(), fetchContacts(), fetchOrganizations()]);
      setMemories(m);
      setContacts(c);
      setOrganizations(o);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const removeMemory = useCallback(async (id: string) => {
    await deleteMemory(id);
    setMemories((prev) => prev.filter((m) => m.id !== id));
  }, []);

  return { memories, contacts, organizations, loading, error, reload, removeMemory };
}
