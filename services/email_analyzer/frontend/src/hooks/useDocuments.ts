import { useCallback, useEffect, useState } from "react";
import {
  deleteDocument,
  fetchDocuments,
  listProjectOptions,
  uploadDocument,
  type Document,
  type ProjectOption,
} from "../documentsApi";

// Charge le Contexte Documents (domaine #7) pour la page /documents — backend
// fonctionnel (upload/liste/téléchargement/suppression, extraction PDF/texte)
// depuis le 2026-07-31, jamais appelé depuis le frontend avant cette unité
// (même situation que Memory/Knowledge avant MemoryPage.tsx, Unit 53).
export function useDocuments(enabled: boolean) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [projects, setProjects] = useState<ProjectOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(
    async (filters?: { projectId?: string; keyword?: string }) => {
      if (!enabled) return;
      setLoading(true);
      setError(null);
      try {
        const [docs, projectOptions] = await Promise.all([
          fetchDocuments(filters),
          listProjectOptions(),
        ]);
        setDocuments(docs);
        setProjects(projectOptions);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Erreur");
      } finally {
        setLoading(false);
      }
    },
    [enabled],
  );

  useEffect(() => {
    void reload();
  }, [reload]);

  const upload = useCallback(async (file: File, projectId?: string) => {
    setUploading(true);
    setError(null);
    try {
      const created = await uploadDocument(file, projectId);
      setDocuments((prev) => [created, ...prev]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
      throw e;
    } finally {
      setUploading(false);
    }
  }, []);

  const remove = useCallback(async (id: string) => {
    await deleteDocument(id);
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  }, []);

  return { documents, projects, loading, uploading, error, reload, upload, remove };
}
