import { useRef, useState } from "react";
import { Navigate } from "react-router-dom";
import { getAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import { useSaasSession } from "../SaasPanels";
import { useDocuments } from "../hooks/useDocuments";
import { downloadDocument } from "../documentsApi";
import BackfillAttachmentsModal from "../BackfillAttachmentsModal";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

// Page "Documents" — première surface frontend pour le Contexte Documents
// (domaine #7 de la feuille de route, project-overview.md). Backend complet
// depuis le 2026-07-31 (upload, extraction PDF/texte, téléchargement,
// suppression) mais sans UI jusqu'ici — même situation que Memory/Knowledge
// avant MemoryPage.tsx (Unit 53).
export default function DocumentsPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);
  const { documents, projects, loading, uploading, error, reload, upload, remove } =
    useDocuments(saasEnabled);
  const [uploadProjectId, setUploadProjectId] = useState("");
  const [filterProjectId, setFilterProjectId] = useState("");
  const [keyword, setKeyword] = useState("");
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [backfillOpen, setBackfillOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!saasEnabled) {
    return <Navigate to="/" replace />;
  }
  if (!getAccessToken()) {
    return <Navigate to="/login" replace />;
  }
  if (!me) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-bg-primary px-4">
        <p className="text-sm text-text-muted">Chargement du compte…</p>
      </div>
    );
  }

  const projectNameById = new Map(projects.map((p) => [p.id, p.name]));

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    try {
      await upload(file, uploadProjectId || undefined);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch {
      // L'erreur est déjà exposée via `error` (état du hook).
    }
  }

  async function handleFilter(e: React.FormEvent) {
    e.preventDefault();
    await reload({ projectId: filterProjectId || undefined, keyword: keyword.trim() || undefined });
  }

  async function handleDownload(id: string, filename: string) {
    setDownloadError(null);
    try {
      await downloadDocument(id, filename);
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : "Erreur de téléchargement");
    }
  }

  return (
    <AppShell me={me}>
      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Documents</h1>
        <p className="mt-1 text-sm text-text-muted">
          Fichiers déposés (PDF, texte), optionnellement rattachés à un projet.
        </p>
      </header>

      {error ? <p className="mb-4 text-sm text-danger">{error}</p> : null}
      {downloadError ? <p className="mb-4 text-sm text-danger">{downloadError}</p> : null}

      <section className="mb-6 rounded-lg border border-border-default bg-surface p-4">
        <h2 className="mb-3 text-sm font-semibold text-text-primary">Déposer un document</h2>
        <form onSubmit={handleUpload} className="flex flex-wrap items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,.csv,application/pdf,text/plain,text/markdown,text/csv"
            className="flex-1 rounded-lg border border-border-default bg-bg-tertiary/40 px-3 py-1.5 text-sm text-text-primary"
          />
          <select
            value={uploadProjectId}
            onChange={(e) => setUploadProjectId(e.target.value)}
            className="rounded-lg border border-border-default bg-bg-tertiary/40 px-2 py-1.5 text-sm text-text-primary"
          >
            <option value="">Aucun projet</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <button
            type="submit"
            disabled={uploading}
            className="shrink-0 rounded-lg bg-accent-primary px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-60"
          >
            {uploading ? "Envoi…" : "Téléverser"}
          </button>
        </form>
        <div className="mt-3 flex items-center justify-between gap-2 border-t border-border-default pt-3">
          <p className="text-xs text-text-muted">
            Récupère les pièces jointes (PDF, texte) des emails déjà analysés, en arrière-plan.
          </p>
          <button
            type="button"
            onClick={() => setBackfillOpen(true)}
            className="shrink-0 rounded-lg px-3 py-1.5 text-xs font-semibold text-text-secondary hover:bg-bg-tertiary"
          >
            Rattraper depuis les emails
          </button>
        </div>
      </section>

      {backfillOpen ? (
        <BackfillAttachmentsModal
          onClose={() => setBackfillOpen(false)}
          onCompleted={() => {
            setBackfillOpen(false);
            void reload({ projectId: filterProjectId || undefined, keyword: keyword.trim() || undefined });
          }}
        />
      ) : null}

      <section className="mb-4 rounded-lg border border-border-default bg-surface p-4">
        <form onSubmit={handleFilter} className="flex flex-wrap items-center gap-2">
          <input
            type="search"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            placeholder="Rechercher (nom, contenu extrait)…"
            className="flex-1 rounded-lg border border-border-default bg-bg-tertiary/40 px-3 py-1.5 text-sm text-text-primary placeholder:text-text-muted"
          />
          <select
            value={filterProjectId}
            onChange={(e) => setFilterProjectId(e.target.value)}
            className="rounded-lg border border-border-default bg-bg-tertiary/40 px-2 py-1.5 text-sm text-text-primary"
          >
            <option value="">Tous les projets</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <button
            type="submit"
            className="shrink-0 rounded-lg px-3 py-1.5 text-xs font-semibold text-text-secondary hover:bg-bg-tertiary"
          >
            Filtrer
          </button>
        </form>
      </section>

      {loading ? <p className="text-sm text-text-muted">Chargement…</p> : null}

      {!loading ? (
        <section className="rounded-lg border border-border-default bg-surface p-4">
          <h2 className="mb-3 text-sm font-semibold text-text-primary">
            Documents ({documents.length})
          </h2>
          {documents.length === 0 ? (
            <p className="text-sm text-text-muted">Aucun document déposé pour l'instant.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {documents.map((doc) => (
                <li
                  key={doc.id}
                  className="flex items-start justify-between gap-3 rounded-lg border border-border-default bg-bg-tertiary/40 p-3"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-text-primary">
                      {doc.filename}
                      {doc.source === "email" ? (
                        <span className="ml-1.5 rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent-primary">
                          Email
                        </span>
                      ) : null}
                    </p>
                    <p className="mt-0.5 text-xs text-text-muted">
                      {formatSize(doc.size_bytes)}
                      {doc.project_id ? ` · ${projectNameById.get(doc.project_id) ?? "Projet inconnu"}` : " · Sans projet"}
                      {doc.has_extracted_text ? " · Texte extrait" : ""}
                      {" · "}
                      {new Date(doc.created_at).toLocaleDateString("fr-FR")}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <button
                      type="button"
                      onClick={() => void handleDownload(doc.id, doc.filename)}
                      className="rounded-lg px-2 py-1 text-xs text-text-secondary hover:bg-bg-tertiary hover:text-accent-primary"
                    >
                      Télécharger
                    </button>
                    <button
                      type="button"
                      onClick={() => void remove(doc.id)}
                      className="rounded-lg px-2 py-1 text-xs text-text-secondary hover:bg-bg-tertiary hover:text-danger"
                    >
                      Supprimer
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : null}
    </AppShell>
  );
}
