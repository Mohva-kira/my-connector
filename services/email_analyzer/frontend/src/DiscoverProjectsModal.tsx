import { useEffect, useState } from "react";
import {
  createProjectFromDomain,
  getResumableDiscoveryJobId,
  pollDomainDiscoveryJob,
  startDomainDiscovery,
  updateProjectFromDomain,
  type DiscoveredDomain,
  type DiscoverProgress,
} from "./discoverApi";

// Au-delà de ce total, on prévient que le scan peut prendre du temps mais
// continue en arrière-plan — miroir de large_scan_threshold() côté backend
// (config.py), qui bascule aussi la granularité de progression à ce seuil.
const LARGE_SCAN_HINT_THRESHOLD = 1000;

// Messages courts qui tournent pendant le chargement — même convention que
// LOADING_STATUS_MESSAGES (HomePage.tsx) : donner un retour vivant sans
// inventer de pipeline visuel spéculatif (Unit 7 mise en pause).
const LOADING_MESSAGES = [
  "Connexion à votre messagerie…",
  "Lecture des 3 derniers mois…",
  "Regroupement par expéditeur…",
];

type Phase = "loading" | "results" | "message";

// Libellé du bouton de confirmation : distingue les créations des mises à
// jour (domaines déjà couverts par un projet existant) plutôt qu'un seul
// compteur "Ajouter N", pour rendre visible que rien n'est dupliqué.
function confirmLabel(domains: DiscoveredDomain[], selected: Set<string>): string {
  const chosen = domains.filter((d) => selected.has(d.domain));
  const toCreate = chosen.filter((d) => !d.existing_project_id).length;
  const toUpdate = chosen.filter((d) => d.existing_project_id).length;
  const parts: string[] = [];
  if (toCreate > 0) parts.push(`Ajouter ${toCreate} projet${toCreate > 1 ? "s" : ""}`);
  if (toUpdate > 0) parts.push(`mettre à jour ${toUpdate} projet${toUpdate > 1 ? "s" : ""}`);
  return parts.join(" et ") || "Confirmer";
}

export default function DiscoverProjectsModal({
  onClose,
  onProjectsProcessed,
}: {
  onClose: () => void;
  onProjectsProcessed: (created: number, updated: number) => void;
}) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [statusMsgIndex, setStatusMsgIndex] = useState(0);
  const [progress, setProgress] = useState<DiscoverProgress | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [domains, setDomains] = useState<DiscoveredDomain[]>([]);
  const [totalScanned, setTotalScanned] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [creating, setCreating] = useState(false);
  const [createErrors, setCreateErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (phase !== "loading") return;
    const id = setInterval(() => {
      setStatusMsgIndex((i) => (i + 1) % LOADING_MESSAGES.length);
    }, 2000);
    return () => clearInterval(id);
  }, [phase]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const jobId = getResumableDiscoveryJobId() ?? (await startDomainDiscovery());
        const result = await pollDomainDiscoveryJob(jobId, (p) => {
          if (!cancelled) setProgress(p);
        });
        if (cancelled) return;
        if (result._error) {
          setMessage(result._error);
          setPhase("message");
          return;
        }
        if (result._empty || !result.domains || result.domains.length === 0) {
          setMessage(result._message || "Aucun expéditeur externe trouvé sur les 3 derniers mois.");
          setPhase("message");
          return;
        }
        setDomains(result.domains);
        setTotalScanned(result.total_emails_scanned ?? 0);
        setSelected(new Set(result.domains.map((d) => d.domain)));
        setPhase("results");
      } catch (err) {
        if (cancelled) return;
        setMessage(err instanceof Error ? err.message : "Erreur inconnue pendant l'analyse.");
        setPhase("message");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function toggle(domain: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(domain)) next.delete(domain);
      else next.add(domain);
      return next;
    });
  }

  async function handleConfirm() {
    setCreating(true);
    setCreateErrors({});
    const toProcess = domains.filter((d) => selected.has(d.domain));
    const errors: Record<string, string> = {};
    let createdCount = 0;
    let updatedCount = 0;
    for (const d of toProcess) {
      try {
        // Un domaine déjà couvert par un projet existant est mis à jour
        // (Fast-Track) plutôt que redupliqué — voir
        // analysis_tasks.py::_run_domain_discovery_sync (annotation
        // existing_project_id).
        if (d.existing_project_id) {
          await updateProjectFromDomain(d.existing_project_id);
          updatedCount += 1;
        } else {
          await createProjectFromDomain(d.domain);
          createdCount += 1;
        }
      } catch (err) {
        errors[d.domain] = err instanceof Error ? err.message : "Échec du traitement.";
      }
    }
    setCreating(false);
    if (Object.keys(errors).length > 0) {
      setCreateErrors(errors);
      // On ne retire de la sélection que les domaines traités avec succès —
      // ceux en erreur restent affichés et sélectionnés pour un nouvel essai.
      setSelected(new Set(Object.keys(errors)));
      setDomains((prev) => prev.filter((d) => errors[d.domain] !== undefined));
    }
    if (createdCount > 0 || updatedCount > 0) onProjectsProcessed(createdCount, updatedCount);
    if (Object.keys(errors).length === 0) onClose();
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="discover-modal-title"
    >
      <div className="w-full max-w-2xl rounded-2xl bg-surface p-6 shadow-xl">
        {phase === "loading" && (
          <div className="flex flex-col items-center gap-4 py-8 text-center">
            <svg
              className="h-10 w-10 animate-spin text-accent-primary"
              viewBox="0 0 24 24"
              fill="none"
              aria-hidden="true"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <h3 id="discover-modal-title" className="text-lg font-semibold text-text-primary">
              Analyse de vos emails
            </h3>
            {progress && progress.total > 0 ? (
              <div className="w-full max-w-xs">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg-tertiary">
                  <div
                    className="h-full rounded-full bg-accent-primary transition-all"
                    style={{ width: `${Math.min(100, (progress.processed / progress.total) * 100)}%` }}
                  />
                </div>
                <p className="mt-2 text-sm text-text-muted">
                  {progress.processed} / {progress.total} emails analysés
                </p>
                {progress.total > LARGE_SCAN_HINT_THRESHOLD && (
                  <p className="mt-1 text-xs text-text-muted">
                    Boîte volumineuse : l'analyse peut prendre du temps. Vous pouvez fermer cette
                    fenêtre, elle continue en arrière-plan — rouvrez-la pour voir la progression.
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-text-muted">{LOADING_MESSAGES[statusMsgIndex]}</p>
            )}
          </div>
        )}

        {phase === "message" && (
          <>
            <h3 id="discover-modal-title" className="text-lg font-semibold text-text-primary">
              Analyse de vos emails
            </h3>
            <p className="mt-1 text-sm text-text-muted">{message}</p>
            <button
              type="button"
              onClick={onClose}
              className="mt-6 rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white"
            >
              Fermer
            </button>
          </>
        )}

        {phase === "results" && (
          <>
            <h3 id="discover-modal-title" className="text-lg font-semibold text-text-primary">
              Projets détectés
            </h3>
            <p className="mt-1 text-sm text-text-muted">
              {domains.length} domaine{domains.length > 1 ? "s" : ""} expéditeur trouvé
              {domains.length > 1 ? "s" : ""} sur les 3 derniers mois ({totalScanned} email
              {totalScanned > 1 ? "s" : ""} analysé{totalScanned > 1 ? "s" : ""}). Un domaine déjà suivi
              sera mis à jour, les autres seront ajoutés comme nouveaux projets.
            </p>

            <div className="mt-4 grid max-h-[50vh] grid-cols-1 gap-3 overflow-y-auto sm:grid-cols-2">
              {domains.map((d) => (
                <DomainCard
                  key={d.domain}
                  domain={d}
                  selected={selected.has(d.domain)}
                  error={createErrors[d.domain]}
                  disabled={creating}
                  onToggle={() => toggle(d.domain)}
                />
              ))}
            </div>

            <div className="mt-6 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={onClose}
                disabled={creating}
                className="rounded-xl px-4 py-2 text-sm text-text-secondary disabled:opacity-50"
              >
                Annuler
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                disabled={selected.size === 0 || creating}
                className="rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
              >
                {creating ? "Traitement en cours…" : confirmLabel(domains, selected)}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function DomainCard({
  domain,
  selected,
  error,
  disabled,
  onToggle,
}: {
  domain: DiscoveredDomain;
  selected: boolean;
  error?: string;
  disabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={`rounded-lg border p-4 text-left transition disabled:cursor-not-allowed disabled:opacity-60 ${
        selected
          ? "border-accent-primary bg-accent-soft ring-1 ring-accent-primary/30"
          : "border-border-default bg-surface hover:border-accent-primary/40"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="font-medium text-text-primary">{domain.domain}</span>
        <span
          aria-hidden="true"
          className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-xs ${
            selected
              ? "border-accent-primary bg-accent-primary text-white"
              : "border-border-default text-transparent"
          }`}
        >
          ✓
        </span>
      </div>
      {domain.existing_project_name && (
        <p className="mt-1 inline-block rounded-full bg-info/10 px-2 py-0.5 text-xs font-medium text-info">
          Déjà suivi : {domain.existing_project_name}
        </p>
      )}
      <p className="mt-1 text-xs text-text-secondary">
        {domain.email_count} email{domain.email_count > 1 ? "s" : ""} · {domain.sender_count} expéditeur
        {domain.sender_count > 1 ? "s" : ""}
      </p>
      {domain.sample_subjects[0] && (
        <p className="mt-2 truncate text-xs text-text-muted">« {domain.sample_subjects[0]} »</p>
      )}
      {error && <p className="mt-2 text-xs text-danger">{error}</p>}
    </button>
  );
}
