import { useEffect, useState } from "react";
import {
  attachSubjectKeyToProject,
  createProjectFromSubjectKey,
  getResumableTicketDiscoveryJobId,
  listProjectOptions,
  pollTicketDiscoveryJob,
  startTicketDiscovery,
  updateProjectFromSubjectKey,
  type DiscoveredSubjectKey,
  type DiscoverProgress,
  type ProjectOption,
} from "./discoverTicketsApi";

// Mirroring de DiscoverProjectsModal.tsx (découverte de domaines), en sibling
// plutôt qu'un flag de mode : ce modal a une troisième action que l'autre n'a
// pas (rattachement manuel à un projet existant, cf. décision utilisateur —
// "créer un nouveau projet OU choisir manuellement un projet existant").
const LARGE_SCAN_HINT_THRESHOLD = 1000;

const LOADING_MESSAGES = [
  "Connexion à votre messagerie…",
  "Lecture des 3 derniers mois…",
  "Regroupement par ticket détecté dans les sujets…",
];

type Phase = "loading" | "results" | "message";

// Valeur du sélecteur "Rattacher à un projet existant" pour une clé sans
// correspondance certaine : "" = créer un nouveau projet, sinon l'id du
// projet choisi (pré-rempli à suggested_project_id si présent).
type Choice = string;

function confirmLabel(
  keys: DiscoveredSubjectKey[],
  selected: Set<string>,
  choices: Record<string, Choice>,
): string {
  const chosen = keys.filter((k) => selected.has(k.subject_key));
  const toUpdate = chosen.filter((k) => k.existing_project_id).length;
  const toCreate = chosen.filter((k) => !k.existing_project_id && !choices[k.subject_key]).length;
  const toAttach = chosen.filter((k) => !k.existing_project_id && choices[k.subject_key]).length;
  const parts: string[] = [];
  if (toCreate > 0) parts.push(`créer ${toCreate} projet${toCreate > 1 ? "s" : ""}`);
  if (toAttach > 0) parts.push(`rattacher ${toAttach} ticket${toAttach > 1 ? "s" : ""}`);
  if (toUpdate > 0) parts.push(`mettre à jour ${toUpdate} projet${toUpdate > 1 ? "s" : ""}`);
  if (parts.length === 0) return "Confirmer";
  return parts[0].charAt(0).toUpperCase() + parts[0].slice(1) + parts.slice(1).map((p) => ` et ${p}`).join("");
}

export default function DiscoverTicketsModal({
  onClose,
  onTicketsProcessed,
}: {
  onClose: () => void;
  onTicketsProcessed: (created: number, updated: number, attached: number) => void;
}) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [statusMsgIndex, setStatusMsgIndex] = useState(0);
  const [progress, setProgress] = useState<DiscoverProgress | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [subjectKeys, setSubjectKeys] = useState<DiscoveredSubjectKey[]>([]);
  const [totalScanned, setTotalScanned] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // Choix du sélecteur "Rattacher à un projet existant" par clé sans match
  // certain — "" (non présent dans la map) = créer un nouveau projet.
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const [projectOptions, setProjectOptions] = useState<ProjectOption[]>([]);
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
        const jobId = getResumableTicketDiscoveryJobId() ?? (await startTicketDiscovery());
        const result = await pollTicketDiscoveryJob(jobId, (p) => {
          if (!cancelled) setProgress(p);
        });
        if (cancelled) return;
        if (result._error) {
          setMessage(result._error);
          setPhase("message");
          return;
        }
        if (result._empty || !result.subject_keys || result.subject_keys.length === 0) {
          setMessage(result._message || "Aucun ticket détecté dans les sujets sur les 3 derniers mois.");
          setPhase("message");
          return;
        }
        const keys = result.subject_keys;
        setSubjectKeys(keys);
        setTotalScanned(result.total_emails_scanned ?? 0);
        setSelected(new Set(keys.map((k) => k.subject_key)));
        const initialChoices: Record<string, Choice> = {};
        for (const k of keys) {
          if (!k.existing_project_id && k.suggested_project_id) {
            initialChoices[k.subject_key] = k.suggested_project_id;
          }
        }
        setChoices(initialChoices);
        // Peuple le sélecteur une seule fois — mêmes projets pour toutes les
        // cartes, pas besoin de refetch par carte.
        try {
          setProjectOptions(await listProjectOptions());
        } catch {
          // Le sélecteur reste vide (fallback "Créer un nouveau projet" only) ;
          // ne bloque pas l'affichage des résultats de la découverte.
        }
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

  function toggle(key: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function setChoice(key: string, projectId: string) {
    setChoices((prev) => {
      const next = { ...prev };
      if (projectId) next[key] = projectId;
      else delete next[key];
      return next;
    });
  }

  async function handleConfirm() {
    setCreating(true);
    setCreateErrors({});
    const toProcess = subjectKeys.filter((k) => selected.has(k.subject_key));
    const errors: Record<string, string> = {};
    let createdCount = 0;
    let updatedCount = 0;
    let attachedCount = 0;
    for (const k of toProcess) {
      try {
        if (k.existing_project_id) {
          await updateProjectFromSubjectKey(k.existing_project_id);
          updatedCount += 1;
        } else if (choices[k.subject_key]) {
          await attachSubjectKeyToProject(choices[k.subject_key], k.subject_key);
          attachedCount += 1;
        } else {
          await createProjectFromSubjectKey(k.subject_key);
          createdCount += 1;
        }
      } catch (err) {
        errors[k.subject_key] = err instanceof Error ? err.message : "Échec du traitement.";
      }
    }
    setCreating(false);
    if (Object.keys(errors).length > 0) {
      setCreateErrors(errors);
      setSelected(new Set(Object.keys(errors)));
      setSubjectKeys((prev) => prev.filter((k) => errors[k.subject_key] !== undefined));
    }
    if (createdCount > 0 || updatedCount > 0 || attachedCount > 0) {
      onTicketsProcessed(createdCount, updatedCount, attachedCount);
    }
    if (Object.keys(errors).length === 0) onClose();
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="discover-tickets-modal-title"
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
            <h3 id="discover-tickets-modal-title" className="text-lg font-semibold text-text-primary">
              Détection des tickets
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
            <h3 id="discover-tickets-modal-title" className="text-lg font-semibold text-text-primary">
              Détection des tickets
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
            <h3 id="discover-tickets-modal-title" className="text-lg font-semibold text-text-primary">
              Tickets détectés
            </h3>
            <p className="mt-1 text-sm text-text-muted">
              {subjectKeys.length} clé{subjectKeys.length > 1 ? "s" : ""} de ticket trouvée
              {subjectKeys.length > 1 ? "s" : ""} sur les 3 derniers mois ({totalScanned} email
              {totalScanned > 1 ? "s" : ""} analysé{totalScanned > 1 ? "s" : ""}). Une clé déjà suivie sera
              mise à jour ; pour les autres, choisissez de créer un nouveau projet ou de rattacher à un
              projet existant.
            </p>

            <div className="mt-4 flex max-h-[50vh] flex-col gap-3 overflow-y-auto">
              {subjectKeys.map((k) => (
                <SubjectKeyCard
                  key={k.subject_key}
                  subjectKey={k}
                  selected={selected.has(k.subject_key)}
                  choice={choices[k.subject_key] ?? ""}
                  projectOptions={projectOptions}
                  error={createErrors[k.subject_key]}
                  disabled={creating}
                  onToggle={() => toggle(k.subject_key)}
                  onChoiceChange={(projectId) => setChoice(k.subject_key, projectId)}
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
                {creating ? "Traitement en cours…" : confirmLabel(subjectKeys, selected, choices)}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function SubjectKeyCard({
  subjectKey,
  selected,
  choice,
  projectOptions,
  error,
  disabled,
  onToggle,
  onChoiceChange,
}: {
  subjectKey: DiscoveredSubjectKey;
  selected: boolean;
  choice: Choice;
  projectOptions: ProjectOption[];
  error?: string;
  disabled: boolean;
  onToggle: () => void;
  onChoiceChange: (projectId: string) => void;
}) {
  const isAutoAttached = Boolean(subjectKey.existing_project_id);
  return (
    <div
      className={`rounded-lg border p-4 transition ${
        selected
          ? "border-accent-primary bg-accent-soft ring-1 ring-accent-primary/30"
          : "border-border-default bg-surface"
      }`}
    >
      <button type="button" onClick={onToggle} disabled={disabled} className="flex w-full items-start justify-between gap-2 text-left disabled:cursor-not-allowed">
        <span className="font-medium text-text-primary">{subjectKey.subject_key}</span>
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
      </button>

      {isAutoAttached && (
        <p className="mt-1 inline-block rounded-full bg-info/10 px-2 py-0.5 text-xs font-medium text-info">
          Déjà suivi : {subjectKey.existing_project_name}
        </p>
      )}
      {!isAutoAttached && subjectKey.suggested_project_name && (
        <p className="mt-1 inline-block rounded-full bg-warning/10 px-2 py-0.5 text-xs font-medium text-warning">
          Suggestion : rattacher à {subjectKey.suggested_project_name} ?
        </p>
      )}

      <p className="mt-1 text-xs text-text-secondary">
        {subjectKey.email_count} email{subjectKey.email_count > 1 ? "s" : ""} ·{" "}
        {subjectKey.distinct_ticket_count} ticket{subjectKey.distinct_ticket_count > 1 ? "s" : ""} distinct
        {subjectKey.distinct_ticket_count > 1 ? "s" : ""}
      </p>
      {subjectKey.sample_subjects[0] && (
        <p className="mt-2 truncate text-xs text-text-muted">« {subjectKey.sample_subjects[0]} »</p>
      )}

      {!isAutoAttached && (
        <label className="mt-3 block text-xs text-text-secondary">
          Rattacher à un projet existant
          <select
            value={choice}
            disabled={disabled}
            onChange={(e) => onChoiceChange(e.target.value)}
            className="mt-1 w-full rounded-lg border border-border-default bg-surface px-2 py-1.5 text-sm text-text-primary disabled:opacity-50"
          >
            <option value="">Créer un nouveau projet</option>
            {projectOptions.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      )}

      {error && <p className="mt-2 text-xs text-danger">{error}</p>}
    </div>
  );
}
