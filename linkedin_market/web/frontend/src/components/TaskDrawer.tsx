import { createContext, ReactNode, useContext, useEffect, useId, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { FINAL_STATUSES, RunResult, RunSourceResult, SOURCE_LABEL, SourceCode, Task, TaskStatus, api, blockedSourcesOf, isBlockedRun, sourceLabel } from "../api";
import { invalidateAfterTask, useTask, useTasks } from "../hooks";
import { Badge, Button, ErrorBox, Modal, Notice, formatDate, inputClass } from "../ui";

interface TaskUi {
  /** Ouvre le tiroir sur une tâche venant d'être lancée. */
  track: (task: Task) => void;
}

const Ctx = createContext<TaskUi>({ track: () => undefined });
export const useTaskUi = () => useContext(Ctx);

const toneOf = (status: Task["status"]) =>
  status === "terminée" ? "success" : status === "échec" ? "danger" : status === "annulée" ? "neutral"
  : status === "en_cours" ? "info" : "warning";

/** Libellés français des statuts de tâche (frontend/02 § 4.3). */
export const TASK_STATUS_LABEL: Record<TaskStatus, string> = {
  en_attente: "En attente",
  en_cours: "En cours",
  attente_saisie: "Réponse attendue",
  attente_verification_manuelle: "Vérification manuelle à faire",
  terminée: "Terminée",
  échec: "Échec",
  annulée: "Annulée",
};

/**
 * E-4 option A (G-10) : analyse terminée sur un rapport partiel, la collecte ayant été interrompue. Nomme les sources
 * bloquées (`result.blocked_sources`), « LinkedIn » à défaut (tâche antérieure à la vague 5).
 */
export const blockedRunNotice = (task: Task) =>
  `Collecte interrompue par ${blockedSourcesOf(task).map(sourceLabel).join(", ")} : rapport établi sur les données déjà collectées.`;

const SOURCE_STATUS_TONE: Record<RunSourceResult["status"], "success" | "warning" | "neutral" | "danger"> = {
  ok: "success", bloquée: "warning", indisponible: "danger", annulée: "neutral",
};

/** Résultat par source d'une collecte (vague 5) ; rien si la tâche n'en porte pas (tâche antérieure ou sans collecte). */
function RunSources({ task }: { task: Task }) {
  if (task.kind !== "run") return null;
  const sources = (task.result as Partial<RunResult> | null)?.sources;
  if (!Array.isArray(sources) || sources.length === 0) return null;
  return (
    <ul className="space-y-1.5 rounded-md border border-border-default p-2 text-sm" aria-label="Résultat par source">
      {sources.map((source) => (
        <li key={source.code}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>{sourceLabel(source.code)} — {source.items} offre{source.items > 1 ? "s" : ""}</span>
            <Badge tone={SOURCE_STATUS_TONE[source.status] ?? "neutral"}>{source.status}</Badge>
          </div>
          {source.detail && <p className="text-xs text-text-secondary">{source.detail}</p>}
        </li>
      ))}
    </ul>
  );
}

/** Source nommée par la tâche en attente de vérification manuelle, si `meta.source` en désigne une connue. */
function manualCheckSource(task: Task): string | null {
  const code = task.meta.source;
  return typeof code === "string" && code in SOURCE_LABEL ? SOURCE_LABEL[code as SourceCode] : null;
}

/** Statut affiché d'une tâche, avec la mention « collecte interrompue » pour une analyse partielle. */
function StatusBadges({ task }: { task: Task }) {
  return (
    <>
      <Badge tone={toneOf(task.status)}>{TASK_STATUS_LABEL[task.status]}</Badge>
      {isBlockedRun(task) && <Badge tone="warning">Collecte interrompue</Badge>}
    </>
  );
}

const isWaiting = (task: Task) => task.status === "attente_saisie" || task.status === "attente_verification_manuelle";
const isConfirm = (task: Task) => !!task.needs_input?.field.startsWith("confirm");

function TaskDetail({ id }: { id: string }) {
  const { data: task } = useTask(id);
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState<unknown>(null);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [task?.logs.length]);

  if (!task) return <p className="text-sm text-text-muted">Chargement…</p>;
  const final = FINAL_STATUSES.includes(task.status);

  const send = async () => {
    try {
      setError(null);
      await api(`/tasks/${task.id}/answer`, { method: "POST", body: { value: answer, remember: true } });
      setAnswer("");
    } catch (e) { setError(e); }
  };
  const choose = async (option: string) => {
    try {
      setError(null);
      await api(`/tasks/${task.id}/answer`, { method: "POST", body: { value: option, remember: true } });
    } catch (e) { setError(e); }
  };
  const cancel = async () => {
    try { setError(null); await api(`/tasks/${task.id}/cancel`, { method: "POST" }); } catch (e) { setError(e); }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2" aria-live="polite">
        <StatusBadges task={task} />
        {!final && <Button small variant="danger" onClick={cancel}>Annuler</Button>}
      </div>
      {isBlockedRun(task) && <Notice tone="warning">{blockedRunNotice(task)}</Notice>}
      <RunSources task={task} />
      {task.status === "attente_verification_manuelle" && (
        <p className="rounded-md bg-warning/10 p-2 text-sm text-warning-strong">
          {manualCheckSource(task) ?? "Le site"} demande une vérification : termine-la dans la fenêtre du navigateur ouverte par l'outil.
        </p>
      )}
      {task.needs_input && (
        <div className="space-y-2 rounded-md border border-warning/40 bg-warning/10 p-3">
          <p className="text-sm font-medium">{task.needs_input.question}</p>
          {isConfirm(task) ? (
            // I-04 : une relecture (`confirm_*`) ne se répond JAMAIS depuis le tiroir, sans le texte `review` sous les
            // yeux ; seule la modale « Relire avant l'envoi » propose ses options.
            <p className="text-sm">Relecture ouverte : confirme ou annule dans la fenêtre « Relire avant l'envoi »</p>
          ) : task.needs_input.options.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {task.needs_input.options.map((option) => (
                <Button key={option} small onClick={() => choose(option)}>{option}</Button>
              ))}
            </div>
          ) : (
            <div className="flex gap-2">
              <input className={inputClass} value={answer} onChange={(e) => setAnswer(e.target.value)} placeholder={task.needs_input.suggestion ? `Suggestion : ${task.needs_input.suggestion}` : "Ta réponse (mémorisée pour la suite)"} />
              <Button variant="primary" onClick={send} disabled={!answer.trim()}>Envoyer</Button>
            </div>
          )}
        </div>
      )}
      {task.error && <ErrorBox error={task.error} />}
      <ErrorBox error={error} />
      {task.result && (
        <pre className="max-h-32 overflow-auto rounded-md bg-bg-secondary p-2 text-xs">{JSON.stringify(task.result, null, 2)}</pre>
      )}
      <pre ref={logRef} className="h-56 overflow-auto whitespace-pre-wrap rounded-md bg-bg-secondary p-2 font-mono text-xs text-text-secondary">
        {task.logs.join("\n") || "Aucun message pour l'instant."}
      </pre>
    </div>
  );
}

/** Référence stable : un `[]` recréé à chaque rendu relancerait sans fin le calcul d'ouverture automatique. */
const NO_TASKS: Task[] = [];

export function TaskProvider({ children }: { children: ReactNode }) {
  const { data: tasks = NO_TASKS } = useTasks();
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const seen = useRef<Map<string, string>>(new Map());
  const drawerId = useId();

  // Quand une tâche se termine, les données qu'elle a pu changer sont périmées (invalidation selon son type).
  useEffect(() => {
    for (const task of tasks) {
      const before = seen.current.get(task.id);
      if (before && before !== task.status && FINAL_STATUSES.includes(task.status)) invalidateAfterTask(client, task);
      seen.current.set(task.id, task.status);
    }
  }, [tasks, client]);

  // Une tâche qui attend l'utilisateur (question, vérification manuelle) ouvre le tiroir une seule fois. Calculé
  // pendant le rendu, à chaque nouvelle liste de tâches (pas de setState dans un effet).
  const [prompted, setPrompted] = useState<{ tasks: Task[]; keys: ReadonlySet<string> }>({ tasks: NO_TASKS, keys: new Set() });
  if (prompted.tasks !== tasks) {
    const keys = new Set(prompted.keys);
    let toOpen: Task | undefined;
    for (const task of tasks) {
      const key = `${task.id}:${task.needs_input?.question ?? task.status}`;
      if (isWaiting(task) && !keys.has(key)) {
        keys.add(key);
        if (!isConfirm(task)) toOpen = toOpen ?? task;
      }
    }
    setPrompted({ tasks, keys });
    if (toOpen) { setSelected(toOpen.id); setOpen(true); }
  }

  const confirming = tasks.find((t) => t.status === "attente_saisie" && isConfirm(t));
  // Relecture (I-04) : toute fermeture (Échap, ✕) répond TOUJOURS « Annuler » (le serveur traite toute valeur autre
  // que « Soumettre » comme une annulation). Le focus initial va sur le bouton « Annuler » s'il est proposé, sinon sur
  // le titre : aucune action d'envoi n'est jamais l'action par défaut.
  const CANCEL = "Annuler";
  const hasCancel = (confirming?.needs_input?.options ?? []).includes(CANCEL);
  const safeButton = useRef<HTMLButtonElement>(null);
  const [confirmError, setConfirmError] = useState<unknown>(null);
  const answerConfirm = async (task: Task, value: string) => {
    try {
      setConfirmError(null);
      await api(`/tasks/${task.id}/answer`, { method: "POST", body: { value, remember: false } });
      client.invalidateQueries({ queryKey: ["tasks"] });
    } catch (e) { setConfirmError(e); }
  };

  const active = tasks.filter((t) => !FINAL_STATUSES.includes(t.status));
  const needsAttention = active.some(isWaiting);

  return (
    <Ctx.Provider value={{ track: (task) => { setSelected(task.id); setOpen(true); client.invalidateQueries({ queryKey: ["tasks"] }); } }}>
      {children}
      {confirming?.needs_input && (
        <Modal title="Relire avant l'envoi" onClose={() => answerConfirm(confirming, CANCEL)} wide dismissible={false}
          initialFocus={hasCancel ? safeButton : undefined} focusTitle={!hasCancel}>
          <p className="mb-2 text-sm font-medium">{confirming.needs_input.question}</p>
          <p className="mb-2 text-xs text-text-muted">Voici ce que LinkedIn affiche à l'écran de relecture. Rien n'est envoyé tant que tu ne cliques pas sur « Soumettre ». Le navigateur ouvert par l'outil montre la même page.</p>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md bg-bg-secondary p-3 text-xs">{confirming.needs_input.review || "Aucun résumé disponible."}</pre>
          <div className="mt-3"><ErrorBox error={confirmError} /></div>
          <div className="mt-4 flex justify-end gap-2">
            {confirming.needs_input.options.map((option, i) => (
              <Button key={option} ref={option === CANCEL ? safeButton : undefined} variant={i === 0 ? "primary" : "secondary"}
                onClick={() => answerConfirm(confirming, option)}>{option}</Button>
            ))}
          </div>
        </Modal>
      )}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={open ? drawerId : undefined}
        className={`fixed bottom-4 right-4 z-40 rounded-full border px-4 py-2 text-sm font-medium shadow-lift ${needsAttention ? "border-warning bg-warning/15 text-warning-strong" : "border-border-default bg-surface"}`}
      >
        Tâches{active.length > 0 ? ` · ${active.length} en cours` : ""}
      </button>
      {open && (
        <aside id={drawerId} className="fixed bottom-16 right-4 z-40 flex max-h-[75vh] w-[min(28rem,calc(100vw-2rem))] flex-col overflow-hidden rounded-xl border border-border-default bg-surface shadow-lift" aria-label="Tâches">
          <div className="flex items-center justify-between border-b border-border-default px-4 py-2.5">
            <h2 className="text-sm font-semibold">{selected ? "Détail de la tâche" : "Tâches"}</h2>
            <div className="flex gap-1">
              {selected && <Button small variant="ghost" onClick={() => setSelected(null)}>← Liste</Button>}
              <Button small variant="ghost" onClick={() => setOpen(false)}>Fermer</Button>
            </div>
          </div>
          <div className="overflow-y-auto p-4">
            {selected ? (
              <TaskDetail id={selected} />
            ) : tasks.length === 0 ? (
              <p className="text-sm text-text-muted">Aucune tâche.</p>
            ) : (
              <ul className="space-y-2">
                {tasks.map((task) => (
                  <li key={task.id}>
                    <button onClick={() => setSelected(task.id)} className="w-full rounded-md border border-border-default p-2 text-left hover:bg-bg-secondary">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-sm font-medium">{task.title}</span>
                        <span className="flex shrink-0 gap-1"><StatusBadges task={task} /></span>
                      </div>
                      <div className="mt-0.5 text-xs text-text-muted">{formatDate(task.created_at)}</div>
                      {task.error && <div className="mt-1 truncate text-xs text-danger-strong">{task.error}</div>}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      )}
    </Ctx.Provider>
  );
}
