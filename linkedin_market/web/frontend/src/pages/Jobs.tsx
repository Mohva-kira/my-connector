import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Job, ManualItemCreated, SOURCE_LABEL, SourceCode, Task, api, enc, sourceLabel } from "../api";
import { Icon } from "../components/icons";
import { ManualImportModal } from "../components/ManualImportModal";
import { RunModal } from "../components/RunModal";
import { useTaskUi } from "../components/TaskDrawer";
import { useAction, useJob, useJobs, useLabelJob, useSources } from "../hooks";
import { Badge, Button, Card, Empty, ErrorBox, Modal, Notice, PageHeader, formatDate, inputClass, scoreTone, statusTone } from "../ui";

export function JobDetailModal({ id, onClose }: { id: string; onClose: () => void }) {
  const { data: job, error } = useJob(id);
  const { track } = useTaskUi();
  const navigate = useNavigate();
  const prepare = useAction(async () => api<Task>(`/jobs/${enc(id)}/prepare`, { method: "POST" }), [["tasks"]]);
  const [showText, setShowText] = useState(false);

  return (
    <Modal title={job ? job.title : "Offre"} onClose={onClose} wide>
      <ErrorBox error={error} />
      {job && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-sm text-text-secondary">
            <Badge tone={scoreTone(job.score)}>{job.score} %</Badge>
            <Badge>{sourceLabel(job.source_code)}</Badge>
            <span>{job.company || "Entreprise inconnue"}</span>
            {job.location && <span>· {job.location}</span>}
            {job.posted_at && <span>· publiée le {job.posted_at}</span>}
            {job.freelance && <Badge tone="info">Freelance</Badge>}
            {job.remote && <Badge tone="info">Remote</Badge>}
            {job.years_required ? <Badge>{job.years_required}+ ans</Badge> : null}
            {job.seniority_required && <Badge>{job.seniority_required}</Badge>}
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Compétences que tu as</h3>
              <div className="flex flex-wrap gap-1">{job.matched_skills.length ? job.matched_skills.map((s) => <Badge key={s} tone="success">✓ {s}</Badge>) : <span className="text-sm text-text-muted">Aucune</span>}</div>
            </div>
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Compétences manquantes</h3>
              <div className="flex flex-wrap gap-1">{job.missing_skills.length ? job.missing_skills.map((s) => <Badge key={s} tone="danger">✗ {s}</Badge>) : <span className="text-sm text-text-muted">Aucune</span>}</div>
            </div>
          </div>
          <div>
            <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Critères du score</h3>
            <ul className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
              {Object.entries(job.criteria).map(([name, v]) => (
                <li key={name} className="flex items-center gap-2 text-xs">
                  <span className="w-40 shrink-0 truncate text-text-secondary">{name.replace(/_/g, " ")}</span>
                  <span className="h-1.5 flex-1 rounded-full bg-bg-tertiary"><span className="block h-1.5 rounded-full bg-accent-primary" style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} /></span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <Button small variant="ghost" onClick={() => setShowText((v) => !v)}>{showText ? "Masquer" : "Lire"} le texte de l'offre</Button>
            {showText && <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded-md bg-bg-secondary p-3 text-xs">{job.text || "Texte indisponible."}</pre>}
          </div>
          <ErrorBox error={prepare.error} />
          <div className="flex flex-wrap justify-end gap-2">
            <a href={job.url} target="_blank" rel="noopener noreferrer" className="rounded-md border border-border-default px-3.5 py-2 text-sm hover:bg-bg-secondary">Ouvrir l'offre</a>
            {job.application_status ? (
              <Button variant="primary" onClick={() => navigate(`/candidatures/${enc(job.id)}`)}>Voir la candidature ({job.application_status})</Button>
            ) : (
              <Button variant="primary" loading={prepare.isPending} onClick={() => prepare.mutate(undefined, { onSuccess: (task) => { track(task); } })}>Préparer la candidature</Button>
            )}
          </div>
        </div>
      )}
    </Modal>
  );
}

/** Filtres de la page, conservés dans l'URL (`/offres?q=&min_score=&label=&status=&source=`) ; valeur par défaut = absente. */
const FILTER_DEFAULTS = { q: "", min_score: "0", label: "", status: "", source: "" };
type FilterKey = keyof typeof FILTER_DEFAULTS;

export default function Jobs() {
  const [params, setParams] = useSearchParams();
  const filters: Record<FilterKey, string> = {
    q: params.get("q") ?? FILTER_DEFAULTS.q,
    min_score: params.get("min_score") ?? FILTER_DEFAULTS.min_score,
    label: params.get("label") ?? FILTER_DEFAULTS.label,
    status: params.get("status") ?? FILTER_DEFAULTS.status,
    source: params.get("source") ?? FILTER_DEFAULTS.source,
  };
  const { data, error, isLoading } = useJobs(filters);
  const { track } = useTaskUi();
  const [open, setOpen] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [importing, setImporting] = useState(false);
  const [imported, setImported] = useState<ManualItemCreated | null>(null);
  const { data: sources } = useSources();
  const sourceOptions: { code: string; label: string }[] = sources
    ? sources.map((s) => ({ code: s.code, label: s.label }))
    : (Object.keys(SOURCE_LABEL) as SourceCode[]).map((code) => ({ code, label: SOURCE_LABEL[code] }));
  const label = useLabelJob();
  const train = useAction(() => api<Task>("/model/train", { method: "POST" }), [["tasks"]]);
  const rescore = useAction(() => api<Task>("/runs/rescore", { method: "POST" }), [["tasks"]]);

  const model = data?.model;
  const nLabels = model?.n_labels ?? 0;
  const set = (key: FilterKey, value: string) =>
    setParams((current) => {
      const next = new URLSearchParams(current);
      if (value === FILTER_DEFAULTS[key]) next.delete(key);
      else next.set(key, value);
      return next;
    }, { replace: true });

  return (
    <>
      <PageHeader
        title="Offres"
        subtitle={data?.keyword ? `Dernière analyse : « ${data.keyword} » · ${formatDate(data.generated_at)}` : "Offres classées par score de correspondance"}
        actions={<>
          <Button onClick={() => { setImported(null); setImporting(true); }}>Importer une offre</Button>
          <Button onClick={() => rescore.mutate(undefined, { onSuccess: track })} loading={rescore.isPending}>Recalculer les scores</Button>
          <Button variant="primary" onClick={() => setRunning(true)}>Lancer une analyse</Button>
        </>}
      />
      {running && <RunModal onClose={() => setRunning(false)} />}
      {importing && <ManualImportModal onClose={() => setImporting(false)} onImported={(item) => { setImporting(false); setImported(item); }} />}
      {imported && (
        <div className="mb-4">
          <Notice tone="success">
            Offre « {imported.title} » importée. Elle sera scorée au prochain « Recalculer les scores ».
          </Notice>
        </div>
      )}
      <ErrorBox error={error ?? label.error ?? train.error ?? rescore.error} />

      <div className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <input className={inputClass} placeholder="Rechercher (titre, entreprise…)" value={filters.q} onChange={(e) => set("q", e.target.value)} aria-label="Rechercher" />
        <label className="flex items-center gap-2 text-sm">Score ≥ {filters.min_score} %
          <input type="range" min={0} max={100} step={5} value={filters.min_score} onChange={(e) => set("min_score", e.target.value)} className="flex-1" />
        </label>
        <select className={inputClass} value={filters.label} onChange={(e) => set("label", e.target.value)} aria-label="Annotation">
          <option value="">Toutes les annotations</option>
          <option value="pertinente">Pertinentes</option>
          <option value="non_pertinente">Pas pertinentes</option>
          <option value="non_annotee">Non annotées</option>
        </select>
        <select className={inputClass} value={filters.status} onChange={(e) => set("status", e.target.value)} aria-label="Candidature">
          <option value="">Toutes les candidatures</option>
          <option value="aucune">Sans candidature</option>
          {["préparée", "contactée", "postulée", "entretien", "refusée", "abandonnée", "externe", "échec"].map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select className={inputClass} value={filters.source} onChange={(e) => set("source", e.target.value)} aria-label="Source">
          <option value="">Toutes les sources</option>
          {sourceOptions.map((o) => <option key={o.code} value={o.code}>{o.label}</option>)}
        </select>
      </div>

      {model && (
        <div className="mb-4">
          <Notice tone={model.trained ? "success" : "info"}>
            {model.trained
              ? `Modèle appris sur ${nLabels} annotations (précision ${Math.round((model.train_accuracy ?? 0) * 100)} %). `
              : `Poids a priori : ${nLabels} annotation(s), il en faut au moins ${data?.min_labels ?? 6} des deux types pour apprendre. `}
            Annote les offres (<Icon name="thumbUp" className="inline h-3.5 w-3.5 align-[-2px]" /> pertinente /{" "}
            <Icon name="thumbDown" className="inline h-3.5 w-3.5 align-[-2px]" /> pas pertinente) puis{" "}
            <button className="underline" onClick={() => train.mutate(undefined, { onSuccess: track })}>réentraîne le modèle</button>.
          </Notice>
        </div>
      )}

      {isLoading ? <p className="text-sm text-text-muted">Chargement…</p> : !data?.jobs.length ? (
        <Empty>Aucune offre. Lance une analyse, ou assouplis les filtres.</Empty>
      ) : (
        <Card className="overflow-x-auto p-0">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="border-b border-border-default text-left text-xs uppercase text-text-muted">
              <tr><th className="p-3">Score</th><th className="p-3">Offre</th><th className="p-3">Compétences</th><th className="p-3">Annotation</th><th className="p-3">Candidature</th></tr>
            </thead>
            <tbody>
              {data.jobs.map((job: Job) => (
                <tr key={job.id} className="border-b border-border-subtle last:border-0 hover:bg-bg-secondary">
                  <td className="p-3"><Badge tone={scoreTone(job.score)}>{job.score} %</Badge></td>
                  <td className="p-3">
                    <button className="text-left font-medium hover:text-accent-primary" onClick={() => setOpen(job.id)}>{job.title || "Sans titre"}</button>
                    <div className="flex flex-wrap items-center gap-1.5 text-xs text-text-secondary">
                      <Badge>{sourceLabel(job.source_code)}</Badge>
                      <span>{[job.company, job.location].filter(Boolean).join(" · ")}</span>
                    </div>
                  </td>
                  <td className="p-3 text-xs"><span className="text-success">✓ {job.matched_skills.length}</span> <span className="ml-2 text-danger">✗ {job.missing_skills.length}</span></td>
                  <td className="p-3">
                    <div className="flex gap-1" role="group" aria-label={`Annotation de ${job.title || "l'offre"}`}>
                      <Button small variant={job.label === 1 ? "primary" : "secondary"} title="Pertinente" aria-label="Pertinente" aria-pressed={job.label === 1}
                        onClick={() => label.mutate({ id: job.id, value: job.label === 1 ? null : 1 })}><Icon name="thumbUp" /></Button>
                      <Button small variant={job.label === 0 ? "danger" : "secondary"} title="Pas pertinente" aria-label="Pas pertinente" aria-pressed={job.label === 0}
                        onClick={() => label.mutate({ id: job.id, value: job.label === 0 ? null : 0 })}><Icon name="thumbDown" /></Button>
                    </div>
                  </td>
                  <td className="p-3">{job.application_status ? <Badge tone={statusTone(job.application_status)}>{job.application_status}</Badge> : <span className="text-xs text-text-muted">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
      {open && <JobDetailModal id={open} onClose={() => setOpen(null)} />}
    </>
  );
}
