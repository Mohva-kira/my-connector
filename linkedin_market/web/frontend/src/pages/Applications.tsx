import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useNavigate, useParams } from "react-router-dom";
import { ApplicationDetail, Contact, FINAL_STATUSES, SOURCE_LABEL, Task, api, authedUrl, enc, sourceOfRef } from "../api";
import { ComposeModal } from "../components/ComposeModal";
import { OppositionDialog } from "../components/OppositionDialog";
import { useConsentGuard } from "../components/RiskConsentDialog";
import { useTaskUi } from "../components/TaskDrawer";
import { useApplication, useApplications, useAction, useStatus, useTasks } from "../hooks";
import { Badge, Button, Card, Empty, ErrorBox, Field, Notice, PageHeader, Tabs, formatDate, inputClass, scoreTone, statusTone } from "../ui";

type Tab = "dossier" | "postuler" | "contacts" | "historique";

function Dossier({ app }: { app: ApplicationDetail }) {
  const [message, setMessage] = useState(app.message);
  const [cv, setCv] = useState(app.cv);
  const [preview, setPreview] = useState(true);
  // Nouvelle version serveur (autre candidature ou enregistrement) : le brouillon est remplacé (calculé au rendu).
  const [source, setSource] = useState({ id: app.id, message: app.message, cv: app.cv });
  if (source.id !== app.id || source.message !== app.message || source.cv !== app.cv) {
    setSource({ id: app.id, message: app.message, cv: app.cv });
    setMessage(app.message);
    setCv(app.cv);
  }

  const save = useAction(
    (body: { message?: string; cv?: string }) => api<ApplicationDetail>(`/applications/${enc(app.id)}`, { method: "PATCH", body }),
    [["application", app.id], ["applications"]]);
  const linkedin = sourceOfRef(app.id) === "linkedin";
  const dirtyMessage = message !== app.message;
  const dirtyCv = cv !== app.cv;

  return (
    <div className="space-y-4">
      <ErrorBox error={save.error} />
      <Card title="Message de candidature" actions={<Button small variant="primary" disabled={!dirtyMessage} loading={save.isPending} onClick={() => save.mutate({ message })}>Enregistrer</Button>}>
        {linkedin && message.length > 1500 && <div className="mb-2"><Notice tone="warning">Ce message est très long : il contient sans doute du texte brut de la page LinkedIn. Relis-le et raccourcis-le avant de l'envoyer.</Notice></div>}
        <textarea className={`${inputClass} font-sans`} rows={7} value={message} onChange={(e) => setMessage(e.target.value)} aria-label="Message de candidature" />
        <div className="mt-2 flex items-center gap-2 text-xs text-text-muted">
          <span>{message.length} caractères</span>
          <Button small variant="ghost" onClick={() => navigator.clipboard?.writeText(message)}>Copier</Button>
        </div>
      </Card>

      <Card title="Conseils">
        {app.advice ? <div className="prose-md text-sm"><ReactMarkdown remarkPlugins={[remarkGfm]}>{app.advice}</ReactMarkdown></div> : <p className="text-sm text-text-muted">Aucun conseil.</p>}
      </Card>

      <Card
        title={`CV adapté (${app.generator === "claude" ? "rédigé par Claude" : "assemblé localement"})`}
        actions={<div className="flex gap-2">
          <Button small variant="ghost" onClick={() => setPreview((v) => !v)}>{preview ? "Modifier" : "Aperçu"}</Button>
          <a className="rounded-md border border-border-default px-2.5 py-1 text-xs hover:bg-bg-secondary" target="_blank" rel="noopener noreferrer" href={authedUrl(`/applications/${enc(app.id)}/cv.html`)}>Version imprimable / PDF</a>
          <Button small variant="primary" disabled={!dirtyCv} loading={save.isPending} onClick={() => save.mutate({ cv })}>Enregistrer</Button>
        </div>}
      >
        {preview
          ? <div className="prose-md max-h-[28rem] overflow-auto rounded-md border border-border-subtle p-4 text-sm"><ReactMarkdown remarkPlugins={[remarkGfm]}>{cv}</ReactMarkdown></div>
          : <textarea className={`${inputClass} font-mono`} rows={18} value={cv} onChange={(e) => setCv(e.target.value)} aria-label="CV en Markdown" />}
      </Card>
    </div>
  );
}

const APPLY_STATE_LABEL: Record<string, string> = {
  en_cours: "En cours", postulée: "Envoyée", incertaine: "Envoi non confirmé", échec: "Échec",
};

/** Référence stable pour l'absence de tâches (évite un nouveau tableau à chaque rendu). */
const NO_TASKS: Task[] = [];

function Apply({ app }: { app: ApplicationDetail }) {
  const { track } = useTaskUi();
  const { data: status } = useStatus();
  const { data: tasks = NO_TASKS } = useTasks();
  const running = tasks.find((t) => t.kind === "apply" && t.meta.job_id === app.id && !FINAL_STATUSES.includes(t.status));
  const start = useAction(() => api<Task>(`/applications/${enc(app.id)}/apply`, { method: "POST" }), [["tasks"], ["application", app.id]]);
  // Popup de risques AVANT la requête (I-16) ; le 428 du serveur reste le filet.
  const consent = useConsentGuard();
  const quota = status?.quotas.apply;
  const done = app.status === "postulée" || app.status === "entretien";
  const origin = sourceOfRef(app.id);

  // Candidature simplifiée : LinkedIn uniquement (le serveur répond 422 sinon). Aucun bouton « Lancer », aucune popup de risques.
  if (origin !== "linkedin") {
    return (
      <Card title="Postuler">
        <Notice>
          La candidature simplifiée n'existe que pour LinkedIn : ouvre l'offre sur {SOURCE_LABEL[origin]} et postule toi-même.
        </Notice>
        <div className="mt-3">
          <a className="inline-block rounded-md border border-border-default px-3.5 py-2 text-sm hover:bg-bg-secondary" target="_blank" rel="noopener noreferrer" href={app.url}>Ouvrir l'offre</a>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card title="Postuler sur LinkedIn (candidature simplifiée)">
        <ol className="mb-4 list-decimal space-y-1 pl-5 text-sm text-text-secondary">
          <li>Un navigateur s'ouvre sur l'offre avec ta session LinkedIn (identifiants du fichier .env).</li>
          <li>L'outil remplit les étapes : email et téléphone des paramètres, CV adapté (PDF), réponses déjà mémorisées.</li>
          <li>Toute autre question t'est posée : rien n'est deviné, ta réponse est mémorisée pour les prochaines.</li>
          <li>À l'écran de relecture, tu vois exactement ce qui part et tu confirmes — ou tu annules.</li>
        </ol>
        {app.apply_state && <p className="mb-3 text-sm">État : <Badge tone={app.apply_state === "postulée" ? "success" : app.apply_state === "échec" ? "danger" : "warning"}>{APPLY_STATE_LABEL[app.apply_state] ?? app.apply_state}</Badge></p>}
        {app.apply_state === "incertaine" && <div className="mb-3"><Notice tone="warning">L'envoi n'a pas pu être confirmé. Vérifie sur LinkedIn (Emplois → Candidatures) avant de réessayer, puis passe le statut à « postulée » si besoin.</Notice></div>}
        {app.status === "externe" && <div className="mb-3"><Notice>Cette offre n'a pas de candidature simplifiée : postule sur le site de l'entreprise (bouton « Ouvrir l'offre »).</Notice></div>}
        {status && !status.credentials_configured && <div className="mb-3"><Notice tone="warning">Identifiants LinkedIn absents (.env).</Notice></div>}
        <ErrorBox error={start.error} />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          {running ? (
            <Button variant="primary" onClick={() => track(running)}>Suivre la candidature en cours</Button>
          ) : (
            <Button variant="primary" disabled={done || !status?.credentials_configured || !!status?.blocked || (quota ? quota.used >= quota.limit : false)}
              loading={start.isPending} onClick={() => consent.guard("linkedin", () => start.mutate(undefined, { onSuccess: track }))}>
              {done ? "Déjà postulée" : "Lancer la candidature"}
            </Button>
          )}
          {quota && <span className="text-xs text-text-muted">Aujourd'hui : {quota.used} / {quota.limit} candidatures</span>}
        </div>
        {consent.dialog}
      </Card>
      <Notice tone="warning">L'automatisation de LinkedIn est contraire à leurs conditions d'utilisation : garde un rythme humain. Les plafonds quotidiens se règlent dans Paramètres.</Notice>
    </div>
  );
}

function Contacts({ app }: { app: ApplicationDetail }) {
  const [rows, setRows] = useState<Contact[]>(app.contacts);
  // Contacts relus depuis le serveur : la saisie locale repart de cette version (calculé au rendu).
  const [source, setSource] = useState(app.contacts);
  if (source !== app.contacts) {
    setSource(app.contacts);
    setRows(app.contacts);
  }
  const save = useAction(
    (contacts: Contact[]) => api<ApplicationDetail>(`/applications/${enc(app.id)}`, { method: "PATCH", body: { contacts } }),
    [["application", app.id]]);
  const update = (i: number, patch: Partial<Contact>) => setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const { track } = useTaskUi();
  const { data: status } = useStatus();
  const lookup = useAction(() => api<Task>(`/applications/${enc(app.id)}/contacts/lookup`, { method: "POST" }), [["tasks"]]);
  const [composing, setComposing] = useState<{ contact?: Contact } | null>(null);
  const [opposing, setOpposing] = useState<Contact | null>(null);
  const consent = useConsentGuard();
  const dirty = JSON.stringify(rows) !== JSON.stringify(app.contacts);
  const linkedin = sourceOfRef(app.id) === "linkedin";

  return (
    <Card title="Contacts de l'entreprise" actions={<div className="flex flex-wrap gap-2">
      <Button small onClick={() => consent.guard("linkedin", () => lookup.mutate(undefined, { onSuccess: track }))} loading={lookup.isPending} disabled={!linkedin || !status?.credentials_configured || !!status?.blocked}
        title={linkedin ? undefined : "La recherche de recruteurs n'existe que pour les offres LinkedIn."}>Chercher les recruteurs de l'offre</Button>
      <Button small onClick={() => setRows([...rows, { full_name: "", profile_url: "", email: null, source: "manuel" }])}>Ajouter un contact</Button>
      <Button small variant="primary" onClick={() => setComposing({})}>Écrire à l'entreprise</Button>
    </div>}>
      <ErrorBox error={save.error ?? lookup.error} />
      {composing && <ComposeModal app={app} contact={composing.contact} onClose={() => setComposing(null)} />}
      {opposing && <OppositionDialog contact={opposing} applicationId={app.id} onClose={() => setOpposing(null)} />}
      {consent.dialog}
      {linkedin && <p className="mb-3 text-xs text-text-muted">La recherche lit le bloc « équipe de recrutement » de l'offre et les coordonnées que LinkedIn t'affiche : l'email n'apparaît que s'il est visible pour toi.</p>}
      {rows.length === 0 ? <Empty>Aucun contact. Ajoute le recruteur à la main (nom, profil LinkedIn, email).</Empty> : (
        <ul className="space-y-3">
          {rows.map((c, i) => (
            <li key={i} className="grid gap-2 rounded-md border border-border-default p-3 sm:grid-cols-3">
              <Field label="Nom"><input className={inputClass} value={c.full_name} onChange={(e) => update(i, { full_name: e.target.value })} /></Field>
              {/* Contact opposé : coordonnées effacées par le serveur et non ressaisissables (backend/01 § 5.7.1). */}
              <Field label="Profil LinkedIn"><input className={inputClass} value={c.profile_url} disabled={c.opposed} onChange={(e) => update(i, { profile_url: e.target.value })} placeholder="https://www.linkedin.com/in/…" /></Field>
              <Field label="Email"><input className={inputClass} type="email" value={c.email ?? ""} disabled={c.opposed} onChange={(e) => update(i, { email: e.target.value || null })} /></Field>
              <div className="flex flex-wrap items-center justify-between gap-2 sm:col-span-3">
                <span className="text-xs text-text-muted">
                  {c.headline} {c.source === "linkedin_equipe_recrutement" && <Badge>trouvé via LinkedIn</Badge>}{" "}
                  {c.opposed && <Badge tone="danger">Opposé</Badge>}
                </span>
                <div className="flex flex-wrap gap-1">
                  <Button small disabled={dirty || c.opposed}
                    title={c.opposed ? "Cette personne a demandé à ne plus être contactée." : dirty ? "Enregistre d'abord les contacts" : undefined}
                    onClick={() => setComposing({ contact: c })}>Écrire</Button>
                  {!c.opposed && (
                    <Button small variant="ghost" disabled={dirty} title={dirty ? "Enregistre d'abord les contacts" : undefined}
                      onClick={() => setOpposing(c)}>Ce contact refuse d'être recontacté</Button>
                  )}
                  <Button small variant="ghost" onClick={() => setRows(rows.filter((_, j) => j !== i))}>Supprimer</Button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
      <div className="mt-3 flex justify-end"><Button variant="primary" loading={save.isPending} onClick={() => save.mutate(rows)}>Enregistrer les contacts</Button></div>
    </Card>
  );
}

/** Libellés français de `history[].kind` (contrat vague 1, C-6). */
const HISTORY_KIND_LABEL: Record<string, string> = {
  status: "Statut", apply: "Candidature", contacts: "Contacts", message: "Message",
};

function History({ app }: { app: ApplicationDetail }) {
  const entries = [...app.history].reverse();
  return (
    <Card title="Historique">
      {entries.length === 0 ? <p className="text-sm text-text-muted">Rien pour l'instant.</p> : (
        <ul className="divide-y divide-border-subtle text-sm">
          {entries.map((h, i) => (
            <li key={i} className="flex gap-3 py-2">
              <span className="w-32 shrink-0 text-xs text-text-muted">{formatDate(h.at)}</span>
              <Badge>{HISTORY_KIND_LABEL[h.kind] ?? h.kind}</Badge>
              <span className="min-w-0 truncate text-text-secondary">{Object.entries(h).filter(([k]) => !["at", "kind"].includes(k)).map(([, v]) => String(v)).join(" · ")}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function Detail({ id }: { id: string }) {
  const { data: app, error } = useApplication(id);
  const [tab, setTab] = useState<Tab>("dossier");
  const setStatus = useAction(
    (status: string) => api<ApplicationDetail>(`/applications/${enc(id)}`, { method: "PATCH", body: { status } }),
    [["application", id], ["applications"], ["jobs"], ["monitor"]]);

  if (error) return <ErrorBox error={error} />;
  if (!app) return <p className="text-sm text-text-muted">Chargement…</p>;
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold">{app.title || "Sans titre"}</h2>
          <p className="text-sm text-text-secondary">{app.company || "Entreprise inconnue"} · <Badge tone={scoreTone(app.score)}>{app.score} %</Badge> · préparée le {formatDate(app.prepared_at)}</p>
          {sourceOfRef(app.id) === "linkedin" && (app.title.length > 140 || app.company.length > 100) && <div className="mt-2"><Notice tone="warning">Le titre ou l'entreprise semblent mal extraits de la page LinkedIn (texte trop long). Vérifie le dossier avant d'envoyer quoi que ce soit.</Notice></div>}
        </div>
        <div className="flex items-center gap-2">
          <select className={inputClass} value={app.status} onChange={(e) => setStatus.mutate(e.target.value)} aria-label="Statut">
            {app.statuses.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <a className="whitespace-nowrap rounded-md border border-border-default px-3 py-2 text-sm hover:bg-bg-secondary" target="_blank" rel="noopener noreferrer" href={app.url}>Ouvrir l'offre</a>
        </div>
      </div>
      <ErrorBox error={setStatus.error} />
      <Tabs<Tab> tabs={[{ id: "dossier", label: "Dossier" }, { id: "postuler", label: "Postuler" }, { id: "contacts", label: `Contacts (${app.contacts.length})` }, { id: "historique", label: "Historique" }]} value={tab} onChange={setTab}>
        {tab === "dossier" && <Dossier app={app} />}
        {tab === "postuler" && <Apply app={app} />}
        {tab === "contacts" && <Contacts app={app} />}
        {tab === "historique" && <History app={app} />}
      </Tabs>
    </div>
  );
}

export default function Applications() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { data: apps = [], error } = useApplications();
  const [filter, setFilter] = useState("");
  const rows = apps.filter((a) => !filter || a.status === filter);
  const selected = jobId ? decodeURIComponent(jobId) : undefined;

  useEffect(() => {
    if (!selected && apps.length > 0) navigate(`/candidatures/${enc(apps[0].id)}`, { replace: true });
  }, [selected, apps, navigate]);

  return (
    <>
      <PageHeader title="Candidatures" subtitle="Dossiers préparés, statut et contacts" />
      <ErrorBox error={error} />
      {apps.length === 0 ? <Empty>Aucune candidature préparée. Ouvre une offre dans « Offres » et clique sur « Préparer la candidature ».</Empty> : (
        <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
          <div>
            <select className={`${inputClass} mb-2`} value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filtrer par statut">
              <option value="">Tous les statuts</option>
              {[...new Set(apps.map((a) => a.status))].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <ul className="space-y-2">
              {rows.map((a) => (
                <li key={a.id}>
                  <button onClick={() => navigate(`/candidatures/${enc(a.id)}`)} aria-current={a.id === selected}
                    className={`w-full rounded-md border p-3 text-left hover:bg-bg-secondary ${a.id === selected ? "border-accent-primary bg-accent-soft" : "border-border-default bg-surface"}`}>
                    <div className="truncate text-sm font-medium">{a.title || "Sans titre"}</div>
                    <div className="truncate text-xs text-text-secondary">{a.company}</div>
                    <div className="mt-1 flex items-center gap-2"><Badge tone={statusTone(a.status)}>{a.status}</Badge><span className="text-xs text-text-muted">{a.score} %</span></div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <div className="min-w-0">{selected ? <Detail id={selected} /> : null}</div>
        </div>
      )}
    </>
  );
}
