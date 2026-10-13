import { useEffect, useRef, useState } from "react";
import { Settings, SourceInfo, api } from "../api";
import { RiskConsentDialog } from "../components/RiskConsentDialog";
import { useAction, useRevokeConsent, useSettings, useSources, useStatus } from "../hooks";
import { CAUTION_MAX, fieldErrors, settingsSchema } from "../schemas";
import { Badge, Button, Card, ErrorBox, Field, Modal, Notice, PageHeader, formatDate, inputClass } from "../ui";

const ACCESS_MODE_LABEL: Record<SourceInfo["access_mode"], string> = {
  navigateur: "Navigateur", api_officielle: "API officielle", import_manuel: "Import manuel",
};

/**
 * Carte « Sources et consentements » (#sources, frontend/04 § 1.3) : mode d'accès, état, consentement aux risques,
 * lecture des risques et révocation confirmée (I-16).
 */
function SourcesCard() {
  const { data: sources, error } = useSources();
  const revoke = useRevokeConsent();
  const [reading, setReading] = useState<SourceInfo | null>(null);
  const [revoking, setRevoking] = useState<SourceInfo | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  // Lien direct /parametres#sources : la carte n'existe qu'une fois les sources chargées.
  useEffect(() => {
    if (sources && window.location.hash === "#sources") cardRef.current?.scrollIntoView({ block: "start" });
  }, [sources]);

  return (
    <div id="sources" ref={cardRef} className="scroll-mt-4 lg:col-span-2">
      <Card title="Sources et consentements">
        <ErrorBox error={error ?? revoke.error} />
        {!sources ? (!error && <p className="text-sm text-text-muted">Chargement…</p>) : (
          <ul className="divide-y divide-border-subtle text-sm">
            {sources.map((source) => (
              <li key={source.code} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2">
                <span className="min-w-32 font-medium">{source.label}</span>
                <Badge tone={source.access_mode === "api_officielle" ? "success" : source.access_mode === "navigateur" ? "warning" : "neutral"}>
                  {ACCESS_MODE_LABEL[source.access_mode]}
                </Badge>
                {!source.enabled && <Badge>bientôt</Badge>}
                {source.code === "france_travail" && source.configured === false && (
                  <span className="text-xs text-text-muted">Identifiants France Travail absents du fichier .env</span>
                )}
                {source.access_mode === "navigateur" && (
                  <span className="text-xs text-text-secondary">
                    {source.consent
                      ? `Risques acceptés le ${formatDate(source.consent.accepted_at)} (version ${source.consent.risk_version})`
                      : "Risques non acceptés"}
                    {source.consent_required && source.consent ? ` — nouvelle version ${source.risk_version} à relire` : ""}
                  </span>
                )}
                {source.access_mode === "navigateur" && source.enabled && (
                  <span className="ml-auto flex gap-1">
                    <Button small variant="ghost" onClick={() => setReading(source)}>Lire les risques</Button>
                    {source.consent && <Button small variant="ghost" onClick={() => setRevoking(source)}>Révoquer</Button>}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>
      {reading && <RiskConsentDialog source={reading} onAccepted={() => setReading(null)} onRefused={() => setReading(null)} />}
      {revoking && (
        <Modal title={`Révoquer ton consentement pour ${revoking.label} ?`} onClose={() => setRevoking(null)}>
          <p className="text-sm text-text-secondary">
            Les actions qui utilisent {revoking.label} (collecte, candidature, messages) seront refusées tant que tu n'auras
            pas de nouveau accepté les risques. Les données déjà collectées sont conservées.
          </p>
          <div className="mt-4 flex justify-end gap-2">
            <Button onClick={() => setRevoking(null)}>Annuler</Button>
            <Button variant="danger" loading={revoke.isPending}
              onClick={() => revoke.mutate(revoking.code, { onSuccess: () => setRevoking(null) })}>Révoquer</Button>
          </div>
        </Modal>
      )}
    </div>
  );
}

export default function SettingsPage() {
  const { data: saved, error } = useSettings();
  const { data: status } = useStatus();
  const [form, setForm] = useState<Settings | null>(null);
  const [done, setDone] = useState(false);
  // Chaque nouvelle version des réglages enregistrés remplace le formulaire (calculé au rendu, sans effet).
  const [syncedFrom, setSyncedFrom] = useState<Settings | undefined>(undefined);
  if (saved !== syncedFrom) {
    setSyncedFrom(saved);
    if (saved) setForm(saved);
  }

  const save = useAction((body: Partial<Settings>) => api<Settings>("/settings", { method: "PUT", body }), [["settings"], ["status"], ["monitor"]]);
  const testMail = useAction(() => api<{ to: string }>("/smtp/test", { method: "POST", body: {} }), []);
  if (!form) return <ErrorBox error={error} />;
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => { setDone(false); setForm({ ...form, [key]: value }); };
  // Champ numérique vide ou illisible : NaN, signalé par la validation (jamais 0 implicite).
  const num = (v: string) => (v === "" ? Number.NaN : Number(v));
  const shown = (v: number) => (Number.isNaN(v) ? "" : v);

  // Validation de confort (bornes et messages du serveur) ; le serveur reste seul garant.
  const checked = settingsSchema.safeParse(form);
  const errors = checked.success ? {} : fieldErrors(checked.error);
  const submit = () => { if (checked.success) save.mutate(form, { onSuccess: () => setDone(true) }); };
  const forget = (question: string) => {
    const answers = { ...form.answers };
    delete answers[question];
    save.mutate({ answers }, { onSuccess: () => setDone(true) });
  };

  return (
    <>
      <PageHeader
        title="Paramètres"
        subtitle="Valeurs par défaut des analyses, plafonds et informations utilisées dans les candidatures"
        actions={<Button variant="primary" onClick={submit} loading={save.isPending} disabled={!checked.success}
          title={checked.success ? undefined : "Corrige les champs signalés"}>Enregistrer</Button>}
      />
      <ErrorBox error={save.error} />
      {done && <div className="mb-4"><Notice tone="success">Paramètres enregistrés.</Notice></div>}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Recherche par défaut">
          <div className="space-y-3">
            <Field label="Mot-clé du poste"><input className={inputClass} value={form.keyword} onChange={(e) => set("keyword", e.target.value)} /></Field>
            <Field label="Localisation" hint="Vide = monde entier"><input className={inputClass} value={form.location} onChange={(e) => set("location", e.target.value)} /></Field>
            <Field label="Ton lieu de résidence" hint="Utilisé dans les analyses et les CV ; vide = non mentionné" error={errors.home_location}><input className={inputClass} value={form.home_location} onChange={(e) => set("home_location", e.target.value)} /></Field>
            <Field label="Ton fuseau horaire" hint="Ex. Africa/Abidjan, Europe/Paris ; vide = non mentionné" error={errors.timezone}><input className={inputClass} value={form.timezone} onChange={(e) => set("timezone", e.target.value)} /></Field>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={form.remote_only} onChange={(e) => set("remote_only", e.target.checked)} />Full remote uniquement</label>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Prudence">
                <select className={inputClass} value={form.caution} onChange={(e) => set("caution", e.target.value as Settings["caution"])}>
                  <option value="prudent">Prudent</option><option value="normal">Normal</option><option value="agressif">Agressif</option>
                </select>
              </Field>
              <Field label="Éléments par source" hint={`Vide = selon la prudence (${CAUTION_MAX[form.caution]} au plus)`} error={errors.max_items}>
                <input className={inputClass} type="number" min={1} max={CAUTION_MAX[form.caution]} value={form.max_items ?? ""} onChange={(e) => set("max_items", e.target.value === "" ? null : Number(e.target.value))} />
              </Field>
              <Field label="Source du profil">
                <select className={inputClass} value={form.profile_source} onChange={(e) => set("profile_source", e.target.value as Settings["profile_source"])}>
                  <option value="auto">Automatique</option><option value="file">Fichier (profil.*)</option><option value="linkedin">LinkedIn</option>
                </select>
              </Field>
              <Field label="Interprétation / CV">
                <select className={inputClass} value={form.interpreter} onChange={(e) => set("interpreter", e.target.value as Settings["interpreter"])}>
                  <option value="claude">Claude</option><option value="local">Local (règles)</option>
                </select>
              </Field>
            </div>
            {form.interpreter === "claude" && (
              <Notice tone="warning">Avec Claude, ton profil et le texte de l'offre sont envoyés à l'API Anthropic, sans noms, emails, téléphones ni liens de profils de personnes (remplacés par [personne]).</Notice>
            )}
          </div>
        </Card>

        <Card title="Plafonds et candidatures">
          <div className="space-y-3">
            <Field label="Score minimal pour proposer une offre" hint="Pourcentage, 70 par défaut" error={errors.apply_min_score}><input className={inputClass} type="number" min={0} max={100} value={shown(form.apply_min_score)} onChange={(e) => set("apply_min_score", num(e.target.value))} /></Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Candidatures / jour" error={errors.apply_daily_max}><input className={inputClass} type="number" min={0} max={200} value={shown(form.apply_daily_max)} onChange={(e) => set("apply_daily_max", num(e.target.value))} /></Field>
              <Field label="Messages / jour" error={errors.message_daily_max}><input className={inputClass} type="number" min={0} max={200} value={shown(form.message_daily_max)} onChange={(e) => set("message_daily_max", num(e.target.value))} /></Field>
            </div>
            <Field label="Email de contact" hint="Renseigné dans « Candidature simplifiée »" error={errors.contact_email}><input className={inputClass} type="email" value={form.contact_email} onChange={(e) => set("contact_email", e.target.value)} /></Field>
            <Field label="Téléphone"><input className={inputClass} value={form.contact_phone} onChange={(e) => set("contact_phone", e.target.value)} /></Field>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={form.linkedin_skill_sync} onChange={(e) => set("linkedin_skill_sync", e.target.checked)} />Ajouter à mon profil LinkedIn chaque compétence ajoutée dans l'outil (« J'ai », page Compétences)</label>
            <Field label="Signature des emails"><textarea className={inputClass} rows={3} value={form.signature} onChange={(e) => set("signature", e.target.value)} /></Field>
          </div>
        </Card>

        <Card title="Réponses mémorisées" className="lg:col-span-2">
          {Object.keys(form.answers).length === 0 ? (
            <p className="text-sm text-text-muted">Aucune. Quand une candidature pose une question inconnue, ta réponse est mémorisée ici pour les suivantes.</p>
          ) : (
            <ul className="divide-y divide-border-subtle text-sm">
              {Object.entries(form.answers).map(([q, a]) => (
                <li key={q} className="flex items-center justify-between gap-3 py-2">
                  <span className="min-w-0 flex-1"><span className="block truncate font-medium">{q}</span><span className="block truncate text-text-secondary">{a}</span></span>
                  <Button small variant="ghost" onClick={() => forget(q)}>Oublier</Button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="État de l'environnement (.env)" className="lg:col-span-2">
          <ul className="grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
            <li>Identifiants LinkedIn : <Badge tone={status?.credentials_configured ? "success" : "danger"}>{status?.credentials_configured ? "configurés" : "absents"}</Badge></li>
            <li>SMTP : <Badge tone={status?.smtp_configured ? "success" : "warning"}>{status?.smtp_configured ? "configuré" : "non configuré"}</Badge></li>
            <li>Clé Anthropic : <Badge tone={status?.anthropic_configured ? "success" : "neutral"}>{status?.anthropic_configured ? "présente" : "absente"}</Badge></li>
            <li>Fichier de profil : <Badge tone={status?.profile_file ? "success" : "neutral"}>{status?.profile_file ? "trouvé" : "aucun"}</Badge></li>
          </ul>
          <p className="mt-3 text-xs text-text-muted">Les secrets restent dans le fichier .env : l'interface n'affiche et ne stocke jamais de mot de passe. Navigateur : LINKEDIN_BROWSER / LINKEDIN_HEADLESS (une vérification manuelle LinkedIn exige un navigateur visible).</p>
          {status?.smtp_configured && (
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button small onClick={() => testMail.mutate()} loading={testMail.isPending}>Envoyer un email de test à moi-même</Button>
              {testMail.isSuccess && <span className="text-sm text-success">Envoyé à {testMail.data.to}.</span>}
            </div>
          )}
          <ErrorBox error={testMail.error} />
          <div className="mt-3"><Notice tone="warning">Rappel : l'automatisation de LinkedIn (collecte, candidatures, messages) est contraire à leurs conditions d'utilisation et peut entraîner la restriction de ton compte. Les plafonds ci-dessus limitent le risque sans le supprimer.</Notice></div>
        </Card>

        <SourcesCard />
      </div>
    </>
  );
}
