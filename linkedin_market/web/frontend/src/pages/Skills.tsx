import { useState } from "react";
import { Profile, SkillAdded, Task, api, enc } from "../api";
import { useConsentGuard } from "../components/RiskConsentDialog";
import { useTaskUi } from "../components/TaskDrawer";
import { useAction, useProfile, useSettings } from "../hooks";
import { Badge, Button, Card, ErrorBox, Field, Notice, PageHeader, inputClass } from "../ui";

export default function Skills() {
  const { data: profile, error } = useProfile();
  const { track } = useTaskUi();
  const [skill, setSkill] = useState("");
  const [form, setForm] = useState({ title: "", years: "", seniority: "" });

  // Le formulaire est réinitialisé depuis le profil quand son identité change (calculé au rendu, sans effet).
  const identityKey = profile?.available ? JSON.stringify([profile.title, profile.years_experience, profile.seniority]) : null;
  const [syncedKey, setSyncedKey] = useState<string | null>(null);
  if (identityKey !== syncedKey) {
    setSyncedKey(identityKey);
    if (profile?.available) {
      setForm({
        title: profile.overrides.title || profile.title || "",
        years: String(profile.overrides.years_experience ?? profile.years_experience ?? ""),
        seniority: profile.overrides.seniority || profile.seniority || "",
      });
    }
  }

  const refresh = [["profile"], ["jobs"], ["monitor"]];
  const [syncNote, setSyncNote] = useState("");
  const add = useAction((s: string) => api<SkillAdded>("/profile/skills", { method: "POST", body: { skill: s } }), [...refresh, ["tasks"]]);
  // Popup de risques AVANT la requête (I-16) : import LinkedIn, et ajout de compétence si la synchronisation
  // LinkedIn est active. Refus = rien n'est envoyé ; après acceptation, l'utilisateur clique de nouveau.
  const consent = useConsentGuard();
  const { data: settings } = useSettings();
  const addSkill = (s: string, after?: () => void) => {
    const go = () => add.mutate(s, {
      onSuccess: (res) => {
        after?.();
        setSyncNote(res.linkedin_sync.status === "unavailable" ? res.linkedin_sync.message ?? "" : "");
        if (res.linkedin_sync.task) track(res.linkedin_sync.task);
      },
    });
    if (settings?.linkedin_skill_sync) consent.guard("linkedin", go);
    else go();
  };
  const remove = useAction((s: string) => api<Profile>(`/profile/skills/${enc(s)}`, { method: "DELETE" }), refresh);
  const save = useAction(
    () => api<Profile>("/profile", {
      method: "PUT",
      body: { title: form.title, seniority: form.seniority, ...(form.years === "" ? { clear_years: true } : { years_experience: Number(form.years) }) },
    }), refresh);
  const reset = useAction(() => api<Profile>("/profile/reset-overrides", { method: "POST" }), refresh);
  const rescore = useAction(() => api<Task>("/runs/rescore", { method: "POST" }), [["tasks"]]);
  const importLi = useAction(() => api<Task>("/profile/import-linkedin", { method: "POST" }), [["tasks"]]);

  const submit = () => {
    if (!skill.trim()) return;
    addSkill(skill, () => setSkill(""));
  };

  const added = new Set(profile?.overrides.added_skills ?? []);
  const removedList = profile?.overrides.removed_skills ?? [];
  const hasOverrides = added.size > 0 || removedList.length > 0 || !!profile?.overrides.title || profile?.overrides.years_experience != null || !!profile?.overrides.seniority;

  return (
    <>
      <PageHeader
        title="Compétences et profil"
        subtitle="Ajuste ce que l'outil sait de toi : les scores, les CV et les messages s'appuient sur ces éléments"
        actions={<>
          <Button onClick={() => consent.guard("linkedin", () => importLi.mutate(undefined, { onSuccess: track }))} loading={importLi.isPending}>Importer depuis LinkedIn</Button>
          <Button variant="primary" onClick={() => rescore.mutate(undefined, { onSuccess: track })} loading={rescore.isPending}>Recalculer les scores</Button>
        </>}
      />
      {consent.dialog}
      <ErrorBox error={error ?? add.error ?? remove.error ?? save.error ?? reset.error ?? rescore.error ?? importLi.error} />

      {profile && !profile.available && <Notice tone="warning">{profile.message}</Notice>}
      {syncNote && <div className="mb-4"><Notice tone="warning">{syncNote}</Notice></div>}
      {profile?.scores_outdated && (
        <div className="mb-4"><Notice tone="warning">Tes compétences ont changé depuis la dernière analyse : les scores affichés sont périmés. Clique sur « Recalculer les scores ».</Notice></div>
      )}

      {profile && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card title="Mes compétences" className="lg:col-span-2" actions={hasOverrides ? <Button small variant="ghost" onClick={() => reset.mutate()}>Annuler mes ajustements</Button> : undefined}>
            <form className="mb-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); submit(); }}>
              <input className={inputClass} value={skill} onChange={(e) => setSkill(e.target.value)} placeholder="Ajouter une compétence (ex. Kubernetes)" aria-label="Nouvelle compétence" />
              <Button type="submit" variant="primary" disabled={!skill.trim()} loading={add.isPending}>Ajouter</Button>
            </form>
            <ul className="flex flex-wrap gap-2" aria-label="Compétences">
              {profile.skills.map((s) => (
                <li key={s} className="flex items-center gap-1 rounded-full border border-border-default bg-bg-secondary py-1 pl-3 pr-1 text-sm">
                  {s}
                  {added.has(s) && <Badge tone="accent">ajoutée</Badge>}
                  <button aria-label={`Retirer ${s}`} onClick={() => remove.mutate(s)} className="min-h-6 min-w-6 rounded-full px-1.5 text-text-muted hover:bg-danger/15 hover:text-danger">✕</button>
                </li>
              ))}
              {profile.skills.length === 0 && <li className="text-sm text-text-muted">Aucune compétence détectée.</li>}
            </ul>
            {removedList.length > 0 && (
              <div className="mt-4">
                <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Retirées par toi</h3>
                <div className="flex flex-wrap gap-2">
                  {removedList.map((s) => <button key={s} onClick={() => addSkill(s)} className="rounded-full border border-dashed border-border-default px-3 py-1 text-sm text-text-muted line-through hover:text-text-primary" title="Remettre">{s}</button>)}
                </div>
              </div>
            )}
          </Card>

          <Card title="Identité professionnelle">
            <div className="space-y-3">
              <Field label="Titre"><input className={inputClass} value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} /></Field>
              <Field label="Années d'expérience"><input className={inputClass} type="number" min={0} max={60} value={form.years} onChange={(e) => setForm({ ...form, years: e.target.value })} /></Field>
              <Field label="Séniorité">
                <select className={inputClass} value={form.seniority} onChange={(e) => setForm({ ...form, seniority: e.target.value })}>
                  <option value="">Détectée automatiquement</option>
                  {["junior", "confirme", "senior", "lead"].map((l) => <option key={l} value={l}>{l}</option>)}
                </select>
              </Field>
              <Button variant="primary" onClick={() => save.mutate()} loading={save.isPending}>Enregistrer</Button>
              <p className="text-xs text-text-muted">Source : {profile.origin ?? "—"}. Ton fichier de profil n'est jamais modifié : les ajustements s'appliquent par-dessus.</p>
            </div>
          </Card>

          <Card title="Demandées sur le marché, absentes de ton profil" className="lg:col-span-2">
            {profile.suggestions.missing.length === 0 ? (
              <p className="text-sm text-text-muted">Aucune suggestion (lance une analyse pour en obtenir).</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {profile.suggestions.missing.map((s) => (
                  <button key={s.skill} onClick={() => addSkill(s.skill)} className="rounded-full border border-border-default px-3 py-1 text-sm hover:bg-accent-soft" title="Ajouter à mon profil">
                    + {s.skill} <span className="text-xs text-text-muted">{Math.round(s.weight * 100)} %</span>
                  </button>
                ))}
              </div>
            )}
            <p className="mt-3 text-xs text-text-muted">N'ajoute que ce que tu maîtrises réellement : ces compétences servent aussi à rédiger tes CV et sont ajoutées à ton profil LinkedIn.</p>
          </Card>

          <Card title="En hausse sur 30 jours">
            {profile.suggestions.rising.length === 0 ? <p className="text-sm text-text-muted">Rien de notable.</p> : (
              <div className="flex flex-wrap gap-2">
                {profile.suggestions.rising.map((s) => (
                  <button key={s.skill} onClick={() => addSkill(s.skill)} className="rounded-full border border-border-default px-3 py-1 text-sm hover:bg-accent-soft">
                    + {s.skill} <span className="text-xs text-success">↗ +{Math.round(s.growth * 100)} %</span>
                  </button>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}
    </>
  );
}
