import { useRef, useState } from "react";
import { Settings, SourceCode, SourceInfo, Task, api } from "../api";
import { CAUTION_MAX, fieldErrors, runBodySchema } from "../schemas";
import { useSettings, useSources } from "../hooks";
import { RiskConsentDialog } from "./RiskConsentDialog";
import { SourcePicker, collectableSources, unavailableReason } from "./SourcePicker";
import { Button, ErrorBox, Field, Modal, Notice, inputClass } from "../ui";
import { useTaskUi } from "./TaskDrawer";

export function RunModal({ onClose }: { onClose: () => void }) {
  const { data: settings } = useSettings();
  const { track } = useTaskUi();
  const [form, setForm] = useState<Partial<Settings> & { scrape: boolean }>({ scrape: true });
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const keywordRef = useRef<HTMLInputElement>(null);
  // Sources (vague 3) : absentes si l'API ne les expose pas encore ; le serveur prend alors ["linkedin"].
  const { data: sources } = useSources();
  const [picked, setPicked] = useState<SourceCode[] | null>(null);
  const [consentFor, setConsentFor] = useState<SourceInfo | null>(null);

  if (!settings) return null;
  const available = collectableSources(sources ?? []).filter((s) => !unavailableReason(s));
  const defaultPick: SourceCode[] = available.some((s) => s.code === "linkedin") ? ["linkedin"] : available.slice(0, 1).map((s) => s.code);
  const selected = picked ?? defaultPick;
  // Une valeur saisie (même `null` : champ vidé) prime sur le réglage enregistré.
  const value = <K extends keyof Settings>(key: K) => (key in form ? form[key] : settings[key]) as Settings[K];
  const set = <K extends keyof Settings>(key: K, v: Settings[K]) => setForm((f) => ({ ...f, [key]: v }));

  // Corps validé avec les bornes du serveur (G-3) : max_items ≤ plafond de la prudence choisie.
  const checked = runBodySchema.safeParse({
    scrape: form.scrape,
    keyword: value("keyword"),
    location: value("location"),
    remote_only: value("remote_only"),
    caution: value("caution"),
    max_items: value("max_items"),
    ...(sources && form.scrape ? { sources: selected } : {}),
  });
  const errors = checked.success ? {} : fieldErrors(checked.error);
  const cap = CAUTION_MAX[value("caution")];

  const start = async () => {
    if (!checked.success) return;
    // Popup de risques AVANT toute requête si une source cochée l'exige (I-16) ; après acceptation, rien n'est
    // lancé automatiquement : l'utilisateur clique de nouveau sur « Lancer ».
    const needsConsent = form.scrape ? sources?.find((s) => selected.includes(s.code) && s.consent_required) : undefined;
    if (needsConsent) {
      setConsentFor(needsConsent);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { max_items, ...rest } = checked.data;
      const task = await api<Task>("/runs", { method: "POST", body: max_items == null ? rest : { ...rest, max_items } });
      track(task);
      onClose();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  // Prudence passée à « prudent » depuis la popup : le formulaire suit, et un nombre d'éléments au-delà du plafond
  // « prudent » est vidé (le plafond s'applique alors).
  const onConsentAccepted = ({ switchedToPrudent }: { switchedToPrudent: boolean }) => {
    setConsentFor(null);
    if (!switchedToPrudent) return;
    const items = value("max_items");
    setForm((f) => ({ ...f, caution: "prudent", ...(items != null && items > CAUTION_MAX.prudent ? { max_items: null } : {}) }));
  };

  return (
    <Modal title="Lancer une analyse" onClose={onClose} initialFocus={keywordRef}>
      <div className="space-y-3">
        <Field label="Poste visé (mot-clé)">
          <input className={inputClass} value={value("keyword")} onChange={(e) => set("keyword", e.target.value)} placeholder="Data Engineer" ref={keywordRef} />
        </Field>
        <Field label="Localisation" hint="Vide = monde entier">
          <input className={inputClass} value={value("location")} onChange={(e) => set("location", e.target.value)} />
        </Field>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Prudence anti-blocage">
            <select className={inputClass} value={value("caution")} onChange={(e) => set("caution", e.target.value as Settings["caution"])}>
              <option value="prudent">Prudent (15 offres)</option>
              <option value="normal">Normal (30)</option>
              <option value="agressif">Agressif (60)</option>
            </select>
          </Field>
          <Field label="Éléments par source" hint={`Vide = ${cap} (plafond de la prudence)`} error={errors.max_items}>
            <input className={inputClass} type="number" min={1} max={cap} value={value("max_items") ?? ""} placeholder={String(cap)}
              onChange={(e) => set("max_items", e.target.value === "" ? null : Number(e.target.value))} />
          </Field>
          <label className="flex items-center gap-2 text-sm sm:col-span-2">
            <input type="checkbox" checked={value("remote_only")} onChange={(e) => set("remote_only", e.target.checked)} />
            Full remote uniquement
          </label>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={form.scrape} onChange={(e) => setForm((f) => ({ ...f, scrape: e.target.checked }))} />
          Collecter de nouvelles données sur LinkedIn (sinon : réanalyser l'historique)
        </label>
        {form.scrape && (
          <Notice tone="warning">
            Un navigateur va s'ouvrir et parcourir LinkedIn avec ton compte. Garde des volumes faibles : le scraping automatisé est
            contraire aux conditions d'utilisation de LinkedIn.
          </Notice>
        )}
        {form.scrape && sources && (
          <SourcePicker sources={sources} value={selected} onChange={setPicked} onReadRisks={setConsentFor} error={errors.sources} />
        )}
        <ErrorBox error={error} />
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Annuler</Button>
          <Button variant="primary" onClick={start} loading={busy} disabled={!checked.success}>Lancer</Button>
        </div>
      </div>
      {consentFor && (
        <RiskConsentDialog source={consentFor} onAccepted={onConsentAccepted} onRefused={() => setConsentFor(null)} />
      )}
    </Modal>
  );
}
