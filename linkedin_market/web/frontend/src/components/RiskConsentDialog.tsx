import { ReactNode, useId, useState } from "react";
import { Settings, SourceCode, SourceInfo, api } from "../api";
import { fillRiskText, riskTextOf } from "../content/risks";
import { useAcceptConsent, useAction, useSettings, useSources } from "../hooks";
import { Button, ErrorBox, Modal, Notice, formatDate } from "../ui";

interface RiskConsentDialogProps {
  /** Source `navigateur` dont les risques sont à accepter (`risk_version` non nul). */
  source: SourceInfo;
  /**
   * Consentement enregistré : l'appelant reprend la main (aucune action relancée automatiquement).
   * `switchedToPrudent` : la prudence vient d'être passée à « prudent » (l'appelant aligne son formulaire).
   */
  onAccepted: (result: { switchedToPrudent: boolean }) => void;
  /** Refus (bouton, Échap ou ✕) : aucune requête mutative. */
  onRefused: () => void;
}

/** Passage de la prudence à « prudent » (`PUT /settings`) : réglages et état sont périmés après succès. */
const useSwitchToPrudent = () =>
  useAction<void, Settings>(() => api<Settings>("/settings", { method: "PUT", body: { caution: "prudent" } }), [["settings"], ["status"]]);

/**
 * Popup de risques d'une source (I-16, frontend/02 § 4.1). Pas de fermeture par le voile ; Échap vaut « Refuser » ;
 * case d'acceptation jamais pré-cochée ; texte inconnu de l'application ⇒ acceptation impossible. Focus initial sur
 * le titre pour une lecture depuis le début.
 */
export function RiskConsentDialog({ source, onAccepted, onRefused }: RiskConsentDialogProps) {
  const version = source.risk_version ?? "";
  const text = riskTextOf(source.code, version);
  const { data: settings } = useSettings();
  const accept = useAcceptConsent();
  const switchToPrudent = useSwitchToPrudent();
  const [accepted, setAccepted] = useState(false);
  // Facultatif (spécification) : proposé coché si la prudence courante n'est pas « prudent ».
  const [switchPrudent, setSwitchPrudent] = useState<boolean | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const bodyId = useId();

  const caution = settings?.caution ?? "prudent";
  const offerPrudent = caution !== "prudent";
  const switchChecked = switchPrudent ?? offerPrudent;
  const values = {
    prudence: caution,
    apply_daily_max: settings?.apply_daily_max ?? "…",
    message_daily_max: settings?.message_daily_max ?? "…",
  };
  const changedSince = source.consent && source.consent.risk_version !== version ? source.consent : null;

  const confirm = async () => {
    if (!text || !accepted) return;
    setError(null);
    setSaving(true);
    try {
      await accept.mutateAsync({ code: source.code, risk_version: version });
      const switchedToPrudent = offerPrudent && switchChecked;
      if (switchedToPrudent) await switchToPrudent.mutateAsync();
      onAccepted({ switchedToPrudent });
    } catch (e) {
      setError(e);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal title={text?.title ?? `Risques liés à ${source.label}`} onClose={onRefused} dismissible={false} wide describedBy={bodyId} focusTitle>
      <div id={bodyId} className="max-h-[50vh] space-y-3 overflow-y-auto pr-1 text-sm">
        {changedSince && (
          <Notice tone="warning">Ce texte a changé depuis ton acceptation du {formatDate(changedSince.accepted_at)} : relis-le.</Notice>
        )}
        {text ? (
          <ol className="list-decimal space-y-3 pl-5">
            {text.sections.map((section) => (
              <li key={section.heading}>
                <p className="font-semibold">{section.heading}</p>
                <p className="mt-0.5 text-text-secondary">{fillRiskText(section.body, values)}</p>
              </li>
            ))}
          </ol>
        ) : (
          <Notice tone="danger">Texte de risques introuvable pour la version {version} : mets à jour l'application</Notice>
        )}
        <p className="text-xs text-text-muted">Version des risques : {version}</p>
      </div>

      <div className="mt-4 space-y-2 border-t border-border-subtle pt-3">
        <label className="flex items-start gap-2 text-sm">
          <input type="checkbox" className="mt-0.5" checked={accepted} disabled={!text} onChange={(e) => setAccepted(e.target.checked)} />
          <span>J'ai lu ces risques et je les accepte pour mon compte {source.label}</span>
        </label>
        {offerPrudent && (
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" className="mt-0.5" checked={switchChecked} disabled={!text} onChange={(e) => setSwitchPrudent(e.target.checked)} />
            <span>Passer la prudence à « prudent »</span>
          </label>
        )}
        <ErrorBox error={error} />
        <div className="flex justify-end gap-2 pt-1">
          <Button onClick={onRefused}>Refuser</Button>
          <Button variant="primary" disabled={!text || !accepted} loading={saving} onClick={confirm}>J'accepte les risques</Button>
        </div>
      </div>
    </Modal>
  );
}

/**
 * Popup de risques PRÉVENTIVE (I-16, frontend/02 § 4.1) : `guard(code, action)` exécute `action` seulement si la
 * source n'exige pas de consentement ; sinon la popup s'ouvre et aucune requête n'est envoyée. Après acceptation,
 * rien n'est relancé automatiquement (décision du 2026-10-09) : l'utilisateur clique de nouveau. Si la liste des
 * sources est indisponible (erreur), l'action part et le 428 du serveur reste le filet ; si elle n'est pas encore
 * chargée, elle est attendue avant de décider. `dialog` est à rendre par l'appelant.
 */
export function useConsentGuard(): { guard: (code: SourceCode, action: () => void) => void; dialog: ReactNode } {
  const { data: sources, refetch } = useSources();
  const [consentFor, setConsentFor] = useState<SourceInfo | null>(null);
  const guard = (code: SourceCode, action: () => void) => {
    const decide = (list?: SourceInfo[]) => {
      const source = list?.find((s) => s.code === code && s.consent_required);
      if (source) setConsentFor(source);
      else action();
    };
    // Liste pas encore chargée (clic juste après l'ouverture de la page) : on l'attend avant toute requête.
    if (sources) decide(sources);
    else void refetch().then((result) => decide(result.data));
  };
  const close = () => setConsentFor(null);
  const dialog = consentFor ? <RiskConsentDialog source={consentFor} onAccepted={close} onRefused={close} /> : null;
  return { guard, dialog };
}
