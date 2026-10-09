import { SourceCode, SourceInfo } from "../api";
import { Badge, Button, formatDate } from "../ui";

interface SourcePickerProps {
  sources: SourceInfo[];
  value: SourceCode[];
  onChange: (value: SourceCode[]) => void;
  /** Ouvre la popup de risques de la source (aucune requête mutative). */
  onReadRisks: (source: SourceInfo) => void;
  error?: string;
}

/** Sources sélectionnables pour une collecte : l'import manuel (Indeed) n'est jamais proposé (R-6). */
export const collectableSources = (sources: SourceInfo[]) => sources.filter((s) => s.access_mode !== "import_manuel");

/** Raison pour laquelle une source ne peut pas être cochée, ou `null`. */
export function unavailableReason(source: SourceInfo): string | null {
  if (!source.enabled) return "bientôt";
  if (source.code === "france_travail" && source.configured === false) {
    return "identifiants FRANCE_TRAVAIL_CLIENT_ID / FRANCE_TRAVAIL_CLIENT_SECRET absents du fichier .env";
  }
  return null;
}

/** Choix des sources d'une collecte (frontend/02 § 4.5). */
export function SourcePicker({ sources, value, onChange, onReadRisks, error }: SourcePickerProps) {
  const toggle = (code: SourceCode, checked: boolean) =>
    onChange(checked ? [...value.filter((c) => c !== code), code] : value.filter((c) => c !== code));
  return (
    <fieldset className="space-y-2 text-sm" aria-describedby={error ? "source-picker-error" : undefined}>
      <legend className="mb-1 font-medium text-text-secondary">Sources</legend>
      {collectableSources(sources).map((source) => {
        const reason = unavailableReason(source);
        return (
          <div key={source.code} className="flex flex-wrap items-center gap-2">
            <label className={`flex items-center gap-2 ${reason ? "text-text-muted" : ""}`}>
              <input type="checkbox" disabled={!!reason} checked={!reason && value.includes(source.code)}
                onChange={(e) => toggle(source.code, e.target.checked)} />
              {source.label}
            </label>
            {source.access_mode === "api_officielle" && <Badge tone="success">API officielle</Badge>}
            {source.access_mode === "navigateur" && (source.consent_required
              ? <Badge tone="warning">Risques à accepter</Badge>
              : <Badge tone="warning">Navigateur — risques acceptés le {formatDate(source.consent?.accepted_at)}</Badge>)}
            {reason && <span className="text-xs text-text-muted">({reason})</span>}
            {!reason && source.access_mode === "navigateur" && source.consent_required && (
              <Button small variant="ghost" onClick={() => onReadRisks(source)}>Lire et accepter les risques</Button>
            )}
          </div>
        );
      })}
      {error && <p id="source-picker-error" className="text-xs text-danger-strong">{error}</p>}
    </fieldset>
  );
}
