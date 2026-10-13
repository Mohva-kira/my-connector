import {
  ButtonHTMLAttributes, KeyboardEvent, ReactElement, ReactNode, RefObject, cloneElement, forwardRef, isValidElement, useEffect, useId,
  useLayoutEffect, useRef,
} from "react";
import { ApiError } from "./api";

type Tone = "neutral" | "success" | "warning" | "danger" | "info" | "accent";

// Texte en jeton `*-strong` : contraste ≥ 4,5:1 sur le fond du ton à 15 % (frontend/01 § 4.1).
const TONES: Record<Tone, string> = {
  neutral: "bg-bg-tertiary text-text-secondary",
  success: "bg-success/15 text-success-strong",
  warning: "bg-warning/15 text-warning-strong",
  danger: "bg-danger/15 text-danger-strong",
  info: "bg-info/15 text-info-strong",
  accent: "bg-accent-soft text-accent-strong",
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${TONES[tone]}`}>{children}</span>;
}

export function Card({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-border-default bg-surface p-4 ${className}`}>
      {(title || actions) && (
        <header className="mb-3 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-text-primary">{title}</h2>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

type ButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type"> & {
  children: ReactNode;
  variant?: "primary" | "secondary" | "danger" | "ghost";
  loading?: boolean;
  type?: "button" | "submit";
  small?: boolean;
};

/**
 * Bouton partagé ; les attributs HTML restants (`aria-label`, `aria-pressed`…) sont transmis tels quels et la `ref`
 * atteint le <button> (focus initial d'une modale).
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { children, variant = "secondary", disabled, loading, type = "button", small, className = "", ...rest },
  ref,
) {
  const styles = {
    primary: "bg-accent-primary text-on-accent hover:bg-accent-hover",
    secondary: "border border-border-default bg-surface text-text-primary hover:bg-bg-secondary",
    danger: "bg-danger text-on-accent hover:opacity-90",
    ghost: "text-text-secondary hover:bg-bg-secondary",
  }[variant];
  return (
    <button
      {...rest}
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`rounded-md font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${small ? "px-2.5 py-1 text-xs" : "px-3.5 py-2 text-sm"} ${styles} ${className}`}
    >
      {loading ? "…" : children}
    </button>
  );
});

type FieldControlProps = { "aria-invalid"?: boolean; "aria-describedby"?: string };

/** Champ libellé ; `error` s'affiche sous le champ, relié par `aria-describedby` avec `aria-invalid`. */
export function Field({ label, hint, error, children }: { label: string; hint?: string; error?: string; children: ReactNode }) {
  const errorId = useId();
  const control = error && isValidElement(children)
    ? cloneElement(children as ReactElement<FieldControlProps>, { "aria-invalid": true, "aria-describedby": errorId })
    : children;
  return (
    <label className="block text-sm">
      <span className="mb-1 block font-medium text-text-secondary">{label}</span>
      {control}
      {hint && <span className="mt-1 block text-xs text-text-muted">{hint}</span>}
      {error && <span id={errorId} className="mt-1 block text-xs text-danger-strong">{error}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-md border border-border-strong bg-bg-primary px-3 py-2 text-sm text-text-primary focus:border-accent-primary";

/** Délai lisible d'un `Retry-After` (secondes) : « 3 h », « 40 min ». */
function retryDelay(seconds: number): string {
  return seconds >= 3600 ? `${Math.round(seconds / 3600)} h` : `${Math.max(1, Math.round(seconds / 60))} min`;
}

/** Message d'erreur affiché tel quel ; un 429 avec `Retry-After` indique en plus quand réessayer (G-2). */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.status === 429 && error.retryAfter ? `${error.detail} Réessaie dans ${retryDelay(error.retryAfter)}.` : error.detail;
  }
  return error instanceof Error ? error.message : String(error);
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const message = errorMessage(error);
  return <div role="alert" className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger-strong">{message}</div>;
}

/**
 * Bandeau annoncé aux lecteurs d'écran quand il apparaît (WCAG 4.1.3) : `alert` (assertif) pour le ton `danger`,
 * `status` (poli) pour les autres. Ne pas l'envelopper dans un autre `role="alert"`.
 */
export function Notice({ tone = "info", children }: { tone?: Tone; children: ReactNode }) {
  return <div role={tone === "danger" ? "alert" : "status"} className={`rounded-md px-3 py-2 text-sm ${TONES[tone]}`}>{children}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="rounded-md border border-dashed border-border-default px-4 py-8 text-center text-sm text-text-muted">{children}</p>;
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 tabIndex={-1} className="text-xl font-semibold focus:outline-none">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-text-secondary">{subtitle}</p>}
      </div>
      <div className="flex flex-wrap gap-2">{actions}</div>
    </header>
  );
}

interface ModalProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
  /** Élément focalisé à l'ouverture (sinon : premier élément interactif, choisi par le navigateur). */
  initialFocus?: RefObject<HTMLElement | null>;
  /**
   * `false` : un clic sur le voile ne ferme pas (dialogues de décision : relecture avant envoi, risques). Échap
   * reste actif et appelle `onClose`, qui doit alors valoir refus / annulation.
   */
  dismissible?: boolean;
  /** `id` de l'élément qui décrit la modale (`aria-describedby`), ex. le texte des risques. */
  describedBy?: string;
  /**
   * Focus initial sur le titre (`tabIndex=-1`) : un lecteur d'écran lit le contenu depuis le début (popup de risques).
   * Prioritaire sur `initialFocus`.
   */
  focusTitle?: boolean;
}

/**
 * Modale sur l'élément natif `<dialog>` + `showModal()` : le reste de la page devient inerte (piège de focus), Échap
 * et la pile de modales sont gérés par le navigateur. Le focus est rendu à l'élément déclencheur à la fermeture.
 */
export function Modal({ title, onClose, children, wide, initialFocus, dismissible = true, describedBy, focusTitle }: ModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const titleId = useId();
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; });

  useLayoutEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const trigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    // Échap (évènement `cancel`) ou fermeture native : le parent démonte la modale.
    const onCancel = (event: Event) => { event.preventDefault(); onCloseRef.current(); };
    const onNativeClose = () => onCloseRef.current();
    // Clic sur le voile : la cible est le <dialog> lui-même, hors de sa boîte.
    const onBackdrop = (event: MouseEvent) => {
      if (!dismissible || event.target !== dialog) return;
      const box = dialog.getBoundingClientRect();
      const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
      if (!inside) onCloseRef.current();
    };
    dialog.addEventListener("cancel", onCancel);
    dialog.addEventListener("close", onNativeClose);
    dialog.addEventListener("mousedown", onBackdrop);
    if (!dialog.open) dialog.showModal();
    if (focusTitle) titleRef.current?.focus();
    else initialFocus?.current?.focus();
    return () => {
      // Retirer les écouteurs AVANT close() : un démontage ne doit jamais rappeler onClose (ex. « Annuler » d'une
      // relecture confirm_*).
      dialog.removeEventListener("cancel", onCancel);
      dialog.removeEventListener("close", onNativeClose);
      dialog.removeEventListener("mousedown", onBackdrop);
      if (dialog.open) dialog.close();
      if (trigger?.isConnected) trigger.focus();
    };
  }, [initialFocus, dismissible, focusTitle]);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby={titleId}
      aria-describedby={describedBy}
      className={`max-h-[90vh] w-[calc(100%-2rem)] overflow-y-auto rounded-xl border border-border-default bg-surface p-5 text-text-primary shadow-lift backdrop:bg-overlay/40 ${wide ? "max-w-3xl" : "max-w-lg"}`}
    >
      <div className="mb-4 flex items-center justify-between">
        <h2 id={titleId} ref={titleRef} tabIndex={focusTitle ? -1 : undefined} className="text-base font-semibold focus:outline-none">{title}</h2>
        <button type="button" aria-label="Fermer" onClick={onClose} className="min-h-6 min-w-6 rounded p-1 text-text-muted hover:bg-bg-secondary">✕</button>
      </div>
      {children}
    </dialog>
  );
}

interface TabsProps<T extends string> {
  tabs: { id: T; label: string }[];
  value: T;
  onChange: (id: T) => void;
  /** Contenu de l'onglet actif : s'il est fourni, il est rendu dans un `role="tabpanel"` relié par `aria-controls`. */
  children?: ReactNode;
}

/** Onglets au clavier (WAI-ARIA) : flèches gauche / droite, Début / Fin, `tabIndex` itinérant, activation immédiate. */
export function Tabs<T extends string>({ tabs, value, onChange, children }: TabsProps<T>) {
  const baseId = useId();
  const listRef = useRef<HTMLDivElement>(null);
  const tabId = (id: T) => `${baseId}-tab-${id}`;
  const panelId = `${baseId}-panel`;
  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const index = tabs.findIndex((tab) => tab.id === value);
    const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1 }[event.key];
    if (next === undefined || tabs.length === 0) return;
    event.preventDefault();
    const target = tabs[(next + tabs.length) % tabs.length];
    onChange(target.id);
    listRef.current?.querySelector<HTMLButtonElement>(`[data-tab="${CSS.escape(target.id)}"]`)?.focus();
  };
  return (
    <>
      <div ref={listRef} role="tablist" className="mb-4 flex gap-1 border-b border-border-default">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            id={tabId(tab.id)}
            data-tab={tab.id}
            role="tab"
            aria-selected={tab.id === value}
            aria-controls={children !== undefined ? panelId : undefined}
            tabIndex={tab.id === value ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={onKeyDown}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${tab.id === value ? "border-accent-primary font-medium text-accent-primary" : "border-transparent text-text-secondary hover:text-text-primary"}`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {children !== undefined && (
        <div role="tabpanel" id={panelId} aria-labelledby={tabId(value)} tabIndex={0}>
          {children}
        </div>
      )}
    </>
  );
}

export function scoreTone(score: number): Tone {
  return score >= 75 ? "success" : score >= 50 ? "warning" : "neutral";
}

export function statusTone(status: string | null): Tone {
  switch (status) {
    case "postulée": return "success";
    case "entretien": return "accent";
    case "contactée": return "info";
    case "refusée": case "échec": return "danger";
    case "abandonnée": case "externe": return "neutral";
    default: return "warning";
  }
}

/** Date et heure locales ; accepte `null` / absent (C-2) et l'ISO UTC avec `Z` (C-1). */
export function formatDate(value?: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
}
