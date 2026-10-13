import { useRef, useState } from "react";
import { ApiError, ManualItemCreated } from "../api";
import { useImportItem } from "../hooks";
import { MANUAL_TEXT_MAX, fieldErrors, manualImportSchema } from "../schemas";
import { Button, ErrorBox, Field, Modal, Notice, inputClass } from "../ui";

const FIELDS = ["url", "title", "company", "location", "keyword", "text"] as const;
type FieldName = (typeof FIELDS)[number];
type FormState = Record<FieldName, string>;

const EMPTY: FormState = { url: "", title: "", company: "", location: "", keyword: "", text: "" };

const isField = (name: string): name is FieldName => (FIELDS as readonly string[]).includes(name);

/**
 * `detail` d'un 422 (« champ : message », plusieurs séparés par « ; ») → messages rattachés aux champs du formulaire.
 * Ce qui ne se rattache à aucun champ reste dans `rest` (affiché en tête, tel quel).
 */
function splitServerErrors(detail: string): { byField: Partial<Record<FieldName, string>>; rest: string[] } {
  const byField: Partial<Record<FieldName, string>> = {};
  const rest: string[] = [];
  for (const part of detail.split(" ; ")) {
    const match = /^(\w+) : (.+)$/s.exec(part);
    if (match && isField(match[1])) byField[match[1]] ??= match[2];
    else if (part.trim()) rest.push(part);
  }
  return { byField, rest };
}

interface ManualImportModalProps {
  onClose: () => void;
  /** Appelé après le `201` (la modale doit alors être fermée par le parent). */
  onImported: (item: ManualItemCreated) => void;
}

/**
 * Import manuel d'une offre Indeed (R-6) : Indeed n'autorise pas la collecte, l'utilisateur colle lui-même le texte.
 * Aucun appel vers Indeed, aucune prévisualisation de l'adresse, aucun pré-remplissage ni lecture du presse-papiers.
 */
export function ManualImportModal({ onClose, onImported }: ManualImportModalProps) {
  const [form, setForm] = useState<FormState>(EMPTY);
  const [touched, setTouched] = useState<ReadonlySet<FieldName>>(new Set());
  // Erreurs 422 du serveur par champ, retirées dès que le champ est modifié.
  const [serverErrors, setServerErrors] = useState<Partial<Record<FieldName, string>>>({});
  const [otherError, setOtherError] = useState<unknown>(null);
  const urlRef = useRef<HTMLInputElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const importItem = useImportItem();

  const checked = manualImportSchema.safeParse(form);
  const clientErrors = checked.success ? {} : fieldErrors(checked.error);
  const textLength = Array.from(form.text.trim()).length;

  const set = (name: FieldName, value: string) => {
    setForm((f) => ({ ...f, [name]: value }));
    setServerErrors((current) => {
      const next = { ...current };
      delete next[name];
      return next;
    });
  };
  const touch = (name: FieldName) => setTouched((t) => new Set(t).add(name));
  const errorOf = (name: FieldName): string | undefined =>
    serverErrors[name] ?? (touched.has(name) ? clientErrors[name] : undefined);

  const submit = () => {
    if (!checked.success) {
      setTouched(new Set(FIELDS));
      return;
    }
    setOtherError(null);
    const { keyword, ...rest } = checked.data;
    importItem.mutate(
      { source_code: "indeed", ...rest, ...(keyword ? { keyword } : {}) },
      {
        onSuccess: onImported,
        onError: (error) => {
          // 409 et autres : bandeau, saisie conservée. 422 : détail rattaché aux champs.
          if (error instanceof ApiError && error.status === 422) {
            const { byField, rest: others } = splitServerErrors(error.detail);
            setServerErrors(byField);
            // Une erreur rattachée à un champ y déplace le focus (le champ est relié à son message par aria-describedby).
            if (Object.keys(byField).length > 0) {
              setTimeout(() => formRef.current?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus(), 0);
            }
            setOtherError(others.length > 0 ? others.join(" ; ") : Object.keys(byField).length > 0 ? null : error);
          } else {
            setServerErrors({});
            setOtherError(error);
          }
        },
      },
    );
  };

  return (
    <Modal title="Importer une offre Indeed" onClose={onClose} initialFocus={urlRef} wide>
      <form
        ref={formRef}
        className="space-y-3"
        noValidate
        onSubmit={(event) => { event.preventDefault(); submit(); }}
      >
        <Notice>
          Indeed n'autorise pas la collecte automatique : copie toi-même le texte de l'offre. Rien n'est téléchargé depuis
          Indeed : l'adresse est seulement conservée pour que tu puisses rouvrir l'offre.
        </Notice>
        <Field label="Adresse de l'offre (https)" error={errorOf("url")}>
          <input ref={urlRef} className={inputClass} type="text" inputMode="url" autoComplete="off" spellCheck={false}
            value={form.url} onChange={(e) => set("url", e.target.value)} onBlur={() => touch("url")} placeholder="https://fr.indeed.com/viewjob?jk=…" />
        </Field>
        <Field label="Titre" error={errorOf("title")}>
          <input className={inputClass} value={form.title} onChange={(e) => set("title", e.target.value)} onBlur={() => touch("title")} />
        </Field>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Entreprise" error={errorOf("company")}>
            <input className={inputClass} value={form.company} onChange={(e) => set("company", e.target.value)} onBlur={() => touch("company")} />
          </Field>
          <Field label="Lieu" error={errorOf("location")}>
            <input className={inputClass} value={form.location} onChange={(e) => set("location", e.target.value)} onBlur={() => touch("location")} />
          </Field>
        </div>
        <Field label="Mot-clé (facultatif)" hint="Vide = mot-clé de ta dernière analyse ou de Paramètres" error={errorOf("keyword")}>
          <input className={inputClass} value={form.keyword} onChange={(e) => set("keyword", e.target.value)} onBlur={() => touch("keyword")} />
        </Field>
        <Field label="Texte de l'offre" error={errorOf("text")}>
          <textarea className={`${inputClass} font-sans`} rows={10} value={form.text}
            onChange={(e) => set("text", e.target.value)} onBlur={() => touch("text")} />
        </Field>
        <p className="-mt-2 text-right text-xs text-text-muted">
          {textLength.toLocaleString("fr-FR")} / {MANUAL_TEXT_MAX.toLocaleString("fr-FR")}
        </p>
        <ErrorBox error={otherError} />
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Annuler</Button>
          <Button type="submit" variant="primary" loading={importItem.isPending} disabled={!checked.success}>Importer</Button>
        </div>
      </form>
    </Modal>
  );
}
