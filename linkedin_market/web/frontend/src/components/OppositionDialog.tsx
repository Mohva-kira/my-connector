import { useRef, useState } from "react";
import { Contact, SUPPRESSION_REASONS, SuppressionReason, api } from "../api";
import { useAction } from "../hooks";
import { fieldErrors, suppressionSchema } from "../schemas";
import { Button, ErrorBox, Field, Modal, Notice, inputClass } from "../ui";

const REASON_LABELS: Record<SuppressionReason, string> = {
  désinscription: "Demande de désinscription (réponse « STOP », email)",
  refus_oral: "Refus exprimé de vive voix",
  plainte: "Plainte",
};

interface OppositionDialogProps {
  /** Contact enregistré (coordonnées telles que connues du serveur). */
  contact: Contact;
  /** Candidature du contact : son détail est relu après l'enregistrement (badge « Opposé »). */
  applicationId: string;
  onClose: () => void;
}

/** Coordonnées du contact à bloquer : email puis profil LinkedIn, chacun s'il est renseigné. */
export const oppositionRecipients = (contact: Contact) =>
  [contact.email ?? "", contact.profile_url].map((v) => v.trim()).filter(Boolean);

/**
 * Opposition d'un destinataire (I-13, frontend/02 § 4.7) : définitive. Le motif n'est jamais présélectionné ; rien
 * n'est envoyé avant le clic sur « Enregistrer l'opposition ». Une opposition est enregistrée pour l'email ET pour le
 * profil LinkedIn du contact. Le serveur bloque déjà toutes les coordonnées liées au contact du destinataire reçu
 * (backend/01 § 5.7.1, V4-1) ; envoyer `POST /api/suppressions` une fois par coordonnée reste sûr (201 idempotent).
 */
export function OppositionDialog({ contact, applicationId, onClose }: OppositionDialogProps) {
  const recipients = oppositionRecipients(contact);
  const [reason, setReason] = useState<SuppressionReason | "">("");
  const [error, setError] = useState<string>("");
  const reasonRef = useRef<HTMLSelectElement>(null);

  const save = useAction(
    async (bodies: { recipient: string; reason: SuppressionReason }[]) => {
      for (const body of bodies) await api<{ ok: boolean }>("/suppressions", { method: "POST", body });
    },
    [["application", applicationId], ["applications"]]);

  const submit = () => {
    setError("");
    if (!reason) return;
    const bodies = [];
    for (const recipient of recipients) {
      const checked = suppressionSchema.safeParse({ recipient, reason });
      if (!checked.success) {
        setError(Object.values(fieldErrors(checked.error))[0] ?? "");
        return;
      }
      bodies.push(checked.data);
    }
    save.mutate(bodies, { onSuccess: onClose });
  };

  return (
    <Modal title="Ce contact refuse d'être recontacté" onClose={onClose} dismissible={false} initialFocus={reasonRef}>
      <div className="space-y-3 text-sm">
        <p>
          <span className="font-medium">{contact.full_name || "Contact sans nom"}</span>
          {recipients.length > 0 && <span className="block break-all text-text-secondary">{recipients.join(" · ")}</span>}
        </p>
        <Field label="Motif">
          <select ref={reasonRef} className={inputClass} value={reason} onChange={(e) => setReason(e.target.value as SuppressionReason | "")}>
            <option value="">Choisis un motif</option>
            {SUPPRESSION_REASONS.map((r) => <option key={r} value={r}>{REASON_LABELS[r]}</option>)}
          </select>
        </Field>
        <Notice tone="danger">Définitif : aucun message ne pourra plus être envoyé à cette adresse ni à ce profil.</Notice>
        {recipients.length === 0 && <Notice tone="warning">Ce contact n'a ni email ni profil LinkedIn : rien à enregistrer.</Notice>}
        <ErrorBox error={error || save.error} />
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Annuler</Button>
          <Button variant="danger" disabled={!reason || recipients.length === 0} loading={save.isPending} onClick={submit}>
            Enregistrer l'opposition
          </Button>
        </div>
      </div>
    </Modal>
  );
}
