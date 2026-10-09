import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { ApiError, ApplicationDetail, Contact, MessageDraft, OPPOSED_DETAIL, Task, api, enc } from "../api";
import { useAction, useStatus } from "../hooks";
import { Button, ErrorBox, Field, Modal, Notice, Tabs, formatDate, inputClass } from "../ui";
import { useConsentGuard } from "./RiskConsentDialog";
import { useTaskUi } from "./TaskDrawer";

export type Channel = "email" | "linkedin";

/** Champs d'un message gardés en mémoire (relance après une popup de risques) ; jamais `confirm`. */
export interface ComposeInitial {
  channel: Channel;
  to: string;
  subject: string;
  body: string;
  attach_cv: boolean;
}

interface ComposeModalProps {
  app: ApplicationDetail;
  contact?: Contact;
  onClose: () => void;
  /** Fourni : la modale s'ouvre directement sur la relecture avec ces champs (rien n'est envoyé sans clic). */
  initial?: ComposeInitial;
}

/** Note de connexion LinkedIn (`LINKEDIN_NOTE_MAX`) : pied compris, le corps est tronqué avant le pied. */
const LINKEDIN_NOTE_MAX = 300;
/** Séparateur corps / mention ajouté par le serveur (backend/01 § 5.7.1, V4-4). */
const FOOTER_SEPARATOR = "\n\n";
/** Longueur en caractères Unicode, comme `len()` côté serveur (pas `s.length`, qui compte des unités UTF-16). */
const charCount = (s: string) => Array.from(s).length;

const draftUrl = (appId: string, to: string) =>
  `/applications/${enc(appId)}/message-draft${to.trim() ? `?to=${enc(to.trim())}` : ""}`;

/**
 * Message à l'entreprise (I-04, I-12, I-13 ; frontend/02 § 4.2 b) : rédaction puis relecture. La relecture affiche
 * l'origine de l'adresse et le pied imposé renvoyé par le serveur (lecture seule), exige la case « J'ai relu ce
 * message et je confirme l'envoi » (jamais pré-cochée, remise à zéro à chaque relecture) et bloque tout envoi vers
 * un destinataire opposé, sans moyen de forcer. « Confirmer l'envoi » n'a jamais le focus initial.
 */
export function ComposeModal({ app, contact, onClose, initial }: ComposeModalProps) {
  const { track } = useTaskUi();
  const { data: status } = useStatus();
  const consent = useConsentGuard();
  const [channel, setChannel] = useState<Channel>(
    initial?.channel ?? (contact?.email || !contact?.profile_url ? "email" : "linkedin"));
  const [to, setTo] = useState({
    email: initial?.channel === "email" ? initial.to : contact?.email ?? "",
    linkedin: initial?.channel === "linkedin" ? initial.to : contact?.profile_url ?? "",
  });
  const [subject, setSubject] = useState(initial?.subject ?? "");
  const [body, setBody] = useState(initial?.body ?? "");
  const [attach, setAttach] = useState(initial?.attach_cv ?? true);
  const [review, setReview] = useState(!!initial);
  // Confirmation de relecture : jamais pré-cochée ni mémorisée (I-04).
  const [reviewed, setReviewed] = useState(false);
  const reviewRef = useRef<HTMLDivElement>(null);

  const target = channel === "email" ? to.email : to.linkedin;
  const initialTarget = useRef(target);

  // Brouillon initial (objet, corps sans pied) : ignoré si les champs viennent de la mémoire (relance).
  useEffect(() => {
    if (initial) return;
    api<MessageDraft>(draftUrl(app.id, initialTarget.current))
      .then((d) => { setSubject(d.subject); setBody(d.body); })
      .catch(() => undefined);
  }, [app.id, initial]);

  // Relecture : pied exact, origine et opposition pour le destinataire saisi, toujours relus depuis le serveur.
  const draft = useQuery({
    queryKey: ["message-draft", app.id, target.trim()],
    queryFn: () => api<MessageDraft>(draftUrl(app.id, target)),
    enabled: review && !!target.trim(),
    staleTime: 0,
    gcTime: 0,
  });

  const send = useAction(
    () => api<Task>(`/applications/${enc(app.id)}/messages`, {
      method: "POST",
      body: { channel, to: target, subject, body, attach_cv: attach, confirm: true },
    }), [["tasks"], ["application", app.id]]);

  // Focus initial de la relecture sur son premier élément (jamais sur « Confirmer l'envoi »).
  useEffect(() => {
    if (review) reviewRef.current?.focus();
  }, [review]);

  const quota = status?.quotas.messages;
  const overQuota = quota ? quota.used >= quota.limit : false;
  const bodyLength = charCount(body);
  const tooLongNote = channel === "linkedin" && bodyLength > LINKEDIN_NOTE_MAX;
  const smtpMissing = channel === "email" && status && !status.smtp_configured;
  const ready = target.trim() && body.trim() && (channel === "linkedin" || subject.trim());
  const contactOpposed = !!contact?.opposed;
  const sendOpposed = send.error instanceof ApiError && send.error.status === 409 && send.error.detail === OPPOSED_DETAIL;
  const opposed = contactOpposed || sendOpposed || !!draft.data?.opposed;

  // Contact correspondant au destinataire : date de collecte pour la ligne « Origine de l'adresse ».
  const matching = [contact, ...app.contacts].find((c) => c && target.trim() && (c.email === target.trim() || c.profile_url === target.trim()));
  const origin = draft.data
    ? draft.data.origin === "linkedin_equipe_recrutement"
      ? `Affichée par LinkedIn dans l'équipe de recrutement de l'offre${matching?.collected_at ? `, le ${formatDate(matching.collected_at)}` : ""}`
      : "Saisie par toi"
    : "";
  const footer = draft.data?.footer ?? "";
  // Compté comme le serveur (V4-4) : corps sans blancs finaux + "\n\n" + mention > 300 ⇒ corps tronqué, mention gardée.
  const noteOverflow = channel === "linkedin" && !!footer
    && charCount(body.trimEnd()) + charCount(FOOTER_SEPARATOR) + charCount(footer) > LINKEDIN_NOTE_MAX;
  // Place laissée au texte dans une note de connexion une fois la mention connue.
  const noteLimit = footer ? LINKEDIN_NOTE_MAX - charCount(FOOTER_SEPARATOR) - charCount(footer) : LINKEDIN_NOTE_MAX;

  const openReview = () => {
    const go = () => { send.reset(); setReviewed(false); setReview(true); };
    if (channel === "linkedin") consent.guard("linkedin", go);
    else go();
  };
  const backToEdit = () => { send.reset(); setReviewed(false); setReview(false); };
  const confirmSend = () => {
    // Après un échec (409, 428…), la case est décochée : un nouvel essai exige une nouvelle confirmation.
    const go = () => send.mutate(undefined, { onSuccess: (task) => { track(task); onClose(); }, onError: () => setReviewed(false) });
    if (channel === "linkedin") consent.guard("linkedin", go);
    else go();
  };

  const opposedBanner = opposed && (
    <Notice tone="danger">Cette personne a demandé à ne plus être contactée : envoi impossible.</Notice>
  );
  const label = "text-xs font-semibold uppercase text-text-muted";

  return (
    <Modal title={review ? "Confirmer l'envoi" : "Envoyer un message à l'entreprise"} onClose={onClose} dismissible={false} wide>
      {!review ? (
        <div className="space-y-3">
          <Tabs<Channel> tabs={[{ id: "email", label: "Email (SMTP)" }, { id: "linkedin", label: "Message LinkedIn" }]} value={channel} onChange={setChannel}>
            <div className="space-y-3">
              {channel === "email" ? (
                <>
                  <Field label="Destinataire" hint={contact ? `${contact.full_name}${contact.email ? "" : " — aucun email affiché par LinkedIn : saisis-le à la main"}` : undefined}>
                    <input className={inputClass} type="email" value={to.email} onChange={(e) => setTo({ ...to, email: e.target.value })} placeholder="recruteur@entreprise.com" />
                  </Field>
                  <Field label="Objet"><input className={inputClass} value={subject} onChange={(e) => setSubject(e.target.value)} /></Field>
                  <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={attach} onChange={(e) => setAttach(e.target.checked)} />Joindre le CV (PDF)</label>
                  {smtpMissing && <Notice tone="warning">SMTP non configuré : renseigne SMTP_HOST et SMTP_FROM (et SMTP_USER / SMTP_PASSWORD) dans le fichier .env.</Notice>}
                </>
              ) : (
                <>
                  <Field label="Profil LinkedIn du destinataire">
                    <input className={inputClass} value={to.linkedin} onChange={(e) => setTo({ ...to, linkedin: e.target.value })} placeholder="https://www.linkedin.com/in/…" />
                  </Field>
                  <Notice>Si cette personne fait partie de tes relations, le message est envoyé directement. Sinon l'outil envoie une demande de connexion avec ce texte comme note (300 caractères maximum, mention de désinscription comprise).</Notice>
                </>
              )}
              <Field label="Message" hint="Une mention imposée (origine de l'adresse et désinscription) sera ajoutée à la fin : tu la verras à la relecture.">
                <textarea className={inputClass} rows={9} value={body} onChange={(e) => setBody(e.target.value)} />
              </Field>
              <div className="text-xs text-text-muted">
                {bodyLength} caractères{channel === "linkedin" && footer && ` · place pour ton texte dans une note de connexion : ${noteLimit} (mention déduite)`}
                {(tooLongNote || noteOverflow) && <span className="ml-2 text-warning-strong">— au-delà de {noteLimit}, la note de connexion sera tronquée</span>}
              </div>
            </div>
          </Tabs>
          {contactOpposed && opposedBanner}
          {overQuota && <Notice tone="danger">Plafond quotidien de messages atteint ({quota?.used} / {quota?.limit}).</Notice>}
          <div className="flex justify-end gap-2">
            <Button onClick={onClose}>Annuler</Button>
            <Button variant="primary" disabled={!ready || overQuota || !!smtpMissing || contactOpposed} onClick={openReview}>Relire avant l'envoi</Button>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div ref={reviewRef} tabIndex={-1} className="space-y-3 focus:outline-none">
            {opposedBanner || <Notice tone="warning">Ce message sera réellement envoyé. Vérifie le destinataire et le contenu.</Notice>}
            <dl className="space-y-2 text-sm">
              <div><dt className={label}>Canal</dt><dd>{channel === "email" ? "Email (SMTP)" : "Message LinkedIn"}</dd></div>
              <div><dt className={label}>Destinataire</dt><dd className="break-all">{target}</dd></div>
              <div><dt className={label}>Origine de l'adresse</dt><dd>{origin || "…"}</dd></div>
              {channel === "email" && <div><dt className={label}>Objet</dt><dd>{subject}</dd></div>}
              {channel === "email" && <div><dt className={label}>Pièce jointe</dt><dd>{attach ? "CV (PDF)" : "Aucune"}</dd></div>}
              <div><dt className={label}>Message</dt><dd><pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-bg-secondary p-3 font-sans text-sm">{body}</pre></dd></div>
              <div>
                <dt className={label}>Mention ajoutée à la fin (imposée, non modifiable)</dt>
                <dd><p className="whitespace-pre-wrap rounded-md border border-dashed border-border-strong p-3 text-sm text-text-secondary">{footer || "…"}</p></dd>
              </div>
            </dl>
            {noteOverflow && <Notice tone="warning">Avec la mention, la note de connexion dépasse 300 caractères : ton texte sera tronqué avant la mention, qui est toujours conservée.</Notice>}
          </div>
          <ErrorBox error={draft.error} />
          {!sendOpposed && <ErrorBox error={send.error} />}
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" className="mt-0.5" checked={reviewed} disabled={opposed} onChange={(e) => setReviewed(e.target.checked)} />
            <span>J'ai relu ce message et je confirme l'envoi</span>
          </label>
          <div className="flex justify-end gap-2">
            <Button onClick={backToEdit}>Modifier</Button>
            <Button variant="primary" disabled={!reviewed || opposed || !draft.data || overQuota || !!smtpMissing} loading={send.isPending} onClick={confirmSend}>Confirmer l'envoi</Button>
          </div>
        </div>
      )}
      {consent.dialog}
    </Modal>
  );
}
