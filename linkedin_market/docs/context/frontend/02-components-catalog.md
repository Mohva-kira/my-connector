# Catalogue de Composants UI

> Vérifié dans `web/frontend/src/` : `ui.tsx` (primitives), `components/` (`TaskDrawer.tsx`, `RunModal.tsx`,
> `ComposeModal.tsx`, `icons.tsx`), composants locaux aux pages (`pages/*.tsx`). Aucune bibliothèque de
> composants externe (ni Radix, ni Headless UI) : uniquement React 18 + Tailwind ; Markdown rendu par
> `react-markdown` 10 + `remark-gfm` 4 (sans HTML brut : `rehype-raw` n'est pas utilisé, donc pas d'injection HTML).

## 1. Organisation des Composants

### 1.1 Arborescence réelle `[ACTUEL]`

```
src/
├── main.tsx               QueryClient, BrowserRouter, TaskProvider, police Inter auto-hébergée
├── App.tsx                coquille : navigation, BlockBanners (repos par source), ConsentGate (428), <Routes>
│                          dans ErrorBoundary
├── api.ts                 client HTTP (ApiError, onConsentRequired) + types partagés (Task, Settings, SourceInfo…)
├── hooks.ts               hooks TanStack Query (useStatus, useSources…) + useAction, invalidateAfterTask
├── schemas.ts             schémas Zod (réglages, RunBody, opposition, garde du rapport)
├── content/risks.ts       textes de risques versionnés (source unique, I-16)
├── ui.tsx                 primitives : Badge, Card, Button, Field, inputClass, ErrorBox, errorMessage, Notice,
│                          Empty, PageHeader, Modal, Tabs, scoreTone, statusTone, formatDate
├── index.css              jetons CSS (2 thèmes), focus visible, .prose-md, animations
├── components/
│   ├── TaskDrawer.tsx     TaskProvider (contexte useTaskUi), tiroir des tâches, modale de relecture confirm_*
│   ├── RunModal.tsx       formulaire « Lancer une analyse » (sources, max_items, popup préventive)
│   ├── ComposeModal.tsx   rédaction + relecture d'un message à l'entreprise (pied imposé, opposition)
│   ├── RiskConsentDialog.tsx  popup de risques + hook useConsentGuard (popup préventive)
│   ├── SourcePicker.tsx   choix des sources d'une collecte
│   ├── OppositionDialog.tsx   enregistrement d'une opposition (I-13)
│   ├── ErrorBoundary.tsx  frontière d'erreur de rendu autour des routes
│   └── icons.tsx          Icon (SVG au trait 24×24, 21 noms)
└── pages/
    ├── Dashboard.tsx      QuotaBar, ActivityChart (locaux)
    ├── Jobs.tsx           JobDetailModal (exporté, réutilisé par Market)
    ├── Market.tsx         Panel, SectionHeading, ScoreRing, Bars, WeeklyChart, Kpi, OfferCard, CopyButton, MarketBody
    ├── Applications.tsx   Dossier, Apply, Contacts, History, Detail (locaux)
    ├── Skills.tsx
    └── SettingsPage.tsx
```

### 1.2 Organisation cible `[CIBLE]`
> `[ACTUEL]` : `components/ErrorBoundary.tsx`, `src/schemas.ts` (vague 2), `components/RiskConsentDialog.tsx`,
> `components/SourcePicker.tsx`, `src/content/risks.ts` (vague 3), `components/OppositionDialog.tsx` (vague 4).
> Restent `[CIBLE]` : `ManualImportModal` (vague 5), `InterpreterBadge` (vague 6), `ConfirmDialog` (la révocation
> d'un consentement utilise aujourd'hui une `Modal` locale à Paramètres).


La structure plate reste adaptée à la taille du projet (≈ 2 000 lignes). Règles d'évolution :
- une primitive réutilisée par **deux pages ou plus** va dans `ui.tsx` (ex. `Panel`, `ScoreRing`, `Bars` si
  réutilisés hors de Marché) ; au-delà de ~300 lignes, éclater `ui.tsx` en `ui/` (un fichier par primitive,
  réexport par `ui/index.ts`) ;
- un composant métier transverse va dans `components/` : nouveaux `RiskConsentDialog.tsx`,
  `SourcePicker.tsx`, `OppositionDialog.tsx`, `ManualImportModal.tsx`, `InterpreterBadge.tsx`,
  `ErrorBoundary.tsx`, `ConfirmDialog.tsx` ;
- les textes de risques versionnés vont dans `src/content/risks.ts` (cf. § 4.1) ;
- les schémas de formulaire Zod dans `src/schemas.ts` (cf. `05-forms-and-validation.md`).

## 2. États Obligatoires par Composant

| État | Implémentation actuelle `[ACTUEL]` | Règle `[CIBLE]` |
| :--- | :--- | :--- |
| Default | Classes de `Button` / `inputClass`. | Inchangé. |
| Hover | `hover:bg-bg-secondary`, `hover:bg-accent-hover`, `hover:opacity-90` (danger), `transition` sur `Button`. | Inchangé. |
| Focus | Contour natif sur les boutons ; champs : `focus:border-accent-primary` seulement. | Anneau `focus-visible:ring-2` partout (cf. `01-ui-design-system.md` § 4.2). |
| Active / Pressed | Non stylé ; boutons bascule (annotation) via `variant` ou `aria-pressed` (Marché). | `aria-pressed` sur tout bouton bascule. |
| Disabled | `disabled:cursor-not-allowed disabled:opacity-50` ; `disabled` aussi pendant `loading`. | Expliquer la raison (texte ou `title` + `aria-describedby`) quand le blocage vient d'une règle métier (plafond, blocage, identifiants). |
| Loading / Busy | `Button loading` remplace le libellé par « … » ; pages : « Chargement… » ; Marché : squelette `h-40 animate-pulse`. | `aria-busy="true"` + libellé conservé avec indicateur (« Lancer… ») pour ne pas perdre le nom accessible ; largeur minimale pour éviter le décalage. |
| Error | `ErrorBox` (`role="alert"`, bordure `danger/40`). | Erreur de champ sous le champ, `aria-invalid` + `aria-describedby` (cf. `05`). |
| Empty | `Empty` (bordure pointillée, texte d'action : « Lance une analyse, ou assouplis les filtres. »). | Toujours proposer l'action suivante. |

## 3. Primitives (`ui.tsx`) `[ACTUEL]`

### 3.1 `Badge`
`{ tone?: Tone = "neutral"; children: ReactNode }` — `span` arrondi `rounded-full px-2 py-0.5 text-xs font-medium`.
Usage : scores, statuts de candidature et de tâche, compétences (« ✓ python », « ✗ aws »), types d'événement.

### 3.2 `Card`
`{ title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }` — `<section>` avec
en-tête `<h2>` si `title` ou `actions`. Bordure, `bg-surface`, `rounded-lg`, `p-4`, **sans ombre**
(contrairement à `Panel` de Marché qui a `shadow-card` et `rounded-xl`).

### 3.3 `Button` — type `ButtonProps` `[ACTUEL]` (vague 7)

```ts
type ButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type"> & {
  children: ReactNode;
  variant?: "primary" | "secondary" | "danger" | "ghost";   // défaut "secondary"
  loading?: boolean;                                         // loading ⇒ disabled + aria-busy + libellé « … »
  type?: "button" | "submit";                                // défaut "button"
  small?: boolean;                                           // px-2.5 py-1 text-xs, sinon px-3.5 py-2 text-sm
};
```
`forwardRef` : la `ref` atteint le `<button>` (focus initial d'une modale) ; les autres attributs (`aria-label`,
`aria-pressed`, `title`…) sont transmis par `...rest`. Reste `[CIBLE]` : garder le libellé pendant `loading`
(« Lancer… ») au lieu de « … » pour ne pas perdre le nom accessible.

### 3.4 `Field` et `inputClass` `[ACTUEL]`
`Field { label: string; hint?: string; error?: string; children: ReactNode }` — `<label>` englobant : libellé
(`font-medium text-text-secondary`), champ, aide (`text-xs text-text-muted`), erreur (`text-danger-strong`) avec
`id` généré par `useId()` ; le champ enfant reçoit `aria-invalid` + `aria-describedby`. Messages issus de
`schemas.ts` (mêmes textes que le serveur). `inputClass` : chaîne de classes partagée par `input`, `select`,
`textarea` (bordure `border-strong`).

### 3.5 `ErrorBox`, `Notice`, `Empty`, `PageHeader` `[ACTUEL]`
- `ErrorBox { error: unknown }` : rien si `error` est falsy ; sinon `errorMessage(error)` = `ApiError.detail`
  (chaîne normalisée G-1) et, pour un 429 avec `Retry-After`, « Réessaie dans X h / min » (G-2). `role="alert"`.
- `Notice { tone?: Tone = "info"; children }` : bandeau ; `role="status"` pour `success` / `info`.
- `Empty { children }` : état vide.
- `PageHeader { title: string; subtitle?: string; actions?: ReactNode }` : `<h1 tabIndex={-1}>` de page (focus au
  changement de section). Marché a son propre en-tête (animé), mais garde un `<h1>`.

### 3.6 `Modal` `[ACTUEL]` (vague 7)

```ts
interface ModalProps {
  title: string; onClose: () => void; children: ReactNode; wide?: boolean;   // max-w-lg, wide : max-w-3xl
  initialFocus?: RefObject<HTMLElement | null>;   // focus initial (sinon : choix du navigateur)
  dismissible?: boolean;                           // défaut true ; false : clic sur le voile sans effet
  describedBy?: string;                            // aria-describedby (ex. texte des risques)
  focusTitle?: boolean;                            // focus initial sur le titre (tabIndex=-1), prioritaire
}
```
- Élément natif `<dialog>` + `showModal()` : piège de focus, reste de la page inerte, pile de modales gérée par
  le navigateur (seule la plus haute reçoit Échap) ; `aria-labelledby` vers le `<h2>` ; voile
  `backdrop:bg-overlay/40` ; boîte `max-h-[90vh]` défilante ; bouton « ✕ » `aria-label="Fermer"`.
- **Échap** (`cancel`) et « ✕ » appellent `onClose` même si `dismissible = false` : dans un dialogue de décision,
  `onClose` **vaut refus / annulation** (popup de risques : Refuser ; relecture `confirm_*` : Annuler ; message :
  fermeture sans envoi).
- Focus rendu au déclencheur à la fermeture ; les écouteurs sont retirés **avant** `close()` au démontage, si bien
  qu'un démontage ne rappelle jamais `onClose`.
- `dismissible = false` : `RiskConsentDialog`, relecture `confirm_*`, `ComposeModal`, `OppositionDialog`.

### 3.7 `Tabs<T extends string>` `[ACTUEL]` (vague 7)
`{ tabs: { id: T; label: string }[]; value: T; onChange: (id: T) => void; children?: ReactNode }` —
`role="tablist"` / `role="tab"` / `aria-selected` ; flèches gauche / droite, Début / Fin, `tabIndex` itinérant,
activation immédiate ; si `children` est fourni, il est rendu dans un `role="tabpanel"` relié par `aria-controls` /
`aria-labelledby`. Utilisé avec `children` par la candidature (Dossier, Postuler, Contacts (n), Historique) et
`ComposeModal` (Email (SMTP), Message LinkedIn).

### 3.8 Fonctions utilitaires
`scoreTone`, `statusTone` (cf. `01-ui-design-system.md` § 1.2), `formatDate(value?)` →
`toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" })`, valeur brute si date invalide.

### 3.9 `Icon` (`components/icons.tsx`)
`{ name: IconName; className?: string = "h-4 w-4" }` — SVG `stroke="currentColor"`, `aria-hidden`. Noms :
`chart, trend, target, scale, user, info, spark, play, external, download, copy, check, plus, thumbUp,
thumbDown, doc, users, clock, pin, refresh, arrowRight`. Choisi pour éviter les emojis — la page Offres utilise
encore 👍 / 👎 / ✓ / ✗ en texte : `[CIBLE]` remplacer par `Icon thumbUp` / `thumbDown`.

## 4. Composants Métier

### 4.1 `RiskConsentDialog` — popup de risques d'une source `[ACTUEL]` côté UI, backend `[CIBLE]` (I-16, R-1, R-3, R-6)
> **`[ACTUEL]` côté frontend (vagues 3-4, revue ACCEPTÉE le 2026-10-09)**, vérifié contre l'API simulée ; les routes
> `GET /api/sources`, `POST|DELETE /api/sources/{code}/consent` et le 428 restent `[CIBLE]` côté backend : `risks.ts` recopié tel quel ; popup montée
> globalement par un `ConsentGate` qui écoute `onConsentRequired()` (abonnement exposé par `api()` sur tout 428) ;
> aucun rejeu (bandeau « Relancer : … ») ; case « Passer la prudence à « prudent » » cochée par défaut si la
> prudence courante est plus risquée — **conforme** (décision 2026-10-09 : la règle « aucune case pré-cochée »
> vise les confirmations d'envoi et de risques, pas un choix qui réduit le risque ; la case reste visible et
> décochable). Focus initial sur le titre et `aria-describedby` vers le texte : props `focusTitle` / `describedBy`
> de `Modal` (vague 4). Popup **préventive** (avant la requête) : `RunModal`, et via `useConsentGuard()` (attend le
> chargement de `GET /api/sources` avant de décider) « Lancer la candidature », « Chercher les recruteurs »,
> « Importer depuis LinkedIn », ajout de compétence si `linkedin_skill_sync`, relecture et envoi d'un message LinkedIn.
> Case de prudence cochée (confirmée par l'utilisateur, Q4) : un seul `PUT /settings {caution: "prudent"}` après
> l'acceptation (`useSwitchToPrudent`, invalide `["settings"]` et `["status"]`), puis
> `onAccepted({ switchedToPrudent: true })` : `RunModal` passe à « prudent » et vide un `max_items` > 15 ;
> `ConsentGate` réécrit le corps d'un `POST /api/runs` à relancer (`caution: "prudent"`, sans `max_items` > 15).


**Props** :
```ts
interface RiskConsentDialogProps {
  source: SourceInfo;              // { code: "linkedin" | "malt" | "wttj"; label: string; access_mode: "navigateur";
                                   //   risk_version: string; consent: { risk_version: string; accepted_at: string } | null }
  onAccepted: (result: { switchedToPrudent: boolean }) => void;
                                   // consentement enregistré ; l'appelant reprend la main (aucune relance
                                   // automatique) et aligne son formulaire si la prudence est passée à « prudent »
  onRefused: () => void;           // abandonne l'action, aucune requête mutative
}
```

**Quand l'afficher** — avant toute action qui utilise une source `navigateur` si
`consent === null || consent.risk_version !== source.risk_version`. Points d'entrée : `RunModal` (sources
cochées), « Lancer la candidature », « Chercher les recruteurs de l'offre », message LinkedIn,
« Importer depuis LinkedIn », ajout de compétence avec `linkedin_skill_sync`. Si l'API répond **428** malgré tout
(consentement révoqué ailleurs, version changée entre-temps), le client ouvre la popup pour la source indiquée.
**Décision 2026-10-09 : aucun rejeu automatique.** Après acceptation, la popup se ferme et un bandeau propose le
bouton « Relancer : {libellé de l'action} » ; un clic explicite relance l'action d'origine. Pour un message
LinkedIn, la relance rouvre l'écran de relecture (I-04), elle n'envoie rien directement.

**Contenu** (texte versionné dans `src/content/risks.ts`, clé `"{code}@{risk_version}"` ; si la clé est absente,
la popup affiche « Texte de risques introuvable pour la version {v} : mets à jour l'application » et n'autorise
pas l'acceptation) — exemple pour `linkedin@2026-10` :

1. **Titre** : « Avant d'utiliser LinkedIn avec cet outil ».
2. **Ton compte peut être restreint ou suspendu.** Les conditions d'utilisation de LinkedIn (*User Agreement*,
   version du 3 novembre 2025, § 8.2 « À ne pas faire », points 2, 3 et 13) interdisent les logiciels de collecte
   automatisée, le contournement des limites d'usage et les bots pour accéder au service, envoyer des messages ou
   ajouter des contacts. Cet outil pilote un navigateur avec **ton** compte : LinkedIn peut le détecter et
   restreindre ou fermer ce compte. Ce risque est le tien.
3. **Tu traites des données personnelles de tiers (RGPD).** En installant l'outil, tu es responsable de
   traitement pour les profils, posts et recruteurs collectés. L'outil limite ce qu'il garde (titre et compétences
   seulement pour un profil, aucune donnée sensible — I-09), supprime posts et profils après 30 jours (I-10) et
   n'envoie jamais à Claude de donnée identifiable sur un tiers : l'interprétation ne reçoit que des agrégats ;
   la préparation d'un CV, si tu utilises Claude, lui transmet ton propre profil et le texte de l'offre (I-11, E-6). Tu dois répondre aux demandes des personnes concernées.
4. **Des plafonds de prudence s'appliquent et ne peuvent pas être dépassés.** `prudent` : 15 éléments par type,
   pause de 6 à 14 s ; `normal` : 30, 4 à 9 s ; `agressif` : 60, 2,5 à 6 s ; au plus {apply_daily_max}
   candidatures et {message_daily_max} messages par jour (I-07). Au premier signe de vérification (checkpoint,
   CAPTCHA), l'outil **s'arrête** et attend 6 h ; il ne tente jamais de contourner la vérification (I-06).
5. **Conseil** : commence en **`prudent`**, garde le navigateur visible, et préfère la source **France Travail**
   (API officielle, sans risque pour ton compte) quand elle suffit.
6. Ligne de version : « Version des risques : 2026-10 ». Si `consent` existe avec une autre version :
   bandeau « Ce texte a changé depuis ton acceptation du {date} : relis-le. »

Pour `malt` et `wttj`, mêmes sections 3 à 6 ; la section 2 cite « les conditions d'utilisation de Malt /
Welcome to the Jungle, qui ne proposent pas d'API publique de lecture des offres » sans numéro d'article (aucun
article n'est référencé dans le lot 1).

**Commandes** :
- case à cocher `J'ai lu ces risques et je les accepte pour mon compte {label}` (non cochée à l'ouverture) ;
- case facultative, cochée par défaut si la prudence courante n'est pas `prudent` : `Passer la prudence à
  « prudent »` (enregistre `caution: "prudent"` via `PUT /api/settings` après acceptation) ;
- bouton secondaire **« Refuser »** ; bouton primaire **« J'accepte les risques »**, `disabled` tant que la case
  n'est pas cochée, `loading` pendant `POST /api/sources/{code}/consent`
  `{"risk_version": "...", "accepted": true}` (201).

**Accessibilité** : `<dialog>` modal, `aria-labelledby` (titre), `aria-describedby` (texte des risques) ;
`dismissible = false` (clic sur le voile sans effet) ; **Échap = Refuser** ; focus initial sur le **titre**
(`tabIndex=-1`) pour que le lecteur d'écran lise le texte depuis le début, puis Tab → case → boutons ; piège de
focus ; restitution du focus au bouton déclencheur ; corps défilant si > 90 vh, boutons toujours visibles
(pied fixe, critère 2.4.11).

**Erreurs** : `ErrorBox` dans la popup, case conservée ; un 409 / 422 renvoyé par l'enregistrement est affiché
tel quel ; un échec réseau laisse la popup ouverte.

**Révocation** : carte « Sources et consentements » de la page Paramètres (cf. `04-routing-and-pages.md`) —
pour chaque source `navigateur` : mode d'accès, version acceptée, date, bouton « Révoquer » (dialogue de
confirmation, puis `DELETE /api/sources/{code}/consent`). Après révocation, la prochaine action réaffiche la popup.

```mermaid
flowchart TD
    A["Action utilisant une source navigateur"] --> B{"Consentement actif et risk_version identique ?"}
    B -- "oui" --> E["Requête mutative envoyée"]
    B -- "non" --> C["RiskConsentDialog ouvert"]
    C -- "Refuser ou Échap" --> R["Action abandonnée, focus rendu au déclencheur"]
    C -- "Case cochée puis J'accepte les risques" --> D["POST /api/sources/code/consent"]
    D -- "201" --> E
    D -- "erreur" --> C
    E -- "202 tâche" --> T["Tiroir des tâches"]
    E -- "428 Precondition Required" --> C
    E -- "409 ou 429" --> X["Message d'erreur contextualisé"]
```

#### Textes de référence `2026-10` (à recopier dans `src/content/risks.ts`, rédigés le 2026-10-09, **validés par l'utilisateur le 2026-10-09**)
Structure : `RISKS: Record<"linkedin@2026-10" | "malt@2026-10" | "wttj@2026-10", { title: string; sections:
{ heading: string; body: string }[] }>`. `{prudence}`, `{apply_daily_max}`, `{message_daily_max}` sont remplacés par
les réglages courants. Tout changement de texte = nouvelle version (`2026-11`…) + migration `risk_version` (I-16).

**`linkedin@2026-10`** — titre « Avant d'utiliser LinkedIn avec cet outil »
1. *Ton compte LinkedIn peut être restreint ou fermé.* Les conditions d'utilisation de LinkedIn interdisent les
   logiciels qui collectent des données ou agissent automatiquement sur le service (*User Agreement*, § 8.2
   « À ne pas faire »). Cet outil pilote un navigateur avec ton propre compte : LinkedIn peut le détecter et
   restreindre, suspendre ou fermer ce compte, sans préavis. Ce risque est le tien ; l'outil ne peut pas l'éviter.
2. *Tu deviens responsable des données des personnes collectées (RGPD).* Les profils, posts et recruteurs
   collectés sont des données personnelles. L'outil n'en garde que le minimum (titre et compétences pour un
   profil, aucune donnée sensible), efface posts et profils après 30 jours et n'envoie jamais à Claude
   d'information permettant d'identifier une autre personne. C'est à toi de répondre si quelqu'un te demande
   l'accès ou l'effacement de ses données.
3. *Des limites protègent ton compte et ne peuvent pas être dépassées.* Prudence actuelle : « {prudence} »
   (prudent : 15 éléments par type, pause de 6 à 14 s ; normal : 30, 4 à 9 s ; agressif : 60, 2,5 à 6 s). Au plus
   {apply_daily_max} candidatures et {message_daily_max} messages par jour. Aucune candidature ni aucun message
   ne part sans ta confirmation au moment de l'envoi.
4. *Au premier signe de vérification, tout s'arrête.* Si LinkedIn affiche une vérification (checkpoint,
   CAPTCHA, page de connexion), l'outil s'arrête immédiatement et ne touche plus à LinkedIn pendant 6 h. Il ne
   tente jamais de contourner une vérification : c'est toi qui la résous dans la fenêtre du navigateur.
5. *Conseil.* Commence en « prudent », garde la fenêtre du navigateur visible et, quand elle suffit, préfère la
   source France Travail (API officielle, sans risque pour ton compte).

**`malt@2026-10`** — titre « Avant d'utiliser Malt avec cet outil »
1. *Ton compte Malt peut être restreint.* Malt ne propose pas d'interface officielle pour lire ses offres : l'outil
   lit les pages dans un navigateur, avec ton compte si la lecture l'exige. Ses conditions d'utilisation peuvent interdire cet
   usage automatisé ; Malt peut alors limiter ou fermer ton compte. Ce risque est le tien.
2. *Lecture seule.* Sur Malt, l'outil ne fait que lire des offres : il ne postule pas, n'envoie aucun message et
   ne modifie pas ton profil.
3. *Données personnelles (RGPD).* Une offre peut contenir le nom d'un client ou d'un contact : l'outil ne l'envoie
   jamais à Claude, et efface les offres sans candidature après 90 jours.
4. *Limites et arrêt.* Prudence actuelle : « {prudence} » (15, 30 ou 60 éléments par recherche, avec pauses). Au
   premier signe de vérification, l'outil s'arrête et ne touche plus à Malt pendant 6 h, sans jamais contourner
   la vérification.

**`wttj@2026-10`** — titre « Avant d'utiliser Welcome to the Jungle avec cet outil »
1. *Ton accès peut être bloqué.* Welcome to the Jungle ne propose pas d'interface officielle pour lire ses offres :
   l'outil lit les pages dans un navigateur. Ses conditions d'utilisation peuvent interdire cet usage automatisé ;
   le site peut alors bloquer ton accès ou ton compte. Ce risque est le tien.
2. *Lecture seule.* L'outil ne fait que lire des offres : il ne postule pas et n'écrit rien sur le site.
3. *Données personnelles (RGPD).* Les noms de recruteurs éventuellement présents ne sont jamais envoyés à Claude ;
   les offres sans candidature sont effacées après 90 jours.
4. *Limites et arrêt.* Prudence actuelle : « {prudence} ». Au premier signe de vérification, l'outil s'arrête et ne
   touche plus au site pendant 6 h, sans jamais contourner la vérification.

Pied commun : « Version des risques : 2026-10 » ; case « J'ai lu ces risques et je les accepte pour mon compte
{label} » (non cochée) ; boutons « Refuser » / « J'accepte les risques ». Pas de numéro d'article pour Malt et WTTJ
(non vérifié) ; la référence LinkedIn (§ 8.2) vient du lot 1 et doit être revérifiée à chaque nouvelle version.

### 4.2 Écrans de relecture avant envoi (I-04)

**a) Candidature simplifiée — modale « Relire avant l'envoi »** (`TaskProvider`) `[ACTUEL]`
- Ouverte automatiquement dès qu'une tâche de `useTasks()` est en `attente_saisie` avec `needs_input.field`
  commençant par `confirm` (aujourd'hui uniquement `confirm_submit`, options `["Soumettre", "Annuler"]`,
  `review` = texte de l'écran de relecture LinkedIn tronqué à 3 500 caractères).
- Affiche la question, une explication (« Rien n'est envoyé tant que tu ne cliques pas sur « Soumettre » »), le
  `review` dans un `<pre>`, un bouton par option (le premier en `primary`).
- Réponse via `POST /api/tasks/{id}/answer` avec **`remember: false`** ; le serveur refuse aussi de mémoriser un
  champ `confirm*` (double garde).
- `[ACTUEL]` (revue du 2026-10-09) : `dismissible = false` (voile sans effet) ; fermeture (Échap, « ✕ ») =
  **toujours `"Annuler"`**, quelle que soit la liste d'options (le serveur traite toute valeur autre que
  `Soumettre` comme une annulation) ; focus initial sur « Annuler » s'il est proposé, sinon sur le titre, jamais sur
  « Soumettre ». Le tiroir des tâches n'affiche **aucune** option d'un champ `confirm*` : seule cette modale, avec le
  texte `review`, permet de répondre.
- Reste `[CIBLE]` : afficher le titre de l'offre et l'entreprise, et le délai restant (600 s,
  `INPUT_TIMEOUT_SECONDS`).

**b) Message à l'entreprise — `ComposeModal`** `[ACTUEL]` (I-12 / I-13 côté UI depuis la vague 4 ; backend
`message-draft?to=`, pied serveur et 409 d'opposition `[CIBLE]`)
- Props : `{ app: ApplicationDetail; contact?: Contact; onClose: () => void; initial?: ComposeInitial }` —
  `initial` (canal, destinataire, objet, corps, pièce jointe ; jamais `confirm`) ouvre directement la relecture,
  case décochée, après une relance explicite consécutive à un 428. `Modal` en `dismissible = false` ; fermer
  n'envoie jamais rien.
- Étape 1 (rédaction) : onglets canal, destinataire, objet (email), « Joindre le CV (PDF) », message, compteur ;
  bouton « Relire avant l'envoi » désactivé si champ requis vide, plafond atteint ou SMTP absent (email).
- Étape 2 (relecture) : bandeau « Ce message sera réellement envoyé », liste `<dl>` (Canal, Destinataire, Objet,
  Pièce jointe, Message) ; « Modifier » revient à l'étape 1 sans perte ; « Confirmer l'envoi » →
  `POST /api/applications/{id}/messages` avec `confirm: true` → `track(task)` puis fermeture.
- `[ACTUEL]` côté UI (I-12, I-13, R-4 ; revue du 2026-10-09) — chaque relecture relit
  `GET …/message-draft?to=` (`footer`, `origin`, `opposed`) ; « Confirmer l'envoi » reste désactivé tant qu'ils ne
  sont pas chargés ; longueur LinkedIn comptée comme le serveur (`Array.from`, `len(body.trimEnd()) + 2 +
  len(footer) > 300`, place utile `300 − 2 − len(footer)` affichée) ; popup de risques préventive pour un message
  LinkedIn ; après un échec la case est décochée :
  - ligne **Origine de l'adresse** : « Affichée par LinkedIn dans l'équipe de recrutement de l'offre, le
    {collected_at} » (`origin = linkedin_equipe_recrutement`, date si le contact correspondant porte
    `collected_at`) ou « Saisie par toi » (`manuel`) ;
  - **mention de désinscription** ajoutée par le serveur au gabarit, affichée en lecture seule sous le message
    (non modifiable, non supprimable), texte exact = « pied imposé » défini dans `backend/05-queues-and-integrations.md` (source unique, renvoyé
    par `GET /api/applications/{job_id}/message-draft`) ;
  - case obligatoire **« J'ai relu ce message et je confirme l'envoi »** avant d'activer « Confirmer l'envoi » ;
  - si le destinataire est dans la liste d'opposition (réponse 409 du serveur, ou indicateur `opposed: true` du
    contact) : bandeau `danger` « Cette personne a demandé à ne plus être contactée : envoi impossible » et
    bouton désactivé — aucun moyen de forcer ;
  - focus initial sur le premier élément de la relecture ; « Confirmer l'envoi » n'a jamais le focus initial.

### 4.3 `TaskProvider` / tiroir des tâches (`TaskDrawer.tsx`) `[ACTUEL]`
> Vague 2 : bandeau « Collecte interrompue par LinkedIn : rapport établi sur les données déjà collectées. » si
> `result.blocked` (E-4, option A), aussi dans le tableau de bord.

> Vague 7 : libellés de statut par table (plus de `status.replace(/_/g, " ")`).

> Revue du 2026-10-09 `[ACTUEL]` : `TaskDetail` ne rend aucune option d'un champ `confirm*` (« Relecture ouverte :
> confirme ou annule dans la fenêtre « Relire avant l'envoi » ») ; erreurs des boutons d'options affichées dans
> l'`ErrorBox` du détail ; erreurs de la liste en `text-danger-strong`.


- **Contexte** `useTaskUi(): { track(task: Task): void }` — ouvre le tiroir sur une tâche venant d'être créée et
  invalide `["tasks"]`. Utilisé après chaque mutation qui renvoie une tâche (202).
- **Bouton flottant** « Tâches · N en cours » (bas droite, `z-40`), orange quand une tâche attend l'utilisateur.
  `aria-expanded`, `aria-controls`, contraste `warning-strong` `[ACTUEL]` (vague 7, cf. `01` § 4.1).
- **Tiroir** (`<aside aria-label="Tâches">`) : liste (titre, badge de statut, date, erreur tronquée) ou détail
  (`TaskDetail`).
- **Ouverture automatique** une seule fois par couple tâche / question quand une tâche passe en
  `attente_saisie` (hors `confirm*`) ou `attente_verification_manuelle`.
- **Invalidation globale** : quand une tâche observée passe à un statut final, `client.invalidateQueries()`.
- **`TaskDetail`** (`{ id: string }`) : badge, bouton « Annuler » (`POST /api/tasks/{id}/cancel`) tant que la tâche
  n'est pas finale, message de vérification manuelle, question (`needs_input`) avec boutons d'options **ou** champ
  libre + « Envoyer » (`remember: true`, suggestion en placeholder), erreur de tâche, `result` JSON, journal
  (`logs`, défilement automatique en bas, `font-mono`).

Libellés de statut : affichés aujourd'hui par `status.replace(/_/g, " ")` (« attente saisie »). `[CIBLE]` table
de libellés : `en_attente` « En attente », `en_cours` « En cours », `attente_saisie` « Réponse attendue »,
`attente_verification_manuelle` « Vérification manuelle à faire », `terminée` « Terminée », `échec` « Échec »,
`annulée` « Annulée ».

```mermaid
stateDiagram-v2
    state "en_attente (badge warning)" as en_attente
    state "en_cours (badge info)" as en_cours
    state "attente_saisie : tiroir ouvert, question" as attente_saisie
    state "attente_saisie confirm_* : modale de relecture" as confirmation
    state "attente_verification_manuelle : à résoudre dans la fenêtre du navigateur" as verification
    state "terminée (success) : invalidation globale" as terminee
    state "échec (danger) : erreur affichée" as echec
    state "annulée (neutral)" as annulee

    [*] --> en_attente : POST renvoie 202
    en_attente --> en_cours : verrou navigateur libre
    en_cours --> attente_saisie : question inconnue (I-05)
    attente_saisie --> en_cours : réponse envoyée et mémorisée
    en_cours --> confirmation : écran de relecture atteint
    confirmation --> en_cours : Soumettre ou Annuler (remember false)
    en_cours --> verification : checkpoint détecté
    verification --> en_cours : utilisateur a terminé la vérification
    attente_saisie --> echec : 600 s sans réponse
    confirmation --> echec : 600 s sans réponse
    verification --> echec : délai dépassé, blocage 6 h
    en_cours --> terminee
    en_cours --> echec
    en_attente --> annulee : bouton Annuler
    en_cours --> annulee : bouton Annuler
    attente_saisie --> annulee : bouton Annuler
    terminee --> [*]
    echec --> [*]
    annulee --> [*]
```

`[CIBLE]` pour `attente_verification_manuelle`, le détail affiche : « {Source} demande une vérification.
Termine-la toi-même dans la fenêtre du navigateur ouverte par l'outil : l'outil reprendra seul. Il ne remplit
jamais un CAPTCHA à ta place. » — sans bouton « Continuer » (la reprise est détectée côté serveur).

### 4.4 `RunModal` `[ACTUEL]`
> Vague 2 : `max_items` borné selon la prudence choisie (15 / 30 / 60, message G-3), validation `runBodySchema`.
> Vague 3 (UI `[ACTUEL]`, backend `[CIBLE]`) : `SourcePicker` (si `GET /api/sources` répond ; sinon le serveur prend
> `["linkedin"]`), popup de risques préventive pour la première source cochée qui l'exige, aucun lancement
> automatique après acceptation ; `onAccepted({ switchedToPrudent })` aligne la prudence du formulaire. Reste
> `[CIBLE]` : choix de l'interprète.

Props `{ onClose: () => void }`. Lit `useSettings()` (rend `null` tant qu'ils ne sont pas chargés), état local
`Partial<Settings> & { scrape: boolean }` superposé aux réglages, `POST /api/runs`
`{ scrape, keyword, location, remote_only, caution }` puis `track(task)` et fermeture. Détails de validation :
`05-forms-and-validation.md` § 3.3. `[CIBLE]` : `SourcePicker`, champ `max_items`, choix de l'interprète
(`claude` / `local`), popup de risques, état de chargement visible.

### 4.5 `SourcePicker` `[ACTUEL]` côté UI, backend `[CIBLE]` (vague 3)
> Props `{ sources; value: SourceCode[]; onChange; onReadRisks(source); error? }`. Sources de `GET /api/sources`
> hors `import_manuel` (Indeed jamais proposée, R-6) ; `enabled = false` affichée « bientôt » et non cochable ;
> France Travail non cochable si `configured = false` ; défaut `linkedin` ; bouton « Lire et accepter les risques »
> (aucune requête mutative). Le lien « Importer une offre » reste `[CIBLE]` (vague 5).

Liste de cases à cocher issue de `GET /api/sources` (`enabled = true`) : libellé, badge de mode
(`api_officielle` → `success` « API officielle », `navigateur` → `warning` « Navigateur — risques acceptés le
{date} » ou « Risques à accepter », `import_manuel` → non sélectionnable, lien « Importer une offre »). France
Travail désactivée avec explication si `FRANCE_TRAVAIL_CLIENT_ID` / `SECRET` absents. Au moins une source
cochée si `scrape = true`.

### 4.6 `InterpreterBadge` `[CIBLE]` (I-11)
`{ interpreter: "claude" | "local"; model?: string }` — badge `accent` « Interprétation : Claude ({model}) » ou
`neutral` « Interprétation : locale (règles) ». Affiché dans l'en-tête de Marché et la section « Analyse ».
Aujourd'hui Marché affiche le texte brut `interprétation {report.interpreter}` et « Lecture rédigée par
l'interprète {interpreter} » `[ACTUEL]` — la valeur `none` produit un rapport sans interprétation, que l'UI
affiche « Pas d'interprétation pour ce rapport. » ; ce cas disparaît avec la dépréciation de `none`.

### 4.7 `OppositionDialog` `[ACTUEL]` côté UI, backend `[CIBLE]` (I-13, vague 4)
> Props `{ contact: Contact; applicationId: string; onClose }` ; `Modal` `dismissible = false`, focus initial sur le
> motif, jamais présélectionné ; rien n'est envoyé avant « Enregistrer l'opposition ». Une requête
> `POST /api/suppressions` par coordonnée renseignée (email puis profil), 201 idempotent ; le serveur bloque en plus
> toutes les coordonnées liées (V4-1). Validation `suppressionSchema` (`05` § 4). Après succès : invalidation de la
> candidature ; badge « Opposé », « Écrire » désactivé, champs Email / Profil du contact désactivés.

Bouton « Ce contact refuse d'être recontacté » sur chaque contact (onglet Contacts) → dialogue : motif
(`désinscription`, `refus_oral`, `plainte`), avertissement « Définitif : aucun message ne pourra plus être
envoyé à cette adresse ni à ce profil », bouton `danger` « Enregistrer l'opposition » →
`POST /api/suppressions` `{ recipient, reason }`. Le contact affiche ensuite un badge `danger` « Opposé » et ses
boutons « Écrire » sont désactivés.

### 4.8 `ManualImportModal` (Indeed) `[CIBLE]` (R-6)
Formulaire URL + texte collé (cf. `05-forms-and-validation.md` § 3.7) ; notice « Indeed n'autorise pas la
collecte automatique : copie toi-même le texte de l'offre ».

### 4.9 Composants de pages `[ACTUEL]`
| Composant | Fichier | Props principales | Rôle |
| :--- | :--- | :--- | :--- |
| `JobDetailModal` | `Jobs.tsx` (exporté) | `{ id: string; onClose }` | Détail d'offre : score, compétences acquises / manquantes, critères du score, texte (≤ 8 000 car.), « Préparer la candidature » ou « Voir la candidature ». |
| `OfferCard` | `Market.tsx` | `{ job; index; actionable; onOpen }` | Carte d'offre : `ScoreRing`, lieu, date, concurrence (« Plus de … » en `warning`), annotation, Préparer. `actionable = report.is_latest`. |
| `Kpi` | `Market.tsx` | `{ label; value; suffix?; hint?; delta?; index }` | Indicateur animé (`useCountUp`), écart en points vs rapport précédent du même mot-clé. |
| `Panel`, `SectionHeading`, `Bars`, `WeeklyChart`, `ScoreRing`, `CopyButton` | `Market.tsx` | — | Présentation du rapport. |
| `QuotaBar`, `ActivityChart` | `Dashboard.tsx` | `{ label; used; limit }`, `{ data }` | Quotas du jour, activité 14 jours empilée par type. |
| `Dossier`, `Apply`, `Contacts`, `History`, `Detail` | `Applications.tsx` | `{ app }` / `{ id }` | Onglets de la candidature. |

## 5. Conventions de Nommage & Exports

> Contrat vague 1 (`backend/01` § 1.1) : `Contact.full_name` (et non `name`), `Contact.source` ∈
> `linkedin_equipe_recrutement` | `manuel` ; libellés d'historique `status` → « Statut », `apply` → « Candidature »,
> `contacts` → « Contacts », `message` → « Message ».


- **Fichiers** : PascalCase pour les composants et pages (`TaskDrawer.tsx`, `SettingsPage.tsx`), minuscules
  pour les modules utilitaires (`api.ts`, `hooks.ts`, `ui.tsx`, `icons.tsx`) `[ACTUEL]`.
- **Exports** : pages en `export default` ; composants partagés en exports nommés (`RunModal`, `ComposeModal`,
  `TaskProvider`, `useTaskUi`, `JobDetailModal`, `Icon`) `[ACTUEL]`.
- **Props** : aujourd'hui typées en ligne (`{ id: string; onClose: () => void }`), sauf `ButtonProps`. `[CIBLE]`
  interface `NomComposantProps` dès que les props dépassent trois champs ou que le composant est exporté.
- **Vocabulaire** : identifiants et clés JSON du glossaire du lot 1 uniquement (`job`, `application`,
  `source_code`, `caution`, `label`…), jamais `offer`, `candidate`, `scrape_run`. Libellés UI en français.
  Écart `[ACTUEL]` : le type `Contact.name` (front et API) correspond à `full_name` dans l'ERD cible.
- **TypeScript** : `strict: true` (`tsconfig.json`) ; pas de `any` ; les réponses API sont typées dans `api.ts`.
- **Textes** : chaînes en dur dans les composants (pas de i18n) — acceptable tant que l'interface est
  monolingue ; les textes juridiques (risques, mention de désinscription) sont **centralisés** et versionnés.
