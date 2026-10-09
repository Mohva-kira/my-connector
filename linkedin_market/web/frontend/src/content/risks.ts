// Textes de la popup de risques (I-16), versionnés : clé « {code}@{risk_version} ». Source unique du texte des
// risques (backend/01 § 5.8.1). Textes de référence 2026-10 recopiés tels quels depuis frontend/02 § 4.1, validés par
// l'utilisateur le 2026-10-09. Tout changement de texte = nouvelle version (2026-11…) + migration de `risk_version`.
// `{prudence}`, `{apply_daily_max}` et `{message_daily_max}` sont remplacés par les réglages courants.

export interface RiskText {
  title: string;
  sections: { heading: string; body: string }[];
}

export const RISKS: Record<string, RiskText> = {
  "linkedin@2026-10": {
    title: "Avant d'utiliser LinkedIn avec cet outil",
    sections: [
      {
        heading: "Ton compte LinkedIn peut être restreint ou fermé.",
        body: "Les conditions d'utilisation de LinkedIn interdisent les logiciels qui collectent des données ou agissent automatiquement sur le service (User Agreement, § 8.2 « À ne pas faire »). Cet outil pilote un navigateur avec ton propre compte : LinkedIn peut le détecter et restreindre, suspendre ou fermer ce compte, sans préavis. Ce risque est le tien ; l'outil ne peut pas l'éviter.",
      },
      {
        heading: "Tu deviens responsable des données des personnes collectées (RGPD).",
        body: "Les profils, posts et recruteurs collectés sont des données personnelles. L'outil n'en garde que le minimum (titre et compétences pour un profil, aucune donnée sensible), efface posts et profils après 30 jours et n'envoie jamais à Claude d'information permettant d'identifier une autre personne. C'est à toi de répondre si quelqu'un te demande l'accès ou l'effacement de ses données.",
      },
      {
        heading: "Des limites protègent ton compte et ne peuvent pas être dépassées.",
        body: "Prudence actuelle : « {prudence} » (prudent : 15 éléments par type, pause de 6 à 14 s ; normal : 30, 4 à 9 s ; agressif : 60, 2,5 à 6 s). Au plus {apply_daily_max} candidatures et {message_daily_max} messages par jour. Aucune candidature ni aucun message ne part sans ta confirmation au moment de l'envoi.",
      },
      {
        heading: "Au premier signe de vérification, tout s'arrête.",
        body: "Si LinkedIn affiche une vérification (checkpoint, CAPTCHA, page de connexion), l'outil s'arrête immédiatement et ne touche plus à LinkedIn pendant 6 h. Il ne tente jamais de contourner une vérification : c'est toi qui la résous dans la fenêtre du navigateur.",
      },
      {
        heading: "Conseil.",
        body: "Commence en « prudent », garde la fenêtre du navigateur visible et, quand elle suffit, préfère la source France Travail (API officielle, sans risque pour ton compte).",
      },
    ],
  },
  "malt@2026-10": {
    title: "Avant d'utiliser Malt avec cet outil",
    sections: [
      {
        heading: "Ton compte Malt peut être restreint.",
        body: "Malt ne propose pas d'interface officielle pour lire ses offres : l'outil lit les pages dans un navigateur, avec ton compte si la lecture l'exige. Ses conditions d'utilisation peuvent interdire cet usage automatisé ; Malt peut alors limiter ou fermer ton compte. Ce risque est le tien.",
      },
      {
        heading: "Lecture seule.",
        body: "Sur Malt, l'outil ne fait que lire des offres : il ne postule pas, n'envoie aucun message et ne modifie pas ton profil.",
      },
      {
        heading: "Données personnelles (RGPD).",
        body: "Une offre peut contenir le nom d'un client ou d'un contact : l'outil ne l'envoie jamais à Claude, et efface les offres sans candidature après 90 jours.",
      },
      {
        heading: "Limites et arrêt.",
        body: "Prudence actuelle : « {prudence} » (15, 30 ou 60 éléments par recherche, avec pauses). Au premier signe de vérification, l'outil s'arrête et ne touche plus à Malt pendant 6 h, sans jamais contourner la vérification.",
      },
    ],
  },
  "wttj@2026-10": {
    title: "Avant d'utiliser Welcome to the Jungle avec cet outil",
    sections: [
      {
        heading: "Ton accès peut être bloqué.",
        body: "Welcome to the Jungle ne propose pas d'interface officielle pour lire ses offres : l'outil lit les pages dans un navigateur. Ses conditions d'utilisation peuvent interdire cet usage automatisé ; le site peut alors bloquer ton accès ou ton compte. Ce risque est le tien.",
      },
      {
        heading: "Lecture seule.",
        body: "L'outil ne fait que lire des offres : il ne postule pas et n'écrit rien sur le site.",
      },
      {
        heading: "Données personnelles (RGPD).",
        body: "Les noms de recruteurs éventuellement présents ne sont jamais envoyés à Claude ; les offres sans candidature sont effacées après 90 jours.",
      },
      {
        heading: "Limites et arrêt.",
        body: "Prudence actuelle : « {prudence} ». Au premier signe de vérification, l'outil s'arrête et ne touche plus au site pendant 6 h, sans jamais contourner la vérification.",
      },
    ],
  },
};

/** Texte d'une source à une version donnée ; `undefined` si l'application ne le connaît pas (acceptation impossible). */
export function riskTextOf(code: string, version: string): RiskText | undefined {
  return RISKS[`${code}@${version}`];
}

/** Remplace `{prudence}`, `{apply_daily_max}`, `{message_daily_max}` par les réglages courants. */
export function fillRiskText(text: string, values: Record<string, string | number>): string {
  return text.replace(/\{(\w+)\}/g, (match, key: string) => (key in values ? String(values[key]) : match));
}
