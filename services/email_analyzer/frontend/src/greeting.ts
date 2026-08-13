// Salutation selon l'heure locale + rotation de sous-titres pour l'écran
// d'accueil du chat (ChatHome). Rotation indexée par jour-de-l'année (pas
// d'aléatoire au montage) : ChatHome peut se remonter en navigant vers une
// conversation puis en revenant à "/" — un tirage aléatoire à chaque montage
// donnerait une impression d'instabilité, alors qu'un index par jour reste
// stable dans la journée tout en variant réellement d'un jour à l'autre.
export function getTimeOfDaySalutation(date: Date = new Date()): string {
  const hour = date.getHours();
  if (hour >= 5 && hour < 18) return "Bonjour";
  if (hour >= 18 && hour < 22) return "Bonsoir";
  return "Bonne nuit";
}

function dayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0);
  return Math.floor((date.getTime() - start.getTime()) / 86_400_000);
}

export function pickVariant<T>(variants: readonly T[], date: Date = new Date()): T {
  return variants[dayOfYear(date) % variants.length];
}

export const SAAS_BRIEF_SUBTITLES: ReadonlyArray<(since: string) => string> = [
  (since) => `Voici ce qui a changé depuis ${since}.`,
  (since) => `Un point rapide sur les nouveautés depuis ${since}.`,
  (since) => `Voici où en sont vos projets depuis ${since}.`,
  (since) => `Un résumé de votre activité depuis ${since}.`,
];

export const SAAS_LOADING_SUBTITLES: readonly string[] = [
  "Je prépare votre point du jour…",
  "Je regarde ce qui a changé pour vous…",
  "Un instant, je rassemble vos dernières infos…",
];

export const LEGACY_SUBTITLES: readonly string[] = [
  "Comment puis-je vous aider aujourd'hui ?",
  "Qu'est-ce qu'on regarde aujourd'hui ?",
  "Prêt(e) à faire le tri dans vos emails ?",
  "Une question sur vos emails ?",
];
