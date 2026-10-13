import { Badge } from "../ui";

/** Indique qui a rédigé l'analyse : Claude (modèle) ou les règles locales (frontend/02 § 4.6). */
export function InterpreterBadge({ interpreter, model }: { interpreter: string; model?: string }) {
  if (interpreter === "claude") {
    return <Badge tone="accent">Interprétation : Claude{model ? ` (${model})` : ""}</Badge>;
  }
  // `local`, `none` (ancien rapport) ou valeur inconnue : lecture par règles.
  return <Badge tone="neutral">Interprétation : locale (règles)</Badge>;
}
