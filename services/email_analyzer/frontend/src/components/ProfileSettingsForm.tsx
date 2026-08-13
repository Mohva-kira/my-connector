import React, { useEffect, useState } from "react";
import { useProfile } from "../hooks/useProfile";
import type { FamilyMember, ProfileUpdate } from "../profileApi";

// Contexte Personnel (domaine #1, "la base de tout") : formulaire profil +
// habitudes, injecté dans le Context Engine (ContextService, voir
// build_profile_context) — permet à l'assistant de répondre par ex.
// "Tu travailles généralement entre 8h et 18h" une fois ces champs remplis.
// Volontairement pas de sélecteur de contacts favoris ici (favorite_contact_ids
// existe côté API mais n'a pas d'UI en v1 — nécessiterait un multi-select
// croisant la liste de contacts de /memory, hors périmètre de ce formulaire).
const inputClass =
  "mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-2 text-sm text-text-primary";
const labelClass = "block text-sm text-text-secondary";

function parseCsv(value: string): string[] | null {
  const items = value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return items.length ? items : null;
}

function parseFamilyMembers(value: string): FamilyMember[] | null {
  const lines = value
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (!lines.length) return null;
  return lines.map((line) => {
    const [name, relation] = line.split(",").map((s) => s.trim());
    return relation ? { name, relation } : { name };
  });
}

function parsePreferences(value: string): Record<string, string> | null {
  const lines = value
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (!lines.length) return null;
  const out: Record<string, string> = {};
  for (const line of lines) {
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    out[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  }
  return Object.keys(out).length ? out : null;
}

export default function ProfileSettingsForm() {
  const { profile, loading, error, save } = useProfile(true);

  const [displayName, setDisplayName] = useState("");
  const [languages, setLanguages] = useState("");
  const [timezone, setTimezone] = useState("");
  const [location, setLocation] = useState("");
  const [wakeTime, setWakeTime] = useState("");
  const [sleepTime, setSleepTime] = useState("");
  const [workStart, setWorkStart] = useState("");
  const [workEnd, setWorkEnd] = useState("");
  const [sportDays, setSportDays] = useState("");
  const [focusHours, setFocusHours] = useState("");
  const [familyMembers, setFamilyMembers] = useState("");
  const [preferences, setPreferences] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!profile) return;
    setDisplayName(profile.display_name ?? "");
    setLanguages((profile.languages ?? []).join(", "));
    setTimezone(profile.timezone ?? "");
    setLocation(profile.location ?? "");
    setWakeTime(profile.wake_time?.slice(0, 5) ?? "");
    setSleepTime(profile.sleep_time?.slice(0, 5) ?? "");
    setWorkStart(profile.work_start_time?.slice(0, 5) ?? "");
    setWorkEnd(profile.work_end_time?.slice(0, 5) ?? "");
    setSportDays((profile.sport_days ?? []).join(", "));
    setFocusHours(profile.focus_hours ?? "");
    setFamilyMembers(
      (profile.family_members ?? []).map((m) => (m.relation ? `${m.name}, ${m.relation}` : m.name)).join("\n"),
    );
    setPreferences(
      Object.entries(profile.preferences ?? {})
        .map(([k, v]) => `${k}: ${v}`)
        .join("\n"),
    );
  }, [profile]);

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    setBusy(true);
    try {
      const patch: ProfileUpdate = {
        display_name: displayName.trim() || null,
        languages: parseCsv(languages),
        timezone: timezone.trim() || null,
        location: location.trim() || null,
        wake_time: wakeTime || null,
        sleep_time: sleepTime || null,
        work_start_time: workStart || null,
        work_end_time: workEnd || null,
        sport_days: parseCsv(sportDays),
        focus_hours: focusHours.trim() || null,
        family_members: parseFamilyMembers(familyMembers),
        preferences: parsePreferences(preferences),
      };
      await save(patch);
      setMsg("Profil enregistré");
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Erreur");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-2xl border border-border-default bg-surface p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-text-primary">Profil & habitudes</h2>
      <p className="mt-1 text-sm text-text-muted">
        Sert de base au Context Engine : l&apos;assistant en tient compte pour ne pas proposer, par
        exemple, de réunion en dehors de vos horaires de travail habituels.
      </p>

      {loading ? <p className="mt-4 text-sm text-text-muted">Chargement…</p> : null}
      {error ? <p className="mt-4 text-sm text-danger">{error}</p> : null}

      {!loading && !error ? (
        <form className="mt-4 space-y-3" onSubmit={(e) => void handleSave(e)}>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className={labelClass}>
              Nom affiché
              <input className={inputClass} value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
            </label>
            <label className={labelClass}>
              Langues (séparées par des virgules)
              <input
                className={inputClass}
                value={languages}
                onChange={(e) => setLanguages(e.target.value)}
                placeholder="fr, en"
              />
            </label>
            <label className={labelClass}>
              Fuseau horaire
              <input
                className={inputClass}
                value={timezone}
                onChange={(e) => setTimezone(e.target.value)}
                placeholder="Africa/Abidjan"
              />
            </label>
            <label className={labelClass}>
              Localisation
              <input className={inputClass} value={location} onChange={(e) => setLocation(e.target.value)} />
            </label>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className={labelClass}>
              Début horaires de travail
              <input type="time" className={inputClass} value={workStart} onChange={(e) => setWorkStart(e.target.value)} />
            </label>
            <label className={labelClass}>
              Fin horaires de travail
              <input type="time" className={inputClass} value={workEnd} onChange={(e) => setWorkEnd(e.target.value)} />
            </label>
            <label className={labelClass}>
              Heure de réveil
              <input type="time" className={inputClass} value={wakeTime} onChange={(e) => setWakeTime(e.target.value)} />
            </label>
            <label className={labelClass}>
              Heure de sommeil
              <input type="time" className={inputClass} value={sleepTime} onChange={(e) => setSleepTime(e.target.value)} />
            </label>
            <label className={labelClass}>
              Jours de sport (séparés par des virgules)
              <input
                className={inputClass}
                value={sportDays}
                onChange={(e) => setSportDays(e.target.value)}
                placeholder="mardi, jeudi"
              />
            </label>
            <label className={labelClass}>
              Plage de concentration préférée
              <input
                className={inputClass}
                value={focusHours}
                onChange={(e) => setFocusHours(e.target.value)}
                placeholder="9h-11h"
              />
            </label>
          </div>

          <label className={labelClass}>
            Famille (une personne par ligne, format « Nom, relation »)
            <textarea
              className={`${inputClass} min-h-[4rem]`}
              value={familyMembers}
              onChange={(e) => setFamilyMembers(e.target.value)}
              placeholder={"Awa, conjointe\nLéa, fille"}
            />
          </label>

          <label className={labelClass}>
            Préférences (une par ligne, format « clé: valeur »)
            <textarea
              className={`${inputClass} min-h-[4rem]`}
              value={preferences}
              onChange={(e) => setPreferences(e.target.value)}
              placeholder={"stack_frontend: React"}
            />
          </label>

          {msg ? <p className="text-sm text-text-secondary">{msg}</p> : null}
          <div className="pt-2">
            <button
              type="submit"
              disabled={busy}
              className="rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              Enregistrer
            </button>
          </div>
        </form>
      ) : null}
    </section>
  );
}
