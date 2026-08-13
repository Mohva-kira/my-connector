import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de ProfileOut/ProfileUpdateBody (api/routers/profile.py) —
// domaine "Contexte Personnel" (profil + habitudes), injecté dans le
// Context Engine (ContextService._build_profile_section).
export type FamilyMember = {
  name: string;
  relation?: string | null;
};

export type Profile = {
  display_name: string | null;
  languages: string[] | null;
  timezone: string | null;
  location: string | null;
  preferences: Record<string, string> | null;
  family_members: FamilyMember[] | null;
  favorite_contact_ids: string[] | null;
  wake_time: string | null;
  sleep_time: string | null;
  work_start_time: string | null;
  work_end_time: string | null;
  sport_days: string[] | null;
  focus_hours: string | null;
};

export type ProfileUpdate = Partial<Profile>;

export async function fetchProfile(): Promise<Profile> {
  const res = await apiFetch("/api/profile");
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Profile;
}

export async function updateProfile(patch: ProfileUpdate): Promise<Profile> {
  const res = await apiFetch("/api/profile", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Profile;
}
