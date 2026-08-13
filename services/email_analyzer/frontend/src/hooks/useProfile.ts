import { useCallback, useEffect, useState } from "react";
import { fetchProfile, updateProfile, type Profile, type ProfileUpdate } from "../profileApi";

// Charge le Contexte Personnel (profil + habitudes) pour la section
// correspondante de SettingsPage — même patron que useMemoryAndKnowledge.
// `enabled=false` (mode legacy) n'appelle jamais /api/profile (503 sans SaaS).
export function useProfile(enabled: boolean) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError(null);
    try {
      setProfile(await fetchProfile());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const save = useCallback(async (patch: ProfileUpdate) => {
    const updated = await updateProfile(patch);
    setProfile(updated);
    return updated;
  }, []);

  return { profile, loading, error, reload, save };
}
