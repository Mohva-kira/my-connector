import React, { useEffect, useMemo, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { apiFetch } from "./apiClient";
import { parseResponseJson, type Health } from "./apiUtils";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import SettingsPage from "./pages/SettingsPage";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sessionTick, setSessionTick] = useState(0);
  const [billingReturnBanner, setBillingReturnBanner] = useState(false);

  useEffect(() => {
    const onHash = () => setBillingReturnBanner(window.location.hash.includes("billing/return"));
    onHash();
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    apiFetch("/api/health")
      .then((r) => parseResponseJson(r))
      .then((d) => {
        setHealth(d as Health);
        setHealthError(null);
      })
      .catch((e: unknown) => {
        setHealth(null);
        setHealthError(e instanceof Error ? e.message : "API indisponible");
      });
  }, []);

  const healthReady = health !== null || healthError !== null;
  const saasEnabled = useMemo(() => health?.saas_enabled === true, [health]);

  if (!healthReady) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-stone-100/80 px-4">
        <p className="text-sm text-slate-500">Chargement…</p>
      </div>
    );
  }

  return (
    <Routes>
      <Route
        path="/login"
        element={
          <LoginPage
            saasEnabled={saasEnabled}
            healthError={healthError}
            onAuthed={() => setSessionTick((s) => s + 1)}
          />
        }
      />
      <Route
        path="/"
        element={
          <HomePage
            health={health}
            healthError={healthError}
            saasEnabled={saasEnabled}
            billingReturnBanner={billingReturnBanner}
            sessionTick={sessionTick}
          />
        }
      />
      <Route
        path="/settings"
        element={
          <SettingsPage
            healthError={healthError}
            saasEnabled={saasEnabled}
            billingReturnBanner={billingReturnBanner}
            sessionTick={sessionTick}
            setSessionTick={setSessionTick}
          />
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
