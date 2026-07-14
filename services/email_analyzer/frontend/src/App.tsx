import React, { useEffect, useMemo, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { apiFetch } from "./apiClient";
import { parseResponseJson, type Health } from "./apiUtils";
import ActionsPage from "./pages/ActionsPage";
import AgendaPage from "./pages/AgendaPage";
import AssistantPage from "./pages/AssistantPage";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import ProjectDetailPage from "./pages/ProjectDetailPage";
import ProjectHubPage from "./pages/ProjectHubPage";
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
      <div className="flex min-h-screen flex-col items-center justify-center bg-bg-primary px-4">
        <p className="text-sm text-text-muted">Chargement…</p>
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
      <Route
        path="/projects"
        element={<ProjectHubPage saasEnabled={saasEnabled} sessionTick={sessionTick} />}
      />
      <Route
        path="/projects/:projectId"
        element={<ProjectDetailPage saasEnabled={saasEnabled} sessionTick={sessionTick} />}
      />
      <Route
        path="/agenda"
        element={<AgendaPage saasEnabled={saasEnabled} sessionTick={sessionTick} />}
      />
      <Route
        path="/actions"
        element={<ActionsPage saasEnabled={saasEnabled} sessionTick={sessionTick} />}
      />
      <Route
        path="/assistant"
        element={<AssistantPage saasEnabled={saasEnabled} sessionTick={sessionTick} />}
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
