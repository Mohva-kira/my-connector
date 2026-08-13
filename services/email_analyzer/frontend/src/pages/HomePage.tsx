import React from "react";
import { Navigate } from "react-router-dom";
import { getAccessToken } from "../apiClient";
import type { Health } from "../apiUtils";
import { useSaasSession } from "../SaasPanels";
import ChatHome from "./ChatHome";

export default function HomePage({
  health,
  healthError,
  saasEnabled,
  billingReturnBanner,
  sessionTick,
}: {
  health: Health | null;
  healthError: string | null;
  saasEnabled: boolean;
  billingReturnBanner: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);

  if (saasEnabled && getAccessToken() && !me) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-bg-primary px-4">
        <p className="text-sm text-text-muted">Chargement du compte…</p>
      </div>
    );
  }

  if (saasEnabled && !getAccessToken()) {
    return <Navigate to="/login" replace />;
  }

  return (
    <ChatHome
      mode={saasEnabled ? "saas" : "legacy"}
      me={me}
      saasEnabled={saasEnabled}
      health={health}
      healthError={healthError}
      billingReturnBanner={billingReturnBanner}
    />
  );
}
