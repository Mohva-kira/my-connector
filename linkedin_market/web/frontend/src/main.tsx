import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "@fontsource-variable/inter";
import App from "./App";
import { TaskProvider } from "./components/TaskDrawer";
import "./index.css";

const client = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: false, staleTime: 5000 } },
});

// Le thème est posé avant le premier rendu par le script inline d'index.html (pas de flash clair en sombre).

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <TaskProvider>
          <App />
        </TaskProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
