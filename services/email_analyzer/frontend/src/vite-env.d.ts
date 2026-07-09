/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL de l'API (ex. https://api.exemple.com). Vide = même origine. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
