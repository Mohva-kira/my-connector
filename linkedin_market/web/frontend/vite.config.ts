import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Développement : `python -m linkedin_market.web --dev` (API sans jeton) puis `npm run dev`.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Morceaux vendeurs séparés (bundle principal < 500 ko) : React, Zod, Markdown.
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("/zod/")) return "zod";
          if (/\/(react-markdown|remark-|rehype-|micromark|mdast-|hast-|unified|unist-|vfile|property-information|space-separated-tokens|comma-separated-tokens|decode-named-character-reference|character-entities|html-url-attributes|devlop|bail|trough|is-plain-obj|ccount|longest-streak|markdown-table|zwitch|trim-lines|estree-util|style-to|inline-style-parser)/.test(id)) return "markdown";
          if (/\/(react|react-dom|scheduler|react-router|react-router-dom|@remix-run|@tanstack)\//.test(id)) return "react";
          return undefined;
        },
      },
    },
  },
  server: {
    port: 5174,
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: false } },
  },
});
