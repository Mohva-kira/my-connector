import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import jsxA11y from "eslint-plugin-jsx-a11y";
import globals from "globals";

export default tseslint.config(
  { ignores: ["dist"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  reactHooks.configs.flat.recommended,
  jsxA11y.flatConfigs.recommended,
  { files: ["src/**/*.{ts,tsx}"], languageOptions: { globals: globals.browser } },
  // Dette lint (frontend/04 § 4.3) soldée à la vague 7 : set-state-in-effect, no-autofocus,
  // no-static-element-interactions et no-noninteractive-element-interactions sont en "error" (préréglages).
);
