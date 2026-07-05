# Code Standards

## Application Overview

This application is an AI-powered email intelligence assistant. It connects to Gmail and Outlook via OAuth2, ingests emails, applies a hybrid rules + AI scoring model, and delivers structured daily briefings and urgent notifications. The backend is Node.js (NestJS or Express), the frontend is React, background jobs run via BullMQ + Redis, and data is stored in PostgreSQL.

---

## 1. Language and Runtime

- **Backend**: Node.js 20+ (TypeScript strict mode)
- **Frontend**: React 18+ with TypeScript strict mode
- **AI scripts / data utilities**: Python 3.11+ (if used alongside the Node.js backend)
- All TypeScript files must pass `tsc --noEmit` with zero errors before commit.

---

## 2. Project Structure

```
/frontend          React SPA
/backend           NestJS or Express API server
/workers           BullMQ background job processors
/integrations      Gmail API + Microsoft Graph adapters
/ai-engine         Prompt orchestration and schema validation
/shared            Shared types, utilities, scoring rules
/migrations        Alembic (if Python DB) or Knex/Prisma migrations
```

All boundaries must be respected. No cross-boundary imports without going through the API or shared layer.

---

## 3. TypeScript Rules

- `strict: true` in all tsconfig files — no exceptions.
- No `any`. Use `unknown` and narrow with type guards.
- All function parameters and return values must be explicitly typed.
- Use `interface` for data shapes, `type` for unions and mapped types.
- Prefer `readonly` arrays and properties where mutation is not needed.
- No `// @ts-ignore` or `// @ts-expect-error` without a comment explaining why.

---

## 4. API Design

- All endpoints return JSON with a consistent envelope:
  ```json
  { "data": ..., "error": null }
  { "data": null, "error": { "code": "...", "message": "..." } }
  ```
- HTTP status codes must be semantically correct (200, 201, 400, 401, 403, 404, 422, 500).
- All request bodies and query params must be validated (use Zod or class-validator).
- Paginated endpoints return `{ items, total, page, limit }`.
- Never expose internal stack traces in error responses.

---

## 5. Authentication and Security

- All routes except `/auth/*` and `/health` require a valid JWT Bearer token.
- OAuth tokens (Gmail, Outlook) must be encrypted at rest using AES-256 (or Fernet-equivalent).
- Access tokens are never forwarded to the frontend — backend fetches email data directly.
- All user data is scoped by `user_id`. Never expose another user's data.
- Passwords must be hashed with bcrypt (min 12 rounds).
- Environment variables are never committed to version control.
- CORS must be locked to the frontend origin in production.

---

## 6. Database

- Use PostgreSQL only. No other databases in the primary data path.
- All schema changes go through versioned migration files — never alter tables manually.
- Foreign keys and indexes must be explicitly defined for all join columns.
- All queries must be parameterized — no string interpolation in SQL.
- Use transactions for multi-table writes.
- Column naming: `snake_case`.

---

## 7. AI Integration

- All AI calls must produce output that validates against a predefined JSON schema:
  ```json
  {
    "type": "info | action | urgent",
    "importance_score": 0–100,
    "action_required": true | false,
    "deadline": "ISO date string | null",
    "category": "client | personal | admin | spam",
    "confidence": 0–1
  }
  ```
- AI output must be stored before any action is taken on it.
- AI must never send emails, delete emails permanently, or perform external writes.
- Confidence below 0.5 must be flagged for human review before actioning.
- AI prompts live in `/ai-engine/prompts/`. Do not embed prompts inline in business logic.

---

## 8. Background Jobs

- All jobs must be idempotent — re-running must produce the same result as running once.
- Jobs must deduplicate by `(user_id, email_external_id)` before processing.
- Failed jobs must be re-queued with exponential backoff (max 5 retries).
- No background job reads from or writes to the frontend state directly.
- Job payloads must be typed and validated on entry.

---

## 9. Error Handling

- Use structured error classes with `code`, `message`, and optional `context`.
- Catch at the boundary (controller / job handler) — do not swallow errors in service code.
- Log all unexpected errors with correlation IDs.
- Never throw raw strings — always throw proper Error instances.

---

## 10. Code Style

- Indentation: 2 spaces (TypeScript/JavaScript), 4 spaces (Python).
- Max line length: 100 characters.
- No trailing whitespace.
- No commented-out code committed to the repository.
- Function names: `camelCase` (TS/JS), `snake_case` (Python).
- Constants: `UPPER_SNAKE_CASE`.
- File names: `kebab-case.ts` for modules, `PascalCase.tsx` for React components.
- Imports are sorted: external packages first, then internal modules, then types.

---

## 11. Testing

- Unit tests for all scoring rules and AI schema validation.
- Integration tests for all API endpoints (at least happy path + auth failure).
- Background jobs must have at least one end-to-end test.
- No mocking of the database in integration tests.
- Test file naming: `*.spec.ts` (unit), `*.e2e-spec.ts` (integration).

---

## 12. Logging

- Use structured JSON logging in production.
- Log levels: `debug`, `info`, `warn`, `error`.
- Every background job logs start, end, and result count.
- AI calls log model used, token count, and latency.
- Never log email body content or OAuth tokens.

---

## 13. Comments

- Write comments only for non-obvious business rules, security invariants, or workarounds.
- Never write comments that restate what the code does.
- No TODO comments committed without a linked issue.
