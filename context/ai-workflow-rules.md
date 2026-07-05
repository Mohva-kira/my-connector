# AI Workflow Rules

This document defines strict execution rules for any AI coding agent working on this project. These rules are mandatory and must be followed at all times. Violations are considered a failure of the implementation process.

---

## 1. Overall Development Approach

- Follow a **spec-driven development approach** at all times.
- Implement features **incrementally**, never in large uncontrolled batches.
- Treat each feature as a **self-contained unit of work** with clear inputs and outputs.
- Always prioritize **working software over speculative architecture**.
- Do not implement features that are not explicitly defined in the current scope.

---

## 2. Scoping Rules

- Implement **one unit of work at a time only**.
- A "unit" is defined as one of the following:
  - A single backend endpoint
  - A single background job
  - A single AI processing step
  - A single frontend screen or component group
  - A single integration (e.g., Gmail sync)

- Do NOT combine multiple unrelated units into one implementation step.
- Do NOT refactor unrelated code while implementing a new feature.
- Do NOT introduce future features “for convenience”.

---

## 3. When to Split Work into Smaller Steps

Always split work if:

- The implementation involves more than one system layer (frontend + backend + AI + DB).
- The feature requires external integrations (Gmail, Outlook, OpenAI).
- The feature includes both data modeling and UI changes.
- The feature involves background jobs or async processing.
- The implementation exceeds ~150–200 lines of code in a single step.

When splitting:
- First implement backend logic.
- Then implement data persistence.
- Then implement API layer.
- Finally implement frontend integration.

---

## 4. Handling Missing or Ambiguous Requirements

- If a requirement is unclear, STOP implementation immediately.
- Do NOT assume missing business logic.
- Ask for clarification or define a minimal safe default only if explicitly allowed.
- If a default must be chosen:
  - Prefer the simplest implementation.
  - Avoid AI autonomy without explicit rules.
  - Avoid hidden behavior or implicit logic.

Examples:
- If scoring rules are unclear → implement minimal rule set only.
- If UI behavior is unclear → implement static safe display only.

---

## 5. File Modification Rules

NEVER modify the following without explicit instruction:

### UI / Generated Components
- `/frontend/components/ui/*`
- Any third-party or generated UI libraries

### Core AI Engine Contracts
- `/ai-engine/prompts/*`
- `/ai-engine/schemas/*`

### Authentication Layer
- `/backend/auth/*`
- OAuth token handling logic

### Database Schema Migrations
- `/migrations/*` unless explicitly instructed

---

## 6. Code Consistency Rules

- Always reuse existing utilities before creating new ones.
- Do not duplicate logic across services.
- All AI outputs MUST follow predefined JSON schema strictly.
- All API responses MUST be typed and consistent.

---

## 7. Documentation Sync Rules

- Every implemented feature MUST update documentation immediately.
- If a new API endpoint is created, update:
  - `/docs/api.md`
- If architecture changes, update:
  - `/docs/architecture.md`
- If workflow changes, update:
  - `/docs/project-overview.md`

- Documentation must reflect implementation truth. No speculative documentation is allowed.

---

## 8. AI Processing Rules

- AI must NEVER execute actions directly without validation rules.
- All AI outputs must be:
  - validated against schema
  - stored before execution
- AI decisions must be traceable (store reasoning metadata when available)
- AI must NEVER send emails or external messages autonomously.

---

## 9. Background Job Rules

- All background tasks must be idempotent.
- Batch processing must not duplicate email processing.
- Failed jobs must be retry-safe.
- No background job should directly modify frontend state.

---

## 10. Verification Checklist (MANDATORY BEFORE COMPLETION)

Before marking any unit as complete, verify:

### Functionality
- [ ] Feature works end-to-end
- [ ] No runtime errors in logs
- [ ] API returns expected structured output

### Data Integrity
- [ ] Data is correctly stored in database
- [ ] No duplicate or corrupted entries
- [ ] AI output conforms to schema

### Integration
- [ ] Frontend correctly consumes backend data
- [ ] Background jobs execute successfully (if applicable)

### Safety
- [ ] No unauthorized email sending
- [ ] No cross-user data leakage
- [ ] No bypass of authentication layer

### Documentation
- [ ] Relevant docs updated
- [ ] No mismatch between code and documentation

---

## 11. Completion Rule

A unit is NOT complete unless:
- It is fully implemented
- It is tested manually or via logs
- It is integrated into the system flow
- Documentation is updated
- Verification checklist is fully satisfied

No partial completion is allowed.
