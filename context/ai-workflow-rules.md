# AI Coding Agent: Workflow Rules & Enforcement

You are an automated coding agent tasked with implementing and refactoring this codebase. You must follow these explicit rules under penalty of code rejection. Your operation model is strictly deterministic, spec-driven, and incremental.

---

## 1. Overall Approach & Scoping Rules

1. **Read and Follow Specifications:** You must read `architecture.md`, `project-overview.md`, and `code-standards.md` before writing any code. Every implementation must align with the real stack (FastAPI backend, Redis + `arq` for background jobs, PostgreSQL, React/Vite web frontend — no Celery, no React Native).
2. **Execute Single-Unit Changes Only:** Implement exactly one logical unit of work at a time (e.g., one database migration, one API route, one React hook). Never batch multiple unrelated changes into a single file modification.
3. **Ban Speculative Coding:** Do not write boilerplate code, placeholders, or "future-use" parameters for features that are out of scope or slated for later phases. If code is not needed for the current task, do not write it.
4. **Enforce Invariants Continuously:** You must never write code that violates the system invariants specified in `architecture.md` §8 (Strict 1:1 Email-Project mapping, no raw SQL, tenant isolation, No Heavy I/O in the API Main Thread — delegate to `arq` jobs).
5. **Conversation-First Scoping (pivot, 2026-07-17):** Per `project-overview.md`'s Executive Assistant vision, any new feature should satisfy at least one of: improves understanding of the user's work, improves memory, improves reasoning, improves available context, or improves conversation. Do not add UI complexity that serves none of these. Do not implement Memory Engine / Knowledge Graph / Agent-Tools primitives speculatively — only when a task explicitly asks for that phase of work (see `project-overview.md` roadmap).

---

## 2. Granularity and Splitting Rules

1. **Split Heavy Tasks:** If a task requires modifying more than three files, or writing more than 150 lines of code, stop immediately. Split the work into smaller, independent technical units.
2. **Sequential Dependecy Chain:** Build from the data layer outward. When implementing a new feature, follow this exact sequence across separate step iterations:
   * Step A: Database schemas and migrations.
   * Step B: Core business logic, encryption utilities, and unit tests.
   * Step C: `arq` background tasks (`email_analyzer/analysis_tasks.py`), polling engines, and Redis job-state.
   * Step D: FastAPI endpoints and API routing.
   * Step E: Frontend state management, hooks, and UI components.

---

## 3. Handling Requirements and Ambiguities

1. **Stop on Ambiguity:** If a required field, API contract, or error-handling scenario is missing from the provided prompt or specifications, **do not guess**. Halt execution and explicitly state the missing requirement to the user.
2. **Flag Out-of-Scope Requests:** If a user instruction asks you to build a feature explicitly designated as non-applicable (e.g., automated email sending, image attachment parsing, or local browser LLM inference), reject the task immediately and cite the `project-overview.md` Out-of-Scope section.

---

## 4. Protected Files & Code Modification Bounds

1. **Lock Generated UI Components:** Do not modify automatically generated UI library files, core icon packs, or base tailwind configurations unless explicitly ordered to do so. 
2. **Lock Cryptographic Modules:** Never modify the encryption/decryption routines located in `backend/crypto/` without an explicit security audit instruction.
3. **Preserve Third-Party Contracts:** Do not edit vendor-specific API wrappers or client definitions (e.g., Google OAuth scopes or enterprise IMAP parsing abstractions) without validating compatibility against the `architecture.md` storage model.

---

## 5. Documentation and Synchronization

1. **Sync Specs Instantly:** When you change an API path, database schema, or environment variable schema, you must update the corresponding structural description inside `architecture.md` or `code-standards.md` in the exact same tool call execution.
2. **Document via Self-Describing Code:** Write explicit Type Hints (Python) and strict types (TypeScript). Do not write comments that state what the code does; use comments *only* to justify an un-obvious architectural necessity.
3. **Log with Masking Strictness:** Ensure all new logging instructions use structural masking. Never log variables containing customer email bodies, attachment strings, tokens, or plaintext credentials.

---

## 6. Pre-Flight Verification Checklist

You must run and pass this verification checklist before announcing that a unit of work is complete. Do not ask the user to verify code until you confirm these checks pass:

- [ ] **Type Check:** `mypy .` (Backend) and `tsc --noEmit` (Frontend) execute with zero errors.
- [ ] **Linter Check:** `ruff check .` (or Black) and `eslint` return zero style or syntax warnings.
- [ ] **Log Masking:** The code guarantees that raw email or attachment strings are never written to standard logs, tokens, or credentials (note: at-rest encryption of email bodies is not yet implemented — see `architecture.md` §4/§6 — do not claim it is unless you implemented it).
- [ ] **No Bloat:** Every line of code added directly serves the immediate unit task; all temporary debugging blocks have been deleted.
- [ ] **Task Isolation:** No synchronous internet connection calls (IMAP, Gmail, LLMs) or heavy file parsing loops occur inside FastAPI endpoints. They are successfully routed through `arq` tasks.