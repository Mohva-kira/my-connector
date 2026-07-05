# Architecture Context
# Architecture Overview

This document defines the technical architecture of the AI Email Assistant that transforms Gmail and Outlook emails into actionable intelligence and daily briefings.

---

## 1. Stack

| Layer | Technology | Role |
|------|------------|------|
| Frontend | React (Web App) | User dashboard, briefing view, notifications, settings |
| Backend API | Node.js (NestJS or Express) | Core API, orchestration of email processing and user actions |
| Background Jobs | BullMQ + Redis | Batch processing (2x/day), email sync, AI processing queues |
| Database | PostgreSQL | Primary storage for users, emails, insights, actions |
| Cache / Queue | Redis | Job queues, rate limiting, temporary processing state |
| Email Integration | Gmail API + Microsoft Graph API | Email ingestion and sync |
| AI Layer | OpenAI GPT-4.1 / GPT-4o-mini | Email classification, extraction of actions, urgency detection |
| Search | PostgreSQL Full-Text Search | Email and metadata search (MVP) |
| Authentication | OAuth2 (Google + Microsoft) + JWT | Secure login and session management |
| Deployment | Docker + Cloud VPS (or AWS/GCP) | Containerized deployment and scaling |

---

## 2. System Boundaries (Codebase Responsibility)

### `/frontend`
Responsible for:
- User authentication UI
- Dashboard (daily briefing)
- Notification display
- Settings (email accounts, preferences)
- Action management (view tasks, archive suggestions)

---

### `/backend`
Responsible for:
- API gateway
- Authentication validation
- User management
- Email sync orchestration
- Business logic (scoring, categorization)
- Action execution (archive, tag, task creation)

---

### `/workers`
Responsible for:
- Batch email processing (2x/day)
- Email ingestion pipelines
- AI classification calls
- Notification generation
- Task extraction and updates

---

### `/integrations`
Responsible for:
- Gmail API integration
- Microsoft Graph integration
- OAuth token refresh and management
- Email normalization layer

---

### `/ai-engine`
Responsible for:
- Prompt orchestration
- Email classification (importance, urgency, action)
- Structured extraction (JSON outputs)
- Confidence scoring

---

### `/shared`
Responsible for:
- Types (Email, User, Insight, Action)
- Utility functions
- Scoring rules engine

---

## 3. Storage Model

### PostgreSQL (Primary Database)

Stores structured and relational data:

#### Users
- id
- email
- oauth_provider_tokens
- settings

#### Emails (raw)
- id
- user_id
- provider (gmail/outlook)
- external_email_id
- from
- to
- subject
- body
- received_at
- encrypted_body

#### Email Insights
- id
- email_id
- importance_score (0–100)
- type (info | action | urgent)
- category (client | personal | admin | spam)
- deadline (nullable)
- ai_confidence

#### Actions
- id
- email_id
- type (archive | tag | task)
- status (pending | completed)
- created_at

---

### Redis (Cache + Queue)

Used for:
- Job queues (email sync, AI processing)
- Temporary processing states
- Rate limiting (API calls to Gmail/OpenAI)
- Session caching (optional)

---

### File Storage (Optional / Future)

Not used in MVP, but reserved for:
- Attachments (if enabled later)
- Large email bodies archival snapshots

---

## 4. Authentication & Access Model

### Authentication Flow

1. User signs up via email/password or OAuth (Google/Microsoft optional)
2. User connects Gmail/Outlook via OAuth2
3. Backend stores encrypted OAuth tokens
4. JWT issued for session authentication

---

### Access Control Rules

- All data is **user-scoped**
- Emails belong strictly to one user (`user_id`)
- No cross-user data access is allowed
- AI processing is executed per-user context only

---

### Token Handling

- OAuth tokens stored encrypted in DB
- Refresh tokens used for long-term sync
- Access tokens are never exposed to frontend

---

## 5. AI & Background Task Model

### Batch Processing (2x/day)

Triggered via cron + worker system:

1. Fetch new emails (Gmail + Outlook)
2. Normalize email format
3. Apply rules engine scoring
4. Call AI model for classification
5. Store insights in database
6. Generate daily briefing

---

### Urgent Processing (Near real-time batch trigger)

- Triggered after sync job
- Filters emails with high risk indicators
- If score ≥ threshold → notification generated immediately

---

### AI Processing Pipeline

Input:
- email subject
- email body
- metadata (sender, recipient, thread)

Output:
```json
{
  "type": "info | action | urgent",
  "importance_score": 0-100,
  "action_required": true/false,
  "deadline": "date|null",
  "category": "client | personal | admin",
  "confidence": 0-1
}
