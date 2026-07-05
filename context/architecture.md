# my-connector

# Technical Architecture

## Architecture Context & Overview

This document defines the technical architecture of **my-connector**, an AI-powered email intelligence platform that transforms Gmail and Outlook email streams into structured project workspaces, actionable insights, and intelligent daily briefings.

The architecture is designed around five core principles:

- Scalability
- Responsiveness
- Asynchronous processing
- AI-driven contextualization
- Strict user-centric email filtering

Long-running AI operations are completely decoupled from the user interface to guarantee a fluid experience even during intensive processing.

---

# Technology Stack

| Layer               | Technology                      | Responsibility                                                                   |
| ------------------- | ------------------------------- | -------------------------------------------------------------------------------- |
| **Frontend**        | React                           | Temporal Briefing dashboard, Project Hub, Fast-Track refresh, Discovery Alerts   |
| **Backend API**     | FastAPI (Python)                | REST API, authentication, project orchestration, routing, OAuth token management |
| **Background Jobs** | Redis + arq                     | Historical clustering, scheduled synchronizations, asynchronous AI execution     |
| **Database**        | PostgreSQL                      | Users, projects, emails, summaries, AI insights                                  |
| **Cache & Queue**   | Redis                           | Queue management, job states, API rate limiting, session caching                 |
| **Email Providers** | Gmail API + Microsoft Graph API | Secure email ingestion                                                           |
| **AI Engine**       | OpenAI GPT-4o / GPT-4o-mini     | Semantic clustering, summarization, sentiment analysis, suggestion generation    |
| **Search Engine**   | PostgreSQL Full Text Search     | Project-scoped search (MVP)                                                      |
| **Authentication**  | OAuth2 + JWT                    | Secure authentication and authorization                                          |

---

# High-Level Architecture

```
                Gmail API
                    │
                    │
            Microsoft Graph API
                    │
                    ▼
        ┌──────────────────────────┐
        │ Normalized Email Layer   │
        └──────────────────────────┘
                    │
                    ▼
          Recipient Matrix Filter
        (Only To / CC are accepted)
                    │
                    ▼
          FastAPI API Gateway
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
 PostgreSQL               Redis + arq
         │                     │
         │                     ▼
         │              AI Worker Pool
         │                     │
         └──────────────┬──────┘
                        ▼
                OpenAI GPT-4o
                        │
                        ▼
              Project Summaries
                        │
                        ▼
               React Frontend
```

---

# Codebase Responsibilities

## `/frontend`

The frontend provides the entire user experience.

### Temporal Briefing

The main dashboard presents:

- Morning priorities
- Evening recap
- Upcoming deadlines
- Outstanding actions

Tasks from every project are aggregated and sorted by urgency.

---

### Project Hub

Displays validated projects as a responsive grid.

Each project card includes:

- Project summary
- Health indicator
- AI sentiment
- Outstanding actions
- Last update timestamp

---

### Fast-Track Refresh

Users can refresh a single project without waiting for the next scheduled synchronization.

The frontend issues a scoped refresh request and monitors execution progress.

---

### Discovery Alerts

Non-intrusive toast notifications appear whenever AI discovers a potential new project.

---

# `/backend`

The backend orchestrates every business process.

## API Gateway

Responsible for:

- JWT validation
- OAuth token verification
- Secure token rotation
- Request authorization

---

## Project Controller

Handles project lifecycle operations:

- Validate discovered projects
- Rename projects
- Merge duplicate projects
- Archive projects

---

## Fast-Track Controller

```
POST /api/projects/{id}/refresh
```

Instead of executing heavy processing immediately, the endpoint:

- creates a background job
- returns HTTP **202 Accepted**
- provides a unique `job_id`

---

# `/workers`

Workers execute long-running asynchronous operations.

## Historical Clustering Worker

Executed only during onboarding.

Responsibilities:

- Analyze historical emails
- Detect conversation clusters
- Build initial projects
- Generate first summaries

---

## Batch Synchronization Worker

Runs twice daily.

Responsibilities:

- Retrieve incremental emails
- Update project states
- Refresh summaries
- Generate new suggested actions

---

## Scoped Project Worker

Dedicated high-priority worker for Fast-Track refreshes.

Only processes one project at a time.

---

# `/integrations`

## Normalized Email Layer

Provides a unified abstraction over:

- Gmail API
- Microsoft Graph API

Regardless of the provider, emails are converted into a common internal schema.

---

## Recipient Matrix Filter

Before an email enters the platform it is validated.

Accepted:

- To
- CC

Ignored automatically:

- BCC
- Mailing lists
- Marketing emails
- Newsletters
- Broadcast communications

---

# `/ai-engine`

The AI Engine transforms email conversations into structured business knowledge.

## Historical Clustering

Groups conversations using:

- Semantic similarity
- Subject analysis
- Recipient patterns
- Thread history

---

## Summary Generator

Produces structured summaries including:

- Executive overview
- Current status
- Project sentiment
- Pending actions
- Suggested follow-ups

The engine returns structured JSON instead of plain text whenever possible.

---

# PostgreSQL Data Model

```
Users
   │
   ├───────────────┐
   ▼               ▼
Projects         Emails
   │               │
   ▼               ▼
ProjectSummaries EmailInsights
   │
   ▼
SuggestedActions
```

---

# Database Schema

## users

| Column       | Type            |
| ------------ | --------------- |
| id           | UUID            |
| email        | VARCHAR         |
| oauth_tokens | Encrypted JSONB |
| created_at   | TIMESTAMP       |

---

## projects

| Column       | Type              |
| ------------ | ----------------- |
| id           | UUID              |
| user_id      | FK                |
| name         | VARCHAR           |
| status       | active / archived |
| rules_matrix | JSONB             |

The `rules_matrix` stores:

- keywords
- sender addresses
- domains
- matching heuristics

---

## emails

| Column           | Type           |
| ---------------- | -------------- |
| id               | UUID           |
| project_id       | FK (nullable)  |
| user_id          | FK             |
| recipient_status | direct_to / cc |
| subject          | TEXT           |
| body_encrypted   | TEXT           |
| received_at      | TIMESTAMP      |

---

## project_summaries

| Column                         | Type                                         |
| ------------------------------ | -------------------------------------------- |
| id                             | UUID                                         |
| project_id                     | FK                                           |
| updated_at                     | TIMESTAMP                                    |
| content                        | TEXT                                         |
| sentiment                      | on_track / under_tension / awaiting_feedback |
| last_processed_email_timestamp | TIMESTAMPTZ                                  |

This timestamp enables efficient incremental synchronization.

---

## suggested_actions

| Column      | Type                            |
| ----------- | ------------------------------- |
| id          | UUID                            |
| project_id  | FK                              |
| description | TEXT                            |
| deadline    | TIMESTAMP (nullable)            |
| status      | pending / completed / dismissed |

---

# AI Processing Pipeline

## Process 1 — Historical Project Discovery

### Step 1

Retrieve historical email metadata.

Typical synchronization window:

- 30 days
- 60 days
- 90 days

---

### Step 2

Discard every email where the authenticated user is neither:

- To
- CC

---

### Step 3

Extract:

- Subjects
- Thread IDs
- Senders
- Recipients

---

### Step 4

Send normalized data to the AI Engine.

---

### Step 5

AI returns structured project proposals.

Example:

```json
{
  "proposed_projects": [
    {
      "name": "BUMDA Core Module",
      "thread_ids": ["t1", "t2"]
    },
    {
      "name": "RecoPro SaaS Setup",
      "thread_ids": ["t3"]
    }
  ]
}
```

---

### Step 6

The user validates, edits, or merges projects before they are persisted into PostgreSQL.

---

# Process 2 — Fast-Track Project Refresh

## Trigger

The user clicks:

**Refresh Summary**

on a project card.

---

## Execution

A dedicated arq queue (Redis-backed) receives the request.

```
project-fasttrack-queue
```

The request bypasses all scheduled background jobs.

---

## Workflow

### 1.

Read the project's:

```
last_processed_email_timestamp
```

---

### 2.

Retrieve only incremental emails received after that timestamp.

---

### 3.

Append newly received emails into PostgreSQL.

---

### 4.

Send only the email delta to the AI Engine.

---

### 5.

Regenerate:

- Summary
- Sentiment
- Suggested actions

---

### 6.

Update the frontend through the polling engine.

---

# Asynchronous Job Architecture

Long-running AI operations never execute inside HTTP request lifecycles.

Instead, every expensive operation becomes a background job.

```
React Frontend
       │
       │ POST /api/projects/{id}/refresh
       ▼
┌─────────────────────┐
│  FastAPI API Gateway│
└─────────────────────┘
       │
       │
 HTTP 202 Accepted
 { job_id }
       │
       ▼
┌─────────────────────┐
│    Redis + arq      │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Background Workers  │
└─────────────────────┘
       │
       ▼
   OpenAI GPT-4o
       │
       ▼
 PostgreSQL Update
       │
       ▼
 Polling Response
       │
       ▼
 React Component Update
```

---

# Job Lifecycle

## Step 1 — Immediate Response

Every expensive endpoint immediately returns:

```
HTTP 202 Accepted
```

along with

- job_id
- current status
- metadata

No HTTP request waits for AI execution.

---

## Step 2 — Queue Processing

arq stores and schedules execution.

Redis maintains:

- pending
- processing
- completed
- failed

states.

---

## Step 3 — Client Polling

The frontend periodically requests:

```
GET /api/jobs/:id
```

until the job completes.

Possible states:

```
pending
```

↓

```
processing
```

↓

```
completed
```

or

```
failed
```

---

## Step 4 — Local UI Refresh

Only the affected project card is updated.

The rest of the interface remains untouched, providing a fast and responsive user experience.

---

# Worker Isolation

Dedicated worker pools maintain separate database connections from the public REST API.

Benefits include:

- No API blocking during AI processing
- Stable CRUD performance
- Better database resource management
- Protection against reverse proxy timeouts (504 Gateway Timeout)
- Horizontal scalability of AI workloads

This architecture ensures that expensive operations such as historical clustering, semantic analysis, and project summarization never impact the responsiveness of the user-facing application.
