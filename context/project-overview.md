# my-connector

# Project Overview

## Overview

This application is an AI-powered email intelligence assistant that connects to a user’s Gmail and Outlook accounts, analyzes incoming emails, and transforms them into a structured, actionable daily briefing. Instead of requiring users to read and manually process their inbox, the system extracts key information such as important messages, required actions, and deadlines, and presents them as clear notifications and summaries. The goal is to help users understand exactly what they need to do each day without directly interacting with their email inbox.

---

## Goals

1. Reduce the need for users to manually read and process emails.
2. Automatically identify important emails using a hybrid rules + AI scoring system.
3. Extract actionable tasks, deadlines, and priorities from email content.
4. Provide a structured daily briefing twice per day (morning and evening).
5. Deliver urgent notifications for high-priority emails in near real-time (batch-triggered).
6. Support Gmail and Outlook integration from the first version.
7. Maintain a secure and structured storage system for email data and insights.

---

## Core User Flow

1. User signs up and creates an account.
2. User connects Gmail and/or Outlook accounts via OAuth authentication.
3. The system performs an initial synchronization of emails.
4. Emails are stored securely and normalized into a unified internal format.
5. A batch processing system runs twice per day:
   - Applies rule-based scoring (VIP sender, keywords, recipient context).
   - Applies AI-based classification (importance, action, deadline, category).
6. Each email is transformed into structured insights:
   - Action required or not
   - Importance score (0–100)
   - Deadline detection
   - Categorization
7. The system generates a “Daily Briefing” twice per day:
   - Morning briefing: priorities for the day
   - Evening briefing: remaining actions and updates
8. If an email exceeds a critical threshold, an urgent notification is triggered immediately after processing.
9. Users interact with a dashboard showing:
   - Daily briefing
   - Important notifications
   - Suggested actions

---

## Features

### 1. Email Integration
- Gmail OAuth integration
- Outlook (Microsoft Graph) integration
- Secure token-based authentication
- Initial and incremental email synchronization

### 2. Email Processing Engine
- Email normalization (Gmail + Outlook unified format)
- Rule-based scoring system (VIP contacts, recipient status, keywords)
- AI-based classification (action, urgency, deadline, category)
- Hybrid scoring model combining rules + AI output

### 3. Intelligence & Insights Layer
- Importance scoring (0–100)
- Action extraction (task detection)
- Deadline detection from email content
- Email categorization (client, personal, admin, etc.)
- Structured metadata generation per email

### 4. Action Engine (Limited Automation)
- Automatic email archiving for low-priority messages
- Automatic tagging and categorization
- Task creation from action-required emails
- No automatic sending of emails

### 5. Notifications System
- Urgent notifications for high-priority emails (score ≥ threshold)
- Batch-based processing (not full real-time streaming)
- Optional alerts between processing cycles

### 6. Daily Briefing System
- Twice daily generation (morning and evening)
- Summarized view of:
  - Actions to complete
  - Important emails
  - Urgent items
- Human-readable explanation of why items are important

### 7. Data & Storage
- Secure storage of raw emails (encrypted)
- Structured storage of processed insights
- Full-text search capability for email content
- Audit trail of AI decisions and actions

---

## In Scope

- Gmail and Outlook integration (OAuth authentication)
- Email ingestion and synchronization system
- Hybrid rules + AI-based email classification engine
- Email importance scoring system
- Deadline and action extraction from email content
- Twice-daily batch processing system
- Daily briefing generation (morning and evening)
- Urgent notification system for high-priority emails
- Basic dashboard showing briefings and actions
- Email archiving, tagging, and task creation automation
- Secure storage of email data and metadata
- Full-text search over emails (PostgreSQL-based MVP)

---

## Out of Scope

- Full inbox replacement UI (no full email client experience)
- AI-generated email replies or automatic sending of emails
- Real-time streaming email processing for all messages
- Team collaboration or shared inbox features
- Advanced CRM or sales pipeline management features
- Elasticsearch or large-scale distributed search systems (initial version)
- Mobile-native applications (initial MVP focuses on web)
- Multi-provider integrations beyond Gmail and Outlook
- Fully autonomous AI agents executing high-risk actions (sending emails, deleting emails permanently)
- Semantic/vector search (embeddings) in the initial version

---

## Success Criteria

The project is considered successful when:

1. A user can connect Gmail or Outlook and successfully import emails.
2. The system correctly identifies important emails with a hybrid rules + AI model.
3. Each email is transformed into a structured insight (action, urgency, deadline, category).
4. Users receive two clear and useful daily briefings (morning and evening).
5. Urgent emails trigger timely notifications without overwhelming the user.
6. Users can clearly understand why an email was marked important.
7. The system reduces the need for users to manually read their inbox to understand priorities.
8. The application reliably processes emails in batch mode without data loss or inconsistency.
