# Project Overview: Intelligent Project Management & Email Assistant

An AI-powered project management assistant that automatically ingests, organizes, and analyzes communications across professional communication channels to eliminate manual tracking for Project Managers. By connecting to Gmail and enterprise IMAP servers, the application securely captures historical and incoming emails, extracts distinct client and product entities, resolves duplicates through an intelligent aliasing system, and builds a tiered, hybrid knowledge base (local 30-day device cache coupled with an encrypted PostgreSQL cloud archive). The system proactively crafts context-aware draft responses, surfaces deep analytical project insights, and parses complex document attachments (PDFs and Excel spreadsheets) without impacting device performance or exposing raw customer data.

## Project Goals

1. **Automate Zero-Touch Project Creation:** Eliminate manual project onboarding by discovering and clustering technical and business initiatives from raw email streams within 5 minutes of setup.
2. **Implement Privacy-by-Design Architecture:** Protect corporate data through a strict hybrid storage model where full historical archives are encrypted in the cloud, while immediate 30-day context remains isolated on the user's physical device.
3. **Minimize Context-Switching Overhead:** Empower Project Managers to track deadlines, monitor scope shifts, and generate production-ready email replies directly from a unified assistant interface.
4. **Build a Fault-Tolerant Ingestion Pipeline:** Create a decoupled asynchronous ingestion architecture capable of handling 10,000+ initial project emails and heavy attachments without service degradation, timeouts, or API rate limit blocks.

## Step-by-Step Core User Flow

1. **Authentication & Setup:** The user logs into the ReactJS/React Native application and authenticates their email accounts using Google OAuth or explicit IMAP credentials. The user specifies a historical synchronization window (e.g., past 30 days).
2. **Asynchronous Discover & Delta Scan:** A lightweight Python background cron job logs the session, saves the sync timestamp, and pushes individual message identifiers into a Celery/Redis queue.
3. **Entity Extraction & Aliasing:** Asynchronous workers pull emails from the queue, extract text blocks and attachments, and evaluate them via the Cloud LLM. The system searches for existing project names or known synonyms in the database aliases (e.g., matching "RP" or "Reco Pro" to the canonical project "RecoPro").
4. **Categorization & Ingestion:**
   - **If a match is found:** The email is bound to the existing project (Strict 1:1 relationship) and its contents update the knowledge base.
   - **If no match is found:** The system isolates the extracted keywords (Client Name, Product, Company) and queues a pre-composed project creation proposal on the dashboard.
5. **Client Cache Syncing:** The desktop or mobile device downloads the decrypted delta payload for the last 30 days, builds an optimized local vector index (using Orama/Voy/SQLite), and purges any expired local data exceeding the 30-day rolling window.
6. **Interaction & Value Delivery:** The user opens the app, instantly views their structured projects dashboard, reviews auto-generated summaries of recent activities, and clicks an issue to reveal a tailored, pre-written draft response ready to be reviewed and sent.

## Features Breakdown by Category

### 1. Ingestion & Synchronization Pipeline

- **Multi-Provider Authentication:** Secure OAuth 2.0 integration for Gmail accounts and standard credential setups for secure corporate IMAP servers.
- **Timestamp-Based Delta Polling:** Hourly backend cron scripts tracking `last_synced_at` markers to pull only newly arrived messages, preserving computational bandwidth.
- **Distributed Queue Architecture:** Celery workers backed by a Redis message broker to manage heavy, long-running extraction tasks safely out of the web request cycle.

### 2. Entity Resolution & AI Categorization

- **Strict 1:1 Email Routing:** Deterministic categorization algorithms ensuring every individual message is mapped to exactly one primary project context based on dominant thematic weights.
- **Intelligent Alias Mapping:** Relational database registries containing known project shorthand, acronyms, and client specific names to prevent duplicate project entities.
- **Draft Generation Engine:** Context-aware LLM processing to generate ready-to-send email drafts responding to client inquiries, scope adjustments, or blockers.

### 3. Tiered Hybrid Storage & Security

- **Encrypted Cloud Vault:** PostgreSQL schema employing envelope encryption to safeguard historical raw email body copy and processed attachment text at rest.
- **Rolling Client-Side Vector Cache:** Local embedded vector database instances on web/mobile clients restricted strictly to a 30-day sliding window of hot operational data.
- **Automated Device Housekeeping:** Self-executing client scripts running at application startup to strip out vector entries and text metrics older than 30 days.

### 4. Document Processing (Parser)

- **Unstructured Text Extraction:** Specialized Python workers processing incoming PDF documents and Excel files to index structural project tables and requirements text.

## In-Scope (What We Are Building)

- Web frontend built with ReactJS and Mobile apps built with React Native.
- Python backend utilizing Celery for asynchronous background task execution.
- Direct ingestion integration for both standard Gmail API and enterprise IMAP protocols.
- Support for indexing and vectorizing raw email text alongside textual content found inside PDF and Excel attachments.
- Automated LLM-driven generation of email response drafts based on project context.
- A central encrypted PostgreSQL database for full historical storage and a local 30-day embedded vector index on the client device.

## Out-of-Scope (What We Are NOT Building)

- **Automatic Email Dispatching:** The system will _never_ send an email automatically; it only writes drafts that require human review, modification, and manual sending from the interface.
- **Real-time Webhook Ingestion:** No active Push notification webhooks (e.g., Gmail Push Webhooks); all updates rely strictly on the stable hourly polling infrastructure.
- **Multi-Project Cross-Linking:** Emails will not be linked to multiple projects; any message covering multiple projects will be anchored strictly to its singular dominant project entity.
- **Non-Standard Attachments:** No processing, rendering, or indexing of image formats (JPEG/PNG), CAD files, ZIP archives, or rich video/audio attachments.
- **Client-Side Heavy LLM Execution:** No execution of local LLM models (e.g., Llama/Gemma) on the user's phone or browser; all inference is routed through secured Cloud LLM APIs.

## Success Criteria (Definition of Done)

- **Performance Benchmark:** Initial onboarding sync of up to 1,000 emails completes in under 3 minutes without dropped tasks or worker restarts.
- **Data Leak Defenses:** Automated end-to-end testing confirms that a query executed by User A never returns or exposes vector contexts belonging to User B.
- **Storage Footprint Cap:** The local device database storage utilization remains constant and capped after 30 days of continuous operation, verifying the sliding-window purge script works correctly.
- **Extraction Accuracy:** Out-of-the-box alias matching successfully identifies and groups common project variations (e.g., merging "RP", "Reco Pro", and "RecoPro") with an accuracy rate of 95% or higher on test email batches.
- **Functional Integration:** A project manager can connect a live test email account, see auto-generated project folders within minutes, select an email, and copy a valid, contextually accurate AI draft reply to their clipboard.
