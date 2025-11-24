# Driver-sentiment-engine

The Driver Sentiment Engine processes employee trip feedback in real time, calculates sentiment scores, updates performance metrics, and raises alerts for risky drivers — all with efficient, scalable, and secure design principles.

This project is built to demonstrate system design, OOPS, queues, asynchronous processing, thread safety, real-time dashboards, JWT authentication, and logging/monitoring.

🌟 Features
🔐 1. Authentication

Secure JWT-based login for User & Admin.

Separate tokens: user_token and admin_token.

Auto-invalidation of old tokens using INSTANCE_ID — ensures logout on server restart.

Back-button protection (no cached pages after logout).

⚙️ 2. Feedback Processing Engine

Processes feedback asynchronously using an in-memory queue.

Uses a thread-safe, lock-based engine to avoid race conditions.

Deduplicates feedback using UID hashing.

Computes rolling average per entity (driver/trip/app/marshal).

Raises alerts when average score < threshold (2.5).

Supports multiple entities without code changes (fully schema-driven).

📊 3. Real-Time Admin Dashboard

Auto-refresh every 2 seconds.

Displays:

Bar graph of sentiment scores

Recent feedback list

Recent alert list

Graph saved using Matplotlib (static/graph.png).

“Run Agg” button allows manual batch aggregation.

📝 4. Dynamic Feedback Form (Configurable UI)

Admin can:

Add fields

Remove fields

Change field type (text / textarea / select)

Add dropdown options

All changes are saved in form_schema.json — no code change required.

🛠 5. Monitoring & Logs

Uses RotatingFileHandler to store logs in:

logs/app.log


Logs include:

Logins

Failed login attempts

Feedback received

Alerts raised

Queue processing events

Aggregation runs

System startup

💾 6. Lightweight Storage

Used for demo requirements.

Files:

feedback_history.json — raw feedback

data.json — aggregated scores

form_schema.json — UI configuration

🏗 System Architecture
User/Admin UI → Flask App → Queue → SentimentEngine → Data + Alerts → Dashboard


Key components:

Flask: Routing, UI, API

JWT: Authentication

Queue: Real-time stream processing

Threading + Locks: Concurrency control

Matplotlib: Graph generation

JSON Storage: Lightweight data persistence

Logging: Monitoring

📦 Tech Stack
Layer	Technology

Backend	Python, Flask

Authentication	PyJWT

UI	HTML, Bootstrap 5, JavaScript

Graphs	Matplotlib

Storage	JSON-based filesystem storage

Concurrency	Python Queue + Threading

Monitoring	Logging with RotatingFileHandler
