# Beat Book Builder — Changelog

## Session: May 13, 2026 (uncommitted)

### Ingest overhaul

- **Replaced gpt-4o-mini with Claude Haiku** for all normalization — OpenAI is no longer used at any ingest stage (only for `text-embedding-3-small` embeddings)
- **Removed the PDF fast path** — all PDFs now go through the LLM for proper type-specific metadata extraction
- **Added OCR for scanned PDFs** — PyMuPDF renders pages to PNG at 150 DPI; Haiku vision transcribes in batches of 4 pages, capped at 100 pages per document
- **Filename-based type hints** — keywords in the filename (e.g. `suspension`, `minutes`, `newsletter`) are passed to the LLM as a hint before it classifies the document
- **Fixed OCR sentinel bug** — `__SCANNED_PDF__` was being swallowed by the extractor's exception wrapper; changed to a substring check

### Content type system

- New `content_type` field: `article`, `document`, `dataset`, `report`, `transcript`, `press_release`, `post`, `other`
- New `organization` field (institution) separate from `author` (individual)
- Type-specific `metadata` dict — the LLM classifies the document type first, then extracts the matching schema:
  - `document` (legal order/action) → `docket_number`, `action_type`, `subject_name`, `jurisdiction`
  - `document` (license) → `license_number`, `license_status`, `state`, `expiry_date`
  - `dataset` → `source_organization`, `date_range`, `fields`, `row_count`
  - `report` → `issuing_organization`, `report_number`
  - `transcript` → `body_name`, `meeting_format`, `key_attendees`
  - `press_release` → `contact_name`, `contact_email`
  - `post` → `platform`, `handle`
- Preview UI updated: two-row field layout (title + type / org + date + author), metadata chips in story footer, editable type dropdown per story

### Rate limiting fixes

- **Global semaphore** (`threading.Semaphore(3)`) in `claude_client.py` caps all concurrent batch Anthropic calls system-wide — prevents the 4 files × 4 chunks = 16 simultaneous calls storm that was triggering 429s
- **Exponential backoff with jitter** replaces the fixed 60-second sleep; uses the `Retry-After` header when Anthropic provides it (15s → 30s → 60s)
- **`NORMALIZE_MAX_TOKENS`** reduced from 32768 → 4096 — marker JSON is compact; the old value burned ~8× excess TPM budget
- **`_label_cluster` now has retry logic** — previously a single 429 during cluster labeling would silently crash the whole pipeline
- **Pipeline thread pool** capped at `MAX_ANTHROPIC_CONCURRENT` (3) instead of 8, so threads don't idle-spin waiting on the semaphore
- **`agent.py`** updated to use exponential backoff instead of fixed waits

### Interview stage removed

- Removed the Q&A interview between pipeline completion and beat book generation — the agent now goes straight from exploring topics to writing
- Removed: `interview_user` tool, `InterviewCallback`, `on_interview` callback, `interview_log` from agent, app server, and research agent
- Removed interview screen from frontend JS and HTML
- Research agent system prompt no longer references reporter answers

### Performance improvements

- **`read_stories_in_topic` bulk tool** added to the agent — returns all stories in a topic with 2000-char excerpts in one API call instead of one call per story
- **Cluster labeling switched from Sonnet to Haiku** (~10× faster per label call)
- **Research agent switched from Opus 4.7 to Sonnet 4.6** (~3–5× faster per turn)
- **`_INGEST_CONCURRENCY = 4`** — files are processed in parallel at the HTTP level
- **Reload guard** — `beforeunload` event prompts the user before navigating away during active processing

### Housekeeping

- **`Makefile` added** — `make install`, `make dev`, `make run`, `make lint`, `make clean`
- `requirements.txt` updated with all new dependencies (PyMuPDF, python-docx, python-pptx, openpyxl, xlrd, beautifulsoup4, striprtf, ebooklib)
- All OpenAI references removed from the normalization path

---

## Prior commits (from git log)

| Commit | Summary |
|--------|---------|
| `5f5d739` | Surface every Anthropic rate-limit retry instead of waiting silently |
| `ad9e958` | Make ingest schema-agnostic, no-drop, and rate-limit tolerant |
| `c4d65ec` | Switch chat-model slots from Qwen on Ollama to Claude Sonnet 4.6 |
| `ed8e35b` | Guard story-array parsing against non-dict entries |
| `6155dd0` | Add ENABLE_THINKING toggle and surface ingest errors faster |
| `661aa56` | Guard JSON title field against non-string values |
| `9611031` | Raise per-file cap to 25 MB and add example files |
| `aab7f65` | Tie the read-count gate to the reporter's selected topics |
| `b831447` | Gate generate_beat_book on per-topic read-count targets |
| `7b98ccc` | Force the beat-book agent to read more of the corpus before generating |
| `dcf698a` | Push the beat-book agent toward prose, not bullet outlines |
| `3a6c4f8` | Drop serif; use Instrument Sans for body everywhere |
| `306b937` | Cap story body at next story's start to prevent bleed-through |
| `1f576bb` | Sticky preview toolbar with confidence-filter chips |
| `e9fb62e` | Chunked normalization: drop the 120k-character cap |
| `8f1c496` | Initial commit: Beat Book Builder |
