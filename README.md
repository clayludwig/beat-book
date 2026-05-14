# Beat Book Builder

A web application that turns a collection of news articles into an interactive **beat book** — a practical reporting guide for journalists covering a specific topic area. Upload source articles in any common format (Word, PDF, HTML, markdown, plain text, JSON) or paste URLs, and the system extracts the stories, lets you review them, then automatically discovers topics via embedding and clustering before walking the reporter through an AI-guided interview to produce a tailored beat book.

Originally built around [Chicago Public Media](https://chicago.suntimes.com/) story data; works with any news corpus regardless of source format.

---

## Setup & Running

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) (used only for embeddings — `text-embedding-3-small`)
- An [Anthropic API key](https://console.anthropic.com/) (used for `claude-sonnet-4-6` — story normalization, cluster labeling, the interview agent — and `claude-opus-4-7` for the research agent)

No local daemons required — everything runs through hosted APIs, so the project is portable across machines.

### Install

```bash
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root (see `.env.example`):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional — extended thinking on Sonnet 4.6 (slower, higher quality).
# Default: off. Ignored by the ingest normalization step, which forces
# tool_choice and is incompatible with thinking.
# ENABLE_THINKING=true
```

### Run

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

or with uv:

```bash
uv run python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Table of Contents

- [Setup & Running](#setup--running)
- [How It Works](#how-it-works)
- [Architecture Overview](#architecture-overview)
- [Ingest: Files, URLs, and the Preview Screen](#ingest-files-urls-and-the-preview-screen)
  - [Supported Inputs](#supported-inputs)
  - [Stage 1: Extract Text](#stage-1-extract-text)
  - [Stage 2: LLM Normalization](#stage-2-llm-normalization)
- [Pipeline: Step by Step](#pipeline-step-by-step)
  - [1. Embedding](#1-embedding)
  - [2. Dimensionality Reduction](#2-dimensionality-reduction)
  - [3. Clustering](#3-clustering)
  - [4. Topic Labeling](#4-topic-labeling)
- [Agent: The Interview & Beat Book Generation](#agent-the-interview--beat-book-generation)
  - [Agent Tools](#agent-tools)
  - [Agent Loop](#agent-loop)
- [Frontend](#frontend)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)

---

## How It Works

1. **Add sources** — Upload files (Word, PDF, HTML, markdown, plain text, JSON, RTF) or paste URLs through a drag-and-drop web interface.
2. **Detect stories** — The server extracts text from each source and asks Claude Sonnet 4.6 (`claude-sonnet-4-6`) to identify distinct news stories, splitting multi-story documents and inferring missing metadata. The reporter reviews the detected stories on a preview screen and can edit titles/dates/authors or deselect anything before continuing.
3. **Analyze** — The server runs each confirmed story through an NLP pipeline: embed the text (OpenAI `text-embedding-3-small`), reduce dimensions, cluster into topics at two granularities (broad and specific), and label each cluster with an LLM.
4. **Interview** — A Claude-powered AI agent connects over WebSocket, explores the discovered topics, and asks the reporter 3–5 targeted questions about their beat, audience, and goals.
5. **Generate** — The agent synthesizes everything — topics, article content, and reporter answers — into a polished Markdown beat book with sources, story ideas, context, and reporting tips. A second research agent (Claude Opus 4.7) then enriches the draft with public web research.

---

## Architecture Overview

```
Browser (static HTML/JS/CSS)
    │
    ├── POST /ingest          → files + URLs in
    │       │                   stories out (preview JSON)
    │       │
    │       └── ingest.py     → extract_text(...) → markitdown / stdlib
    │                           normalize(...)    → Claude Sonnet 4.6
    │
    ├── POST /process         → streams SSE progress events
    │       │
    │       └── pipeline.py   → embed (OpenAI text-embedding-3-small)
    │                            → UMAP → HDBSCAN → LLM label (Sonnet 4.6)
    │
    └── WS /ws/{session_id}   → bidirectional agent conversation
            │
            └── agent.py      → Claude tool-use loop
                               (reads topics/stories, interviews user, writes beat book)
```

The app server is **FastAPI** running on **Uvicorn**. Ingestion and pipeline work both run in a thread pool so the async server stays responsive. The agent conversation happens over a WebSocket, allowing real-time back-and-forth between the Claude-backed agent and the reporter's browser.

---

## Ingest: Files, URLs, and the Preview Screen

**File:** `ingest.py`

Ingest is a two-stage pipeline that converts any supported source into the `{title, content, date?, author?, link?}` shape the rest of the system expects. No upload format is privileged — JSON, Word, PDF, plain text, and pasted URLs all go through the same two stages.

### Supported Inputs

| Source | How it's handled |
|--------|------------------|
| `.docx`, `.doc`, `.pdf`, `.html`, `.pptx`, `.xlsx`, `.csv`, `.rtf`, `.epub` | Converted to markdown via [markitdown](https://github.com/microsoft/markitdown). |
| `.md`, `.markdown`, `.txt`, `.log` | Read directly as UTF-8 text. |
| `.json` | Parsed and rendered as readable markdown (any known wrapper unwrapped). No special-case schema. |
| URLs (`http`/`https`) | Fetched server-side with `httpx`. SSRF-protected — private, loopback, link-local, and unresolvable addresses are refused. |

**Per-file size cap:** 25 MB. No limit on number of files or URLs per request.

### Stage 1: Extract Text

**Function:** `extract_text(filename, raw_bytes) -> str`

Dispatches on file extension. Office documents and PDFs are written to a temp file and converted to markdown with markitdown; text formats are decoded directly. Unknown extensions fall back to UTF-8 decoding, with markitdown as a hail-mary.

### Stage 2: LLM Normalization

**Function:** `normalize(text, source_label, anthropic_key) -> list[Story]`

A single Claude **Sonnet 4.6** call with forced tool-use (`tool_choice={"type":"tool","name":"register_stories"}`). The LLM:

1. Decides whether the document contains news content. If not (meeting notes, invoices, raw data dumps), returns an empty story list with a `skip_reason`.
2. Identifies each distinct news story in the document — a single file can produce multiple stories, e.g. a notebook of pasted articles.
3. For each story, extracts:
   - `title` (or invents a 6–10 word description if no headline is present),
   - `date` as `YYYY-MM-DD` or empty,
   - `author` (byline only),
   - `link` (source URL if present),
   - `confidence` (`high` | `medium` | `low`) and one-sentence `reasoning`.
4. Returns **character offsets** into the input text marking where each story's body begins and ends. The server slices the body verbatim from the original extracted text — the LLM never rewrites story content.

The preview UI shows the detected stories grouped by source. Excluded sources are shown with their `skip_reason`. The reporter can edit metadata, deselect individual stories, and then click **Run pipeline** to send the confirmed list to `/process`.

---

## Pipeline: Step by Step

The pipeline lives in `pipeline.py` and is called by the `/process` endpoint in `app.py`. It takes a list of story dicts (already normalized by ingest) and returns a `PipelineResult` containing stories, topics, and lookup structures.

### 1. Embedding

**File:** `pipeline.py` — `_story_to_text()`, `_embed_batch()`, `_load_or_embed()`

Each story is converted to a text representation for embedding:
- The **title**, a **section line** (if found in the first 10 lines of content), and the **first 400 words** of content are concatenated.

These text blocks are sent to the **OpenAI Embeddings API** in batches of 100.

- **Model:** `text-embedding-3-small` — produces 1536-dimensional vectors.
- **Caching:** Embeddings are cached to `.cache/embeddings.pkl` keyed by a hash of the first 10 texts + model name. Switching embedding models automatically invalidates the cache.

**Tech:** OpenAI Python SDK (`openai`), NumPy for vector storage.

> **Why not Anthropic for embeddings?** Anthropic doesn't host an embedding API. OpenAI's `text-embedding-3-small` is small, fast, and cheap at this scale. If you'd rather embed on your own infrastructure, point an `OpenAI` client at a remote Ollama server hosting `mxbai-embed-large` and switch the model name; the rest of the pipeline is unchanged.

### 2. Dimensionality Reduction

**File:** `pipeline.py` — `_reduce()`, `_umap_params()`

The high-dimensional embedding vectors (1536-d) are projected down to a lower-dimensional space to make clustering feasible and to capture local structure.

- **Algorithm:** [UMAP](https://umap-learn.readthedocs.io/) (Uniform Manifold Approximation and Projection)
- **Parameters are adaptive** based on corpus size `n`:
  - `n_components`: `min(15, max(5, n // 40))` — more dimensions for larger corpora
  - `n_neighbors`: `min(30, max(5, int(n ** 0.55)))` — balances local vs. global structure
  - `min_dist`: `0.0` — allows tight clusters
  - `metric`: `cosine` — standard for text embeddings

UMAP preserves local neighborhood relationships from the high-dimensional space, meaning articles that are semantically similar end up near each other in the reduced space.

**Tech:** `umap-learn` library (built on NumPy/SciPy/scikit-learn).

### 3. Clustering

**File:** `pipeline.py` — `_cluster()`, `_assign_outliers()`, `_cluster_sizes()`

Stories are clustered at **two granularities** to produce both broad themes and specific sub-topics:

- **Algorithm:** [HDBSCAN](https://hdbscan.readthedocs.io/) (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
- **Broad clusters:** `min_cluster_size = max(4, n // 25)` — produces fewer, larger groups
- **Specific clusters:** `min_cluster_size = max(2, n // 60)` — produces more, finer-grained groups
- Both use `min_samples=2`, Euclidean distance on the UMAP-reduced space, and the "excess of mass" (`eom`) cluster selection method.

**Outlier reassignment:** HDBSCAN labels some points as noise (`-1`). After clustering, every noise point is assigned to its nearest cluster by Euclidean distance to cluster centroids. This ensures every story belongs to at least one topic.

**Why HDBSCAN over K-Means?** HDBSCAN doesn't require specifying the number of clusters in advance — it discovers them from the data's density structure. This is critical because we don't know how many topics a given news corpus will contain.

**Tech:** `hdbscan` library, NumPy.

### 4. Topic Labeling

**File:** `pipeline.py` — `_label_cluster()`, `_label_all()`

Each cluster gets a human-readable topic label generated by an LLM:

1. For each cluster, the **most representative stories** are selected — the ones closest to the cluster's centroid in the reduced space (up to 8 stories).
2. Their headlines and a 30-word excerpt are formatted into a prompt.
3. The LLM is asked to return a concise 2-5 word topic label describing the shared subject matter (e.g. "High School Basketball", "City Budget Disputes", "Immigration Policy").

- **Model:** `claude-sonnet-4-6` via the Anthropic API (configurable in `claude_client.py`)
- **Prompt engineering:** The prompt explicitly instructs the model to focus on *what* the articles are about, not *where* they're from, to avoid generic geographic labels.

This runs once for broad clusters and once for specific clusters.

**Tech:** Anthropic Python SDK (`anthropic`) — `client.messages.create`.

---

## Agent: The Interview & Beat Book Generation

**File:** `agent.py`

After the pipeline finishes, the reporter's browser opens a WebSocket connection and a Claude-powered agent takes over. The agent uses Anthropic's [tool use](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview) on `messages.create` to interact with the pipeline results and the reporter.

### Agent Tools

| Tool | Type | Description |
|------|------|-------------|
| `view_topics` | Local | Returns all broad and specific topics with story counts |
| `list_stories_in_topic` | Local | Lists stories belonging to a given topic |
| `read_story` | Local | Reads full content of a story by index (truncated to 3000 chars) |
| `search_stories` | Local | Keyword search across story titles and content |
| `interview_user` | Interactive | Sends a question to the reporter via WebSocket and awaits their response. Supports `checklist`, `single_choice`, `multiple_choice`, and `free_response` question types |
| `generate_beat_book` | Output | Writes the final Markdown beat book to the `output/` directory and delivers it to the browser |

### Agent Loop

1. The agent receives a system prompt defining its role as a journalism mentor.
2. It starts by calling `view_topics` to survey the topic landscape.
3. It reads representative stories to understand the coverage.
4. It uses `interview_user` to ask the reporter 3-5 questions — starting with a checklist of topics for them to select their beat, then follow-ups about audience, experience, and needs.
5. It digs deeper into relevant stories using `read_story` and `search_stories`.
6. It calls `generate_beat_book` with a complete Markdown document.

The loop runs for up to 40 turns. The loop exits when the beat book is saved, the model stops calling tools, or it hits the per-turn output cap mid-tool-call (in which case it surfaces an error rather than risk a malformed continuation).

- **Model:** `claude-sonnet-4-6` via the Anthropic API
- **Max tokens per turn:** 32,768

**Tech:** Anthropic Python SDK (`anthropic`), async/await for WebSocket communication.

---

## Frontend

**Files:** `static/index.html`, `static/app.js`, `static/style.css`

The frontend is a single-page app with four screens:

### Upload Screen
- Drag-and-drop or file-picker accepting any common document format
- Textarea for pasting URLs (one per line) — fetched server-side and run through the same ingest pipeline
- 25 MB per-file cap

### Preview Screen
- Detected stories grouped by source, with confidence chips and one-line reasoning from the LLM
- Inline editing of title / date / author per story
- Per-story checkbox to include or exclude
- Excluded sources (non-news content, failed extracts) shown separately with the skip reason

### Interview Screen
- Renders the questions the agent asks: checkboxes (checklist/multiple choice), radio buttons (single choice), or a textarea (free response)

### Generating + Done Screen
- Real-time WebSocket connection to the agent
- Stepper UI showing review → write → research → cite progress
- Beat book delivery shows links to the rendered viewer and raw Markdown

**Tech:** Vanilla JavaScript (no framework), CSS custom properties for theming.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web server** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | Async HTTP + WebSocket server |
| **File extraction** | [markitdown](https://github.com/microsoft/markitdown) | Convert docx / pdf / html / pptx / xlsx / rtf to markdown |
| **URL fetching** | [httpx](https://www.python-httpx.org/) | Fetch URLs with SSRF protection (private IPs blocked) |
| **Story normalization** | [Anthropic API](https://docs.claude.com/) (`claude-sonnet-4-6`) | Tool-use call that splits documents into stories and infers title/date/author |
| **Embeddings** | [OpenAI API](https://platform.openai.com/docs/guides/embeddings) (`text-embedding-3-small`) | Convert article text to 1536-d vectors. (Anthropic has no embedding API.) |
| **Dimensionality reduction** | [UMAP](https://umap-learn.readthedocs.io/) | Project embeddings to lower dimensions for clustering |
| **Clustering** | [HDBSCAN](https://hdbscan.readthedocs.io/) | Density-based topic discovery at two granularities |
| **Topic labeling** | [Anthropic API](https://docs.claude.com/) (`claude-sonnet-4-6`) | Generate human-readable labels for each cluster |
| **Agent** | [Anthropic API](https://docs.claude.com/) (`claude-sonnet-4-6`) | Tool-using agent for interview and beat book generation |
| **Research agent** | [Anthropic API](https://docs.claude.com/) (`claude-opus-4-7`) | Public-web research over the draft beat book |
| **Numerical** | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [scikit-learn](https://scikit-learn.org/) | Vector math, distance calculations, preprocessing |
| **Frontend** | Vanilla HTML/CSS/JS | No-framework single-page app |

---

## Project Structure

```
beat-book/
├── app.py                  # FastAPI server — /ingest, /process, WebSocket
├── ingest.py               # Multi-format extraction + LLM normalization
├── pipeline.py             # NLP pipeline — embedding, UMAP, HDBSCAN, LLM labeling
├── agent.py                # Claude agent — tool definitions, system prompt, agent loop
├── claude_client.py        # Shared Anthropic config — model name + client factory
├── research_agent.py       # Sandboxed research agent that revises the draft beat book
├── citation_matcher.py     # Matches beat-book claims back to source sentences
├── requirements.txt        # Python dependencies
├── Makefile                # Install, run, dev, lint, clean targets
├── .env.example            # Template for required API keys
├── static/
│   ├── index.html          # Single-page app markup (upload, preview, interview, done)
│   ├── app.js              # Frontend logic — ingest, preview, SSE pipeline, WebSocket
│   ├── style.css           # Styles
│   └── viewer/             # Inline-citation viewer for finished beat books
├── docs/                   # Architecture deep-dives
├── output/                 # Generated beat books (Markdown + sources JSON)
└── .cache/                 # Embedding cache (auto-generated)
```

---

## Authors

- [Clay Ludwig](https://clayludwig.com/)
- [Cat Murphy](https://github.com/catelizabethmurphy)
- [Derek Willis](https://thescoop.org/)
