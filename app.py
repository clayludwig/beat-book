"""
app.py
------
FastAPI web app.

- POST /ingest             → upload files and/or URLs, run multi-format
                              extraction + LLM normalization, return a
                              preview of detected stories.
- POST /process            → run the embedding/clustering pipeline on a
                              confirmed (and optionally edited) story list.
                              Streams SSE progress; ends with a session_id.
- WS   /ws/{session_id}    → WebSocket for the agent conversation.
- GET  /                   → serves the frontend.
"""

import asyncio
import json
import os
import queue
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from pipeline import run_pipeline, PipelineResult
from agent import run_agent
from research_agent import run_research_agent
from citation_matcher import (
    embed_source_stories,
    markdown_to_beatbook_entries,
    build_sources_file,
)
from ingest import ingest_file, ingest_url

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Beat Book Builder")

# Files-in-flight per /ingest request. Serial so a multi-file upload
# doesn't multiply concurrent Claude calls against Anthropic's per-tier
# concurrent-request limit (ingest.py itself runs chunks serially too).
_INGEST_CONCURRENCY = 4

# In-memory session store: session_id → PipelineResult
sessions: Dict[str, PipelineResult] = {}


@dataclass
class IngestJob:
    job_id: str
    msg_queue: queue.Queue = field(default_factory=queue.Queue)
    done: bool = False
    result: Optional[dict] = None
    error: str = ""


ingest_jobs: Dict[str, IngestJob] = {}


class StoryIn(BaseModel):
    """Content entry payload accepted by /process. The pipeline only requires
    title + content; the rest are passed through if non-empty."""
    title: str
    content: str
    date: str = ""
    author: str = ""
    organization: str = ""
    link: str = ""
    content_type: str = "article"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    stories: List[StoryIn] = Field(default_factory=list)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SANDBOX_ROOT = OUTPUT_DIR / "sandboxes"
SANDBOX_ROOT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


async def _run_ingest_job(
    job: IngestJob,
    buffered_files: List[tuple[str, bytes]],
    url_list: List[str],
    *,
    anthropic_key: str,
) -> None:
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(_INGEST_CONCURRENCY)
    total_sources = len(buffered_files) + len(url_list)

    job.msg_queue.put({"type": "job_started", "total_sources": total_sources})

    async def run_file(name: str, raw: bytes):
        async with semaphore:
            job.msg_queue.put({"type": "source_started", "source_label": name})

            def on_progress(payload: dict):
                job.msg_queue.put({
                    "type": "source_progress",
                    "source_label": name,
                    **payload,
                })

            result = await loop.run_in_executor(
                None,
                lambda: ingest_file(
                    name,
                    raw,
                    anthropic_key,
                    on_progress=on_progress,
                ),
            )
            job.msg_queue.put({
                "type": "source_done",
                "source_label": name,
                "num_entries": len(result.stories),
                "excluded": result.excluded,
            })
            return result

    async def run_url(url: str):
        async with semaphore:
            job.msg_queue.put({"type": "source_started", "source_label": url})

            def on_progress(payload: dict):
                job.msg_queue.put({
                    "type": "source_progress",
                    "source_label": url,
                    **payload,
                })

            result = await loop.run_in_executor(
                None,
                lambda: ingest_url(
                    url,
                    anthropic_key,
                    on_progress=on_progress,
                ),
            )
            job.msg_queue.put({
                "type": "source_done",
                "source_label": url,
                "num_entries": len(result.stories),
                "excluded": result.excluded,
            })
            return result

    tasks = [run_file(name, raw) for name, raw in buffered_files]
    tasks += [run_url(u) for u in url_list]

    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.error = f"Ingestion failed: {type(e).__name__}: {e}"
        job.msg_queue.put({"type": "error", "error": job.error})
        job.done = True
        return

    sources = [r.to_preview_dict() for r in results]
    total_stories = sum(len(r.stories) for r in results)
    job.result = {
        "sources": sources,
        "total_stories": total_stories,
        "total_sources": len(results),
    }
    job.msg_queue.put({"type": "done", **job.result})
    job.done = True


@app.post("/ingest/start")
async def ingest_start(
    files: List[UploadFile] = File(default_factory=list),
    urls: str = Form(""),
):
    """Start ingest in the background and return a job id."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return JSONResponse(
            {"error": "ANTHROPIC_API_KEY not configured."}, status_code=500
        )

    url_list = [u.strip() for u in urls.splitlines() if u.strip()]

    if not files and not url_list:
        return JSONResponse(
            {"error": "No files or URLs provided."}, status_code=400
        )

    buffered_files: List[tuple[str, bytes]] = []
    for f in files:
        raw = await f.read()
        buffered_files.append((f.filename or "upload.bin", raw))

    job_id = str(uuid.uuid4())[:10]
    job = IngestJob(job_id=job_id)
    ingest_jobs[job_id] = job

    asyncio.create_task(
        _run_ingest_job(
            job,
            buffered_files,
            url_list,
            anthropic_key=anthropic_key,
        )
    )

    return JSONResponse({"job_id": job_id})


@app.get("/ingest/stream/{job_id}")
async def ingest_stream(job_id: str):
    job = ingest_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Invalid ingest job."}, status_code=404)

    async def event_stream():
        while not job.done or not job.msg_queue.empty():
            try:
                msg = job.msg_queue.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.1)
        if job.error and job.result is None:
            yield f"data: {json.dumps({'type': 'error', 'error': job.error})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/process")
async def process(body: ProcessRequest):
    """Run the embedding + clustering pipeline on a confirmed list of stories.
    Streams SSE progress events, terminates with a session_id the frontend can
    open over WebSocket for the agent conversation."""
    stories = [
        {k: v for k, v in s.model_dump().items() if v or k in ("title", "content")}
        for s in body.stories
    ]
    if not stories:
        return JSONResponse({"error": "No stories provided."}, status_code=400)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return JSONResponse({"error": "OPENAI_API_KEY not configured (used for embeddings)."}, status_code=500)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not configured (used for cluster labeling)."}, status_code=500)

    progress_queue: queue.Queue = queue.Queue()

    def on_progress(step: str, fraction: float, detail: str):
        progress_queue.put({"step": step, "fraction": fraction, "detail": detail})

    async def event_stream():
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None, run_pipeline, stories, openai_key, anthropic_key, on_progress
        )

        while not future.done():
            try:
                msg = progress_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'progress', **msg})}\n\n"
            except queue.Empty:
                pass
            await asyncio.sleep(0.15)

        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield f"data: {json.dumps({'type': 'progress', **msg})}\n\n"

        try:
            result = future.result()
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': f'{type(e).__name__}: {e}'})}\n\n"
            return

        session_id = str(uuid.uuid4())[:8]
        sessions[session_id] = result

        yield (
            "data: " + json.dumps({
                "type": "done",
                "session_id": session_id,
                "num_stories": len(stories),
                "num_topics": len(result.topics),
                "broad_topics": {k: len(v) for k, v in result.broad_topics.items()},
                "specific_topics": {k: len(v) for k, v in result.specific_topics.items()},
            }) + "\n\n"
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — Agent conversation
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def agent_ws(ws: WebSocket, session_id: str):
    await ws.accept()

    pipeline_result = sessions.get(session_id)
    if not pipeline_result:
        await ws.send_json({"type": "error", "text": "Invalid session. Please upload stories first."})
        await ws.close()
        return

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        await ws.send_json({"type": "error", "text": "ANTHROPIC_API_KEY not configured."})
        await ws.close()
        return

    # ── Wait for topic selection from the client ───────────────────────────
    selected_topics: list[str] = []
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=120)
        msg = json.loads(raw)
        if msg.get("type") == "select_topics":
            valid = set(pipeline_result.topics.keys())
            selected_topics = [t for t in msg.get("topics", []) if t in valid]
    except (asyncio.TimeoutError, Exception):
        pass
    if not selected_topics:
        selected_topics = list(pipeline_result.topics.keys())

    # ── Callbacks ─────────────────────────────────────────────────────────

    async def on_message(text: str):
        """Send agent text to the frontend."""
        await ws.send_json({"type": "message", "text": text})

    async def on_heartbeat():
        """Keep the WebSocket alive during long Anthropic API calls."""
        await ws.send_json({"type": "heartbeat"})

    # research_task is set by on_exploration_done and awaited in on_beat_book.
    _research_task: asyncio.Task | None = None
    _research_filename: str = "beat_book.md"

    async def on_exploration_done(context_doc: str):
        """Start the Opus research agent as soon as exploration is done,
        in parallel with Haiku's beat-book write."""
        nonlocal _research_task, _research_filename
        filename = _research_filename
        sandbox_dir = SANDBOX_ROOT / session_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        await ws.send_json({"type": "research_started", "filename": filename})

        async def on_research_progress(stage: str, detail: str):
            await ws.send_json({"type": "research_progress", "stage": stage, "detail": detail})

        async def on_research_tool_status(tool_name: str, desc: str, detail: str):
            await ws.send_json({"type": "research_tool_status",
                                "tool_name": tool_name, "tool": desc, "detail": detail})

        async def on_research_text(text: str):
            await ws.send_json({"type": "research_message", "text": text})

        async def _run():
            try:
                return await run_research_agent(
                    sandbox_dir=sandbox_dir,
                    markdown_filename=filename,
                    anthropic_api_key=anthropic_key,
                    on_progress=on_research_progress,
                    on_tool_status=on_research_tool_status,
                    on_text=on_research_text,
                    initial_content=context_doc,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None  # fallback: use draft from on_beat_book

        _research_task = asyncio.create_task(_run())

    async def on_beat_book(filename: str, markdown: str):
        """Merge the Haiku draft with the parallel Opus research, then hand
        the combined Markdown to the citation matcher.

        Pipeline: [Haiku write ∥ Opus research] → merge → citations.
        """
        nonlocal _research_filename
        _research_filename = filename

        # ── 1. Persist the raw draft ──────────────────────────────────────
        stem = Path(filename).stem
        draft_path = OUTPUT_DIR / f"{stem}.draft.md"
        draft_path.write_text(markdown, encoding="utf-8")

        # ── 2. Await research (may already be done if write was slow) ─────
        sandbox_dir = SANDBOX_ROOT / session_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        research_result: str | None = None
        if _research_task is not None:
            # Update the sandbox file with the real draft so Opus can
            # reference it if it hasn't already finished.
            (sandbox_dir / filename).write_text(markdown, encoding="utf-8")
            research_result = await _research_task
        else:
            # on_exploration_done never fired (very small corpus) — fall back
            # to sequential research.
            await ws.send_json({"type": "research_started", "filename": filename})
            (sandbox_dir / filename).write_text(markdown, encoding="utf-8")

            async def on_research_progress(stage: str, detail: str):
                await ws.send_json({"type": "research_progress",
                                    "stage": stage, "detail": detail})

            async def on_research_tool_status(tool_name: str, desc: str, detail: str):
                await ws.send_json({"type": "research_tool_status",
                                    "tool_name": tool_name, "tool": desc, "detail": detail})

            async def on_research_text(text: str):
                await ws.send_json({"type": "research_message", "text": text})

            try:
                research_result = await run_research_agent(
                    sandbox_dir=sandbox_dir,
                    markdown_filename=filename,
                    anthropic_api_key=anthropic_key,
                    on_progress=on_research_progress,
                    on_tool_status=on_research_tool_status,
                    on_text=on_research_text,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                await ws.send_json({
                    "type": "error",
                    "text": f"Research agent failed ({type(e).__name__}: {e}). Using draft.",
                })

        # research_result is the Opus-revised content; fall back to draft
        revised_markdown = research_result if research_result else markdown

        await ws.send_json({"type": "research_complete"})

        # ── 3. Canonical output is the revised markdown ──────────────────
        filepath = OUTPUT_DIR / filename
        filepath.write_text(revised_markdown, encoding="utf-8")
        await ws.send_json({
            "type": "beat_book_markdown_saved",
            "filename": filename,
        })

        # Citation matching uses OpenAI embeddings (Anthropic has no embedding API).
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            await ws.send_json({
                "type": "error",
                "text": "OPENAI_API_KEY not configured; skipping citation matching.",
            })
            return

        stories = pipeline_result.stories

        citation_progress_queue: queue.Queue = queue.Queue()

        def on_matcher_progress(stage: str, fraction: float, detail: str):
            citation_progress_queue.put({"stage": stage, "fraction": fraction, "detail": detail})

        def run_matcher():
            source_embeddings = embed_source_stories(stories, openai_key, on_matcher_progress)
            entries = markdown_to_beatbook_entries(revised_markdown, source_embeddings, openai_key, on_matcher_progress)
            sources = build_sources_file(stories, source_embeddings)
            return entries, sources

        await ws.send_json({
            "type": "citation_progress",
            "stage": "starting",
            "fraction": 0.0,
            "detail": "Embedding source sentences…",
        })

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, run_matcher)

        while not future.done():
            try:
                msg = citation_progress_queue.get_nowait()
                await ws.send_json({"type": "citation_progress", **msg})
            except queue.Empty:
                await asyncio.sleep(0.15)

        while not citation_progress_queue.empty():
            msg = citation_progress_queue.get_nowait()
            await ws.send_json({"type": "citation_progress", **msg})

        try:
            entries, sources = future.result()
        except Exception as e:
            await ws.send_json({
                "type": "error",
                "text": f"Citation matching failed: {e}. The raw Markdown is still available at /output/{filename}.",
            })
            return

        json_path = OUTPUT_DIR / f"{stem}.json"
        sources_path = OUTPUT_DIR / f"{stem}_sources.json"
        json_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")

        await ws.send_json({
            "type": "beat_book",
            "filename": filename,
            "markdown_path": f"/output/{quote(filename)}",
            "viewer_url": f"/static/viewer/viewer.html?book={quote(stem)}",
            "stem": stem,
        })

    async def on_tool_status(tool_name: str, tool_desc: str, detail: str):
        """Send tool execution status to frontend."""
        await ws.send_json({
            "type": "tool_status",
            "tool_name": tool_name,
            "tool": tool_desc,
            "detail": detail,
        })

    async def on_agent_progress(pct: float, label: str):
        """Send agent coverage-review progress to frontend (0–100)."""
        await ws.send_json({
            "type": "agent_progress",
            "pct": pct,
            "label": label,
        })

    # ── Run agent ─────────────────────────────────────────────────────────

    try:
        await run_agent(
            pipeline_result=pipeline_result,
            anthropic_key=anthropic_key,
            on_message=on_message,
            on_beat_book=on_beat_book,
            on_tool_status=on_tool_status,
            on_heartbeat=on_heartbeat,
            on_agent_progress=on_agent_progress,
            on_exploration_done=on_exploration_done,
            selected_topics=selected_topics,
        )
    except WebSocketDisconnect:
        print(f"Session {session_id}: client disconnected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await ws.send_json({
                "type": "error",
                "text": f"Agent error ({type(e).__name__}): {e}",
            })
        except Exception:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STATIC FILES (must be last so it doesn't shadow routes)
# ─────────────────────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
