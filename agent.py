"""
agent.py
--------
Claude-powered agent (Sonnet 4.6) that has tools to explore stories/
topics and produces a beat book without an interview stage.
"""

import asyncio
import json
from typing import Callable, Awaitable

import anthropic

from pipeline import PipelineResult
from claude_client import (
    CHAT_MODEL,
    RATE_LIMIT_MAX_RETRIES,
    chat_client,
    rate_limit_pause,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

AGENT_MODEL = CHAT_MODEL
# Exploration (deciding which topics/stories to look at) is mechanical and
# uses Haiku for ~5-10x faster turns. The final beat-book write switches
# back to Sonnet for prose quality.
EXPLORE_MODEL = "claude-haiku-4-5-20251001"

# Generous output cap — beat books can be long. Sonnet 4.6 supports a
# 1M token context window and up to 128k output tokens; 32k per turn is
# plenty for a beat-book draft plus a few tool calls.
MAX_TOKENS_PER_TURN = 32768
# Tighter cap for the forced final generation turn — keeps Sonnet from
# spending 5+ minutes filling the buffer. A beat book is ~5-10k words,
# which is comfortably under 16k tokens of output.
MAX_TOKENS_FINAL_GENERATE = 16384
# Hard cap on any single tool result sent back to the model. Prevents one
# large read_stories_in_topic call from flooding the context.
MAX_TOOL_RESULT_CHARS = 60_000
# When the conversation exceeds this many messages, old tool result turns
# are replaced with a stub so the history stays within the 1M token limit.
MAX_HISTORY_MESSAGES = 40

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS (Anthropic tool-use schema)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "view_topics",
        "description": (
            "View all discovered topics from the uploaded stories. Returns broad "
            "and specific topics with story counts. Use this first to understand "
            "the landscape of coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_stories_in_topic",
        "description": (
            "List all stories that belong to a given topic. Returns story index, "
            "title, and date for each."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The exact topic label to look up.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "read_story",
        "description": (
            "Read the full content of a story by its index number. Returns title, "
            "author, date, and full text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Zero-based story index.",
                },
            },
            "required": ["index"],
        },
    },
    {
        "name": "read_stories_in_topic",
        "description": (
            "Read all stories in a topic at once — title, metadata, and a "
            "content excerpt for every story in the topic in a single call. "
            "Use this as your primary research tool: it satisfies the research "
            "requirement for the topic in one call instead of N separate "
            "read_story calls. After scanning, use read_story only if you need "
            "the full text of a specific document."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The exact topic label to scan.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "search_stories",
        "description": (
            "Search stories by keyword. Returns matching story indices, titles, "
            "and dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for in story titles and content.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "generate_beat_book",
        "description": (
            "Write the final beat book as a Markdown document. Call this once you "
            "have gathered enough information from the topics, stories, and the "
            "reporter's answers. The content you pass will be saved as the output file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "markdown_content": {
                    "type": "string",
                    "description": "The complete beat book in Markdown format.",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename for the beat book (e.g. 'sports_beat_book.md').",
                },
            },
            "required": ["markdown_content", "filename"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert journalism mentor and beat-book author. Your job is to help \
a reporter create a comprehensive "beat book" — a practical reporting guide for \
covering a specific beat (topic area).

You have been given a set of news stories that the reporter has uploaded. These \
stories have already been analyzed and grouped into topics automatically.

Your workflow:
1. **Explore** — Call `view_topics` to see the full topic list. Then call \
`read_stories_in_topic` for each topic — it returns every story with a \
2000-character excerpt and immediately satisfies the research requirement for \
that topic. Use `read_story` only when you need the full text of a specific \
document. Use `search_stories` to surface specific people, institutions, or \
themes across topics. Every tool result includes a `[Research progress]` block \
showing per-topic read status — once all topics show "OK", you can generate.
2. **Generate** — Call `generate_beat_book` once every topic shows "OK" in \
the progress block.

The beat book is a narrative document, not an outline. A reporter should be \
able to read it cover-to-cover the way they'd read a long-form magazine \
feature about their own beat. Structure it as follows.

Open with a **Beat Overview** of two or three paragraphs explaining what the \
beat covers and why it matters — written as prose, not as bullets. Move into \
**Key Topics & Themes**, organized around the topics the reporter selected; \
each topic gets a few paragraphs describing who is doing what, what is at \
stake, and the recurring tension or arc in the coverage. Cover **Key Sources \
& Players** — the people, organizations, and institutions that appear \
repeatedly — by writing about them in sentences, explaining their role and \
how they tend to surface; do not reduce them to a bulleted roster. **Story \
Ideas & Angles** can be a short numbered list because each idea is a \
discrete thought, but introduce the list with a sentence or two framing the \
gap in coverage it addresses. Provide **Background & Context** as flowing \
prose: the history, policy, and institutional knowledge a new reporter would \
need to make sense of the beat. End with **Reporting Tips** (a few sentences \
of practical advice specific to this beat, not generic journalism advice) \
and a **Calendar & Recurring Events** section, where a list is appropriate \
because the items are genuinely list-shaped (a meeting on the second \
Tuesday of every month).

**Writing style.** Write in connected prose, the way a senior reporter would \
brief a colleague picking up the beat — not as an outline. Use bullets only \
for genuinely list-shaped content: a roster of named sources, a short list \
of story ideas, a calendar. Do NOT create per-topic sub-headers like \
"What's happening / Key story / Story angles" — let the prose carry the \
structure. Write complete sentences with concrete subjects and verbs ("The \
school board voted 6-1 last March to raise property taxes" — not "School \
board: 6-1 vote, March, property tax increase"). Reference actual stories, \
names, and details from the corpus, not generic advice. The result should \
read like a piece of journalism about the beat, useful enough that a \
brand-new reporter could pick it up and start producing informed coverage \
the same day.

**Do NOT include a table of contents.** The viewer provides its own \
navigation from the document's headings, so a TOC in the Markdown is \
redundant. Start the document with the title and subtitle, then go directly \
into the Beat Overview.

Keep your conversational messages concise. Use tools frequently.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TOOL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def _target_for_topic(topic_size: int) -> int:
    """Per-topic minimum read count: every story if the topic has fewer than
    15, otherwise half (rounded up)."""
    if topic_size < 15:
        return topic_size
    return (topic_size + 1) // 2


def _progress_report(
    pipeline_result: PipelineResult,
    listed_topics: set,
    read_indices: set,
) -> tuple[str, bool]:
    """Format the agent's research progress, plus a boolean telling whether
    every listed topic has reached its read-count target. The boolean drives
    the generate_beat_book gate."""
    if not listed_topics:
        return (
            "[Research progress] No topics listed yet. Call list_stories_in_topic "
            "for each topic the reporter selected to start exploring.",
            False,
        )

    lines = ["[Research progress]"]
    all_met = True
    for topic in sorted(listed_topics):
        indices = pipeline_result.topics.get(topic, [])
        if not indices:
            continue
        total = len(indices)
        read = sum(1 for i in indices if i in read_indices)
        target = _target_for_topic(total)
        met = read >= target
        if not met:
            all_met = False
        marker = "OK" if met else "needs more"
        lines.append(
            f"  - {topic}: {read}/{total} read (target {target}) — {marker}"
        )
    lines.append(
        f"Total stories read (unique): {len(read_indices)}. "
        "Targets: every story in topics with <15 stories, otherwise half."
    )
    if not all_met:
        lines.append(
            "generate_beat_book will be rejected until every listed topic "
            "meets its target. Keep reading."
        )
    return "\n".join(lines), all_met


def execute_local_tool(name: str, input_data: dict, result: PipelineResult) -> str:
    """Execute a non-interactive tool and return a string result."""
    if name == "view_topics":
        return result.topic_summary()

    if name == "list_stories_in_topic":
        stories = result.stories_for_topic(input_data["topic"])
        if not stories:
            return f"No stories found for topic '{input_data['topic']}'. Check exact spelling."
        return json.dumps(stories, indent=2)

    if name == "read_story":
        story = result.get_story(input_data["index"])
        if not story:
            return f"Invalid index {input_data['index']}. Valid range: 0–{len(result.stories)-1}."
        return json.dumps({
            "index": input_data["index"],
            "title": story.get("title", ""),
            "author": story.get("author", ""),
            "date": story.get("date", ""),
            "topics": result.story_topics[input_data["index"]],
            "content": story.get("content", "")[:1500],
        }, indent=2)

    if name == "read_stories_in_topic":
        topic = input_data.get("topic", "")
        indices = result.topics.get(topic, [])
        if not indices:
            return f"No stories found for topic '{topic}'. Check exact spelling."
        entries = []
        for i in indices:
            story = result.get_story(i)
            if not story:
                continue
            entries.append({
                "index": i,
                "title": story.get("title", ""),
                "date": story.get("date", ""),
                "author": story.get("author", ""),
                "organization": story.get("organization", ""),
                "content_type": story.get("content_type", "article"),
                "metadata": story.get("metadata", {}),
                "excerpt": story.get("content", "")[:600],
            })
        return json.dumps(entries, indent=2)

    if name == "search_stories":
        matches = result.search_stories(input_data["query"])
        if not matches:
            return f"No stories matching '{input_data['query']}'."
        return json.dumps(matches, indent=2)

    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

# Type for the callback that sends agent text messages to the frontend
MessageCallback   = Callable[[str], Awaitable[None]]
# Type for the callback that reports tool execution status
ToolStatusCallback = Callable[[str, str, str], Awaitable[None]]


# Human-friendly descriptions for each tool
TOOL_DESCRIPTIONS = {
    "view_topics": "Reviewing discovered topics",
    "list_stories_in_topic": "Listing stories in topic",
    "read_stories_in_topic": "Scanning all stories in topic",
    "read_story": "Reading a story",
    "search_stories": "Searching stories",
    "generate_beat_book": "Writing the beat book",
}


def _cap_tool_result(content: str) -> str:
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    return content[:MAX_TOOL_RESULT_CHARS] + (
        f"\n\n[Truncated: result was {len(content):,} chars; "
        f"showing first {MAX_TOOL_RESULT_CHARS:,}. "
        "Use read_story with specific indices for full content.]"
    )


def _prune_history(messages: list) -> list:
    """Replace the content of old tool-result turns with a stub once the
    conversation grows past MAX_HISTORY_MESSAGES, keeping the most recent
    turns intact so the model retains full context of recent work."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    # Keep the first message (initial user prompt) and the most recent half.
    keep_tail = MAX_HISTORY_MESSAGES // 2
    pruned = []
    for i, msg in enumerate(messages):
        if i == 0 or i >= len(messages) - keep_tail:
            pruned.append(msg)
        elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
            # Replace tool result content with stubs to free token budget.
            stubbed = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    stubbed.append({**block, "content": "[earlier result omitted to save context]"})
                else:
                    stubbed.append(block)
            pruned.append({**msg, "content": stubbed})
        else:
            pruned.append(msg)
    return pruned


async def run_agent(
    pipeline_result: PipelineResult,
    anthropic_key: str,
    on_message: MessageCallback,
    on_beat_book: Callable[[str, str], Awaitable[None]],
    on_tool_status: ToolStatusCallback = None,
    on_heartbeat: Callable[[], Awaitable[None]] = None,
    on_agent_progress: Callable[[float, str], Awaitable[None]] = None,
    on_exploration_done: Callable[[str], Awaitable[None]] = None,
    selected_topics: list[str] | None = None,
) -> None:
    """
    Run the agent loop.

    Args:
        pipeline_result: Output from the embedding/clustering pipeline.
        anthropic_key: Anthropic API key (claude-sonnet-4-6).
        on_message: async callback(text) — sends agent text to the frontend.
        on_beat_book: async callback(filename, markdown) — saves/delivers the beat book.
        on_tool_status: async callback(tool_name, detail) — reports tool execution status.
        on_heartbeat: optional async callback fired every ~15s during API calls
                      to keep the WebSocket connection alive.
    """
    client = chat_client(anthropic_key)

    def _streamed_create(**kwargs):
        """Blocking streaming call. Using `messages.stream` instead of
        `messages.create` keeps the HTTP connection alive while Sonnet
        writes long output; the non-streaming endpoint silently disconnects
        on multi-minute responses (RemoteProtocolError)."""
        import time as _t
        t0 = _t.time()
        print(f"[agent.stream] entering stream context, model={kwargs.get('model')}, "
              f"max_tokens={kwargs.get('max_tokens')}", flush=True)
        chars = 0
        events = 0
        last_print = t0
        try:
            stream_ctx = client.messages.stream(**kwargs)
        except Exception as e:
            print(f"[agent.stream] stream() raised before entering: {type(e).__name__}: {e}",
                  flush=True)
            raise
        with stream_ctx as stream:
            print(f"[agent.stream] context open at {_t.time()-t0:.1f}s, "
                  f"iterating events…", flush=True)
            for event in stream:
                events += 1
                if events == 1:
                    print(f"[agent.stream] first event at {_t.time()-t0:.1f}s "
                          f"type={type(event).__name__}", flush=True)
                delta = getattr(event, "delta", None)
                if delta is not None:
                    text = getattr(delta, "text", None) or getattr(delta, "partial_json", None)
                    if text:
                        chars += len(text)
                now = _t.time()
                if now - last_print >= 5:
                    print(f"[agent.stream] {now-t0:.1f}s, {events} events, "
                          f"{chars} chars streamed", flush=True)
                    last_print = now
            final = stream.get_final_message()
        print(f"[agent.stream] done in {_t.time()-t0:.1f}s, "
              f"{events} events, stop_reason={final.stop_reason}", flush=True)
        return final

    async def _api_call_with_heartbeat(**kwargs):
        """Run the streaming API call in a thread while sending heartbeats
        every 15 seconds so the WebSocket doesn't time out during long turns."""
        api_task = asyncio.create_task(asyncio.to_thread(_streamed_create, **kwargs))
        while not api_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(api_task), timeout=15)
            except asyncio.TimeoutError:
                if on_heartbeat:
                    try:
                        await on_heartbeat()
                    except Exception:
                        pass
            except Exception:
                # surface the real exception from api_task below
                break
        return await api_task

    # Restrict to reporter-selected topics if provided.
    if selected_topics:
        from pipeline import PipelineResult as _PR
        from dataclasses import replace as _replace
        filtered = {t: v for t, v in pipeline_result.topics.items()
                    if t in set(selected_topics)}
        pipeline_result = _replace(pipeline_result, topics=filtered)

    n_stories = len(pipeline_result.stories)
    n_topics  = len(pipeline_result.topics)

    # Anthropic conversation: `system` is a top-level kwarg, not a message.
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"I've uploaded {n_stories} news stories. The system has automatically "
                f"discovered {n_topics} topics across them. Please build a comprehensive "
                "beat book from these stories. Start by exploring the topics, read all "
                "of them, then generate the beat book."
            ),
        },
    ]

    last_message_text = ""
    beat_book_done = False

    # Research-progress tracking. listed_topics is the set of topic labels the
    # agent has called list_stories_in_topic on (a proxy for "topics the
    # reporter selected"); read_indices is the set of story indices the
    # agent has read via read_story. Both feed _progress_report, which is
    # surfaced in every local tool result AND used to gate generate_beat_book.
    listed_topics: set = set()
    read_indices: set = set()

    MAX_TURNS = 40
    for _turn in range(MAX_TURNS):
        messages = _prune_history(messages)

        # Once coverage is met, FORCE the model to call generate_beat_book.
        # Without forced tool_choice, Sonnet often narrates "I've reviewed
        # everything" and ends the turn instead of producing the book.
        _, threshold_met = _progress_report(
            pipeline_result, listed_topics, read_indices,
        )
        force_generate = threshold_met and not beat_book_done
        if force_generate and on_exploration_done and not getattr(run_agent, "_exploration_fired", False):
            run_agent._exploration_fired = True
            # Build a context doc with topic summaries + story excerpts for
            # the research agent to start on in parallel.
            context_lines = ["# Beat Book Research Context\n"]
            for topic, indices in pipeline_result.topics.items():
                context_lines.append(f"## {topic}\n")
                for i in list(indices)[:6]:
                    s = pipeline_result.get_story(i)
                    if s:
                        excerpt = " ".join(s.get("content", "").split()[:60])
                        context_lines.append(f"- **{s.get('title','')}** — {excerpt}\n")
                context_lines.append("")
            context_doc = "\n".join(context_lines)
            try:
                await on_exploration_done(context_doc)
            except Exception:
                pass
        if force_generate:
            print(f"[agent] turn {_turn}: forcing generate_beat_book "
                  f"(listed={len(listed_topics)}/{len(pipeline_result.topics)}, "
                  f"read={len(read_indices)}/{len(pipeline_result.stories)})",
                  flush=True)
            if on_tool_status:
                await on_tool_status(
                    "generate_beat_book",
                    "Writing the beat book",
                    "Drafting the full Markdown — this takes 1–3 minutes…",
                )
            if on_agent_progress:
                await on_agent_progress(100, "Writing the beat book")
            await on_message(
                "Coverage target met — writing the beat book now. "
                "This step takes 1–3 minutes."
            )
        # SDK blocks the event loop; offload to a worker thread so the
        # WebSocket handler stays responsive. If the SDK exhausts its own
        # 429-retry budget, the outer loop here pauses and tries again
        # before tearing down the session.
        response = None
        for rl_attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                if force_generate:
                    # Final-write strategy: no tools at all, no tool_choice.
                    # The model emits the beat book as plain Markdown text,
                    # which streams reliably (forcing a tool_use that
                    # contains 16k tokens of stringified JSON hangs at TTFT
                    # for both Sonnet and Haiku — confirmed empirically).
                    write_messages = list(messages) + [{
                        "role": "user",
                        "content": (
                            "Now write the full beat book in Markdown. "
                            "Reply with ONLY the Markdown document — no "
                            "preamble, no acknowledgement. Start directly "
                            "with the title (`# ...`)."
                        ),
                    }]
                    request_kwargs = dict(
                        model=EXPLORE_MODEL,
                        max_tokens=MAX_TOKENS_FINAL_GENERATE,
                        system=[{
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=write_messages,
                    )
                else:
                    request_kwargs = dict(
                        model=EXPLORE_MODEL,
                        max_tokens=4096,
                        system=[{
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        tools=TOOLS,
                        messages=messages,
                    )
                print(f"[agent] turn {_turn}: calling Sonnet "
                      f"(force_generate={force_generate}, "
                      f"max_tokens={request_kwargs['max_tokens']}, "
                      f"messages={len(messages)})", flush=True)
                response = await _api_call_with_heartbeat(**request_kwargs)
                print(f"[agent] turn {_turn}: stop_reason={response.stop_reason} "
                      f"blocks={[getattr(b,'type',None) for b in response.content]}",
                      flush=True)
                break
            except anthropic.RateLimitError as e:
                if rl_attempt >= RATE_LIMIT_MAX_RETRIES:
                    await on_message(
                        "⚠️ Anthropic's concurrent-connection rate limit "
                        "is persistently exceeded — please wait a few "
                        "minutes and start a new session."
                    )
                    return
                pause = rate_limit_pause(rl_attempt, e)
                await on_message(
                    f"⏸ Hit Anthropic's rate limit. "
                    f"Waiting {pause:.0f}s before retrying "
                    f"(attempt {rl_attempt + 1}/{RATE_LIMIT_MAX_RETRIES})…"
                )
                await asyncio.sleep(pause)
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                if rl_attempt >= RATE_LIMIT_MAX_RETRIES:
                    await on_message(
                        f"⚠️ Connection to Anthropic kept failing ({e}). "
                        "Please check your network and start a new session."
                    )
                    return
                pause = min(30.0, 5.0 * (2 ** rl_attempt))
                await on_message(
                    f"⏸ Connection error talking to Anthropic ({e}). "
                    f"Retrying in {pause:.0f}s "
                    f"(attempt {rl_attempt + 1}/{RATE_LIMIT_MAX_RETRIES})…"
                )
                await asyncio.sleep(pause)
            except anthropic.APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status and 500 <= status < 600 and rl_attempt < RATE_LIMIT_MAX_RETRIES:
                    pause = min(30.0, 5.0 * (2 ** rl_attempt))
                    await on_message(
                        f"⏸ Anthropic returned {status}. Retrying in "
                        f"{pause:.0f}s (attempt {rl_attempt + 1}/"
                        f"{RATE_LIMIT_MAX_RETRIES})…"
                    )
                    await asyncio.sleep(pause)
                else:
                    raise
        assert response is not None  # loop above either sets or returns

        # Forward any plain-text narration to the frontend (dedup repeats).
        text_combined = "".join(
            b.text for b in response.content
            if getattr(b, "type", None) == "text"
        ).strip()

        if force_generate:
            # Final-write turn: the text body IS the beat book.
            if text_combined:
                await on_beat_book("beat_book.md", text_combined)
                beat_book_done = True
                break
            else:
                await on_message(
                    "⚠️ The model returned no Markdown on the final write. "
                    "Try again."
                )
                break

        if text_combined and text_combined != last_message_text:
            await on_message(text_combined)
            last_message_text = text_combined

        stop_reason = response.stop_reason

        # No tool use this turn = end of conversation, or we already wrote
        # the beat book on the prior turn.
        if stop_reason != "tool_use":
            if stop_reason == "max_tokens":
                await on_message(
                    "⚠️ The model hit its output limit mid-turn. The beat book "
                    "may be incomplete. Please try again with a tighter scope."
                )
                break
            if beat_book_done:
                break
            # Agent stopped narrating without calling generate_beat_book.
            # Push it forward instead of silently dropping the session.
            _, threshold_met = _progress_report(
                pipeline_result, listed_topics, read_indices,
            )
            messages.append({"role": "assistant", "content": response.content})
            if threshold_met:
                nudge = (
                    "You have met the coverage target on every listed topic. "
                    "Call `generate_beat_book` NOW with the full Markdown. "
                    "Do not reply with text — call the tool."
                )
            else:
                nudge = (
                    "You stopped without calling a tool. Continue exploring: "
                    "use `read_stories_in_topic` on topics that still show "
                    "'needs more', then call `generate_beat_book`. Do not "
                    "reply with prose — call a tool."
                )
            messages.append({"role": "user", "content": nudge})
            continue
        if beat_book_done:
            break

        # Preserve the full assistant content (text + thinking + tool_use)
        # verbatim so the next request stays coherent.
        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            tool_name = block.name
            tool_id = block.id
            tool_input = block.input or {}

            # Report tool status to the frontend
            if on_tool_status:
                desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name)
                detail = ""
                if tool_name in ("list_stories_in_topic", "read_stories_in_topic"):
                    detail = tool_input.get("topic", "")
                elif tool_name == "read_story":
                    idx = tool_input.get("index", "")
                    story = pipeline_result.get_story(idx) if isinstance(idx, int) else None
                    detail = story.get("title", f"#{idx}")[:60] if story else f"#{idx}"
                elif tool_name == "search_stories":
                    detail = tool_input.get("query", "")
                await on_tool_status(tool_name, desc, detail)

            if tool_name == "generate_beat_book":
                progress, threshold_met = _progress_report(
                    pipeline_result, listed_topics, read_indices,
                )
                if not threshold_met:
                    content_str = (
                        "generate_beat_book REJECTED — research is incomplete.\n\n"
                        f"{progress}\n\n"
                        "Continue using list_stories_in_topic (for any topics you "
                        "haven't listed yet) and read_story to bring every listed "
                        "topic up to its target, then call generate_beat_book again. "
                        "Do not stop — the user will get no beat book if you abandon "
                        "the loop now."
                    )
                else:
                    await on_beat_book(
                        tool_input.get("filename", "beat_book.md"),
                        tool_input.get("markdown_content", ""),
                    )
                    content_str = "Beat book saved successfully. You may now give a brief closing message."
                    beat_book_done = True

            else:
                content_str = execute_local_tool(tool_name, tool_input, pipeline_result)
                # Track research progress and append a status block so the
                # model can self-check against the per-topic targets.
                if tool_name == "list_stories_in_topic":
                    topic = tool_input.get("topic", "")
                    if topic and topic in pipeline_result.topics:
                        listed_topics.add(topic)
                elif tool_name == "read_stories_in_topic":
                    topic = tool_input.get("topic", "")
                    if topic and topic in pipeline_result.topics:
                        listed_topics.add(topic)
                        read_indices.update(pipeline_result.topics[topic])
                elif tool_name == "read_story":
                    idx = tool_input.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(pipeline_result.stories):
                        read_indices.add(idx)
                progress, _ = _progress_report(
                    pipeline_result, listed_topics, read_indices,
                )
                content_str = f"{content_str}\n\n{progress}"

                if on_agent_progress:
                    total_topics = len(pipeline_result.topics) or 1
                    topic_pct = 0.0
                    for topic, indices in pipeline_result.topics.items():
                        if not indices:
                            continue
                        target = _target_for_topic(len(indices))
                        read = sum(1 for i in indices if i in read_indices)
                        topic_pct += min(1.0, read / target) / total_topics
                    pct = round(topic_pct * 100)
                    label = f"Reviewing coverage — {len(listed_topics)}/{total_topics} topics"
                    try:
                        await on_agent_progress(pct, label)
                    except Exception:
                        pass

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": _cap_tool_result(content_str),
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    if not beat_book_done:
        print(f"[agent] loop exited without writing beat book "
              f"(turn count exhausted or stop_reason mismatch). "
              f"listed_topics={listed_topics}, read={len(read_indices)}", flush=True)
        await on_message(
            "⚠️ Agent stopped before producing a beat book. "
            "Try again, or check the server logs."
        )
