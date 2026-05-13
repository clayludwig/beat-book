"""
agent.py
--------
Haiku-driven planning agent. Its job is narrow:

  1. Survey the topic landscape (the topic briefings produced by
     pipeline.build_briefings are injected into the system prompt).
  2. Run ONE round of interview_user to learn the reporter's beat
     selection, audience, and emphasis.
  3. Call generate_beat_book with a slim plan (filename, beat selection,
     focus notes). The actual prose is synthesized by a single Sonnet
     pass against the same briefings — see _generate_final_beat_book.

The old "must read half of every topic before publishing" research gate
is gone. Briefings already capture the corpus; the original story-level
tools (read_story / list_stories_in_topic / search_stories) remain
available for optional verification.
"""

import asyncio
import json
from typing import Awaitable, Callable, List, Optional

import anthropic

from pipeline import PipelineResult, render_briefings_markdown, strip_quotes
from claude_client import (
    CHAT_MODEL,
    DRAFT_MODEL,
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMIT_PAUSE_SECONDS,
    chat_client,
    thinking_param,
)

# ────────────────────────────────────────────────────────────────────────
# MODELS
# ────────────────────────────────────────────────────────────────────────

AGENT_MODEL  = CHAT_MODEL    # Haiku 4.5 drives the planning / interview loop
WRITER_MODEL = DRAFT_MODEL   # Sonnet 4.6 writes the final prose

LOOP_MAX_TOKENS  = 8192      # per turn in the planning loop
DRAFT_MAX_TOKENS = 32768     # one Sonnet call to produce the full beat book
MAX_TURNS        = 10        # soft cap — most sessions finish in 3-5 turns


# ────────────────────────────────────────────────────────────────────────
# TOOL SCHEMAS
# ────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "view_topics",
        "description": (
            "Re-list the broad and specific topics with story counts. The "
            "topic briefings are already in your system prompt; use this "
            "only if you want a compact recap."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_stories_in_topic",
        "description": (
            "List the stories in a given topic — index, title, date. "
            "Optional: only call this if you want to verify or quote a "
            "specific story beyond what the briefing already captures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Exact topic label (case-sensitive).",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "read_story",
        "description": (
            "Read the full content of a story by index. Optional verification "
            "tool — the briefings already include representative excerpts."
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
        "name": "search_stories",
        "description": (
            "Keyword search across all story titles and content. Optional "
            "verification tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "interview_user",
        "description": (
            "Ask the reporter a BATCH of interview questions in a single "
            "form. Strongly prefer asking 4-6 related questions in one call. "
            "The first question MUST be a checklist of broad topic labels "
            "so the reporter selects which topics form their beat.\n\n"
            "Each question supports one of four input types:\n"
            "- 'checklist': multi-select checkboxes\n"
            "- 'single_choice': radio buttons — pick one\n"
            "- 'multiple_choice': checkboxes — pick one or more\n"
            "- 'free_response': open text input"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intro": {
                    "type": "string",
                    "description": "Optional 1-2 sentence intro framing the form.",
                },
                "questions": {
                    "type": "array",
                    "description": "Ordered questions (4-6 recommended).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question":      {"type": "string"},
                            "question_type": {
                                "type": "string",
                                "enum": ["checklist", "single_choice",
                                         "multiple_choice", "free_response"],
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Options for non-free questions. Empty for free_response.",
                            },
                        },
                        "required": ["question", "question_type"],
                    },
                },
            },
            "required": ["questions"],
        },
    },
    {
        "name": "generate_beat_book",
        "description": (
            "Trigger the final beat book draft. You do NOT write the prose "
            "yourself — a writer model will synthesize the Markdown from the "
            "briefings + interview answers + the inputs you pass here. "
            "Call this once you have the reporter's answers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Output filename, kebab-case, ending in .md (e.g. 'school-board-beat-book.md').",
                },
                "beat_selection": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of broad-topic labels (exact spelling) that the reporter selected as their beat.",
                },
                "focus_notes": {
                    "type": "string",
                    "description": (
                        "2-4 sentences distilling the reporter's interview "
                        "answers into emphasis instructions for the writer: "
                        "which topics matter most, what audience knowledge "
                        "to assume, what angles to lead with, what tone."
                    ),
                },
            },
            "required": ["filename", "beat_selection", "focus_notes"],
        },
    },
]


# ────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (two text blocks: static prefix + cached briefings)
# ────────────────────────────────────────────────────────────────────────

_SYSTEM_PREFIX = """\
You are a journalism mentor running the interview step of a beat-book \
builder. The reporter has uploaded a corpus of news stories; an upstream \
pipeline has clustered them into broad topics and produced a structured \
briefing per topic. Those briefings appear later in this system prompt \
and are your primary source — the final beat book will be written from \
them by a separate writer model.

Your job is narrow and short:

1. Review the topic landscape and the per-topic briefings (below). \
You do not need to call view_topics first — the briefings already \
contain the labels, story counts, and a digest of each topic.

2. Call interview_user ONCE with 4-6 questions:
   - Question 1 MUST be a 'checklist' of the broad topic labels \
exactly as written below, so the reporter selects which topics form \
their beat.
   - Questions 2-6 probe: who the reporter is writing for, their \
experience level on this beat (newcomer vs. veteran), specific \
angles or storylines they care about, and what they most want from \
the beat book (story ideas, sourcing tips, background, etc.).

3. After receiving the answers, immediately call generate_beat_book \
with:
   - filename: kebab-case .md filename
   - beat_selection: the topic labels they checked (exact spelling \
from the briefings)
   - focus_notes: 2-4 sentences of emphasis instructions for the \
writer, distilled from the answers. Be concrete: name topics, \
audiences, and angles. The writer reads these notes and the \
briefings — nothing else from this conversation.

The writer will produce the final prose. Do NOT try to write the \
beat book yourself — your generate_beat_book call kicks off a \
separate Sonnet pass that handles the writing.

Optional verification tools (use only if a briefing seems to be \
missing or wrong about something):
- view_topics  — compact recap of topic labels and counts
- list_stories_in_topic(topic) — story index/title/date for a topic
- read_story(index) — the full text of one story
- search_stories(query) — keyword search across all stories

You have a 10-turn budget. Most sessions finish in 3-5 turns; \
spending turns on verification is fine but not required.

Keep any narration short. Use tools.
"""

_BRIEFINGS_HEADER = "\n# Topic landscape\n\n"


def _build_system_blocks(pipeline_result: PipelineResult) -> list[dict]:
    """Return the `system` value as a list of text blocks. The briefings
    block is marked cache_control: ephemeral so subsequent turns reuse it
    instead of re-encoding the full corpus knowledge."""
    briefings_md = render_briefings_markdown(pipeline_result.briefings or {})
    # Build a compact list of broad-topic labels with story counts to anchor
    # the checklist question.
    label_lines: list[str] = []
    for label, indices in sorted(
        pipeline_result.broad_topics.items(), key=lambda x: -len(x[1])
    ):
        label_lines.append(f"- {label} ({len(indices)} stories)")
    label_block = "## Broad topic labels (use these exactly in the checklist)\n\n" + "\n".join(label_lines)

    return [
        {"type": "text", "text": _SYSTEM_PREFIX},
        {
            "type": "text",
            "text": _BRIEFINGS_HEADER + label_block + "\n\n" + briefings_md,
            "cache_control": {"type": "ephemeral"},
        },
    ]


# ────────────────────────────────────────────────────────────────────────
# LOCAL TOOL EXECUTION
# ────────────────────────────────────────────────────────────────────────

def execute_local_tool(name: str, input_data: dict, result: PipelineResult) -> str:
    """Execute a non-interactive tool and return a string result."""
    if name == "view_topics":
        return result.topic_summary()

    if name == "list_stories_in_topic":
        stories = result.stories_for_topic(input_data.get("topic", ""))
        if not stories:
            return f"No stories found for topic {input_data.get('topic', '')!r}. Check exact spelling."
        return json.dumps(stories, indent=2)

    if name == "read_story":
        idx = input_data.get("index")
        story = result.get_story(idx) if isinstance(idx, int) else None
        if not story:
            return f"Invalid index {idx}. Valid range: 0-{len(result.stories) - 1}."
        # Strip direct quotes from the preview the agent sees — it gives a
        # tighter 3000-char window of actual reportorial prose. The original
        # content is still available to the citation matcher elsewhere.
        return json.dumps({
            "index":   idx,
            "title":   story.get("title", ""),
            "author":  story.get("author", ""),
            "date":    story.get("date", ""),
            "topics":  result.story_topics[idx],
            "content": strip_quotes(story.get("content", ""))[:3000],
        }, indent=2)

    if name == "search_stories":
        matches = result.search_stories(input_data.get("query", ""))
        if not matches:
            return f"No stories matching {input_data.get('query', '')!r}."
        return json.dumps(matches, indent=2)

    return f"Unknown tool: {name}"


# ────────────────────────────────────────────────────────────────────────
# FINAL DRAFT (Sonnet 4.6 prose synthesis)
# ────────────────────────────────────────────────────────────────────────

_WRITER_SYSTEM = """\
You are writing a beat book — a long-form reporting guide a journalist \
will use to cover a specific beat. The reporter has uploaded a news \
corpus; an upstream pipeline produced a structured briefing per topic. \
You will receive those briefings, the reporter's interview answers, \
and a short focus note from the planning agent.

Write the beat book as flowing prose in Markdown. Structure:

- Open with a title (`#`) and a one-line subtitle, then a 2-3 paragraph \
**Beat Overview** explaining what the beat covers and why it matters.
- **Key Topics & Themes** — organize around the topics in beat_selection. \
For each, write a few paragraphs describing who is doing what, what is \
at stake, and the recurring arc in the coverage. Reference real names, \
real organizations, real dates from the briefings.
- **Key Sources & Players** — write about the recurring people and \
organizations in connected sentences; do not reduce them to a \
bulleted roster.
- **Story Ideas & Angles** — a short numbered list, introduced by a \
sentence or two framing the gap each idea addresses.
- **Background & Context** — flowing prose of history, policy, \
institutional knowledge a new reporter would need.
- **Reporting Tips** — a few sentences of practical advice specific \
to this beat. Not generic journalism advice.
- **Calendar & Recurring Events** — a list IS appropriate here \
because items are genuinely list-shaped (meetings, deadlines).

Writing rules:
- Connected prose, not outlines. Use bullets only for genuinely \
list-shaped content (named sources, story ideas, calendar).
- Reference actual names, dates, and details from the briefings. \
Do not invent facts.
- Do NOT include a table of contents. The viewer renders its own.
- Do NOT add per-topic sub-headers like \"What's happening / Key \
story / Story angles\" — let prose carry the structure.

Use the focus_notes to calibrate emphasis. If the reporter said they \
want X, lead with X.
"""


def _format_interview_block(interview_log: list[dict]) -> str:
    if not interview_log:
        return "(The reporter did not answer any interview questions.)"

    parts: list[str] = []
    for round_idx, item in enumerate(interview_log, 1):
        if item.get("intro"):
            parts.append(f"**Round {round_idx} intro:** {item['intro']}")
        for a in item.get("answers") or []:
            q = (a.get("question") or "").strip()
            ans = a.get("answer", "")
            if isinstance(ans, list):
                ans = ", ".join(str(x) for x in ans) if ans else "(no answer)"
            ans = str(ans).strip() or "(no answer)"
            parts.append(f"- Q: {q}\n  A: {ans}")
    return "\n".join(parts)


async def _generate_final_beat_book(
    client,
    pipeline_result: PipelineResult,
    beat_selection: list[str],
    focus_notes: str,
    interview_log: list[dict],
) -> str:
    """Single Sonnet 4.6 call that writes the final Markdown.

    The briefings sit on the user side (not system) because they're
    session-specific; the static writing rules live in the system prompt.
    """
    briefings_md = render_briefings_markdown(pipeline_result.briefings or {})
    interview_md = _format_interview_block(interview_log)
    selection_md = ", ".join(beat_selection) if beat_selection else "(no specific selection — cover the whole corpus)"

    user_msg = (
        "# Beat scope\n\n"
        f"The reporter has identified their beat as: **{selection_md}**.\n\n"
        "# Focus notes from the planning agent\n\n"
        f"{focus_notes or '(none)'}\n\n"
        "# Reporter's interview answers\n\n"
        f"{interview_md}\n\n"
        "# Topic briefings (full corpus knowledge)\n\n"
        f"{briefings_md}\n\n"
        "Now write the complete beat book in flowing Markdown. Begin with "
        "the title and a one-line subtitle, then dive into the Beat Overview."
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model=WRITER_MODEL,
        max_tokens=DRAFT_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": _WRITER_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_msg}],
        thinking=thinking_param(),
    )
    return "".join(
        b.text for b in response.content
        if getattr(b, "type", None) == "text"
    ).strip()


# ────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ────────────────────────────────────────────────────────────────────────

InterviewCallback  = Callable[[dict], Awaitable[str]]
MessageCallback    = Callable[[str], Awaitable[None]]
ToolStatusCallback = Callable[[str, str, str], Awaitable[None]]
BeatBookCallback   = Callable[[str, str], Awaitable[None]]


TOOL_DESCRIPTIONS = {
    "view_topics":          "Reviewing discovered topics",
    "list_stories_in_topic": "Listing stories in topic",
    "read_story":           "Reading a story",
    "search_stories":       "Searching stories",
    "interview_user":       "Asking you a question",
    "generate_beat_book":   "Writing the beat book",
}


async def run_agent(
    pipeline_result: PipelineResult,
    anthropic_key: str,
    interview_log: list[dict],
    on_interview: InterviewCallback,
    on_message: MessageCallback,
    on_beat_book: BeatBookCallback,
    on_tool_status: Optional[ToolStatusCallback] = None,
) -> None:
    """Run the planning loop, then synthesize the final draft via Sonnet.

    Args:
        pipeline_result: PipelineResult (must include .briefings).
        anthropic_key: Anthropic API key.
        interview_log: session-scoped list — the on_interview callback in
            app.py appends one entry per round (intro / questions / answers).
            The writer call reads this list to render the reporter's answers
            into the prompt.
        on_interview: async callback(question_data) -> user's answer string.
        on_message: async callback(text) — agent narration to the frontend.
        on_beat_book: async callback(filename, markdown) — final delivery.
        on_tool_status: async callback(tool_name, desc, detail) — status pings.
    """
    client = chat_client(anthropic_key)

    n_stories = len(pipeline_result.stories)
    n_topics  = len(pipeline_result.broad_topics)

    system_blocks = _build_system_blocks(pipeline_result)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"{n_stories} stories across {n_topics} broad topics have "
                "been uploaded and analyzed. The topic briefings are in your "
                "system prompt. Please run the brief interview now, then "
                "trigger the beat book."
            ),
        },
    ]

    last_message_text = ""
    beat_book_done = False
    thinking_cfg = thinking_param()

    for turn in range(MAX_TURNS):
        response = None
        for rl_attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=AGENT_MODEL,
                    max_tokens=LOOP_MAX_TOKENS,
                    system=system_blocks,
                    tools=TOOLS,
                    messages=messages,
                    thinking=thinking_cfg,
                )
                break
            except anthropic.RateLimitError:
                if rl_attempt >= RATE_LIMIT_MAX_RETRIES:
                    await on_message(
                        "Anthropic's concurrent-connection rate limit is "
                        "persistently exceeded — please wait a minute and "
                        "start a new session."
                    )
                    return
                await on_message(
                    f"Hit Anthropic's concurrent-connection rate limit. "
                    f"Waiting {RATE_LIMIT_PAUSE_SECONDS}s and retrying "
                    f"({rl_attempt + 1}/{RATE_LIMIT_MAX_RETRIES})..."
                )
                await asyncio.sleep(RATE_LIMIT_PAUSE_SECONDS)
        assert response is not None

        text_combined = "".join(
            b.text for b in response.content
            if getattr(b, "type", None) == "text"
        ).strip()
        if text_combined and text_combined != last_message_text:
            await on_message(text_combined)
            last_message_text = text_combined

        stop_reason = response.stop_reason

        if stop_reason != "tool_use":
            if stop_reason == "max_tokens":
                await on_message(
                    "The model hit its output limit mid-turn. The beat "
                    "book may be incomplete. Please try again."
                )
            break
        if beat_book_done:
            break

        # Preserve assistant content verbatim (text + thinking + tool_use).
        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            tool_name  = block.name
            tool_id    = block.id
            tool_input = block.input or {}

            if on_tool_status:
                desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name)
                detail = ""
                if tool_name == "list_stories_in_topic":
                    detail = tool_input.get("topic", "")
                elif tool_name == "read_story":
                    idx = tool_input.get("index")
                    story = pipeline_result.get_story(idx) if isinstance(idx, int) else None
                    detail = (story.get("title", f"#{idx}")[:60] if story else f"#{idx}")
                elif tool_name == "search_stories":
                    detail = tool_input.get("query", "")
                await on_tool_status(tool_name, desc, detail)

            if tool_name == "interview_user":
                # app.py's on_interview appends to the shared interview_log
                # before returning the human-readable answer string.
                answer = await on_interview(tool_input)
                content_str = f"Reporter's answer: {answer}"

            elif tool_name == "generate_beat_book":
                # No more gating: trigger the Sonnet writer pass immediately.
                filename = (tool_input.get("filename") or "beat_book.md").strip()
                if not filename.endswith(".md"):
                    filename = filename + ".md"
                beat_selection = tool_input.get("beat_selection") or []
                focus_notes    = tool_input.get("focus_notes") or ""

                await on_message("Handing off to the writer model...")
                try:
                    markdown = await _generate_final_beat_book(
                        client, pipeline_result,
                        beat_selection, focus_notes,
                        interview_log,
                    )
                except Exception as e:
                    content_str = (
                        f"Writer model failed: {type(e).__name__}: {e}. "
                        "Try generate_beat_book again."
                    )
                else:
                    if not markdown:
                        content_str = (
                            "Writer model returned an empty document. Try "
                            "generate_beat_book again with clearer focus_notes."
                        )
                    else:
                        await on_beat_book(filename, markdown)
                        content_str = "Beat book saved successfully. You may give a brief closing message."
                        beat_book_done = True

            else:
                content_str = execute_local_tool(tool_name, tool_input, pipeline_result)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tool_id,
                "content":     content_str,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    if not beat_book_done:
        await on_message("Agent session ended without saving a beat book.")
