"""
agent.py
--------
Ollama-powered agent (qwen3.5:397b-cloud) that has tools to explore stories/
topics and interview the reporter, then produces a beat book.

The agent runs in an async loop.  Most tools execute locally, but
`interview_user` pauses execution and sends the question to the frontend
via a callback.  The callback returns the user's answer so the loop can
resume.
"""

import json
from typing import Callable, Awaitable, Any

from pipeline import PipelineResult
from ollama_client import CHAT_MODEL, chat_client

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

AGENT_MODEL = CHAT_MODEL

# Generous output cap — beat books can be long. Qwen3.5 supports a 256K total
# context so this leaves plenty of room for the conversation history.
MAX_TOKENS_PER_TURN = 32768

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS (OpenAI tool-use schema, used by Ollama's OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "view_topics",
            "description": (
                "View all discovered topics from the uploaded stories. Returns broad "
                "and specific topics with story counts. Use this first to understand "
                "the landscape of coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stories_in_topic",
            "description": (
                "List all stories that belong to a given topic. Returns story index, "
                "title, and date for each."
            ),
            "parameters": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "read_story",
            "description": (
                "Read the full content of a story by its index number. Returns title, "
                "author, date, and full text."
            ),
            "parameters": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_stories",
            "description": (
                "Search stories by keyword. Returns matching story indices, titles, "
                "and dates."
            ),
            "parameters": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "interview_user",
            "description": (
                "Ask the reporter a BATCH of interview questions in a single form. "
                "The reporter fills out all questions at once and submits. Strongly "
                "prefer asking 3–6 related questions in one call over many separate "
                "calls — it gives the reporter context for what you're trying to "
                "learn and avoids a tedious back-and-forth. Only call this tool a "
                "second time if a follow-up is genuinely necessary based on their "
                "first answers.\n\n"
                "Each question supports one of four input types:\n"
                "- 'checklist': multi-select checkboxes (use for 'select all that apply')\n"
                "- 'single_choice': radio buttons — pick exactly one\n"
                "- 'multiple_choice': checkboxes — pick one or more\n"
                "- 'free_response': open text input\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intro": {
                        "type": "string",
                        "description": "Optional 1–2 sentence intro shown above the form to frame what you're asking and why.",
                    },
                    "questions": {
                        "type": "array",
                        "description": "Ordered list of questions to present in one form (3–6 recommended).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The question text.",
                                },
                                "question_type": {
                                    "type": "string",
                                    "enum": ["checklist", "single_choice", "multiple_choice", "free_response"],
                                },
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Options for checklist / single_choice / multiple_choice. Leave empty for free_response.",
                                },
                            },
                            "required": ["question", "question_type"],
                        },
                    },
                },
                "required": ["questions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_beat_book",
            "description": (
                "Write the final beat book as a Markdown document. Call this once you "
                "have gathered enough information from the topics, stories, and the "
                "reporter's answers. The content you pass will be saved as the output file."
            ),
            "parameters": {
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
1. **Explore** — Start by using `view_topics` to see what topics exist in the \
stories. Read a few representative stories to understand the coverage.
2. **Interview** — Call `interview_user` ONCE with a batch of 4–6 questions. \
Include a short `intro` framing what you're trying to learn. Start the batch \
with a checklist of the discovered topics so they can select which form their \
beat, then add questions about audience, experience level, and what they need \
most from the guide. Only call `interview_user` a second time if a follow-up \
is truly essential based on their answers.
3. **Research thoroughly** — This is the longest step, and the quality of \
the beat book is directly proportional to how much of the corpus you have \
actually read. For each topic the reporter selected, call \
`list_stories_in_topic` first to see what is there, then read a substantial \
sample. At minimum, read half the stories in every selected topic; when a \
topic has fewer than 15 stories, read every one. Use `search_stories` to \
surface specific people, institutions, and themes once you start noticing \
patterns. Every `list_stories_in_topic` / `read_story` / `search_stories` \
tool result includes a `[Research progress]` block that tells you, per \
topic, how many stories you've read out of the target — use it to track \
where you still need depth. The system will refuse `generate_beat_book` \
calls until every listed topic has met its target, so do not bother trying \
to short-circuit; just keep reading.
4. **Generate** — Once every listed topic shows "OK" in the progress block, \
call `generate_beat_book` with a polished Markdown document. You have a \
40-turn budget — using most of it on research is the expected behavior, \
not a problem.

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
            "content": story.get("content", "")[:3000],
        }, indent=2)

    if name == "search_stories":
        matches = result.search_stories(input_data["query"])
        if not matches:
            return f"No stories matching '{input_data['query']}'."
        return json.dumps(matches, indent=2)

    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

# Type for the callback that sends questions to the frontend and gets answers
InterviewCallback = Callable[[dict], Awaitable[str]]
# Type for the callback that sends agent text messages to the frontend
MessageCallback   = Callable[[str], Awaitable[None]]
# Type for the callback that reports tool execution status
ToolStatusCallback = Callable[[str, str, str], Awaitable[None]]


# Human-friendly descriptions for each tool
TOOL_DESCRIPTIONS = {
    "view_topics": "Reviewing discovered topics",
    "list_stories_in_topic": "Listing stories in topic",
    "read_story": "Reading a story",
    "search_stories": "Searching stories",
    "interview_user": "Asking you a question",
    "generate_beat_book": "Writing the beat book",
}


def _assistant_message_dict(msg) -> dict:
    """Convert an OpenAI ChatCompletionMessage into a dict the next request
    can use. Preserves tool_calls verbatim so the API accepts the next turn."""
    out: dict = {"role": "assistant"}
    # content must be present even if null when tool_calls are set
    out["content"] = msg.content if msg.content else None
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


async def run_agent(
    pipeline_result: PipelineResult,
    ollama_key: str,
    on_interview: InterviewCallback,
    on_message: MessageCallback,
    on_beat_book: Callable[[str, str], Awaitable[None]],
    on_tool_status: ToolStatusCallback = None,
) -> None:
    """
    Run the agent loop.

    Args:
        pipeline_result: Output from the embedding/clustering pipeline.
        ollama_key: Ollama Cloud API key (qwen3.5:397b-cloud).
        on_interview: async callback(question_data) → user's answer string.
        on_message: async callback(text) — sends agent text to the frontend.
        on_beat_book: async callback(filename, markdown) — saves/delivers the beat book.
        on_tool_status: async callback(tool_name, detail) — reports tool execution status.
    """
    client = chat_client(ollama_key)

    n_stories = len(pipeline_result.stories)
    n_topics  = len(pipeline_result.topics)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"I've uploaded {n_stories} news stories. The system has automatically "
                f"discovered {n_topics} topics across them. Please help me build a "
                "beat book from these stories. Start by exploring the topics, then "
                "interview me to understand my beat."
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
    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            max_tokens=MAX_TOKENS_PER_TURN,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason

        # Stream the assistant's text content to the frontend (dedup repeats)
        if msg.content and msg.content.strip():
            text = msg.content.strip()
            if text != last_message_text:
                await on_message(msg.content)
                last_message_text = text

        # No tool calls = end of conversation, OR beat book already saved
        if not msg.tool_calls:
            break
        if beat_book_done:
            break

        # If the model was cut off mid tool-call (length limit), the arguments
        # JSON is almost certainly malformed and the API will reject the
        # message if we include it. Surface the error and stop instead of
        # quietly looping.
        if finish_reason == "length":
            await on_message(
                "⚠️ The model hit its output limit mid-tool-call. The beat book "
                "may be incomplete. Please try again with a tighter scope."
            )
            break

        # Append assistant turn (with tool_calls preserved) so the API can
        # correlate the upcoming tool result messages.
        messages.append(_assistant_message_dict(msg))

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_id = tc.id
            try:
                tool_input = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}

            # Report tool status to the frontend
            if on_tool_status:
                desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name)
                detail = ""
                if tool_name == "list_stories_in_topic":
                    detail = tool_input.get("topic", "")
                elif tool_name == "read_story":
                    idx = tool_input.get("index", "")
                    story = pipeline_result.get_story(idx) if isinstance(idx, int) else None
                    detail = story.get("title", f"#{idx}")[:60] if story else f"#{idx}"
                elif tool_name == "search_stories":
                    detail = tool_input.get("query", "")
                await on_tool_status(tool_name, desc, detail)

            if tool_name == "interview_user":
                answer = await on_interview(tool_input)
                content_str = f"Reporter's answer: {answer}"

            elif tool_name == "generate_beat_book":
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
                elif tool_name == "read_story":
                    idx = tool_input.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(pipeline_result.stories):
                        read_indices.add(idx)
                progress, _ = _progress_report(
                    pipeline_result, listed_topics, read_indices,
                )
                content_str = f"{content_str}\n\n{progress}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": content_str,
            })

    if not beat_book_done:
        await on_message("✅ Agent session complete.")
