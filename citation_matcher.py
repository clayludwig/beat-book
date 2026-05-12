"""
citation_matcher.py
-------------------
Sentence-level citation matcher. Takes a markdown beat book and a set of source
stories, embeds each sentence with OpenAI `text-embedding-3-small`, and for
each sentence in the beat book finds the most similar source sentence by
cosine similarity.

Public entry points:
- embed_source_stories(stories, openai_key, on_progress) -> list[dict]
- markdown_to_beatbook_entries(markdown, source_embeddings, openai_key, on_progress) -> list[dict]
- build_sources_file(stories, source_embeddings) -> list[dict]
"""

import math
import re
from typing import Callable, List, Dict, Any, Optional

from openai import OpenAI

# Progress callback signature: (stage_label, fraction_0_to_1, detail)
ProgressCallback = Callable[[str, float, str], None]

# Anthropic doesn't host an embedding API, so embeddings stay on OpenAI.
EMBED_MODEL = "text-embedding-3-small"
# The API accepts up to 2048 inputs per request, but we cap lower to keep
# individual HTTP payloads reasonable.
EMBED_BATCH_SIZE = 256


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTER (ported from generate_story_embeddings.py)
# ─────────────────────────────────────────────────────────────────────────────

_ABBREVIATIONS = [
    (r"\bMr\.", "Mr<<DOT>>"),
    (r"\bMrs\.", "Mrs<<DOT>>"),
    (r"\bMs\.", "Ms<<DOT>>"),
    (r"\bDr\.", "Dr<<DOT>>"),
    (r"\bProf\.", "Prof<<DOT>>"),
    (r"\bSr\.", "Sr<<DOT>>"),
    (r"\bJr\.", "Jr<<DOT>>"),
    (r"\bvs\.", "vs<<DOT>>"),
    (r"\betc\.", "etc<<DOT>>"),
    (r"\bInc\.", "Inc<<DOT>>"),
    (r"\bLtd\.", "Ltd<<DOT>>"),
    (r"\bCo\.", "Co<<DOT>>"),
    (r"\bCorp\.", "Corp<<DOT>>"),
    (r"\bSt\.", "St<<DOT>>"),
    (r"\bAve\.", "Ave<<DOT>>"),
    (r"\bBlvd\.", "Blvd<<DOT>>"),
    (r"\bRd\.", "Rd<<DOT>>"),
    (r"\bPh\.D\.", "Ph<<DOT>>D<<DOT>>"),
    (r"\bM\.D\.", "M<<DOT>>D<<DOT>>"),
    (r"\bB\.A\.", "B<<DOT>>A<<DOT>>"),
    (r"\bB\.S\.", "B<<DOT>>S<<DOT>>"),
    (r"\bM\.A\.", "M<<DOT>>A<<DOT>>"),
    (r"\bM\.S\.", "M<<DOT>>S<<DOT>>"),
    (r"\bU\.S\.", "U<<DOT>>S<<DOT>>"),
    (r"\bU\.K\.", "U<<DOT>>K<<DOT>>"),
    (r"\bD\.C\.", "D<<DOT>>C<<DOT>>"),
    (r"\ba\.m\.", "a<<DOT>>m<<DOT>>"),
    (r"\bp\.m\.", "p<<DOT>>m<<DOT>>"),
    (r"\bNo\.", "No<<DOT>>"),
    (r"\bVol\.", "Vol<<DOT>>"),
    (r"\bGen\.", "Gen<<DOT>>"),
    (r"\bSgt\.", "Sgt<<DOT>>"),
    (r"\bLt\.", "Lt<<DOT>>"),
    (r"\bCapt\.", "Capt<<DOT>>"),
    (r"\bCol\.", "Col<<DOT>>"),
    (r"\bRev\.", "Rev<<DOT>>"),
    (r"\bSen\.", "Sen<<DOT>>"),
    (r"\bRep\.", "Rep<<DOT>>"),
    (r"\bGov\.", "Gov<<DOT>>"),
]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving common abbreviations."""
    if not text or not text.strip():
        return []

    text = re.sub(r"\s+", " ", text).strip()

    protected = text
    for pattern, replacement in _ABBREVIATIONS:
        protected = re.sub(pattern, replacement, protected, flags=re.IGNORECASE)

    # Protect decimals (3.5, $10.99)
    protected = re.sub(r"(\d)\.(\d)", r"\1<<DOT>>\2", protected)

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"']|$)", protected)
    sentences = [s.replace("<<DOT>>", ".").strip() for s in sentences]
    return [s for s in sentences if s and len(s) > 10]


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN SEGMENTATION (ported from md_to_beat_book_format.py)
# ─────────────────────────────────────────────────────────────────────────────

def _is_markdown_heading(line: str) -> bool:
    return re.match(r"^#{1,6}\s+", line.strip()) is not None


def _is_markdown_list_item(line: str) -> bool:
    return bool(
        re.match(r"^\s*[-*+]\s+", line.strip())
        or re.match(r"^\s*\d+\.\s+", line.strip())
    )


def _is_markdown_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def _is_code_block_delimiter(line: str) -> bool:
    return line.strip().startswith("```")


def _segment_markdown(markdown: str) -> List[Dict[str, Any]]:
    """Break markdown into a sequence of entries. Sentences that live inside
    paragraphs get `needs_embedding=True`; headings, list items, table rows,
    blank lines, and code blocks get passed through untouched."""
    entries: List[Dict[str, Any]] = []
    in_code_block = False

    for line in markdown.split("\n"):
        if _is_code_block_delimiter(line):
            in_code_block = not in_code_block
            entries.append({"content": line, "needs_embedding": False})
            continue

        if in_code_block:
            entries.append({"content": line, "needs_embedding": False})
            continue

        if (
            not line.strip()
            or _is_markdown_heading(line)
            or _is_markdown_list_item(line)
            or _is_markdown_table_row(line)
        ):
            entries.append({"content": line, "needs_embedding": False})
            continue

        sentences = split_into_sentences(line)
        if sentences:
            for sentence in sentences:
                entries.append({"content": sentence, "needs_embedding": True})
        else:
            entries.append({"content": line, "needs_embedding": False})

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS (OpenAI batch)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Embed a list of texts in one API call."""
    if not texts:
        return []
    # OpenAI treats empty strings as an error; replace with a single space.
    cleaned = [t if t.strip() else " " for t in texts]
    resp = client.embeddings.create(model=EMBED_MODEL, input=cleaned)
    return [item.embedding for item in resp.data]


def _embed_many(
    client: OpenAI,
    texts: List[str],
    on_progress: Optional[ProgressCallback],
    stage: str,
) -> List[List[float]]:
    """Embed an arbitrary number of texts, batched, with progress reporting."""
    out: List[List[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        chunk = texts[start : start + EMBED_BATCH_SIZE]
        out.extend(_embed_batch(client, chunk))
        if on_progress:
            done = min(start + len(chunk), total)
            on_progress(stage, done / total if total else 1.0, f"{done}/{total} sentences")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE-STORY EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def embed_source_stories(
    stories: List[dict],
    openai_key: str,
    on_progress: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """For each story, split its content into sentences and embed every sentence.

    Returns a list of article dicts in the shape the matcher expects:
        {
          "article_id": str,
          "title": str,
          "date": str,
          "author": str,
          "sentences": [{"text": str, "index": int, "embedding": list[float]}, ...]
        }

    Stories get a synthetic `article_id = "story-{idx}"` if they don't already
    have one.
    """
    client = OpenAI(api_key=openai_key)

    # Prepare per-story sentence lists, then embed all sentences in one batched
    # run so the number of HTTP round-trips is minimized.
    per_story_sentences: List[List[str]] = []
    flat_texts: List[str] = []
    for story in stories:
        sentences = split_into_sentences(story.get("content", ""))
        per_story_sentences.append(sentences)
        flat_texts.extend(sentences)

    if on_progress:
        on_progress("embedding_sources", 0.0, f"{len(flat_texts)} sentences across {len(stories)} stories")

    flat_embeddings = _embed_many(client, flat_texts, on_progress, "embedding_sources")

    # Re-group embeddings back under each story.
    result: List[Dict[str, Any]] = []
    cursor = 0
    for idx, story in enumerate(stories):
        sentences = per_story_sentences[idx]
        article_id = story.get("article_id") or story.get("id") or f"story-{idx}"
        entry: Dict[str, Any] = {
            "article_id": article_id,
            "title": story.get("title", ""),
            "date": story.get("date", ""),
            "author": story.get("author", ""),
            "sentences": [],
        }
        for sent_idx, sentence_text in enumerate(sentences):
            entry["sentences"].append(
                {
                    "text": sentence_text,
                    "index": sent_idx,
                    "embedding": flat_embeddings[cursor],
                }
            )
            cursor += 1
        result.append(entry)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        mag_a += x * x
        mag_b += y * y
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))


def _find_best_match(embedding: List[float], source_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = {
        "article_id": None,
        "sentence_text": "",
        "sentence_index": -1,
        "similarity": -1.0,
        "article_title": "",
        "article_date": "",
        "article_author": "",
    }
    for article in source_embeddings:
        aid = article.get("article_id", "")
        for sentence_data in article.get("sentences", []):
            emb = sentence_data.get("embedding")
            if emb is None:
                continue
            sim = _cosine_similarity(embedding, emb)
            if sim > best["similarity"]:
                best = {
                    "article_id": aid,
                    "sentence_text": sentence_data.get("text", ""),
                    "sentence_index": sentence_data.get("index", 0),
                    "similarity": sim,
                    "article_title": article.get("title", ""),
                    "article_date": article.get("date", ""),
                    "article_author": article.get("author", ""),
                }
    return best


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN → BEAT BOOK JSON
# ─────────────────────────────────────────────────────────────────────────────

def markdown_to_beatbook_entries(
    markdown: str,
    source_embeddings: List[Dict[str, Any]],
    openai_key: str,
    on_progress: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """Convert a Markdown beat book into the Talbot-style JSON entry list.

    Each entry has: content, source, source_sentence, source_sentence_index,
    source_title, similarity.
    """
    client = OpenAI(api_key=openai_key)

    entries = _segment_markdown(markdown)
    to_embed_indices = [i for i, e in enumerate(entries) if e["needs_embedding"] and e["content"].strip()]
    to_embed_texts = [entries[i]["content"] for i in to_embed_indices]

    if on_progress:
        on_progress("embedding_beatbook", 0.0, f"{len(to_embed_texts)} sentences to match")

    embeddings = _embed_many(client, to_embed_texts, on_progress, "embedding_beatbook")

    # Walk through entries and assemble the output.
    out: List[Dict[str, Any]] = []
    embed_iter = iter(zip(to_embed_indices, embeddings))
    next_embed = next(embed_iter, None)

    total_to_match = len(to_embed_indices)
    matched = 0

    for i, entry in enumerate(entries):
        if next_embed is not None and next_embed[0] == i:
            _, emb = next_embed
            match = _find_best_match(emb, source_embeddings)
            out.append(
                {
                    "content": entry["content"],
                    "source": match["article_id"] or "",
                    "source_sentence": match["sentence_text"] or "",
                    "source_sentence_index": match["sentence_index"] if match["sentence_index"] is not None else -1,
                    "source_title": match["article_title"],
                    "similarity": round(match["similarity"], 4),
                }
            )
            next_embed = next(embed_iter, None)
            matched += 1
            if on_progress and (matched % 25 == 0 or matched == total_to_match):
                on_progress("matching", matched / total_to_match if total_to_match else 1.0, f"{matched}/{total_to_match}")
        else:
            out.append(
                {
                    "content": entry["content"],
                    "source": "",
                    "source_sentence": "",
                    "source_sentence_index": -1,
                    "source_title": "",
                    "similarity": 0.0,
                }
            )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# SOURCES FILE (what the viewer loads to render article panels)
# ─────────────────────────────────────────────────────────────────────────────

def build_sources_file(
    stories: List[dict],
    source_embeddings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Produce the `*_sources.json` list the viewer reads. One entry per story,
    with the same `article_id` scheme used during embedding."""
    out: List[Dict[str, Any]] = []
    for idx, story in enumerate(stories):
        article_id = source_embeddings[idx]["article_id"] if idx < len(source_embeddings) else f"story-{idx}"
        out.append(
            {
                "article_id": article_id,
                "title": story.get("title", ""),
                "date": story.get("date", ""),
                "author": story.get("author", ""),
                "content": story.get("content", ""),
                "link": story.get("link", ""),
            }
        )
    return out
