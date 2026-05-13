"""
citation_matcher.py
-------------------
Sentence-level citation matcher. Takes a markdown beat book and a set of source
stories, embeds each sentence with OpenAI `text-embedding-3-small`, and for
each sentence in the beat book finds the most similar source sentence by
cosine similarity.

Matching is fully vectorized: source-sentence embeddings stack into one
(S, 1536) matrix, beat-book sentence embeddings stack into (B, 1536), and a
single `B @ S.T` matrix multiply replaces the pre-numpy nested Python loop.

Public entry points:
- embed_source_stories(stories, openai_key, on_progress) -> list[dict]
- markdown_to_beatbook_entries(markdown, source_embeddings, openai_key, on_progress) -> list[dict]
- build_sources_file(stories, source_embeddings) -> list[dict]
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
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

def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. Zero rows pass through (divisor floor of 1)."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _empty_match() -> Dict[str, Any]:
    return {
        "article_id":      None,
        "sentence_text":   "",
        "sentence_index":  -1,
        "similarity":      -1.0,
        "article_title":   "",
        "article_date":    "",
        "article_author":  "",
    }


def _build_source_matrix(
    source_embeddings: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Stack every source-sentence embedding into one (S, D) matrix and
    return a parallel metadata list of length S. Rows are L2-normalized
    once here so the per-query call is a pure dot product."""
    flat: List[List[float]] = []
    meta: List[Dict[str, Any]] = []
    for article in source_embeddings:
        aid    = article.get("article_id", "")
        title  = article.get("title", "")
        date   = article.get("date", "")
        author = article.get("author", "")
        for sent in article.get("sentences") or []:
            emb = sent.get("embedding")
            if emb is None:
                continue
            flat.append(emb)
            meta.append({
                "article_id":      aid,
                "sentence_text":   sent.get("text", ""),
                "sentence_index":  sent.get("index", 0),
                "article_title":   title,
                "article_date":    date,
                "article_author":  author,
            })
    if not flat:
        return np.zeros((0, 0), dtype=np.float32), []
    matrix = np.asarray(flat, dtype=np.float32)
    return _l2_normalize(matrix), meta


def _find_best_matches_batch(
    query_embs: np.ndarray,
    source_matrix: np.ndarray,
    source_meta: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Best-match every row of `query_embs` against the source matrix in
    one BLAS call. Returns a list parallel to query_embs.

    Both matrices are assumed to share an embedding dimension; the source
    matrix is already L2-normalized (see _build_source_matrix), the
    queries are normalized here.
    """
    if query_embs.size == 0:
        return []
    if source_matrix.size == 0:
        return [_empty_match() for _ in range(query_embs.shape[0])]

    Q = _l2_normalize(query_embs.astype(np.float32))
    sims = Q @ source_matrix.T            # (B, S), cosine since both normalized
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(sims.shape[0]), best_idx]

    out: List[Dict[str, Any]] = []
    for b in range(sims.shape[0]):
        meta = source_meta[int(best_idx[b])]
        out.append({
            "article_id":      meta["article_id"],
            "sentence_text":   meta["sentence_text"],
            "sentence_index":  meta["sentence_index"],
            "similarity":      float(best_sim[b]),
            "article_title":   meta["article_title"],
            "article_date":    meta["article_date"],
            "article_author":  meta["article_author"],
        })
    return out


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

    Matching is vectorized: every beat-book sentence's embedding is dotted
    against the stacked, L2-normalized source matrix in a single BLAS call.
    """
    client = OpenAI(api_key=openai_key)

    entries = _segment_markdown(markdown)
    to_embed_indices = [i for i, e in enumerate(entries) if e["needs_embedding"] and e["content"].strip()]
    to_embed_texts = [entries[i]["content"] for i in to_embed_indices]
    total_to_match = len(to_embed_indices)

    if on_progress:
        on_progress("embedding_beatbook", 0.0, f"{total_to_match} sentences to match")

    raw_embeddings = _embed_many(client, to_embed_texts, on_progress, "embedding_beatbook")

    if on_progress:
        on_progress("matching", 0.0, "Building source matrix...")
    source_matrix, source_meta = _build_source_matrix(source_embeddings)

    if raw_embeddings:
        query_matrix = np.asarray(raw_embeddings, dtype=np.float32)
    else:
        query_matrix = np.zeros((0, 0), dtype=np.float32)

    matches = _find_best_matches_batch(query_matrix, source_matrix, source_meta)
    if on_progress:
        on_progress("matching", 1.0, f"{total_to_match}/{total_to_match}")

    # Reassemble entries in original order, splicing match metadata into
    # the rows that needed embedding.
    out: List[Dict[str, Any]] = []
    match_by_entry_idx = dict(zip(to_embed_indices, matches))
    for i, entry in enumerate(entries):
        m = match_by_entry_idx.get(i)
        if m is None:
            out.append({
                "content":                entry["content"],
                "source":                 "",
                "source_sentence":        "",
                "source_sentence_index":  -1,
                "source_title":           "",
                "similarity":             0.0,
            })
            continue
        out.append({
            "content":                entry["content"],
            "source":                 m["article_id"] or "",
            "source_sentence":        m["sentence_text"] or "",
            "source_sentence_index":  m["sentence_index"] if m["sentence_index"] is not None else -1,
            "source_title":           m["article_title"],
            "similarity":             round(m["similarity"], 4),
        })

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
