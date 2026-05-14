"""
citation_matcher.py
-------------------
Inline citation matcher. Takes a Markdown beat book and a set of source stories
and produces, for each sentence in the beat book, up to K supporting passages
drawn from the sources — each with the specific sub-span that carries the
claim, a per-corpus-calibrated similarity threshold, and metadata for the
viewer to render an inline citation.

Pipeline
~~~~~~~~
1. Chunk each source story into ~100-word sliding windows (16-word overlap),
   keeping the char offset back into the original source so we can resolve a
   passage hit to a quoted span later.
2. Embed every source passage with OpenAI `text-embedding-3-small`. Embed every
   sentence in the beat book the same way, plus a "context-sum" variant that
   adds weighted neighbors (`0.6 * prev + 1.0 * self + 0.4 * next`) — helps
   pronoun-heavy / short sentences attribute correctly.
3. L2-normalize both matrices, multiply once (`B @ S.T`) to get the full
   similarity matrix. Pick the top-K passages per beat-book sentence above a
   per-corpus-calibrated threshold (noise-floor + 3·sigma, with an absolute
   floor of 0.40).
4. For every kept (sentence, passage) pair, run a leave-one-out attribution
   pass over the passage: split it into N overlapping sub-windows, embed
   `passage_minus_subwindow` for each, and rank sub-windows by how much
   removing them hurts similarity. The top 1–2 sub-windows become highlight
   offsets.

Public entry points
~~~~~~~~~~~~~~~~~~~
- `embed_source_stories(stories, openai_key, on_progress)` → source index
- `markdown_to_beatbook_entries(markdown, source_index, openai_key,
   on_progress)` → list of citation entries (one per Markdown line/sentence)
- `build_sources_file(stories, source_index)` → sources JSON for the viewer

The output JSON shape is documented in the docstring of
`markdown_to_beatbook_entries`.
"""

from __future__ import annotations

import concurrent.futures
import re
from typing import Callable, List, Dict, Any, Optional, Tuple

import numpy as np
from openai import OpenAI

# Progress callback signature: (stage_label, fraction_0_to_1, detail)
ProgressCallback = Callable[[str, float, str], None]

# Anthropic doesn't host an embedding API, so embeddings stay on OpenAI.
EMBED_MODEL = "text-embedding-3-small"
# The API accepts up to 2048 inputs per request, but we cap lower to keep
# individual HTTP payloads reasonable.
EMBED_BATCH_SIZE = 256
# Parallel HTTP workers when there are multiple batches to send. OpenAI's
# embedding RPM ceiling for paid accounts comfortably accommodates 6 in-flight
# requests per second; if you see 429s, drop this.
EMBED_PARALLEL_WORKERS = 6

# Passage chunking (source side). 100 words with 16-word overlap maps to
# roughly 128-token windows with ~12% overlap — close to semantra's default.
PASSAGE_WORDS = 100
PASSAGE_OVERLAP_WORDS = 16

# Top-K candidate passages kept per beat-book sentence.
TOP_K = 5

# Calibration parameters. We compute a per-corpus threshold by sampling random
# (beat_book_sentence, source_passage) pairs and taking `noise_mean + N·sigma`,
# clamped to an absolute floor so we don't accept obvious garbage when the
# corpus happens to have a high noise level.
CALIB_RANDOM_SAMPLES = 4000
CALIB_SIGMA = 3.0
CALIB_ABSOLUTE_FLOOR = 0.40

# Context-sum weights for the beat-book side (handles pronoun / short
# sentences). 0.6·prev + 1.0·self + 0.4·next, then re-normalized at the matmul.
CTX_WEIGHT_PREV = 0.6
CTX_WEIGHT_SELF = 1.0
CTX_WEIGHT_NEXT = 0.4

# Leave-one-out sub-window highlight parameters.
LOO_SUBWINDOWS_PER_PASSAGE = 6   # number of overlapping sub-windows
LOO_TOP_HIGHLIGHTS = 2           # how many sub-windows to surface in the UI


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTER (beat-book side)
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

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", protected)
    sentences = [s.replace("<<DOT>>", ".").strip() for s in sentences]
    return [s for s in sentences if s and len(s) > 10]


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN SEGMENTATION
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
    """Break markdown into a sequence of entries. Sentences inside paragraphs
    get `needs_embedding=True`; headings, list items, table rows, blank lines,
    and code blocks pass through untouched (`needs_embedding=False`)."""
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
# WORD-LEVEL TOKENIZER (used for passage windows + leave-one-out)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each "token" here is a whitespace-bounded word. We track each word's start
# and end character offset in the source string so a (start_word, end_word)
# span can be resolved back to a `(char_offset, char_length)` quote.

def _tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Return [(word, char_start, char_end), ...] for every whitespace-bounded
    word in `text`. Punctuation stays attached to the adjacent word."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _passage_windows(
    text: str,
    window_words: int = PASSAGE_WORDS,
    overlap_words: int = PASSAGE_OVERLAP_WORDS,
) -> List[Dict[str, Any]]:
    """Slide a (window_words, overlap_words) window across the text.

    Returns a list of:
        {
          "text": str,            # the passage substring
          "char_offset": int,     # offset in the original source `text`
          "char_length": int,
          "word_start": int,      # word index (inclusive)
          "word_end": int,        # word index (exclusive)
        }
    """
    words = _tokenize_with_offsets(text)
    if not words:
        return []

    step = max(1, window_words - overlap_words)
    out: List[Dict[str, Any]] = []
    i = 0
    n = len(words)
    while i < n:
        j = min(n, i + window_words)
        first_word_start = words[i][1]
        last_word_end = words[j - 1][2]
        passage_text = text[first_word_start:last_word_end]
        out.append(
            {
                "text": passage_text,
                "char_offset": first_word_start,
                "char_length": last_word_end - first_word_start,
                "word_start": i,
                "word_end": j,
            }
        )
        if j >= n:
            break
        i += step
    return out


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
) -> np.ndarray:
    """Embed an arbitrary number of texts in parallel batches and return an
    (n, d) float32 numpy array. Result order matches input order."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    # Slice into ordered batches.
    batches: List[List[str]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batches.append(texts[start : start + EMBED_BATCH_SIZE])

    # Single batch: no point spinning up a thread pool.
    if len(batches) == 1:
        result = _embed_batch(client, batches[0])
        if on_progress:
            on_progress(stage, 1.0, f"{len(texts)}/{len(texts)} embedded")
        return np.asarray(result, dtype=np.float32)

    results: List[Optional[List[List[float]]]] = [None] * len(batches)
    workers = min(EMBED_PARALLEL_WORKERS, len(batches))
    total = len(texts)
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_embed_batch, client, batch_texts): batch_idx
            for batch_idx, batch_texts in enumerate(batches)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done_count += len(batches[idx])
            if on_progress:
                on_progress(
                    stage, done_count / total,
                    f"{done_count}/{total} embedded ({workers}× parallel)",
                )

    flat = [vec for batch_result in results for vec in batch_result]  # type: ignore[union-attr]
    return np.asarray(flat, dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D matrix. Zero rows stay zero."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE-STORY INDEX (passages + offsets + embedding matrix)
# ─────────────────────────────────────────────────────────────────────────────

def embed_source_stories(
    stories: List[dict],
    openai_key: str,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Build the source index: chunk every story into sliding passage windows,
    embed every passage, and return a dict suitable for handing to
    `markdown_to_beatbook_entries`.

    Output:
        {
          "articles": [
            {
              "article_id": str,
              "title": str,
              "date": str,
              "author": str,
              "content": str,
              "passages": [
                {"text", "char_offset", "char_length",
                 "word_start", "word_end", "global_index"}, ...
              ],
            }, ...
          ],
          "global_passages": [...flat list of passage dicts with article ref...],
          "embeddings": np.ndarray of shape (total_passages, embed_dim),
                        L2-normalized (so similarity = embeddings @ q.T).
        }
    """
    client = OpenAI(api_key=openai_key)

    articles: List[Dict[str, Any]] = []
    global_passages: List[Dict[str, Any]] = []
    passage_texts: List[str] = []

    for idx, story in enumerate(stories):
        content = story.get("content", "") or ""
        article_id = story.get("article_id") or story.get("id") or f"story-{idx}"
        passages = _passage_windows(content)

        # Tag each passage with its global index + a back-ref to the article
        # so we can rehydrate hits from the global similarity matrix.
        for p in passages:
            p["global_index"] = len(global_passages)
            global_passages.append({**p, "article_id": article_id, "article_idx": idx})
            passage_texts.append(p["text"])

        articles.append(
            {
                "article_id": article_id,
                "title": story.get("title", ""),
                "date": story.get("date", ""),
                "author": story.get("author", ""),
                "content": content,
                "passages": passages,
            }
        )

    if on_progress:
        on_progress(
            "embedding_sources", 0.0,
            f"{len(passage_texts)} passages across {len(articles)} stories",
        )

    raw = _embed_many(client, passage_texts, on_progress, "embedding_sources")
    normalized = _l2_normalize(raw)

    return {
        "articles": articles,
        "global_passages": global_passages,
        "embeddings": normalized,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION (per-corpus similarity threshold)
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_threshold(
    beat_emb_norm: np.ndarray,
    source_emb_norm: np.ndarray,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, float]:
    """Estimate the noise floor of the similarity distribution and pick a
    cutoff above it. Returns {threshold, noise_mean, noise_std, samples}."""
    if beat_emb_norm.size == 0 or source_emb_norm.size == 0:
        return {
            "threshold": CALIB_ABSOLUTE_FLOOR,
            "noise_mean": 0.0,
            "noise_std": 0.0,
            "samples": 0,
            "sigma": CALIB_SIGMA,
        }

    rng = np.random.default_rng(seed=42)
    n_beat = beat_emb_norm.shape[0]
    n_src = source_emb_norm.shape[0]
    n_samples = min(CALIB_RANDOM_SAMPLES, n_beat * n_src)

    beat_idx = rng.integers(0, n_beat, size=n_samples)
    src_idx = rng.integers(0, n_src, size=n_samples)
    sims = np.einsum("ij,ij->i", beat_emb_norm[beat_idx], source_emb_norm[src_idx])

    noise_mean = float(np.mean(sims))
    noise_std = float(np.std(sims))
    threshold = max(CALIB_ABSOLUTE_FLOOR, noise_mean + CALIB_SIGMA * noise_std)

    if on_progress:
        on_progress(
            "calibrating", 1.0,
            f"threshold={threshold:.3f} (noise_mean={noise_mean:.3f} ± {noise_std:.3f})",
        )

    return {
        "threshold": threshold,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "samples": int(n_samples),
        "sigma": CALIB_SIGMA,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEAVE-ONE-OUT SUB-WINDOW HIGHLIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def _subwindow_ranges(n_words: int, n_sub: int = LOO_SUBWINDOWS_PER_PASSAGE) -> List[Tuple[int, int]]:
    """Return n_sub overlapping (word_start, word_end) sub-windows covering
    [0, n_words). The sub-windows overlap so a sentence that straddles a naive
    boundary still gets surfaced."""
    if n_words <= 0 or n_sub <= 0:
        return []
    if n_words <= n_sub:
        # One word per sub-window; nothing more useful to do.
        return [(i, i + 1) for i in range(n_words)]

    sub_size = max(1, int(round(n_words / (n_sub / 2.0))))
    step = max(1, (n_words - sub_size) // max(1, n_sub - 1))
    ranges: List[Tuple[int, int]] = []
    for i in range(n_sub):
        start = min(i * step, n_words - sub_size)
        end = start + sub_size
        ranges.append((start, end))
    # Deduplicate while preserving order.
    seen = set()
    unique: List[Tuple[int, int]] = []
    for r in ranges:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN → CITATION ENTRIES
# ─────────────────────────────────────────────────────────────────────────────

def _context_sum_embeddings(
    raw: np.ndarray,
    indices_in_sentence_stream: List[int],
) -> np.ndarray:
    """Apply 0.6·prev + 1.0·self + 0.4·next weighting to a stack of
    sentence-level embeddings. `indices_in_sentence_stream[i] = j` means row i
    is the j-th sentence in the running sentence stream (used to detect
    paragraph boundaries — neighbors must be adjacent in the stream to count).
    """
    if raw.size == 0:
        return raw
    out = raw.copy() * CTX_WEIGHT_SELF
    n = raw.shape[0]
    for i in range(n):
        my_pos = indices_in_sentence_stream[i]
        if i - 1 >= 0 and indices_in_sentence_stream[i - 1] == my_pos - 1:
            out[i] += CTX_WEIGHT_PREV * raw[i - 1]
        if i + 1 < n and indices_in_sentence_stream[i + 1] == my_pos + 1:
            out[i] += CTX_WEIGHT_NEXT * raw[i + 1]
    return out


def markdown_to_beatbook_entries(
    markdown: str,
    source_index: Dict[str, Any],
    openai_key: str,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Convert a Markdown beat book into a citation-annotated entry list.

    Returns a dict shaped:

        {
          "calibration": {
            "threshold": float,        # per-corpus similarity cutoff
            "noise_mean": float,
            "noise_std": float,
            "sigma": float,
            "samples": int,
          },
          "entries": [
            {
              "content": str,            # the original Markdown line / sentence
              "passthrough": bool,       # True for headings, list items, etc.
              "supports": [
                {
                  "article_id": str,
                  "article_title": str,
                  "article_date": str,
                  "article_author": str,
                  "passage_text": str,
                  "passage_offset": int,
                  "passage_length": int,
                  "similarity": float,
                  "highlights": [
                    {"char_offset": int, "char_length": int,
                     "contribution": float}, ...
                  ],
                }, ...
              ],
            }, ...
          ],
        }
    """
    client = OpenAI(api_key=openai_key)

    entries = _segment_markdown(markdown)

    # Build the running sentence stream (positions used by context-sum).
    sentence_positions: List[int] = []
    sentence_texts: List[str] = []
    entry_to_embed_idx: Dict[int, int] = {}
    pos = 0
    for i, e in enumerate(entries):
        if e["needs_embedding"] and e["content"].strip():
            entry_to_embed_idx[i] = len(sentence_texts)
            sentence_texts.append(e["content"])
            sentence_positions.append(pos)
            pos += 1
        else:
            # Reset the running paragraph position on a non-sentence break
            # so context-sum doesn't bleed across headings or blank lines.
            pos = 0

    if on_progress:
        on_progress("embedding_beatbook", 0.0, f"{len(sentence_texts)} sentences to match")

    raw_beat = _embed_many(client, sentence_texts, on_progress, "embedding_beatbook")
    # Context-sum, then L2-normalize.
    ctx_beat = _context_sum_embeddings(raw_beat, sentence_positions)
    beat_emb_norm = _l2_normalize(ctx_beat)

    source_emb_norm: np.ndarray = source_index["embeddings"]
    global_passages: List[Dict[str, Any]] = source_index["global_passages"]
    articles_by_id: Dict[str, Dict[str, Any]] = {
        a["article_id"]: a for a in source_index["articles"]
    }

    calibration = _calibrate_threshold(beat_emb_norm, source_emb_norm, on_progress)
    threshold = calibration["threshold"]

    # Full similarity matrix: (n_beat, n_passages). For our corpus sizes (a
    # few hundred beat sentences × a few thousand source passages) this fits
    # comfortably in memory; for bigger corpora switch to batched matmul.
    if beat_emb_norm.size == 0 or source_emb_norm.size == 0:
        sim_matrix = np.zeros((beat_emb_norm.shape[0], 0), dtype=np.float32)
    else:
        sim_matrix = beat_emb_norm @ source_emb_norm.T

    # Pick top-K above threshold per beat-book sentence.
    k = min(TOP_K, sim_matrix.shape[1]) if sim_matrix.shape[1] > 0 else 0
    top_supports_per_sentence: List[List[Dict[str, Any]]] = []
    for row_i in range(sim_matrix.shape[0]):
        row = sim_matrix[row_i]
        if k == 0:
            top_supports_per_sentence.append([])
            continue
        # argpartition is O(n); we sort just the top-K slice afterwards.
        cand_idx = np.argpartition(-row, k - 1)[:k]
        cand_idx = cand_idx[np.argsort(-row[cand_idx])]
        per_sentence: List[Dict[str, Any]] = []
        for col_i in cand_idx:
            sim = float(row[col_i])
            if sim < threshold:
                break
            passage = global_passages[col_i]
            per_sentence.append({"passage": passage, "similarity": sim})
        top_supports_per_sentence.append(per_sentence)

    if on_progress:
        kept_total = sum(len(s) for s in top_supports_per_sentence)
        on_progress(
            "matching", 1.0,
            f"{kept_total} supports across {sum(1 for s in top_supports_per_sentence if s)} of "
            f"{sim_matrix.shape[0]} sentences",
        )

    # ── Phase 1: build the draft supports list and accumulate every LOO
    # sub-window text into one global batch. The earlier per-pair embedding
    # call was the dominant cost in this stage (N supports × 6 embeds × one
    # round-trip each); pooling them lets `_embed_many` send a small number
    # of parallel batches instead.
    draft_supports: Dict[int, List[Dict[str, Any]]] = {}
    loo_minus_texts: List[str] = []
    loo_meta: List[Dict[str, Any]] = []

    for entry_idx, entry in enumerate(entries):
        if entry_idx not in entry_to_embed_idx:
            continue
        embed_i = entry_to_embed_idx[entry_idx]
        query_vec = beat_emb_norm[embed_i]
        supports_for_entry: List[Dict[str, Any]] = []

        for cand in top_supports_per_sentence[embed_i]:
            passage = cand["passage"]
            sim = cand["similarity"]
            article = articles_by_id.get(passage["article_id"], {})

            support_dict: Dict[str, Any] = {
                "article_id": passage["article_id"],
                "article_title": article.get("title", ""),
                "article_date": article.get("date", ""),
                "article_author": article.get("author", ""),
                "passage_text": passage["text"],
                "passage_offset": int(passage["char_offset"]),
                "passage_length": int(passage["char_length"]),
                "similarity": round(sim, 4),
                "highlights": [],
            }
            supports_for_entry.append(support_dict)

            ws, we = passage["word_start"], passage["word_end"]
            n_words = we - ws
            sub_ranges = _subwindow_ranges(n_words)
            if not sub_ranges or n_words <= 2:
                continue

            source_text = article.get("content", "")
            source_words = _tokenize_with_offsets(source_text)
            passage_words = source_words[ws:we]
            minus_texts: List[str] = []
            for (s, e) in sub_ranges:
                if s == 0 and e >= len(passage_words):
                    minus_texts.append("")
                    continue
                run_a = source_text[passage_words[0][1]:passage_words[s - 1][2]] if s > 0 else ""
                run_b = source_text[passage_words[e][1]:passage_words[-1][2]] if e < len(passage_words) else ""
                joined = (run_a + " " + run_b).strip() if (run_a and run_b) else (run_a or run_b)
                minus_texts.append(joined)

            offset = len(loo_minus_texts)
            loo_minus_texts.extend(t if t else " " for t in minus_texts)
            loo_meta.append({
                "support_dict": support_dict,
                "passage_words": passage_words,
                "sub_ranges": sub_ranges,
                "placeholder_mask": [not t for t in minus_texts],
                "minus_offset": offset,
                "minus_count": len(minus_texts),
                "base_similarity": sim,
                "query_vec": query_vec,
            })

        draft_supports[entry_idx] = supports_for_entry

    # ── Phase 2: one big parallelized embed for every sub-window text.
    if loo_minus_texts:
        loo_embs = _embed_many(client, loo_minus_texts, on_progress, "highlighting")
        loo_embs_norm = _l2_normalize(loo_embs)
    else:
        loo_embs_norm = np.zeros((0, 1536), dtype=np.float32)

    # ── Phase 3: compute highlight contributions per (sentence, passage) pair.
    for meta in loo_meta:
        s_off = meta["minus_offset"]
        s_cnt = meta["minus_count"]
        embs = loo_embs_norm[s_off : s_off + s_cnt]
        sims = embs @ meta["query_vec"]
        for i, was_placeholder in enumerate(meta["placeholder_mask"]):
            if was_placeholder:
                sims[i] = np.inf  # never selected (contribution would be -inf)
        contributions = meta["base_similarity"] - sims
        order = np.argsort(-contributions)
        highlights: List[Dict[str, Any]] = []
        for rank in order[:LOO_TOP_HIGHLIGHTS]:
            c = float(contributions[rank])
            if c <= 0:
                continue
            s, e = meta["sub_ranges"][rank]
            first_offset = meta["passage_words"][s][1]
            last_offset = meta["passage_words"][e - 1][2]
            highlights.append({
                "char_offset": int(first_offset),
                "char_length": int(last_offset - first_offset),
                "contribution": round(c, 4),
            })
        meta["support_dict"]["highlights"] = highlights

    # ── Phase 4: assemble the output entry list.
    out_entries: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if i not in entry_to_embed_idx:
            out_entries.append({
                "content": entry["content"],
                "passthrough": True,
                "supports": [],
            })
        else:
            out_entries.append({
                "content": entry["content"],
                "passthrough": False,
                "supports": draft_supports.get(i, []),
            })

    return {"calibration": calibration, "entries": out_entries}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCES FILE (what the viewer loads to render article panels)
# ─────────────────────────────────────────────────────────────────────────────

def build_sources_file(
    stories: List[dict],
    source_index: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Produce the `*_sources.json` list the viewer reads. One entry per story,
    with the same `article_id` scheme used during embedding."""
    out: List[Dict[str, Any]] = []
    articles = source_index.get("articles", [])
    for idx, story in enumerate(stories):
        article_id = articles[idx]["article_id"] if idx < len(articles) else f"story-{idx}"
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
