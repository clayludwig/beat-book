"""
pipeline.py
-----------
Embedding + clustering + topic-labeling pipeline.
Reusable module — called by the web app after file upload.

Returns a PipelineResult with stories, topics, and helper lookups.
"""

import concurrent.futures
import hashlib
import pickle
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from tqdm import tqdm
import numpy as np

# Type for progress callbacks: (step_name, progress_fraction 0.0–1.0, detail_text)
ProgressCallback = Callable[[str, float, str], None]

from openai import OpenAI
import umap
import hdbscan

from claude_client import CHAT_MODEL, chat_client

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Embeddings stay on OpenAI: Anthropic doesn't host an embedding API, and we
# want this to run anywhere without requiring a local model.
EMBED_MODEL = "text-embedding-3-small"
LABEL_MODEL = CHAT_MODEL
CACHE_DIR   = Path(".cache")
SAMPLE_SIZE_FOR_LABEL = 8
EMBED_BATCH_SIZE = 100

# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Everything the agent needs to work with the uploaded stories."""
    stories: List[dict]                          # original story dicts
    topics: Dict[str, List[int]]                 # topic_label → [story indices]
    story_topics: List[List[str]]                # per-story list of topic labels
    broad_topics: Dict[str, List[int]]           # broad topic → [story indices]
    specific_topics: Dict[str, List[int]]        # specific topic → [story indices]
    briefings: Dict[str, dict] = field(default_factory=dict)  # broad topic → structured digest

    def topic_summary(self) -> str:
        """Human-readable summary of discovered topics."""
        lines = ["## Broad Topics"]
        for label, indices in sorted(self.broad_topics.items(), key=lambda x: -len(x[1])):
            lines.append(f"  - **{label}** ({len(indices)} stories)")
        lines.append("\n## Specific Topics")
        for label, indices in sorted(self.specific_topics.items(), key=lambda x: -len(x[1])):
            lines.append(f"  - **{label}** ({len(indices)} stories)")
        return "\n".join(lines)

    def get_story(self, idx: int) -> Optional[dict]:
        if 0 <= idx < len(self.stories):
            return self.stories[idx]
        return None

    def search_stories(self, query: str, max_results: int = 20) -> List[dict]:
        q = query.lower()
        results = []
        for i, s in enumerate(self.stories):
            text = f"{s.get('title','')} {s.get('content','')}".lower()
            if q in text:
                results.append({"index": i, "title": s["title"], "date": s.get("date", "")})
                if len(results) >= max_results:
                    break
        return results

    def stories_for_topic(self, topic: str) -> List[dict]:
        indices = self.topics.get(topic, [])
        return [
            {"index": i, "title": self.stories[i]["title"], "date": self.stories[i].get("date", "")}
            for i in indices
        ]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# ─── Quote stripping ────────────────────────────────────────────────────────
#
# Direct quotations in news prose carry the bulk of an article's word count
# but contribute almost nothing to topic discovery or briefing generation —
# the surrounding reportorial prose already states the gist. Stripping them
# lets the same word budget (400 for embedding, 450 for briefings, etc.)
# cover more of the article.
#
# We only strip when there is a clear attribution verb adjacent to the
# quoted span, in one of two American-newsroom patterns:
#
#     "...," said NAME[.|!|?]
#     "...," NAME said[.|!|?]
#
# Plain in-line uses of double quotes ('the word "amazing"', 'called it
# "a victory"') are left alone — no attribution verb, no match.
#
# Citation matching reads the original story content (not the stripped
# form), so paraphrase-to-source lookups continue to work.

_ATTRIBUTION_VERBS = (
    r"said|says|told|tells|asked|asks|added|adds|noted|notes|"
    r"wrote|writes|explained|explains|stated|states|commented|"
    r"comments|argued|argues|claimed|claims|responded|responds|"
    r"recalled|remarked|admitted|admits|insisted|insists"
)

# Open/close double-quote characters we recognize: ASCII " and smart curly.
_OPEN_QUOTES  = r'["“]'
_CLOSE_QUOTES = r'["”]'
_NOT_CLOSE    = r'[^"”]'

# Common honorifics / abbreviations that contain a period mid-sentence —
# without listing them, "Dr. Jones" prematurely terminates the
# attribution-body match at the period after "Dr".
_ABBREV = (
    r"Dr|Mr|Mrs|Ms|Sr|Jr|Prof|St|U\.S|U\.K|D\.C|"
    r"Inc|Corp|Co|Ltd|Capt|Sgt|Lt|Col|Gen|Sen|"
    r"Rep|Gov|Rev|Ph\.D|M\.D|J\.D|No|Vol"
)
# A "body" character is either a non-terminator OR a known abbreviation
# followed by a period. The non-greedy outer repeat means the trailing
# [.!?] is what actually anchors the end of the match.
_ATTRIB_BODY = r"(?:(?:" + _ABBREV + r")\.|[^.!?\n])"

# Pattern A: "quote," <verb> <subject up to terminator>.
_QUOTE_VERB_FIRST = re.compile(
    _OPEN_QUOTES + _NOT_CLOSE + r'{1,500}?,' + _CLOSE_QUOTES +
    r"\s+(?:" + _ATTRIBUTION_VERBS + r")\b" + _ATTRIB_BODY + r"{0,160}[.!?]",
    re.IGNORECASE,
)
# Pattern B: "quote," <subject> <verb> <optional trailing fragment>.
_QUOTE_SUBJECT_FIRST = re.compile(
    _OPEN_QUOTES + _NOT_CLOSE + r'{1,500}?,' + _CLOSE_QUOTES +
    r"\s+" + _ATTRIB_BODY + r"{1,100}?\s+(?:" + _ATTRIBUTION_VERBS + r")\b" +
    _ATTRIB_BODY + r"{0,160}[.!?]",
    re.IGNORECASE,
)

# Whitespace cleanups after the substitutions punch holes in the prose.
_WS_RUNS    = re.compile(r"[ \t]+")
_WS_LINESEP = re.compile(r"\n[ \t]+")
_WS_BLANKS  = re.compile(r"\n{3,}")


def strip_quotes(text: str) -> str:
    """Remove direct-quotation sentences from news prose.

    Targets the two high-confidence American-newsroom patterns:
        "quote text," said X.
        "quote text," X said.
    Smart quotes ("..." / "...") and the standard attribution-verb roster
    are recognized. Returns the cleaned string with whitespace collapsed.

    This is a preprocessing pass for LLM-facing prompts only — callers
    writing to citation matching should read the original story content
    so paraphrase-to-source lookups still resolve.
    """
    if not text:
        return text
    out = _QUOTE_VERB_FIRST.sub(" ", text)
    out = _QUOTE_SUBJECT_FIRST.sub(" ", out)
    out = _WS_RUNS.sub(" ", out)
    out = _WS_LINESEP.sub("\n", out)
    out = _WS_BLANKS.sub("\n\n", out)
    return out.strip()


def _story_to_text(story: dict) -> str:
    """Build the per-story text the embedding model sees. Direct quotes
    are stripped from the content so the 400-word window covers more of
    the article's structural prose."""
    title   = story.get("title", "")
    content = strip_quotes(story.get("content", ""))
    section = ""
    for line in content.splitlines()[:10]:
        stripped = line.strip()
        if "section:" in stripped.lower():
            section = stripped
            break
    words   = content.split()
    snippet = " ".join(words[:400])
    parts   = [p for p in [title, section, snippet] if p]
    return "\n\n".join(parts)


def _embed_batch(client: OpenAI, texts: List[str]) -> np.ndarray:
    all_vectors = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Embedding"):
        chunk = texts[i : i + EMBED_BATCH_SIZE]
        # OpenAI rejects empty strings; sub a space.
        cleaned = [t if t.strip() else " " for t in chunk]
        resp  = client.embeddings.create(input=cleaned, model=EMBED_MODEL)
        vecs  = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        all_vectors.extend(vecs)
    return np.array(all_vectors, dtype=np.float32)


def _cache_key(texts: List[str]) -> str:
    combined = "\n---\n".join(texts[:10])
    return hashlib.md5((combined + EMBED_MODEL).encode()).hexdigest()


def _load_or_embed(client: OpenAI, texts: List[str]) -> np.ndarray:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "embeddings.pkl"
    key = _cache_key(texts)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == key and len(cached.get("vectors", [])) == len(texts):
            print("✓ Loaded embeddings from cache.")
            return cached["vectors"]
    print(f"Generating embeddings for {len(texts)} stories…")
    vectors = _embed_batch(client, texts)
    with open(cache_file, "wb") as f:
        pickle.dump({"key": key, "vectors": vectors}, f)
    return vectors


def _umap_params(n: int) -> dict:
    n_components = min(15, max(5, n // 40))
    n_neighbors  = min(30, max(5, int(n ** 0.55)))
    return {"n_components": n_components, "n_neighbors": n_neighbors}


def _cluster_sizes(n: int) -> Tuple[int, int]:
    broad    = max(4, n // 25)
    specific = max(2, n // 60)
    return broad, specific


def _reduce(vectors: np.ndarray) -> np.ndarray:
    params = _umap_params(len(vectors))
    print(f"UMAP (n_components={params['n_components']}, n_neighbors={params['n_neighbors']})…")

    def _make_reducer(init: str):
        return umap.UMAP(
            n_components=params["n_components"],
            n_neighbors=params["n_neighbors"],
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            init=init,
        )

    # Newer scipy can raise `Cannot use scipy.linalg.eigh for sparse A with
    # k >= N` inside UMAP's default spectral init when the kNN graph has few
    # connected components. Fall back to pca → random if that happens.
    for init in ("spectral", "pca", "random"):
        try:
            return _make_reducer(init).fit_transform(vectors)
        except TypeError as e:
            if "eigh" not in str(e) and "k >= N" not in str(e):
                raise
            print(f"UMAP init={init!r} hit scipy eigh bug; retrying…")
    raise RuntimeError("UMAP failed with every init strategy")


def _cluster(reduced: np.ndarray, min_cluster_size: int):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  → {n_clusters} clusters, {n_noise} noise (min_cluster_size={min_cluster_size})")
    return labels, clusterer


def _assign_outliers(reduced: np.ndarray, labels: np.ndarray) -> np.ndarray:
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels
    labels          = labels.copy()
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    if not unique_clusters:
        return labels
    cluster_means = np.stack([reduced[labels == c].mean(axis=0) for c in unique_clusters])
    for idx in np.where(noise_mask)[0]:
        dists = np.linalg.norm(cluster_means - reduced[idx], axis=1)
        labels[idx] = unique_clusters[int(dists.argmin())]
    return labels


def _representative_snippets(stories: List[dict], indices: List[int], reduced: np.ndarray) -> List[str]:
    """Pick up to SAMPLE_SIZE_FOR_LABEL stories closest to the cluster centroid
    and format each as a single 'headline -- excerpt' line for the labeler."""
    cluster_vecs = reduced[indices]
    centroid     = cluster_vecs.mean(axis=0)
    dists        = np.linalg.norm(cluster_vecs - centroid, axis=1)
    order        = np.argsort(dists)
    sampled      = [indices[i] for i in order[:SAMPLE_SIZE_FOR_LABEL]]

    snippets = []
    for i in sampled:
        s = stories[i]
        words   = strip_quotes(s.get("content", "")).split()
        excerpt = " ".join(words[10:40])
        snippets.append(f"- {s['title']} -- {excerpt}")
    return snippets


_LABEL_PROMPT_PREFIX = (
    "You are labeling clusters of news articles from a local newspaper. "
    "For each cluster below, produce a concise 2-5 word topic label "
    "describing the SUBJECT MATTER the articles share. Focus on WHAT happens, "
    "not WHERE -- avoid labels like 'Chicago news', 'local community news', "
    "or 'Illinois news' unless the geography itself is the distinguishing "
    "feature (e.g. 'Lake Michigan environment'). Good labels: 'High School "
    "Basketball', 'City Budget Disputes', 'Immigration Policy', 'Crime and "
    "Sentencing', 'City Council', 'Transit'.\n\n"
    "Return your output via the register_labels tool -- one entry per cluster, "
    "keyed by the cluster_id integers shown below.\n\n"
)


_LABEL_BATCH_TOOL = {
    "name": "register_labels",
    "description": "Register a topic label for every cluster in the batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "description": "One entry per cluster. cluster_id must match the integers in the prompt.",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "integer"},
                        "label": {
                            "type": "string",
                            "description": "2-5 word topic label.",
                        },
                    },
                    "required": ["cluster_id", "label"],
                },
            },
        },
        "required": ["labels"],
    },
}


def _label_cluster(client, stories: List[dict], indices: List[int], reduced: np.ndarray) -> str:
    """Label a single cluster. Used by the small-corpus path and as a
    fallback when the batched call misses a cluster."""
    snippets = _representative_snippets(stories, indices, reduced)
    prompt = (
        _LABEL_PROMPT_PREFIX
        + "## Cluster 0\n"
        + "\n".join(snippets)
        + "\n\nReturn one entry with cluster_id=0."
    )
    resp = client.messages.create(
        model=LABEL_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        tools=[_LABEL_BATCH_TOOL],
        tool_choice={"type": "tool", "name": "register_labels"},
    )
    tool_block = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        return "Uncategorized"
    payload = tool_block.input if isinstance(tool_block.input, dict) else {}
    for item in payload.get("labels") or []:
        if isinstance(item, dict):
            label = (item.get("label") or "").strip().strip('"').strip("'")
            if label:
                return label
    return "Uncategorized"


def _label_all(client, stories, labels, reduced, level_name, on_progress=None):
    """Label every cluster in `labels` with ONE batched LLM call.

    Returns {cluster_id: label}. Any cluster the model misses gets a
    single repair call; if that also fails, a generic 'Topic <id>' is
    used so downstream lookups never miss.
    """
    unique = sorted(int(c) for c in np.unique(labels) if c != -1)
    print(f"Labeling {len(unique)} {level_name} clusters in one call...")
    if not unique:
        if on_progress:
            on_progress(f"labeling_{level_name}", 1.0, f"No {level_name} clusters")
        return {}

    if on_progress:
        on_progress(
            f"labeling_{level_name}", 0.0,
            f"Labeling {len(unique)} {level_name} topics in one call...",
        )

    sections: List[str] = []
    for cid in unique:
        indices = list(np.where(labels == cid)[0])
        snippets = _representative_snippets(stories, indices, reduced)
        sections.append(f"## Cluster {cid}\n" + "\n".join(snippets))

    prompt = _LABEL_PROMPT_PREFIX + "\n\n".join(sections)

    # Output budget: ~50 tokens per entry (cluster_id + label + JSON framing).
    # 64 * N with a 1024 floor covers ~80+ clusters comfortably; any cluster
    # the model skips falls through to the per-cluster repair path below.
    resp = client.messages.create(
        model=LABEL_MODEL,
        max_tokens=max(1024, 64 * len(unique)),
        messages=[{"role": "user", "content": prompt}],
        tools=[_LABEL_BATCH_TOOL],
        tool_choice={"type": "tool", "name": "register_labels"},
    )

    tool_block = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    result: Dict[int, str] = {}
    if tool_block is not None:
        payload = tool_block.input if isinstance(tool_block.input, dict) else {}
        for item in payload.get("labels") or []:
            if not isinstance(item, dict):
                continue
            cid = item.get("cluster_id")
            label = (item.get("label") or "").strip().strip('"').strip("'")
            if isinstance(cid, int) and label:
                result[cid] = label

    for cid in unique:
        if cid in result:
            continue
        indices = list(np.where(labels == cid)[0])
        try:
            result[cid] = _label_cluster(client, stories, indices, reduced)
        except Exception:
            result[cid] = f"Topic {cid}"

    if on_progress:
        on_progress(
            f"labeling_{level_name}", 1.0,
            f"Labeled {len(unique)} {level_name} topics",
        )
    return result



# ─────────────────────────────────────────────────────────────────────────────
# PER-TOPIC BRIEFINGS
# ─────────────────────────────────────────────────────────────────────────────
#
# For each broad topic we generate a structured digest in ONE Haiku call.
# These digests replace the agent's old "read half of every topic" gate
# (see project_briefings_over_gate memory): the agent now consumes briefings
# up-front and only falls back to raw read_story when it needs to verify or
# quote directly.
#
# Briefings run in parallel across topics. The number of stories included
# verbatim per topic is capped so even huge topics fit in one prompt.

BRIEFING_DETAILED_STORIES = 20   # full first-N-words for the top stories
BRIEFING_DETAIL_WORDS = 450      # words of body for each detailed story
BRIEFING_TITLE_TAIL_CAP = 80     # additional title/date-only entries listed
BRIEFING_CONCURRENCY = 6         # parallel topics
BRIEFING_MAX_TOKENS = 3072       # generous output cap per topic


_BRIEFING_TOOL = {
    "name": "register_briefing",
    "description": (
        "Register a structured briefing for one news topic. The reporter-"
        "facing beat book will be written from these briefings, so be "
        "specific: real names, real organizations, real dates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key_people": {
                "type": "array",
                "description": "3-8 people who recur across the coverage. Real names, not placeholders.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string", "description": "Their title, affiliation, or role in the beat."},
                        "why_relevant": {"type": "string", "description": "One sentence on why this person matters here."},
                    },
                    "required": ["name", "role", "why_relevant"],
                },
            },
            "key_orgs": {
                "type": "array",
                "description": "3-8 organizations / institutions / agencies that show up repeatedly.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "what_it_is": {"type": "string"},
                        "why_relevant": {"type": "string"},
                    },
                    "required": ["name", "what_it_is", "why_relevant"],
                },
            },
            "key_dates_or_events": {
                "type": "array",
                "description": "Dated events that anchor the recent history of this beat (votes, rulings, openings, scandals, deadlines).",
                "items": {
                    "type": "object",
                    "properties": {
                        "when": {"type": "string", "description": "Year, date, or short phrase ('spring 2024', 'last June')."},
                        "what": {"type": "string", "description": "What happened. One sentence."},
                    },
                    "required": ["when", "what"],
                },
            },
            "recurring_themes": {
                "type": "array",
                "description": "3-6 short phrases describing recurring frames, tensions, or arcs in the coverage.",
                "items": {"type": "string"},
            },
            "representative_excerpts": {
                "type": "array",
                "description": "3-5 short verbatim excerpts (1-3 sentences each) that capture the texture of the coverage.",
                "items": {
                    "type": "object",
                    "properties": {
                        "story_title": {"type": "string"},
                        "excerpt": {"type": "string", "description": "Short verbatim quote, max ~250 chars."},
                    },
                    "required": ["story_title", "excerpt"],
                },
            },
            "open_questions": {
                "type": "array",
                "description": "Questions the coverage raises but does not answer -- angles a new reporter might pursue.",
                "items": {"type": "string"},
            },
            "summary": {
                "type": "string",
                "description": "Two or three sentences summarizing what this topic is about and why it matters on the beat.",
            },
        },
        "required": [
            "summary",
            "key_people",
            "key_orgs",
            "key_dates_or_events",
            "recurring_themes",
            "representative_excerpts",
            "open_questions",
        ],
    },
}


def _topic_briefing_prompt(topic_label, stories, indices, reduced):
    """Build a single-topic briefing prompt.

    Picks the BRIEFING_DETAILED_STORIES closest to the cluster centroid for
    verbatim inclusion, then appends a title-and-date list of up to
    BRIEFING_TITLE_TAIL_CAP additional stories so the model sees the full
    span of coverage without blowing the prompt budget.
    """
    cluster_vecs = reduced[indices]
    centroid = cluster_vecs.mean(axis=0)
    dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
    order = np.argsort(dists)
    ordered_indices = [indices[i] for i in order]

    detailed = ordered_indices[:BRIEFING_DETAILED_STORIES]
    tail = ordered_indices[BRIEFING_DETAILED_STORIES:BRIEFING_DETAILED_STORIES + BRIEFING_TITLE_TAIL_CAP]

    parts = [
        f"You are preparing a structured briefing for a journalist about the topic \"{topic_label}\".\n",
        f"There are {len(indices)} stories grouped under this topic. The {len(detailed)} most representative are shown in full; the next {len(tail)} are listed by headline only.\n",
        "Read them and call register_briefing with a digest the reporter can build their beat book from.\n",
        "Be specific: real names, real organizations, real dates. Quote verbatim from the stories in representative_excerpts.\n",
        "\n# Stories (full text)\n",
    ]
    for rank, idx in enumerate(detailed, 1):
        s = stories[idx]
        title = s.get("title", "") or f"Story #{idx}"
        date = s.get("date", "")
        author = s.get("author", "")
        meta_bits = [b for b in (date, author) if b]
        meta = f" ({'; '.join(meta_bits)})" if meta_bits else ""
        body = " ".join(strip_quotes(s.get("content", "") or "").split()[:BRIEFING_DETAIL_WORDS])
        parts.append(f"\n## {rank}. {title}{meta}\n{body}\n")

    if tail:
        parts.append("\n# Additional headlines (context only, not summarized here)\n")
        for idx in tail:
            s = stories[idx]
            title = s.get("title", "") or f"Story #{idx}"
            date = s.get("date", "")
            parts.append(f"- {title}" + (f" ({date})" if date else "") + "\n")

    return "".join(parts)


def _briefing_for_topic(client, stories, topic_label, indices, reduced):
    """One Haiku call producing the structured briefing for one topic.

    On failure, returns a minimal stub so the pipeline keeps going --
    the agent can fall back to read_story for that one topic.
    """
    empty = {"summary": "", "key_people": [], "key_orgs": [],
             "key_dates_or_events": [], "recurring_themes": [],
             "representative_excerpts": [], "open_questions": [],
             "story_count": len(indices)}
    if not indices:
        return empty

    prompt = _topic_briefing_prompt(topic_label, stories, indices, reduced)
    resp = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=BRIEFING_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        tools=[_BRIEFING_TOOL],
        tool_choice={"type": "tool", "name": "register_briefing"},
    )
    tool_block = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        return empty
    payload = tool_block.input if isinstance(tool_block.input, dict) else {}
    return {
        "summary": (payload.get("summary") or "").strip(),
        "key_people": payload.get("key_people") or [],
        "key_orgs": payload.get("key_orgs") or [],
        "key_dates_or_events": payload.get("key_dates_or_events") or [],
        "recurring_themes": payload.get("recurring_themes") or [],
        "representative_excerpts": payload.get("representative_excerpts") or [],
        "open_questions": payload.get("open_questions") or [],
        "story_count": len(indices),
    }


def build_briefings(client, stories, broad_topics, reduced, on_progress=None):
    """Generate a structured briefing per broad topic, in parallel.

    Returns a {topic_label: briefing_dict} map. Any topic whose call
    raises is given an empty briefing so the pipeline keeps going.
    """
    items = [(label, idx_list) for label, idx_list in broad_topics.items() if idx_list]
    if not items:
        if on_progress:
            on_progress("briefings", 1.0, "No topics to brief")
        return {}

    total = len(items)
    done = 0
    results = {}

    if on_progress:
        on_progress("briefings", 0.0, f"Building {total} topic briefings...")

    def _job(label, indices):
        try:
            return label, _briefing_for_topic(client, stories, label, indices, reduced)
        except Exception as e:
            print(f"[briefings] {label}: failed ({type(e).__name__}: {e})")
            return label, {"summary": "", "key_people": [], "key_orgs": [],
                            "key_dates_or_events": [], "recurring_themes": [],
                            "representative_excerpts": [], "open_questions": [],
                            "story_count": len(indices)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=BRIEFING_CONCURRENCY) as ex:
        futures = [ex.submit(_job, label, idx) for label, idx in items]
        for fut in concurrent.futures.as_completed(futures):
            label, briefing = fut.result()
            results[label] = briefing
            done += 1
            if on_progress:
                on_progress("briefings", done / total, f"Briefed {done}/{total} topics")

    return results


def render_briefings_markdown(briefings):
    """Render the full briefings dict into a single Markdown block. Used by
    the agent and the final-draft synthesis to inject corpus knowledge
    into the prompt without re-reading individual stories."""
    if not briefings:
        return "(No topic briefings were generated.)"

    lines = []
    for label, b in briefings.items():
        lines.append(f"## {label}")
        story_count = b.get("story_count")
        if story_count:
            lines.append(f"_{story_count} stories in this topic._")
        if b.get("summary"):
            lines.append(b["summary"])

        people = b.get("key_people") or []
        if people:
            lines.append("\n**Key people**")
            for p in people:
                if not isinstance(p, dict):
                    continue
                name = (p.get("name") or "").strip()
                role = (p.get("role") or "").strip()
                why = (p.get("why_relevant") or "").strip()
                if name:
                    bits = [name]
                    if role: bits.append(role)
                    if why:  bits.append(why)
                    lines.append("- " + " -- ".join(bits))

        orgs = b.get("key_orgs") or []
        if orgs:
            lines.append("\n**Key organizations**")
            for o in orgs:
                if not isinstance(o, dict):
                    continue
                name = (o.get("name") or "").strip()
                what = (o.get("what_it_is") or "").strip()
                why = (o.get("why_relevant") or "").strip()
                if name:
                    bits = [name]
                    if what: bits.append(what)
                    if why:  bits.append(why)
                    lines.append("- " + " -- ".join(bits))

        events = b.get("key_dates_or_events") or []
        if events:
            lines.append("\n**Key dates and events**")
            for e in events:
                if not isinstance(e, dict):
                    continue
                when = (e.get("when") or "").strip()
                what = (e.get("what") or "").strip()
                if when or what:
                    lines.append(f"- {when}: {what}".strip())

        themes = b.get("recurring_themes") or []
        if themes:
            lines.append("\n**Recurring themes**")
            for t in themes:
                if isinstance(t, str) and t.strip():
                    lines.append(f"- {t.strip()}")

        excerpts = b.get("representative_excerpts") or []
        if excerpts:
            lines.append("\n**Representative excerpts**")
            for e in excerpts:
                if not isinstance(e, dict):
                    continue
                title = (e.get("story_title") or "").strip()
                quote = (e.get("excerpt") or "").strip()
                if quote:
                    if title:
                        lines.append(f"- \"{quote}\" -- _{title}_")
                    else:
                        lines.append(f"- \"{quote}\"")

        questions = b.get("open_questions") or []
        if questions:
            lines.append("\n**Open questions**")
            for q in questions:
                if isinstance(q, str) and q.strip():
                    lines.append(f"- {q.strip()}")

        lines.append("")
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(stories: List[dict], openai_key: str, anthropic_key: str,
                 on_progress: Optional[ProgressCallback] = None) -> PipelineResult:
    """Full pipeline: embed \u2192 reduce \u2192 cluster \u2192 label \u2192 return PipelineResult.

    Embeds via OpenAI (text-embedding-3-small); labels via Claude Sonnet 4.6.
    """
    def _p(step, frac, detail=""):
        if on_progress:
            on_progress(step, frac, detail)

    embed_clt = OpenAI(api_key=openai_key)
    chat_clt  = chat_client(anthropic_key)

    _p("embedding", 0.0, f"Generating embeddings for {len(stories)} stories\u2026")
    texts   = [_story_to_text(s) for s in stories]
    vectors = _load_or_embed(embed_clt, texts)
    _p("embedding", 1.0, "Embeddings complete")

    # Small corpora skip UMAP+HDBSCAN: density-based clustering is meaningless
    # below ~8 stories, and UMAP itself rejects n_neighbors >= n_samples.
    # The agent only needs at least one topic with all stories.
    if len(stories) < 8:
        _p("reducing", 1.0, "Skipping reduction (small corpus)")
        _p("clustering", 1.0, "Skipping clustering (small corpus)")
        _p("labeling", 0.0, "Labeling combined topic\u2026")
        all_indices = list(range(len(stories)))
        label = _label_cluster(chat_clt, stories, all_indices, vectors)
        topics = {label: all_indices}
        story_topics = [[label] for _ in stories]
        _p("labeling", 1.0, "Done")
        _p("briefings", 0.0, "Building topic briefing\u2026")
        briefings = build_briefings(chat_clt, stories, topics, vectors,
                                    on_progress=lambda s, f, d: _p(s, f, d))
        _p("briefings", 1.0, "Briefings complete")
        return PipelineResult(
            stories=stories,
            topics=topics,
            story_topics=story_topics,
            broad_topics=topics,
            specific_topics=topics,
            briefings=briefings,
        )

    _p("reducing", 0.0, "Reducing dimensions\u2026")
    reduced = _reduce(vectors)
    _p("reducing", 1.0, "Dimensionality reduction complete")

    broad_min, specific_min = _cluster_sizes(len(stories))
    print(f"Cluster sizes: broad_min={broad_min}, specific_min={specific_min}")

    _p("clustering", 0.0, "Clustering stories\u2026")
    broad_labels, _  = _cluster(reduced, broad_min)
    broad_labels      = _assign_outliers(reduced, broad_labels)
    _p("clustering", 0.5, "Broad clusters found")

    spec_labels, _   = _cluster(reduced, specific_min)
    spec_labels       = _assign_outliers(reduced, spec_labels)
    _p("clustering", 1.0, "All clusters found")

    _p("labeling", 0.0, "Labeling topics with LLM\u2026")
    broad_map = _label_all(chat_clt, stories, broad_labels, reduced, "broad",
                           lambda s, f, d: _p("labeling", f * 0.4, d))
    spec_map  = _label_all(chat_clt, stories, spec_labels,  reduced, "specific",
                           lambda s, f, d: _p("labeling", 0.4 + f * 0.6, d))

    # Build lookup dicts
    broad_topics    = {}
    specific_topics = {}
    all_topics      = {}
    story_topics    = []

    for i in range(len(stories)):
        bt = broad_map.get(int(broad_labels[i]), "Uncategorized")
        st = spec_map.get(int(spec_labels[i]),   "Uncategorized")

        broad_topics.setdefault(bt, []).append(i)
        specific_topics.setdefault(st, []).append(i)
        all_topics.setdefault(bt, [])
        all_topics.setdefault(st, [])
        if i not in all_topics[bt]:
            all_topics[bt].append(i)
        if i not in all_topics[st]:
            all_topics[st].append(i)

        if st.lower() == bt.lower():
            story_topics.append([bt])
        else:
            story_topics.append([bt, st])

    _p("briefings", 0.0, "Building topic briefings...")
    briefings = build_briefings(
        chat_clt, stories, broad_topics, reduced,
        on_progress=lambda s, f, d: _p(s, f, d),
    )
    _p("briefings", 1.0, "Briefings complete")

    return PipelineResult(
        stories=stories,
        topics=all_topics,
        story_topics=story_topics,
        broad_topics=broad_topics,
        specific_topics=specific_topics,
        briefings=briefings,
    )
