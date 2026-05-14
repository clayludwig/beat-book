"""
pipeline.py
-----------
Embedding + clustering + topic-labeling pipeline.
Reusable module — called by the web app after file upload.

Returns a PipelineResult with stories, topics, and helper lookups.
"""

import hashlib
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
from tqdm import tqdm
import numpy as np

# Type for progress callbacks: (step_name, progress_fraction 0.0–1.0, detail_text)
ProgressCallback = Callable[[str, float, str], None]

from openai import OpenAI
import umap
import hdbscan

import anthropic as anthropic_sdk

from claude_client import (
    ANTHROPIC_SEMAPHORE,
    CHAT_MODEL,
    MAX_ANTHROPIC_CONCURRENT,
    RATE_LIMIT_MAX_RETRIES,
    chat_client,
    rate_limit_pause,
    thinking_param,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Embeddings stay on OpenAI: Anthropic doesn't host an embedding API, and we
# want this to run anywhere without requiring a local model.
EMBED_MODEL = "text-embedding-3-small"
# Cluster labels are 2-5 words — Haiku is sufficient and ~10x faster than Sonnet.
LABEL_MODEL = "claude-haiku-4-5-20251001"
CACHE_DIR   = Path(".cache")
SAMPLE_SIZE_FOR_LABEL = 8
EMBED_BATCH_SIZE = 100
# Labels are 2–5 words; 256 leaves plenty of room with extended thinking off.
LABEL_MAX_TOKENS = 256

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

    def topic_summary(self) -> str:
        """Human-readable summary of the broad topics the agent should explore."""
        lines = ["## Topics"]
        for label, indices in sorted(self.broad_topics.items(), key=lambda x: -len(x[1])):
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

def _story_to_text(story: dict) -> str:
    title   = story.get("title", "")
    content = story.get("content", "")
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


def _label_cluster(client, stories: List[dict], indices: List[int], reduced: np.ndarray) -> str:
    cluster_vecs = reduced[indices]
    centroid     = cluster_vecs.mean(axis=0)
    dists        = np.linalg.norm(cluster_vecs - centroid, axis=1)
    order        = np.argsort(dists)
    sampled      = [indices[i] for i in order[:SAMPLE_SIZE_FOR_LABEL]]

    snippets = []
    for i in sampled:
        s = stories[i]
        words   = s.get("content", "").split()
        excerpt = " ".join(words[10:40])
        snippets.append(f"• {s['title']} — {excerpt}")

    prompt = (
        "You are labeling clusters of news articles from a local newspaper.\n"
        "Below are the most representative headlines and excerpts from one cluster.\n\n"
        + "\n".join(snippets)
        + "\n\nReturn ONLY a concise topic label (2–5 words) describing the SUBJECT MATTER "
        "these articles share. Focus on WHAT happens, not WHERE — avoid labels like "
        "'Chicago news', 'local community news', or 'Illinois news' unless the "
        "geography itself is the distinguishing feature (e.g. 'Lake Michigan environment'). "
        "Good labels: 'High School Basketball', 'City Budget Disputes', "
        "'Immigration Policy', 'Crime and Sentencing', 'City Council', 'Transit'."
    )

    for rl_attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            with ANTHROPIC_SEMAPHORE:
                resp = client.messages.create(
                    model=LABEL_MODEL,
                    max_tokens=LABEL_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                    thinking=thinking_param(),
                )
            break
        except anthropic_sdk.RateLimitError as e:
            if rl_attempt >= RATE_LIMIT_MAX_RETRIES:
                raise
            pause = rate_limit_pause(rl_attempt, e)
            logging.warning("Pipeline label rate limited; waiting %.0fs (attempt %d/%d).",
                            pause, rl_attempt + 1, RATE_LIMIT_MAX_RETRIES)
            time.sleep(pause)

    text = "".join(
        b.text for b in resp.content
        if getattr(b, "type", None) == "text"
    )
    return text.strip().strip('"').strip("'")


def _cluster_snippets(stories, indices, reduced):
    cluster_vecs = reduced[indices]
    centroid     = cluster_vecs.mean(axis=0)
    dists        = np.linalg.norm(cluster_vecs - centroid, axis=1)
    order        = np.argsort(dists)
    sampled      = [indices[i] for i in order[:SAMPLE_SIZE_FOR_LABEL]]
    out = []
    for i in sampled:
        s = stories[i]
        words   = s.get("content", "").split()
        excerpt = " ".join(words[10:40])
        out.append(f"  \u2022 {s['title']} \u2014 {excerpt}")
    return "\n".join(out)


def _label_all(client, stories, labels, reduced, level_name, on_progress=None):
    """Label all clusters at this level in a single Haiku call.

    Falls back to per-cluster labeling if the batched call's JSON parse fails.
    """
    import json
    import re

    unique = sorted(c for c in np.unique(labels) if c != -1)
    print(f"Labeling {len(unique)} {level_name} clusters\u2026")
    if not unique:
        return {}

    blocks = []
    for cid in unique:
        indices = list(np.where(labels == cid)[0])
        blocks.append(f"Cluster {cid}:\n{_cluster_snippets(stories, indices, reduced)}")
    prompt = (
        "You are labeling clusters of news articles from a local newspaper.\n"
        f"There are {len(unique)} clusters below. Each cluster shows its most "
        "representative headlines and excerpts.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn ONLY a JSON object mapping each cluster id (as a string) "
        "to a concise topic label (2\u20135 words) describing the SUBJECT MATTER. "
        "Focus on WHAT happens, not WHERE \u2014 avoid labels like 'Chicago news', "
        "'local community news', or 'Illinois news' unless the geography itself "
        "is the distinguishing feature. Good labels: 'High School Basketball', "
        "'City Budget Disputes', 'Immigration Policy'. "
        'Example output: {"0": "City Council", "1": "Transit", ...}'
    )

    if on_progress:
        on_progress(f"labeling_{level_name}", 0.0,
                    f"Labeling {len(unique)} {level_name} topics in one batch\u2026")

    for rl_attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            with ANTHROPIC_SEMAPHORE:
                resp = client.messages.create(
                    model=LABEL_MODEL,
                    max_tokens=min(2048, 128 + 32 * len(unique)),
                    messages=[{"role": "user", "content": prompt}],
                    thinking=thinking_param(),
                )
            break
        except anthropic_sdk.RateLimitError as e:
            if rl_attempt >= RATE_LIMIT_MAX_RETRIES:
                raise
            pause = rate_limit_pause(rl_attempt, e)
            logging.warning("Batch label rate limited; waiting %.0fs (attempt %d/%d).",
                            pause, rl_attempt + 1, RATE_LIMIT_MAX_RETRIES)
            time.sleep(pause)

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = None
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None

    result = {}
    if isinstance(parsed, dict):
        for cid in unique:
            label = parsed.get(str(cid)) or parsed.get(int(cid)) if isinstance(parsed, dict) else None
            if isinstance(label, str) and label.strip():
                result[cid] = label.strip().strip('"').strip("'")

    missing = [cid for cid in unique if cid not in result]
    if missing:
        logging.warning("Batch label missed %d/%d clusters; filling individually.",
                        len(missing), len(unique))
        for cid in missing:
            indices = list(np.where(labels == cid)[0])
            result[cid] = _label_cluster(client, stories, indices, reduced)

    if on_progress:
        on_progress(f"labeling_{level_name}", 1.0,
                    f"Labeled {len(unique)}/{len(unique)} {level_name} topics")
    return result


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
        return PipelineResult(
            stories=stories,
            topics=topics,
            story_topics=story_topics,
            broad_topics=topics,
            specific_topics=topics,
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

        story_topics.append([bt])

    return PipelineResult(
        stories=stories,
        topics=broad_topics,
        story_topics=story_topics,
        broad_topics=broad_topics,
        specific_topics=specific_topics,
    )
