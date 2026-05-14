# Inline citations from passage embeddings: how the beat book stays attributable

A reporter using the beat book needs to know which sentences came from which source story — both to verify claims and to follow the citation back to the article and byline they can call. The agent doesn't write notes like "according to source #3"; it paraphrases, summarizes, and synthesizes. So we attribute after the fact, by matching each generated sentence against the source corpus the agent had access to.

This is the embeddings-based inline citation pipeline. It lives in `citation_matcher.py`, runs as the last stage of the pipeline (after the draft agent and after the research agent), and produces the per-sentence attribution metadata the viewer uses to wrap clickable spans around beat-book sentences with their supporting source passages, plus the sub-span inside each passage that actually carries the claim.

## The shape of the solution

The earlier version of this matcher was three sentences long: embed every source sentence, embed every beat-book sentence, take the argmax cosine match. It worked, but it had four real problems — paraphrased claims pinned to a single arbitrary source, synthesized sentences losing the contributing sources, no calibrated "no good match" signal, and sentence-vs-sentence matching with no way to point at the specific span inside a paragraph that supported a claim.

The current pipeline still starts with embeddings and cosine similarity but adds five things on top:

1. **Passages, not sentences, on the source side.** Each source story is chopped into 100-word sliding windows with 16-word overlap. The window keeps its character offset back into the original source text, so a passage hit resolves to a quoted span the viewer can highlight.
2. **Top-K retrieval, not argmax.** For each beat-book sentence we keep up to five supporting passages, not just the best one — a sentence that synthesizes two stories can cite both.
3. **A per-corpus calibrated threshold.** We sample random `(beat_book_sentence, source_passage)` pairs to estimate the noise floor of the similarity distribution, then cut off at `noise_mean + 3·sigma` (with an absolute floor of 0.40). Below the threshold, a candidate is dropped — no citation rather than a confidently-wrong one.
4. **Context-sum on the beat-book side.** Pronoun-heavy or short sentences ("He denied it.") have no embedding signal alone. Each beat-book sentence's query vector is `0.6·prev + 1.0·self + 0.4·next`, summing the embeddings of paragraph-adjacent neighbors before matching.
5. **Leave-one-out sub-window highlighting.** Once a passage is picked, we split it into six overlapping sub-windows and re-embed `passage_minus_subwindow` for each. The sub-windows whose removal most degrades similarity are the ones carrying the claim; we surface the top two as highlight offsets the viewer paints in stronger color.

Everything else is plumbing — sentence splitting that doesn't choke on "Mr." or "U.S.", Markdown segmentation that knows not to embed list bullets, NumPy vectorization so the matmul stays milliseconds, and a viewer-side rendering pass that decides what to actually show.

## Step 1: sentence splitting that survives "Mr." and "U.S."

A naïve split-on-period turns "Mr. Johnson said that the U.S. Department of Justice declined to comment." into five sentence-ish fragments. The fix is to mask abbreviations and decimals before splitting, then unmask them after. `_ABBREVIATIONS` is the table of every period-bearing token we expect to see in news copy: titles (`Mr.`, `Mrs.`, `Dr.`, `Sgt.`, `Sen.`), initials (`U.S.`, `D.C.`, `Ph.D.`), latinate filler (`etc.`, `vs.`), street suffixes, time markers (`a.m.`, `p.m.`), and the rest.

```python
def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    protected = text
    for pattern, replacement in _ABBREVIATIONS:
        protected = re.sub(pattern, replacement, protected, flags=re.IGNORECASE)
    # Protect decimals (3.5, $10.99)
    protected = re.sub(r"(\d)\.(\d)", r"\1<<DOT>>\2", protected)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", protected)
    sentences = [s.replace("<<DOT>>", ".").strip() for s in sentences]
    return [s for s in sentences if s and len(s) > 10]
```

The actual split regex requires a sentence-ending punctuation mark followed by whitespace followed by a capital letter or quote. The trailing length filter (`len(s) > 10`) is a cheap way to drop the inevitable two-character noise sentences that escape the abbreviation table — you'd rather miss a real ten-character sentence than create a citation for "OK." or "And.".

This is a regex-and-substitution approach, not a trained sentence-boundary model, and that's a deliberate choice: it's predictable, has no install footprint, and the failure modes are visible in the output rather than hidden in a model's confidence score.

## Step 2: segmenting the beat book Markdown

The beat book is Markdown, not prose. If you embed every line of it indiscriminately, you'll waste tokens on `## Key People`, on `- John Smith, Director of Public Affairs`, and on the literal `|---|---|---|` of a table separator. None of those should ever get a citation: headings aren't claims, list items are usually too short and too schematic to match meaningfully, and table rows would break the Markdown's own GFM parsing if we wrapped them in HTML spans.

So segmentation is a two-pass loop that flags each line as either "embed me" or "pass through":

```python
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
```

The output is a flat list of entries, each with `content` and `needs_embedding`. Paragraph lines explode into one entry per sentence; everything else passes through verbatim. The viewer walks the entry list to assemble HTML, knowing exactly which entries got an embedding pass and which didn't.

## Step 3: sliding-window passages on the source side

Source stories don't get split into sentences any more. They get split into 100-word sliding windows with 16-word overlap — roughly 128-token windows with the kind of ~12% overlap that the semantic-search literature (`semantra`, several embedding-search demos) settled on by trial and error. The motivation is straightforward: a paragraph-shaped chunk gives the embedding model real context to encode against, and that context is what makes paraphrased and summarized claims attribute correctly. A single source sentence is often missing the very subject or verb the beat-book sentence is referring to.

```python
PASSAGE_WORDS = 100
PASSAGE_OVERLAP_WORDS = 16

def _passage_windows(text: str, ...) -> List[Dict[str, Any]]:
    words = _tokenize_with_offsets(text)
    step = max(1, window_words - overlap_words)
    out = []
    i = 0
    while i < n_words:
        j = min(n_words, i + window_words)
        first_word_start = words[i][1]
        last_word_end = words[j - 1][2]
        out.append({
            "text": text[first_word_start:last_word_end],
            "char_offset": first_word_start,
            "char_length": last_word_end - first_word_start,
            "word_start": i,
            "word_end": j,
        })
        if j >= n_words: break
        i += step
    return out
```

Every passage carries the character offset back into the original source text. That's what makes the leave-one-out highlighting later in the pipeline work — once we know the passage carries the claim, we want to point at the exact sub-span inside it, not at "somewhere in this paragraph."

## Step 4: batched embeddings

Both passes — embedding the source passages and embedding the beat-book sentences — go through `_embed_many`, which slices texts into chunks of `EMBED_BATCH_SIZE = 256` and dispatches each chunk to OpenAI's `text-embedding-3-small` in a single HTTP call.

The OpenAI embeddings API accepts up to 2048 inputs per call; the lower cap of 256 keeps individual payloads small enough that a transient connection problem retries cheaply. The empty-string defense (`t if t.strip() else " "`) is mandatory: the API rejects an empty input and the rejection is for the *whole batch*, so a single zero-length sentence anywhere in the list nukes the round-trip.

A typical small corpus — a few hundred articles, each maybe ten passages — is a few thousand passages plus a few hundred beat-book sentences, which is a handful of HTTP calls. Even on an unhurried connection that completes in well under a minute. There's no caching layer between this code and the API; if you re-run the pipeline you re-embed.

## Step 5: context-sum queries

A beat-book sentence like "He denied it." has almost no embedding signal on its own — three closed-class words and a pronoun. The same goes for short verbless headers ("The aftermath."), and for any sentence whose meaning depends on antecedents we'd otherwise be embedding away.

The fix is to weight each beat-book sentence's query vector with its paragraph-adjacent neighbors before matching:

```python
CTX_WEIGHT_PREV = 0.6
CTX_WEIGHT_SELF = 1.0
CTX_WEIGHT_NEXT = 0.4

def _context_sum_embeddings(raw, indices_in_sentence_stream):
    out = raw.copy() * CTX_WEIGHT_SELF
    for i in range(len(raw)):
        my_pos = indices_in_sentence_stream[i]
        if i - 1 >= 0 and indices_in_sentence_stream[i - 1] == my_pos - 1:
            out[i] += CTX_WEIGHT_PREV * raw[i - 1]
        if i + 1 < len(raw) and indices_in_sentence_stream[i + 1] == my_pos + 1:
            out[i] += CTX_WEIGHT_NEXT * raw[i + 1]
    return out
```

The previous sentence gets more weight than the next because pronouns and ellipsis typically refer backward, not forward. Paragraph boundaries gate the sum — a sentence at position zero in its paragraph has no `prev`. Then everything gets L2-normalized so the matmul is a dot product directly.

## Step 6: vectorized matching

Once both sides have embeddings, we L2-normalize both matrices and multiply them. That's it — no Annoy, no FAISS, no IVF index.

```python
beat_emb_norm = _l2_normalize(ctx_beat)               # (M, d)
source_emb_norm = source_index["embeddings"]          # (N, d), already normalized
sim_matrix = beat_emb_norm @ source_emb_norm.T        # (M, N)
```

For our corpus sizes (a few hundred beat-book sentences × a few thousand source passages) this fits comfortably in memory and runs in single-digit milliseconds. Vector databases would help if either side were larger by an order of magnitude or two; today they aren't.

## Step 7: per-corpus calibrated threshold

Cosine similarity always returns *some* number, even between two utterly unrelated sentences. The earlier version of this pipeline hardcoded a per-book threshold in the viewer (`SIMILARITY_THRESHOLDS = { immigration_enforcement_beat_book: 0.67, ... }`) — every new beat book required hand-tuning the cutoff or accepting the 0.65 default. That's not robust.

We replace it with a calibration step that runs once per corpus, against the same embedding matrices we just built:

```python
CALIB_RANDOM_SAMPLES = 4000
CALIB_SIGMA = 3.0
CALIB_ABSOLUTE_FLOOR = 0.40

rng = np.random.default_rng(seed=42)
beat_idx = rng.integers(0, n_beat, size=n_samples)
src_idx = rng.integers(0, n_src, size=n_samples)
sims = np.einsum("ij,ij->i", beat_emb_norm[beat_idx], source_emb_norm[src_idx])
noise_mean = float(np.mean(sims))
noise_std = float(np.std(sims))
threshold = max(CALIB_ABSOLUTE_FLOOR, noise_mean + CALIB_SIGMA * noise_std)
```

Random pairs of beat-book sentences and source passages are, by construction, *not* citations of each other. Their similarities form the noise floor of the corpus — the baseline of "this is what irrelevant looks like, here." Three standard deviations above that floor catches roughly 99.7% of true noise; matches at or above the threshold are the ones standing meaningfully above incidental similarity. The 0.40 absolute floor protects against the degenerate case where the corpus has very high baseline similarity (e.g., a stack of near-duplicate press releases) and the noise-floor formula would produce something laughably low.

The calibration block goes into the output JSON. The viewer reads it and displays the threshold in the header, so a reporter who wants to know "how confident were the citations in this book" can see it at a glance.

## Step 8: top-K retrieval

`argmax` would give us one best support per beat-book sentence. `argpartition` gives us the top K (we use K=5) in O(N) per sentence, and we then sort those K by similarity to pick winners in order:

```python
k = min(TOP_K, sim_matrix.shape[1])
for row_i in range(sim_matrix.shape[0]):
    row = sim_matrix[row_i]
    cand_idx = np.argpartition(-row, k - 1)[:k]
    cand_idx = cand_idx[np.argsort(-row[cand_idx])]
    per_sentence = []
    for col_i in cand_idx:
        sim = float(row[col_i])
        if sim < threshold: break
        per_sentence.append({"passage": global_passages[col_i], "similarity": sim})
```

A sentence that synthesizes claims from two source stories now ends up with two supports — both visible in the data, even if the viewer chooses to show only the top one inline.

## Step 9: leave-one-out sub-window highlighting

A 100-word passage is the unit we matched against, but it's too long to highlight in a viewer pane without losing the value of the highlight. The leave-one-out trick comes from `semantra`'s `/api/explain` endpoint, slightly adapted: split the matched passage into six overlapping sub-windows of about a third of the passage each, re-embed `passage_minus_subwindow` for each one, and rank sub-windows by how much removing them hurts similarity to the query.

```python
sims = passage_minus_subwindow_embeddings @ query_vec
contributions = base_similarity - sims   # bigger = removing this sub-window hurt more
order = np.argsort(-contributions)
top_highlights = [sub_ranges[i] for i in order[:LOO_TOP_HIGHLIGHTS] if contributions[i] > 0]
```

The two sub-windows with the largest "removal hurt most" scores are the ones carrying the claim. They go into the output JSON as `highlights[]` — char offsets the viewer paints in stronger amber inside the lighter-amber passage band.

This is by far the most expensive step in the pipeline: it's an additional embedding batch per kept candidate, with `LOO_SUBWINDOWS_PER_PASSAGE = 6` embeds per passage. We run it only on candidates that passed the threshold and made the top-K cut, so it scales with the number of *useful* citations rather than the size of the corpus.

## Step 10: the JSON shape and the viewer

The matcher returns a single dict:

```json
{
  "calibration": { "threshold": 0.51, "noise_mean": 0.18,
                   "noise_std": 0.11, "sigma": 3.0, "samples": 4000 },
  "entries": [
    { "content": "...", "passthrough": false,
      "supports": [
        { "article_id": "story-3", "article_title": "...", "article_date": "...",
          "passage_text": "the full ~100-word matched window",
          "passage_offset": 1234, "passage_length": 678,
          "similarity": 0.72,
          "highlights": [ { "char_offset": 1289, "char_length": 56,
                            "contribution": 0.18 } ] },
        ...
      ] },
    ...
  ]
}
```

The viewer (`static/viewer/viewer.js`) reads the calibration and surfaces it as a header label. For each entry with at least one support, it wraps the sentence's run-leader with a clickable span (run-grouping is unchanged — consecutive same-source sentences still get one visible link on the first). Clicking the link opens the source article in a side panel and the panel renders the raw article text with two tiers of `<mark>` highlighting: the matched 100-word passage in light amber, and the leave-one-out sub-window(s) inside it in stronger amber. The panel scrolls to the first highlight.

A small score chip after each citation link shows the similarity value — color-coded into three bands (green ≥0.6, amber ≥0.5, grey below). A reporter who wants to know "how confidently is this citation pinned to that source" can see it without opening the side panel.

## What this approach still gets wrong

It's worth being honest about the failure modes:

- **The calibration floor is statistical, not labeled.** We're computing a noise threshold from random pairs, not from labeled "matches" vs. "non-matches." The threshold catches obvious noise but doesn't guarantee that every above-threshold hit is a real attribution — a beat-book sentence and a source passage can share enough surface signal to clear three sigma without actually being about the same thing.
- **Top-K shows the top candidate inline only.** The data carries up to five supports per sentence, but the current viewer only renders the top one. A reporter who wants to see alternates has to open the JSON. (This is an obvious next viewer iteration.)
- **Leave-one-out is local.** It tells you which sub-span inside the matched passage most contributes to the match — but if the *real* supporting sub-span is split across two passage windows, the sub-window we highlight in either window is at best half the story.
- **Synthesized sentences from the open web are still hit-or-miss.** The research agent adds material from web searches that the source corpus doesn't contain. Those sentences should fall below threshold and get no citation. They usually do, but a sentence that happens to share vocabulary with a source story can still cross the threshold and get a misleading link.
- **Run-grouping still hides individual claims.** A paragraph of five sentences all citing the same source gets one visible link on the first sentence. That's good for readability, but a reporter who wants to verify the *third* sentence specifically has to click the run leader and trust that the passage covers all five.

## Why this approach in particular

- **Passages on the source side, sentences on the beat-book side.** The asymmetry is intentional. The beat-book is what we're trying to attribute — sentence-level granularity is the natural unit of a claim. The source corpus is what we're attributing *from* — passage-level chunks give the embedding model real context to encode against and are what make paraphrased and synthesized claims attribute correctly.
- **NumPy, not a vector database.** At a few thousand vectors per side, a single matmul is faster than any approximate-nearest-neighbor index would be after the query overhead. The day a single beat book sources from a million articles, that calculus changes; today it doesn't.
- **Calibration in the data, not in the viewer.** Hardcoding a threshold per book meant every new beat book required hand-tuning. Sampling random pairs is one shot, deterministic (we seed the RNG at 42), and produces a number any reader of the JSON can interrogate.
- **`text-embedding-3-small`, hosted.** Anthropic — which hosts the project's chat slots — does not offer an embedding API. A local Ollama with `mxbai-embed-large` would work, but the project is meant to run anywhere without a local daemon, so a hosted embedding API was the only path that wouldn't bind a fresh checkout to a particular machine.

## How this stage cohabits with the research agent

The pipeline runs in three stages: the draft agent produces a beat book from the source corpus, the research agent deepens it with live web research, and then the citation matcher runs. By the time the matcher sees the markdown, some sentences came from source stories the agent originally read, and some sentences came from the open web.

The matcher doesn't need to know which is which. The web-sourced sentences won't have a meaningful match in the reporter's source corpus and will fall below the calibrated threshold — exactly the right outcome, since those sentences shouldn't link to a source they didn't come from. The threshold is what makes this graceful: the citation matcher doesn't need provenance metadata about who wrote each sentence; the absence of a real match naturally surfaces as the absence of a link in the rendered output.

That's the whole composition: a research agent that adds live context inline, a citation matcher that attributes whatever the data supports above noise and silently drops the rest, and a viewer that renders the surviving citations as discreet, clickable spans with the actual supporting sub-span highlighted in the source. The reporter ends up with a document that reads like prose, behaves like a footnoted article when they hover, and lets them follow any specific claim back to its exact source sentence in a single click.
