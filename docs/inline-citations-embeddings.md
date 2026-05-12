# Inline citations from sentence embeddings: how the beat book stays attributable

A reporter using the beat book needs to know which sentences came from which source story — both to verify claims and to follow the citation back to the article and byline they can call. The agent doesn't write notes like "according to source #3"; it paraphrases, summarizes, and synthesizes. So we attribute after the fact, by matching each generated sentence against the source corpus the agent had access to.

This is the embeddings-based inline citation pipeline. It lives in `citation_matcher.py`, runs as the last stage of the pipeline (after the draft agent and after the research agent), and produces the per-sentence attribution metadata the viewer uses to wrap clickable spans around any beat-book sentence whose nearest source neighbor is similar enough.

## The shape of the solution

The whole approach is three sentences long:

1. Embed every sentence in the source stories.
2. Embed every sentence in the beat book.
3. For each beat-book sentence, pick the source sentence with the highest cosine similarity. If that similarity clears a threshold, treat it as the citation.

Everything else is the texture you need to make those three sentences work in practice — sentence splitting that doesn't choke on "Mr." or "U.S.", Markdown segmentation that knows not to embed list bullets, batched API calls that don't make 800 round-trips, and a viewer-side rendering pass that decides what to actually show.

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
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"']|$)", protected)
    sentences = [s.replace("<<DOT>>", ".").strip() for s in sentences]
    return [s for s in sentences if s and len(s) > 10]
```

The actual split regex requires a sentence-ending punctuation mark followed by whitespace followed by a capital letter, quote, or end-of-string. The trailing length filter (`len(s) > 10`) is a cheap way to drop the inevitable two-character noise sentences that escape the abbreviation table — you'd rather miss a real ten-character sentence than create a citation for "OK." or "And.".

This is a regex-and-substitution approach, not a trained sentence-boundary model, and that's a deliberate choice: it's predictable, has no install footprint, and the failure modes (an over-eager split on an unrecognized abbreviation) are visible in the output rather than hidden in a model's confidence score.

## Step 2: segmenting the beat book Markdown

The beat book is Markdown, not prose. If you embed every line of it indiscriminately, you'll waste tokens on `## Key People`, on `- John Smith, Director of Public Affairs`, and on the literal `|---|---|---|` of a table separator. None of those should ever get a citation: headings aren't claims, list items are usually too short and too schematic to match meaningfully, and table rows would break the Markdown's own GFM parsing if we wrapped them in HTML spans (a row that doesn't start with `|` ceases to be a row).

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

The output is a flat list of entries, each with `content` and `needs_embedding`. Paragraph lines explode into one entry per sentence (so a five-sentence paragraph becomes five separate entries, each independently citable); everything else is a single entry that passes through verbatim. This is what lets the downstream code embed in bulk and then walk the entry list to assemble the JSON output, knowing exactly which entries got an embedding and which didn't.

## Step 3: batched embeddings

Both passes — embedding the source corpus and embedding the beat book — go through `_embed_many`, which slices texts into chunks of `EMBED_BATCH_SIZE = 256` and dispatches each chunk to OpenAI's `text-embedding-3-small` in a single HTTP call:

```python
EMBED_BATCH_SIZE = 256

def _embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    cleaned = [t if t.strip() else " " for t in texts]
    resp = client.embeddings.create(model=EMBED_MODEL, input=cleaned)
    return [item.embedding for item in resp.data]
```

The OpenAI embeddings API accepts up to 2048 inputs per call; the lower cap of 256 keeps individual payloads small enough that a transient connection problem retries cheaply. The empty-string defense (`t if t.strip() else " "`) is mandatory: the API rejects an empty input and the rejection is for the *whole batch*, so a single zero-length sentence anywhere in the list nukes the round-trip.

A typical small corpus — a few hundred articles, each maybe twenty to fifty sentences — is six or eight thousand sentences total, which is thirty-ish HTTP calls. Even on an unhurried connection that completes in well under a minute. There's no caching layer between this code and the API; if you re-run the pipeline you re-embed. That's tolerable here because the source corpus changes between runs (each run is a different reporter with a different set of source stories) and embedding costs at this scale are pennies. The pipeline elsewhere does cache embeddings (see `.cache/embeddings.pkl` in the topic-discovery stage), but the citation matcher has been left uncached on purpose.

## Step 4: brute-force matching

Once both sides have embeddings, matching is straightforward cosine similarity, comparing every beat-book sentence against every source sentence:

```python
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
```

This is `O(M × N)` where `M` is the number of beat-book sentences (low hundreds) and `N` is the number of source sentences (low thousands). It's a few hundred thousand multiply-adds per beat book — milliseconds, even in pure Python. Anything fancier would be premature: no FAISS, no Annoy, no IVF index, no GPU. The day a single beat book sources from a million articles, that calculus changes; today it doesn't, and a brute-force loop is far easier to debug than a vector-database integration.

The match function returns the best source sentence as a plain dict — article id, sentence text, sentence index, similarity score, article title, date, author. That dict gets attached to the entry as the citation payload.

## Step 5: the threshold

Cosine similarity always returns *some* number, even between two utterly unrelated sentences. The "best" match for a beat-book sentence that has no real source — for example, a fact the research agent added from the open web — is still going to be a real number, and the corresponding source sentence is going to look like a citation in the output unless something filters it out.

That something is a threshold, applied in the viewer rather than the matcher:

```javascript
const SIMILARITY_THRESHOLDS = {
    immigration_enforcement_beat_book: 0.67,
};
const SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLDS[bookStem] ?? 0.65;
```

The default is 0.65; one beat book — `immigration_enforcement_beat_book` — has been bumped to 0.67 because its source corpus produced an unusually high baseline of incidental similarity (lots of similar boilerplate around DHS press releases). The right way to tune this for a new beat is to scan the borderline matches by hand: pull the entries with similarity between 0.60 and 0.70 from the JSON, look at the beat-book sentence next to its claimed source sentence, and decide where the line falls between "yes, that sentence really came from this source" and "no, the model is just pattern-matching DHS-shaped text." Bump or lower the threshold for that book accordingly.

The threshold lives in the viewer rather than in the matcher because the matcher's output is the underlying data — every entry's best match, with its real similarity score — and the threshold is a presentation choice that you might want to revisit without re-running the embedding pipeline. If you ever want a "show all citations regardless of confidence" debug mode, it's a one-line change in the viewer.

## Step 6: rendering with run-grouping

A beat book paragraph that summarizes one source story typically has five or six sentences, all of which match (correctly) to the same `article_id`. Wrapping every one of them in a clickable span is visually punishing: the entire paragraph turns into a wall of underlined links, the user can't tell where one source ends and another begins, and the click target becomes ambiguous.

The viewer's fix is run-grouping. It walks the entry list keeping track of the previous entry's source. Only the *first* entry in a run of consecutive same-source entries gets wrapped:

```javascript
let previousSource = null;
const markdown = beatbookData.map((entry, index) => {
    const hasSufficientSimilarity = entry.similarity !== undefined ? entry.similarity >= SIMILARITY_THRESHOLD : true;
    const isValidSource = entry.source && storiesData.some(s => s.article_id === entry.source);
    const isTableRow = entry.content.trimStart().startsWith('|');
    if (isValidSource && hasSufficientSimilarity && !isTableRow) {
        const isFirstInRun = entry.source !== previousSource;
        previousSource = entry.source;
        if (isFirstInRun) {
            return `[[SOURCE:${index}]]${entry.content}[[/SOURCE:${index}]]`;
        }
    } else {
        previousSource = null;
    }
    return entry.content;
}).join('\n');
```

The `[[SOURCE:N]]…[[/SOURCE:N]]` sentinels are a small trick. We can't wrap the sentence in a real `<span>` before handing the text to `marked.parse()`, because raw HTML inside Markdown breaks paragraph wrapping in unpredictable ways. So we wrap it in a unique placeholder, let Marked do its parsing, then post-process the rendered HTML to swap the sentinels for the actual span:

```javascript
let html = marked.parse(markdown);
beatbookData.forEach((entry, index) => {
    if (entry.source) {
        const regex = new RegExp(`\\[\\[SOURCE:${index}\\]\\](.*?)\\[\\[\\/SOURCE:${index}\\]\\]`, 'g');
        html = html.replace(regex, (match, content) => {
            return `<span class="sourced-content" onclick="openArticle('${entry.source}')" onmouseenter="showPreview('${entry.source}', event)" onmouseleave="hidePreview()">${content}</span>`;
        });
    }
});
```

The result: every "run" of same-source sentences gets a single visible link on its first sentence, the whole paragraph reads cleanly, and a click on the link opens the source article in a side panel.

## What this approach gets wrong

It's worth being honest about the failure modes, because they're real and a reporter using this tool should know where to be skeptical of the highlighting:

- **Synthesized sentences collapse to one match.** A beat-book sentence that fuses claims from three different source articles will only get attribution to whichever of the three it happens to be most similar to. The other two contributions disappear from the citation trail.
- **Below-threshold sentences get no attribution at all, even if they have one.** A beat-book sentence that paraphrases its source heavily (different vocabulary, restructured grammar) can fall below the threshold and look uncited. The fix in those cases is to lower the threshold or rewrite the sentence closer to the source — but you don't get a warning that this happened; the sentence just appears un-highlighted.
- **A "best" match always exists.** If the agent inserts a fact it made up — or a fact it pulled from the open web during the research stage — the matcher will still return a closest-source-sentence, and only the threshold prevents that from rendering as a citation. A poorly-chosen threshold (too low) will surface bogus links.
- **Sentence-level granularity is the unit.** No clause-level highlighting; no paragraph-level fall-through; no tracing a single named entity from beat book back to source mention. This was a deliberate simplification — sentences are the natural attributional unit for journalism — but it means a very long sentence with a contestable mid-clause and an uncontestable head-clause gets one citation for the whole thing.

## Why this approach in particular

A few things made this the right shape, given the constraints:

- **No vector database.** A FAISS or LanceDB index would let you scale to millions of sentences, but the corpus per beat book is hundreds to low thousands. Brute-force search over a Python list of dicts is fast enough that the cost of the dependency would dwarf the cost of the search.
- **Sentence-level rather than paragraph-level.** The viewer wants to highlight the specific claim, not the whole bullet. A paragraph-level approach would over-attribute.
- **OpenAI's small embedding model rather than a local one.** Anthropic — which hosts the project's chat slots — does not offer an embedding API, only chat. A self-hosted Ollama with `mxbai-embed-large` would work, but the project is meant to run anywhere without a local daemon, so a hosted embedding API was the only path that wouldn't bind a fresh checkout to a particular machine. The marginal cost of `text-embedding-3-small` at this scale is too low to argue with.
- **Threshold in the viewer, not the matcher.** Keeps the underlying JSON debuggable and tweak-able without re-running embeddings.

## How this stage cohabits with the research agent

The pipeline runs in three stages: the draft agent produces a beat book from the source corpus, the research agent deepens it with live web research, and then the citation matcher runs. By the time the matcher sees the markdown, some sentences came from source stories the agent originally read, and some sentences came from the open web (a Tribune article from last month, an Open Data Chicago dataset, a city council vote).

The matcher doesn't need to know which is which. The web-sourced sentences won't have a meaningful match in the reporter's source corpus and will fall below the threshold — exactly the right outcome, since those sentences shouldn't link to a source they didn't come from. The threshold is what makes this graceful: the citation matcher doesn't need provenance metadata about who wrote each sentence; the absence of a real source naturally surfaces as the absence of a link in the rendered output.

That's the whole composition: a research agent that adds live context inline, a citation matcher that attributes whatever it can attribute and silently drops the rest, and a viewer that renders the surviving citations as discreet, clickable spans. The reporter ends up with a document that reads like prose, behaves like a footnoted article when they hover, and lets them follow any specific claim back to its source in a single click.
