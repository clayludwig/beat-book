# Story extraction: how the ingest pipeline reads any format

A reporter has a folder. Inside the folder is a PDF of a Sunday op-ed, a Word document with three pasted articles, a JSON dump from a CMS export, a scraped RSS feed in JSON, a plain-text newsroom notebook, and a half-dozen URLs to live pages. Their job, before the beat-book pipeline can do anything useful, is to feed all of that into the tool and end up with a clean list of stories — each one with a title, an author, a date, a link, and a body that the rest of the pipeline can embed, cluster, and reason about.

The ingest stage exists to make that single step painless. From the reporter's perspective there is no upload-format menu: drag-and-drop everything, or paste a textarea of URLs, and the preview screen shows what was found.

The mechanism behind that experience is two stages, lives entirely in `ingest.py`, and is intentionally format-blind: nothing about the rest of the pipeline knows or cares whether a given story came from a docx, an RSS feed, or a scraped HTML page.

## The two stages, in one paragraph

**Stage 1** turns bytes into clean text. A format dispatcher routes each input to a converter — `markitdown` for office formats and PDFs, a custom JSON renderer for structured records, stdlib UTF-8 decoding for plain text, and an SSRF-protected HTTPS fetcher for URLs (which then dispatches the response body back through the same dispatcher). The output of stage 1 is always plain text or Markdown.

**Stage 2** turns clean text into structured stories. A single LLM call with a strict tool-use schema reads the document and emits a list of stories, each described by metadata fields (title, date, author, link) and two short verbatim snippets that mark where the body begins and ends. The server then slices the body out of the original text using those markers. The LLM never rewrites story content; it only points at it.

```
bytes ──► extract_text ──► clean text ──► normalize (LLM) ──► Story list
                                                  │
                                                  └── markers ──► server slices verbatim body
```

The rest of this doc walks through each stage and the design choices that fall out of "we want this to work for anything a reporter might upload."

## Stage 1: format dispatch

The dispatcher is a single function — `extract_text(filename, raw_bytes)` — that branches on file extension and produces a string. There are four meaningful branches.

### Office formats and PDFs via markitdown

`.docx`, `.doc`, `.pdf`, `.html`, `.pptx`, `.xlsx`, `.csv`, `.rtf`, `.epub` all go through `markitdown`. We write the bytes to a `NamedTemporaryFile` with the correct suffix so markitdown can dispatch on extension internally, then read back its Markdown output:

```python
def _extract_with_markitdown(filename: str, raw: bytes) -> str:
    from markitdown import MarkItDown
    suffix = _ext_of(filename) or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        md = MarkItDown()
        result = md.convert(tmp.name)
        return (result.text_content or "").strip()
```

Using a temp file rather than an in-memory stream is markitdown's preferred input shape — its dispatcher reads the path's extension to pick the right converter. The temp file is unlinked on close (`delete=True`), so nothing persists.

### JSON gets its own renderer

`.json` is special enough to merit its own path. RSS exports, CMS dumps, and scraped feed archives all show up as JSON, and they all share a rough shape: a list of records (or a dict wrapping a list under `entries` / `stories` / `items`), where each record has some subset of `title`, `headline`, `date`, `published`, `author`, `byline`, `link`, `url`, `content`, `summary`, `body`, `text`.

The JSON renderer, `_extract_json`, unwraps any common wrapper and renders each record as a small Markdown block with a heading and a metadata line:

```
# Council Approves $58M Budget
Date: 2026-04-12 · Author: Jane Smith · Link: https://...

The Talbot County School Board voted 6-1 on Tuesday...

---

# Easton High Football Wins Regional
...
```

Records are separated by `\n\n---\n\n` so that, in stage 2, the LLM sees natural story boundaries it can use to split. There is no special schema knowledge — if a record has fields the renderer doesn't recognize, it falls back to dumping the raw JSON for that record so the LLM at least has the data to look at.

One subtlety in the JSON renderer is HTML scrubbing on body fields. JSON-formatted news feeds frequently store article bodies as HTML strings (especially RSS-style exports where `content` is a `<p>…</p><p>…</p>` blob). If we leave those in, the LLM sees one stream of characters and the markers it returns won't line up with the characters the server actually slices from. `_clean_inline_html` runs first, unescaping entities, converting block tags to paragraph breaks, and stripping inline tags, so what the LLM reads is what the server has to work with.

### Plain text bypasses everything

`.md`, `.markdown`, `.txt`, `.log` are decoded as UTF-8 and passed through. No conversion, no normalization, no surprises.

### URLs become files

URL inputs aren't a separate code path so much as a prepended step. `fetch_url(url)` does the HTTPS fetch with SSRF protection, sniffs the response's `Content-Type`, picks an appropriate extension, and hands `(filename, bytes)` to the same `extract_text` dispatcher. An HTML page becomes a `.html` filename and goes through markitdown; a JSON endpoint becomes a `.json` filename and goes through the JSON renderer.

The SSRF protection is worth its own paragraph.

### URL fetching: every input is hostile until proven otherwise

A reporter pasting a textarea of URLs is one wrong character away from `http://10.0.0.1/admin` or `http://localhost:5432/`. The server is the one making the request, so anything the URL resolves to is reachable from the server's network position — that includes private LAN ranges, cloud metadata endpoints (the AWS `169.254.169.254` one is a classic), and whatever else the host can route to.

`fetch_url` resolves every hostname through `getaddrinfo` and rejects any address whose IP is private, loopback, link-local, multicast, reserved, or unspecified. All A and AAAA records are checked, not just the first one — a hostname that resolves to a public address on IPv4 and a loopback on IPv6 is treated as blocked:

```python
for info in infos:
    ip_str = info[4][0]
    ip = ipaddress.ip_address(ip_str)
    if (ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
        return True  # blocked
```

Unresolvable hostnames are also blocked (the conservative direction — better to refuse a typo than to allow a name that happens to resolve to something private). The HTTP layer is `httpx` with a 15-second timeout, `follow_redirects=True`, and a project-identifying User-Agent. Responses larger than 15 MB are refused; smaller ones are buffered and handed off.

## Stage 2: LLM normalization

By the time we reach stage 2, the input is always a single string of clean text or Markdown, regardless of how it got that way. Now the question is: what stories are in it, and what does each one say?

The naive answer is "ask the LLM to write out each story." Don't do that. The LLM is slow, expensive, and — most importantly — paraphrases. A reporter who needs to cite the exact phrasing of an article cannot afford to read a paraphrased version of it through their own tool.

So stage 2 is built around a different shape: the LLM identifies stories and points at them; the server reads them. The LLM's output is metadata plus pointers, not content.

### The tool schema

There is exactly one tool, `register_stories`, and the model is forced to call it via `tool_choice`. Its parameters define the shape of the answer:

```python
{
    "is_news_content": bool,
    "skip_reason": str,
    "stories": [
        {
            "title": str,
            "date": str,        # YYYY-MM-DD or ""
            "author": str,      # byline only
            "link": str,
            "body_starts_with": str,  # 30-80 VERBATIM chars
            "body_ends_with": str,    # 30-80 VERBATIM chars
            "confidence": "high" | "medium" | "low",
            "reasoning": str,
        }
    ],
}
```

A few things are doing work here:

- **`is_news_content` is the first decision.** If the document is meeting notes, an invoice, a contact sheet, a raw data export — anything that isn't journalism — the LLM is instructed to set this to `false`, leave `stories` empty, and explain what the document looks like in `skip_reason`. The preview UI shows excluded sources separately so the reporter knows the system noticed they uploaded a budget spreadsheet and chose not to invent stories out of it.
- **Metadata is inferred, content is not.** Title, date, and author can all be reasonably guessed from context (a `By Jane Smith` line is a byline, a `Published April 12, 2026` line is a publication date). The LLM has latitude here. But the body — the part the reporter will actually read — is found, not generated.
- **`body_starts_with` and `body_ends_with` are the pointers.** They are short, verbatim snippets from the original document. The server uses them to locate the start and end of the body in the original text and slices everything in between.
- **`confidence` and `reasoning` are for the reviewer.** The preview UI shows them so the reporter can quickly see which stories the LLM was unsure about and verify those first.

### The marker resolver

The model is told the snippets must appear verbatim. In practice, they don't always. The model occasionally:

- Drops or adds a space.
- Substitutes a curly quote for a straight quote.
- Truncates an extra character past where it meant to stop.
- Paraphrases a few words.

The marker resolver, `_resolve_marker_offset`, accepts a marker and a search start position and returns a character offset into the original text — or `-1` if nothing matches. It tries three strategies in order.

**Strategy 1 — exact substring search.** The fast and overwhelmingly common case. If `body_starts_with` appears verbatim, we use that position and stop.

**Strategy 2 — whitespace-normalized search.** If the exact match fails, we collapse all whitespace runs to a single space on both sides, then search again in normalized space. If we find a match, we walk through the original text counting normalized characters until we reach the matched position, and return the corresponding offset in the original. This handles the "the model collapsed double-spaces in the snippet" case without giving up.

```python
norm_text = _normalize_for_match(text[after:])
norm_marker = _normalize_for_match(marker)
norm_pos = norm_text.find(norm_marker)
if norm_pos >= 0:
    # Walk the original counting normalized chars until we reach norm_pos.
    i, j = after, 0
    prev_space = False
    while i < len(text) and j < norm_pos:
        ch = text[i]
        if ch.isspace():
            if not prev_space:
                j += 1
                prev_space = True
        else:
            j += 1
            prev_space = False
        i += 1
    return i
```

**Strategy 3 — first-five-words.** As a last resort, search for just the opening five words of the marker. This catches "the model paraphrased the snippet slightly past the fifth word" cases. The trade-off is that a five-word prefix is more likely to match somewhere by accident than the full snippet — but by this point we've already failed two more specific tests, and a noisy positive is better than no body at all.

If all three strategies fail, the resolver returns `-1` and the body is treated as empty. For single-story documents, the slicer falls back to "use the whole document" in that case — a multi-story document doesn't, because that would duplicate the full text across every story.

### The fallback ladder

Putting it together, `_slice_body` calls the resolver twice:

1. Find the start. If not found, give up on this story.
2. Find the end, starting from after the start. If not found, slice to end of document.

The end-marker miss is more forgiving than the start-marker miss because "this story runs to the end of the file" is a common case — especially for single-article uploads where there is no natural end-of-story signal.

After slicing, the resulting body is filtered: anything shorter than 20 characters is dropped (a sign the markers collided weirdly), and in the single-story-with-no-body case, the full document is substituted.

## The skip path

A reporter uploading a folder will inevitably include some files that aren't news content. The pipeline has two layers of defense against making things up from those:

1. **`is_news_content` at the LLM level.** The system prompt explicitly lists non-news examples (meeting notes, invoices, raw data, empty files) and tells the model to set the flag false and leave `stories` empty.
2. **Body filtering at the server level.** A story whose body slices to fewer than 20 characters is dropped silently; if all stories from a document drop, the document is reclassified as non-news with a fallback reason.

Both layers feed the same preview-UI behavior: the source appears in an "excluded" section with a short explanation, and is not included in the topic-discovery pipeline downstream.

## The 120k-character truncation

`MAX_NORMALIZE_CHARS = 120_000` caps the slice of the document the LLM actually sees. A 120k-character window is ~30k tokens at typical English density — comfortably within the model's input budget while leaving room for the system prompt, the schema, and the response.

If a document exceeds this, the truncation is noted in the user-prefix sent to the model:

> Note: the document was truncated to 120000 characters for parsing. Identify stories within this window only.

This is a deliberate trade-off. A 500-page PDF compendium of a year's coverage will not have every story extracted from a single call — but that's the right cost to pay for a single-pass design. Multi-pass extraction over a windowed document would add latency, complexity, and a join step that the LLM would have to be trained on. In practice, single uploads are rarely that long; a reporter handing the system a 500-page archive can chunk it themselves, and the system happily processes each chunk as a separate source.

## What this gets wrong

Honest failure modes:

- **Bodies that don't appear in the extracted text.** Markitdown sometimes drops the body of a story (image-only PDFs, complex tables, certain HTML layouts). The LLM correctly identifies a title and byline but its body markers fail to resolve, and the story drops. The preview UI shows this as an excluded story with a reason, but the failure mode is "we know there was a story there and we couldn't reach it" — not a great experience.
- **Multi-column PDFs with markitdown.** Two-column newspaper PDFs frequently come out of markitdown with column-1-line-1 immediately followed by column-2-line-1, which means a "story body" can interleave content from two adjacent stories. The LLM does its best with `body_starts_with` / `body_ends_with` but can land on a marker that straddles the two columns.
- **Stories where the title appears inside the body.** A self-referential intro ("This article will explain…") can fool the start-marker into landing on the title line rather than the actual first sentence. The system prompt is explicit about this ("Start with the first sentence of the article body itself, not the title or byline"), but the failure mode still happens occasionally.
- **No structured-field validation.** Date inference can come back as `"sometime in 2026"` or `"April"` if the model gets confused. We trust the schema to enforce shape but not value — a malformed date passes through to the preview UI as-is, where the reporter can correct it.
- **No fuzzy match across the corpus.** Each document is normalized independently; we don't check for duplicate stories across documents. A reporter uploading the same article twice (once as a docx, once as a URL) gets two copies in the pipeline.

The preview screen exists in part to let the reporter catch these. Title, date, author, and inclusion can all be edited inline before the confirmed list goes to `/process`.

## Why this approach

A few choices fall out of the goal of "support any format":

- **One pipeline, every format.** No privileged path. JSON gets a custom renderer because its structure is rich enough to leverage, but the LLM normalization step doesn't know or care whether its input came from JSON or a Word doc — it sees plain text in either case.
- **Marker-based body extraction.** The reporter cites these stories. They have to trust that what the tool says the story says is what the story says. The LLM as pointer-not-paraphraser is the load-bearing decision.
- **One LLM call per document.** Cost is proportional to input length, not story count. A document with twelve articles in it costs roughly the same as a document with one article and twelve paragraphs.
- **Server-side body assembly, not LLM-side.** The LLM's tool call returns a JSON object; assembling that into the final story dicts happens in plain Python, where any error mode is debuggable and any business rule (the 20-character drop, the single-story fallback, the date normalization) lives in one place.
- **Format dispatch is dumb on purpose.** The extension table in `extract_text` is a flat dispatch with no fallbacks-in-the-middle. If a file is `.docx`, markitdown handles it. If markitdown fails, we surface the failure rather than silently retrying as something else. Predictable failure beats clever recovery.

That's the whole story-extraction stage: a format-agnostic dispatcher, a single LLM call that points rather than paraphrases, a marker resolver that survives small imperfections in the LLM's output, and a preview UI that gives the reporter the last word before anything downstream sees the result.
