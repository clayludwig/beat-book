// Beat book viewer.
// Loads `/output/<stem>.json` and `/output/<stem>_sources.json`
// where <stem> comes from the ?book= query param.

const params = new URLSearchParams(window.location.search);
const bookStem = params.get('book') || 'beat_book';
const beatbookFile = `/output/${encodeURIComponent(bookStem)}.json`;
const storiesFile = `/output/${encodeURIComponent(bookStem)}_sources.json`;

// Show the stem (de-underscored, title-cased) in the header.
document.getElementById('siteTitle').textContent = prettifyTitle(bookStem);
document.title = `Beat Book — ${prettifyTitle(bookStem)}`;

function prettifyTitle(stem) {
    return stem
        .replace(/[_\-]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\b\w/g, c => c.toUpperCase());
}

let storiesData = [];
let currentArticleId = null;
// citationsByNumber[N] = { number, sourceKey, articleId, articleTitle,
//   articleAuthor, articleDate, passageText, passageOffset, passageLength,
//   similarity, claimText }  — every visible inline chip has its own
//   sequential number, but several numbers can map to the same source.
const citationsByNumber = {};
// sourcesByKey[key] = { primary, numbers: [N1, N2, ...], firstSeen, claimText }
// — one entry per unique source (article + passage). The footnotes section
// iterates this so each source surfaces ALL the inline numbers that
// reference it.
const sourcesByKey = {};

function closeArticle() {
    document.getElementById('appContainer').classList.remove('split-view');
    currentArticleId = null;
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeArticle();
    }
});

document.addEventListener('click', (e) => {
    const appContainer = document.getElementById('appContainer');
    const articlePanel = document.getElementById('articlePanel');

    if (appContainer.classList.contains('split-view') &&
        !articlePanel.contains(e.target) &&
        !e.target.closest('.footnote-ref, .footnote-link, .footnote-item')) {
        closeArticle();
    }
});

// Open a source article by citation number. Inline [N] markers and footnote
// items both call this with the same number, so the source panel content is
// identical regardless of which entry point the reader used.
function openCitation(number) {
    const c = citationsByNumber[number];
    if (!c) return;
    openArticle(c.articleId, c);
}

// Render the footnotes section. One entry per UNIQUE source — each entry
// shows every inline number that points back to it as a clickable chip.
function renderFootnotesSection(byKey) {
    const sources = Object.values(byKey).sort((a, b) => a.firstSeen - b.firstSeen);
    const items = sources.map(src => {
        const primary = src.primary;
        const meta = [];
        if (primary.article_author) meta.push(formatAuthorName(primary.article_author));
        if (primary.article_date) meta.push(primary.article_date);
        const metaStr = meta.length ? ` — ${meta.join(', ')}` : '';
        const passageHtml = primary.passage_text
            ? `<blockquote class="footnote-passage">${escapeHtml(primary.passage_text)}</blockquote>`
            : '';
        const titleAttr = (primary.article_title ? `Open: ${primary.article_title}` : 'Open source')
            .replace(/"/g, '&quot;');

        // Sorted, comma-separated list of all inline numbers that point here.
        const nums = [...src.numbers].sort((a, b) => a - b);
        const numberChips = nums.map(n =>
            `<a class="footnote-number" onclick="openCitation(${n})" title="Inline citation ${n}">${n}</a>`
        ).join('');

        return `<li id="footnote-source-${src.firstSeen}" class="footnote-item">
            <span class="footnote-numbers">${numberChips}</span>
            <a class="footnote-link" onclick="openCitation(${nums[0]})" title="${titleAttr}">${escapeHtml(primary.article_title || 'Untitled')}</a><span class="footnote-meta">${escapeHtml(metaStr)}</span>
            ${passageHtml}
        </li>`;
    });
    return `<section class="footnotes" aria-label="Sources">
        <h2 class="footnotes-heading">Sources</h2>
        <ol class="footnotes-list">${items.join('')}</ol>
    </section>`;
}

// Hover preview
let previewTimeout = null;
const preview = document.getElementById('sourcePreview');

function showPreview(articleId, event) {
    const story = storiesData.find(s => s.article_id === articleId);
    if (!story) return;

    document.getElementById('previewTitle').textContent = story.title || 'Untitled';
    const authorName = formatAuthorName(story.author);
    document.getElementById('previewAuthor').textContent = authorName !== 'Unknown' ? `By ${authorName}` : '';
    document.getElementById('previewDate').textContent = story.date || '';

    const articleContent = extractArticleContent(story.content);
    const contentPreview = articleContent
        ? articleContent.replace(/\n/g, ' ').substring(0, 300) + '...'
        : 'No content available.';
    document.getElementById('previewContent').textContent = contentPreview;

    positionPreview(event);

    previewTimeout = setTimeout(() => {
        preview.classList.add('visible');
    }, 150);
}

function positionPreview(event) {
    const linkElement = event.target;
    const mouseX = event.clientX;
    const rect = linkElement.getBoundingClientRect();
    const previewWidth = 340;
    const previewHeight = 200;
    const gap = 8;
    const headerHeight = 52;

    preview.classList.remove('above', 'below');

    let left = mouseX - (previewWidth / 2);
    if (left < 10) left = 10;
    if (left + previewWidth > window.innerWidth - 10) {
        left = window.innerWidth - previewWidth - 10;
    }

    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top - headerHeight;

    let top;
    if (spaceBelow >= previewHeight + gap || spaceBelow >= spaceAbove) {
        top = rect.bottom + gap;
        preview.classList.add('below');
    } else {
        top = rect.top - previewHeight - gap;
        preview.classList.add('above');
    }

    if (top < headerHeight + gap) {
        top = headerHeight + gap;
    }

    preview.style.left = left + 'px';
    preview.style.top = top + 'px';
}

function hidePreview() {
    clearTimeout(previewTimeout);
    preview.classList.remove('visible', 'above', 'below');
}

document.addEventListener('DOMContentLoaded', () => {
    const mainPanel = document.querySelector('.main-panel');
    if (mainPanel) {
        mainPanel.addEventListener('scroll', hidePreview, { passive: true });
    }
});

setTimeout(() => {
    const mainPanel = document.querySelector('.main-panel');
    if (mainPanel) {
        mainPanel.addEventListener('scroll', hidePreview, { passive: true });
    }
}, 100);

// Format author name: strip emails, title-case, join multiples.
function formatAuthorName(author) {
    if (!author) return 'Unknown';

    let cleaned = author.replace(/\s*[\w.-]+@[\w.-]+\.\w+\s*/g, ' ').trim();

    const authors = cleaned.split(';').map(name => {
        return name.trim()
            .toLowerCase()
            .replace(/\b\w/g, char => char.toUpperCase());
    }).filter(name => name.length > 0);

    return authors.join(', ') || 'Unknown';
}

// Trim source content. Removes Talbot's "Read News Document" header if present
// (harmless no-op for other sources), strips trailing copyright lines, and
// inserts missing paragraph breaks.
function extractArticleContent(content) {
    if (!content) return '';

    let result = content;

    const marker = 'Read News Document';
    const markerIndex = result.indexOf(marker);
    if (markerIndex !== -1) {
        result = result.substring(markerIndex + marker.length).trim();
    }

    const copyrightIndex = result.indexOf('© Copyright');
    if (copyrightIndex !== -1) {
        result = result.substring(0, copyrightIndex).trim();
    }

    // Defensive HTML cleanup: turn leftover <p>/<br> into paragraph breaks
    // and strip remaining tags. Pipeline-side cleanup should already handle
    // this, but old exports may still carry raw markup.
    if (/<[a-z!\/][^>]*>|&lt;[a-z]/i.test(result)) {
        const decode = (s) => {
            const ta = document.createElement('textarea');
            ta.innerHTML = s;
            return ta.value;
        };
        result = decode(decode(result));
        result = result.replace(/<\s*br\s*\/?\s*>/gi, '\n');
        result = result.replace(
            /<\/\s*(p|div|li|h[1-6]|blockquote|tr|article|section)\s*>/gi,
            '\n\n'
        );
        result = result.replace(
            /<\s*(p|div|li|h[1-6]|blockquote|tr|article|section)(\s[^>]*)?>/gi,
            '\n\n'
        );
        result = result.replace(/<[^>]+>/g, ' ');
        result = result.replace(/[ \t]+/g, ' ')
            .replace(/ *\n */g, '\n')
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    // If the content already has paragraph breaks, don't run sentence-based
    // splitting — it would over-fragment well-formed prose.
    if (/\n\s*\n/.test(result)) {
        return result;
    }

    const abbreviations = [
        ['U.S.', '<<US>>'],
        ['U.K.', '<<UK>>'],
        ['Ph.D.', '<<PHD>>'],
        ['M.D.', '<<MD>>'],
        ['Dr.', '<<DR>>'],
        ['Mr.', '<<MR>>'],
        ['Mrs.', '<<MRS>>'],
        ['Ms.', '<<MS>>'],
        ['Jr.', '<<JR>>'],
        ['Sr.', '<<SR>>']
    ];

    abbreviations.forEach(([abbr, placeholder]) => {
        result = result.replaceAll(abbr, placeholder);
    });

    result = result.replace(/\.([A-Z])/g, '.\n$1');

    abbreviations.forEach(([abbr, placeholder]) => {
        result = result.replaceAll(placeholder, abbr);
    });

    return result;
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[c]);
}

// Snap the matched-passage range onto natural content boundaries so the
// highlight reads as intentional rather than a sliding-window artifact:
//
//  - Drop paragraphs shorter than TINY_PARA_CHARS that fall inside the
//    passage (they're almost always chrome like "Related" / "Read more").
//  - Drop the 1 paragraph after a recognized chrome label (the typical
//    "Related <related-article-headline>" shape).
//  - Drop slivers — a paragraph where less than 30% AND fewer than
//    SLIVER_OVERLAP_CHARS of its length is inside the passage.
//  - Otherwise: if the passage covers ≥50% of a paragraph, expand the
//    highlight to the whole paragraph so it snaps to a clean boundary.
//
// Returns a list of {offset, length} ranges suitable for renderWithHighlights.
const TINY_PARA_CHARS = 30;
const SLIVER_OVERLAP_CHARS = 30;
const CHROME_LABEL_RE = /^(related|read more|read also|see also|related stories|related articles|recommended|more from|trending|advertisement|sponsored)$/i;

function tidyPassageRanges(content, passage) {
    if (!passage || !content) return [];
    const pStart = passage.offset;
    const pEnd = passage.offset + passage.length;

    // Walk `content` once to collect paragraph spans (separated by \n\n+).
    const paragraphs = [];
    const sepRe = /\n\s*\n+/g;
    let cursor = 0;
    let m;
    while ((m = sepRe.exec(content)) !== null) {
        if (m.index > cursor) paragraphs.push({ start: cursor, end: m.index });
        cursor = m.index + m[0].length;
    }
    if (content.length > cursor) paragraphs.push({ start: cursor, end: content.length });

    // Mark chrome paragraphs (label + the 1 paragraph after).
    const dropped = new Set();
    for (let i = 0; i < paragraphs.length; i++) {
        const p = paragraphs[i];
        const text = content.slice(p.start, p.end).trim();
        if (CHROME_LABEL_RE.test(text)) {
            dropped.add(i);
            if (i + 1 < paragraphs.length) dropped.add(i + 1);
        }
    }

    const ranges = [];
    for (let i = 0; i < paragraphs.length; i++) {
        if (dropped.has(i)) continue;
        const p = paragraphs[i];
        const overlapStart = Math.max(p.start, pStart);
        const overlapEnd = Math.min(p.end, pEnd);
        if (overlapEnd <= overlapStart) continue;

        const overlapLen = overlapEnd - overlapStart;
        const paraLen = p.end - p.start;

        if (paraLen < TINY_PARA_CHARS) continue;  // chrome-shaped paragraph
        if (overlapLen < SLIVER_OVERLAP_CHARS && overlapLen / paraLen < 0.3) continue;

        if (overlapLen / paraLen >= 0.5) {
            ranges.push({ offset: p.start, length: paraLen });
        } else {
            ranges.push({ offset: overlapStart, length: overlapLen });
        }
    }
    return ranges;
}

// Render `content` with <mark> spans at the given ranges. Overlapping ranges
// are merged into one mark so the HTML stays valid.
function renderWithHighlights(content, ranges) {
    if (!ranges || !ranges.length) return escapeHtml(content);
    const sorted = [...ranges].sort((a, b) => a.offset - b.offset);
    const merged = [{ ...sorted[0] }];
    for (let i = 1; i < sorted.length; i++) {
        const last = merged[merged.length - 1];
        const cur = sorted[i];
        if (cur.offset <= last.offset + last.length) {
            last.length = Math.max(last.length, cur.offset + cur.length - last.offset);
        } else {
            merged.push({ ...cur });
        }
    }
    let result = '';
    let pos = 0;
    for (const r of merged) {
        if (r.offset > pos) result += escapeHtml(content.slice(pos, r.offset));
        result += `<mark class="passage-match">${escapeHtml(content.slice(r.offset, r.offset + r.length))}</mark>`;
        pos = r.offset + r.length;
    }
    if (pos < content.length) result += escapeHtml(content.slice(pos));
    return result;
}

function openArticle(articleId, matchInfo) {
    hidePreview();

    const appContainer = document.getElementById('appContainer');
    if (currentArticleId === articleId && appContainer.classList.contains('split-view') && !matchInfo) {
        closeArticle();
        return;
    }

    const story = storiesData.find(s => s.article_id === articleId);

    if (!story) {
        console.error('Story not found:', articleId);
        return;
    }

    document.getElementById('articlePanelTitle').textContent = story.title || 'Untitled';

    // Use the raw `story.content` (not the trimmed `extractArticleContent`)
    // when we have passage offsets — those offsets are into the raw text.
    const useRaw = !!(matchInfo && matchInfo.passageLength);
    const articleContent = useRaw ? story.content : extractArticleContent(story.content);
    const passage = useRaw
        ? { offset: matchInfo.passageOffset, length: matchInfo.passageLength }
        : null;
    // Snap the passage range onto natural paragraph boundaries and drop
    // chrome (e.g. "Related" labels and the headlines that follow them) so
    // the highlight reads as intentional content rather than a sliding-window
    // artifact.
    const tidiedRanges = passage ? tidyPassageRanges(articleContent, passage) : [];

    const authorName = formatAuthorName(story.author);
    const bylineHtml = authorName !== 'Unknown'
        ? `<span><strong>By:</strong> ${authorName}</span>`
        : '';

    // When we have a passage/highlight, render the article as a single HTML
    // block so the <mark> tags can span paragraph breaks without being torn
    // apart by `.split(/\n+/)`. Paragraph breaks become block-level <br><br>.
    let bodyHtml;
    if (useRaw && articleContent) {
        const annotated = renderWithHighlights(articleContent, tidiedRanges);
        bodyHtml = `<div class="article-body fade-in" style="animation-delay: 0.15s">${annotated.replace(/\n{2,}/g, '<br><br>').replace(/\n/g, ' ')}</div>`;
    } else {
        const splitter = /\n\s*\n/.test(articleContent) ? /\n\s*\n+/ : /\n+/;
        bodyHtml = articleContent
            ? articleContent.split(splitter)
                .map(p => p.trim())
                .filter(Boolean)
                .map((p, i) => `<p class="fade-in" style="animation-delay: ${0.15 + (i * 0.05)}s">${escapeHtml(p)}</p>`)
                .join('')
            : '<p class="fade-in" style="animation-delay: 0.15s">No content available.</p>';
    }

    const linkHtml = story.link
        ? `<p class="fade-in" style="animation-delay: 0.1s"><a href="${story.link}" target="_blank" rel="noopener">View original →</a></p>`
        : '';

    // When opened from a citation, surface the beat-book claim above the
    // article so the reporter can compare claim vs. highlighted source
    // directly. Without this card, the yellow highlight in the source has no
    // visible counterpart in the beat book.
    let claimCardHtml = '';
    if (matchInfo && matchInfo.claimText) {
        const numberLabel = (typeof matchInfo.number === 'number')
            ? `Source [${matchInfo.number}] cites:`
            : 'Cited for:';
        claimCardHtml = `
            <div class="cited-claim-card fade-in" style="animation-delay: 0.08s">
                <div class="cited-claim-label">${numberLabel}</div>
                <div class="cited-claim-text">${escapeHtml(matchInfo.claimText)}</div>
                <div class="cited-claim-arrow" aria-hidden="true">↓ matched passage highlighted below</div>
            </div>`;
    }

    const articleHtml = `
        <div class="article-meta">
            <h1 class="fade-in" style="animation-delay: 0s">${escapeHtml(story.title || 'Untitled')}</h1>
            <div class="meta-info fade-in" style="animation-delay: 0.05s">
                ${bylineHtml}
                <span><strong>Date:</strong> ${escapeHtml(story.date || 'Unknown')}</span>
            </div>
            ${linkHtml}
        </div>
        ${claimCardHtml}
        <div class="article-content">
            ${bodyHtml}
        </div>
    `;

    const articleContentEl = document.getElementById('articleContent');
    articleContentEl.innerHTML = articleHtml;
    articleContentEl.scrollTop = 0;
    document.getElementById('appContainer').classList.add('split-view');
    currentArticleId = articleId;

    // Scroll the article panel to the first highlight (or the passage start).
    if (useRaw) {
        requestAnimationFrame(() => {
            const target = articleContentEl.querySelector('.passage-highlight') ||
                           articleContentEl.querySelector('.passage-match');
            if (target) {
                const rect = target.getBoundingClientRect();
                const containerRect = articleContentEl.getBoundingClientRect();
                articleContentEl.scrollTop = articleContentEl.scrollTop + (rect.top - containerRect.top) - 60;
            }
        });
    }
}

async function loadData() {
    try {
        try {
            const storiesResponse = await fetch(storiesFile);
            if (storiesResponse.ok) {
                storiesData = await storiesResponse.json();
            } else {
                console.warn('Failed to load sources file');
            }
        } catch (e) {
            console.warn('Error loading sources:', e);
        }

        const response = await fetch(beatbookFile);
        if (!response.ok) {
            throw new Error(`Failed to load ${beatbookFile}`);
        }
        const beatbookData = await response.json();

        // Detect shape: new = {calibration, entries}; old = [entry, ...].
        let entries;
        let isNewShape = false;
        if (Array.isArray(beatbookData)) {
            entries = beatbookData;
        } else if (beatbookData && Array.isArray(beatbookData.entries)) {
            entries = beatbookData.entries;
            isNewShape = true;
        } else {
            throw new Error('Unrecognized beat-book JSON shape');
        }

        // Old-shape fallback threshold (no per-corpus calibration in old data).
        const oldShapeThreshold = 0.65;

        // Pass 1: extract the primary support per entry (or null if uncited).
        const sourceKey = (p) =>
            `${p.article_id}::${p.passage_offset ?? 'x'}::${p.passage_length ?? 'x'}`;
        const primaryByIdx = entries.map(entry => {
            const isTableRow = entry.content.trimStart().startsWith('|');
            if (isTableRow) return null;
            let primary = null;
            if (isNewShape) {
                if (!entry.passthrough && entry.supports && entry.supports.length) {
                    primary = entry.supports[0];
                }
            } else if (entry.source) {
                const meetsThreshold = entry.similarity === undefined || entry.similarity >= oldShapeThreshold;
                if (meetsThreshold) {
                    primary = {
                        article_id: entry.source,
                        article_title: entry.source_title || '',
                        passage_text: entry.source_sentence || '',
                        similarity: entry.similarity,
                    };
                }
            }
            if (!primary) return null;
            const isValidSource = storiesData.some(s => s.article_id === primary.article_id);
            return isValidSource ? primary : null;
        });

        // Pass 2: dedupe consecutive same-source runs. In a stretch of
        // sentences all citing the same source, only the LAST renders a chip
        // (academic convention: "…this paragraph all cites this source.[N]").
        // Blank lines DON'T break the run; non-blank non-cited content
        // (headings, plain prose, list items, table rows) does.
        const showCiteAt = new Set();
        let runKey = null;
        let runLastIdx = -1;
        const flushRun = () => {
            if (runLastIdx >= 0) showCiteAt.add(runLastIdx);
            runKey = null;
            runLastIdx = -1;
        };
        for (let i = 0; i < primaryByIdx.length; i++) {
            const p = primaryByIdx[i];
            if (p) {
                const k = sourceKey(p);
                if (runKey !== null && k !== runKey) flushRun();
                runKey = k;
                runLastIdx = i;
            } else if (entries[i].content.trim() !== '') {
                flushRun();
            }
        }
        flushRun();

        // Pass 3: assign sequential inline numbers to surviving markers in
        // document order. The same source key can collect multiple numbers —
        // those all surface in the bottom footnote so the reader sees every
        // place a given source is cited.
        let nextNumber = 1;
        const numByIdx = {};
        for (let i = 0; i < primaryByIdx.length; i++) {
            if (!showCiteAt.has(i)) continue;
            const primary = primaryByIdx[i];
            const key = sourceKey(primary);
            const number = nextNumber++;
            numByIdx[i] = number;

            citationsByNumber[number] = {
                number,
                sourceKey: key,
                articleId: primary.article_id,
                articleTitle: primary.article_title || '',
                articleAuthor: primary.article_author || '',
                articleDate: primary.article_date || '',
                passageText: primary.passage_text || '',
                passageOffset: primary.passage_offset,
                passageLength: primary.passage_length,
                similarity: primary.similarity,
                claimText: entries[i].content || '',
            };

            if (!sourcesByKey[key]) {
                sourcesByKey[key] = {
                    key,
                    primary,
                    numbers: [],
                    firstSeen: number,
                    claimText: entries[i].content || '',
                };
            }
            sourcesByKey[key].numbers.push(number);
        }

        // Pass 4: build the markdown with [[CITE:N]] markers.
        const markdown = entries.map((entry, i) => {
            if (numByIdx[i] != null) {
                return `${entry.content}[[CITE:${numByIdx[i]}]]`;
            }
            return entry.content;
        }).join('\n');

        let html = marked.parse(markdown);

        // Replace [[CITE:N]] sentinels with [N] superscript markers.
        html = html.replace(/\[\[CITE:(\d+)\]\]/g, (_, n) => {
            const num = parseInt(n, 10);
            const c = citationsByNumber[num];
            const titleAttr = c
                ? (c.articleTitle ? `Source: ${c.articleTitle}` : `Source [${num}]`).replace(/"/g, '&quot;')
                : `Source [${num}]`;
            const safeId = c ? c.articleId.replace(/'/g, "\\'") : '';
            return `<sup class="footnote-ref" onclick="openCitation(${num})" onmouseenter="showPreview('${safeId}', event)" onmouseleave="hidePreview()" title="${titleAttr}">${num}</sup>`;
        });

        // Append the footnotes section at the bottom of the document.
        if (Object.keys(sourcesByKey).length > 0) {
            html += renderFootnotesSection(sourcesByKey);
        }

        document.getElementById('content').innerHTML = html;

        const contentEl = document.getElementById('content');
        const elements = contentEl.querySelectorAll('h1, h2, h3, h4, h5, h6, p, ul, ol, blockquote, table, pre');
        elements.forEach((el, i) => {
            el.classList.add('fade-in');
            el.style.animationDelay = `${i * 0.03}s`;
        });

        setTimeout(initSectionNavigation, 100);
    } catch (error) {
        document.getElementById('content').innerHTML =
            `<p style="color: red;">Error loading beat book: ${error.message}</p>
             <p>Expected files at:</p>
             <ul>
                 <li>${beatbookFile}</li>
                 <li>${storiesFile}</li>
             </ul>`;
    }
}

loadData();

// Reading progress bar
let ticking = false;

function updateReadingProgress() {
    const mainPanel = document.querySelector('.main-panel');
    const progressBar = document.getElementById('readingProgress');

    if (!mainPanel || !progressBar) return;

    const scrollTop = mainPanel.scrollTop;
    const scrollHeight = mainPanel.scrollHeight - mainPanel.clientHeight;

    if (scrollHeight > 0) {
        const progress = scrollTop / scrollHeight;
        progressBar.style.transform = `scaleX(${progress})`;
    }

    ticking = false;
}

function onScroll() {
    if (!ticking) {
        requestAnimationFrame(updateReadingProgress);
        ticking = true;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const mainPanel = document.querySelector('.main-panel');
    if (mainPanel) {
        mainPanel.addEventListener('scroll', onScroll, { passive: true });
    }
});

setTimeout(() => {
    const mainPanel = document.querySelector('.main-panel');
    if (mainPanel) {
        mainPanel.addEventListener('scroll', onScroll, { passive: true });
    }
}, 100);

// Section navigation
let sectionHeaders = [];
let isNavTicking = false;

function initSectionNavigation() {
    const content = document.getElementById('content');
    const headers = content.querySelectorAll('h2');
    const menu = document.getElementById('sectionMenu');
    sectionHeaders = [];
    menu.innerHTML = '';

    const firstItem = document.createElement('button');
    firstItem.className = 'section-menu-item active';
    firstItem.textContent = 'Introduction';
    firstItem.onclick = () => {
        document.querySelector('.main-panel').scrollTo({ top: 0, behavior: 'auto' });
        toggleSectionMenu();
    };
    menu.appendChild(firstItem);

    document.getElementById('currentSectionText').textContent = 'Introduction';

    headers.forEach((header, index) => {
        if (!header.id) {
            header.id = 'section-' + index;
        }

        const fullTitle = header.textContent;
        const title = fullTitle.split(':')[0].trim();

        sectionHeaders.push({
            id: header.id,
            title: title,
            element: header
        });

        const item = document.createElement('button');
        item.className = 'section-menu-item';
        item.textContent = title;
        item.onclick = () => {
            const headerHeight = 52;
            const elementPosition = header.getBoundingClientRect().top;
            const offsetPosition = elementPosition + document.querySelector('.main-panel').scrollTop - headerHeight - 20;

            document.querySelector('.main-panel').scrollTo({
                top: offsetPosition,
                behavior: 'auto'
            });
            toggleSectionMenu();
        };
        menu.appendChild(item);
    });

    document.addEventListener('click', (e) => {
        const nav = document.getElementById('sectionNavigator');
        if (!nav.contains(e.target)) {
            nav.classList.remove('active');
        }
    });

    const mainPanel = document.querySelector('.main-panel');
    if (mainPanel) {
        mainPanel.addEventListener('scroll', onNavScroll, { passive: true });
    }
}

function toggleSectionMenu() {
    document.getElementById('sectionNavigator').classList.toggle('active');
}

function onNavScroll() {
    if (!isNavTicking) {
        requestAnimationFrame(updateActiveSection);
        isNavTicking = true;
    }
}

function updateActiveSection() {
    const headerHeight = 52;
    const offset = 100;

    let currentSection = 'Introduction';

    for (const section of sectionHeaders) {
        const rect = section.element.getBoundingClientRect();
        if (rect.top <= headerHeight + offset) {
            currentSection = section.title;
        }
    }

    const currentText = document.getElementById('currentSectionText');
    if (currentText.textContent !== currentSection) {
        currentText.textContent = currentSection;

        const items = document.querySelectorAll('.section-menu-item');
        items.forEach(item => {
            if (item.textContent === currentSection) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    isNavTicking = false;
}

let resizeTimer;
window.addEventListener('resize', () => {
    document.body.classList.add('resize-animation-stopper');
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        document.body.classList.remove('resize-animation-stopper');
    }, 400);
});
