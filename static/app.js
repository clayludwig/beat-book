// ── Beat Book Builder — Frontend Logic ─────────────────────────────────
(() => {
  "use strict";

  // ── DOM refs ─────────────────────────────────────────────────────────
  const dropZone        = document.getElementById("drop-zone");
  const fileInput       = document.getElementById("file-input");
  const fileListEl      = document.getElementById("file-list");
  const urlInput        = document.getElementById("url-input");
  const uploadBtn       = document.getElementById("upload-btn");
  const uploadStatus    = document.getElementById("upload-status");
  const ingestStep      = document.getElementById("ingest-step");
  const ingestDetail    = document.getElementById("ingest-detail");

  const previewTitle    = document.getElementById("preview-title");
  const previewSummary  = document.getElementById("preview-summary");
  const previewExcluded = document.getElementById("preview-excluded");
  const previewSources  = document.getElementById("preview-sources");
  const previewRunBtn   = document.getElementById("preview-run-btn");
  const previewBackBtn  = document.getElementById("preview-back-btn");
  const previewIncluded = document.getElementById("preview-included-count");
  const previewStatus   = document.getElementById("preview-status");
  const previewProgressStep   = document.getElementById("preview-progress-step");
  const previewProgressBar    = document.getElementById("preview-progress-bar");
  const previewProgressDetail = document.getElementById("preview-progress-detail");
  const filterChips     = document.querySelectorAll(".confidence-filter");
  const filterCountEls  = {
    high:   document.getElementById("filter-count-high"),
    medium: document.getElementById("filter-count-medium"),
    low:    document.getElementById("filter-count-low"),
  };

  const interviewFormHost = document.getElementById("interview-form-host");

  const generatingLabel   = document.getElementById("generating-label");
  const generatingDetail  = document.getElementById("generating-detail");
  const generatingStats   = document.getElementById("generating-stats");
  const generatingElapsed = document.getElementById("generating-elapsed");
  const stepperEl         = document.getElementById("stepper");
  const shimmerBar        = document.querySelector(".shimmer-bar");
  const shimmerFill       = document.querySelector(".shimmer-bar-fill");

  const doneSubtitle    = document.getElementById("done-subtitle");
  const doneViewerLink  = document.getElementById("done-viewer-link");
  const doneMarkdownLink = document.getElementById("done-markdown-link");

  const sessionInfoEls = document.querySelectorAll(
    "#preview-session-info, #interview-session-info, #generating-session-info, #done-session-info"
  );

  // ── State ────────────────────────────────────────────────────────────
  let selectedFiles = [];
  let previewState = [];      // [{source, stories:[{...editable, included}]}]
  let ws = null;
  let activeInterview = null;
  const stats = { storiesRead: 0, searches: 0, topicsListed: 0 };
  // Visibility filter for the preview screen. Toggling a level off hides
  // those rows but does NOT change their `included` state — that stays
  // under user control via the per-row checkbox.
  const confidenceFilter = { high: true, medium: true, low: true };

  let elapsedTimer = null;
  let elapsedStart = null;

  const MAX_FILE_BYTES = 25 * 1024 * 1024;

  // ── Screen routing ───────────────────────────────────────────────────
  function switchScreen(name) {
    document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
    const target = document.getElementById(`${name}-screen`);
    if (target) target.classList.add("active");
  }

  // ── File selection ───────────────────────────────────────────────────
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    addFiles([...e.dataTransfer.files]);
  });
  dropZone.addEventListener("click", (e) => {
    // Clicking the inner label/button should let the label handle the file input directly.
    if (e.target.closest("label.file-btn") || e.target.matches("input")) return;
    fileInput.click();
  });
  fileInput.addEventListener("change", () => {
    addFiles([...fileInput.files]);
    fileInput.value = "";
  });

  urlInput.addEventListener("input", refreshUploadButton);

  function addFiles(files) {
    for (const f of files) {
      if (f.size > MAX_FILE_BYTES) {
        alert(`${f.name} is larger than 25 MB. Please split or compress it before uploading.`);
        continue;
      }
      if (!selectedFiles.find(x => x.name === f.name && x.size === f.size)) {
        selectedFiles.push(f);
      }
    }
    renderFileList();
  }

  function removeFile(idx) {
    selectedFiles.splice(idx, 1);
    renderFileList();
  }

  function renderFileList() {
    if (selectedFiles.length === 0) {
      fileListEl.hidden = true;
    } else {
      fileListEl.hidden = false;
      fileListEl.innerHTML = selectedFiles.map((f, i) =>
        `<div class="file-item">
          <span class="name">${escapeHtml(f.name)}</span>
          <span>${(f.size / 1024).toFixed(1)} KB</span>
          <button type="button" data-remove="${i}" aria-label="Remove" style="background:none;border:none;color:var(--text-faint);cursor:pointer;font-size:1.1rem;line-height:1;padding:0 0.3rem;">×</button>
        </div>`
      ).join("");
      fileListEl.querySelectorAll("[data-remove]").forEach(btn => {
        btn.addEventListener("click", () => removeFile(parseInt(btn.dataset.remove, 10)));
      });
    }
    refreshUploadButton();
  }

  function refreshUploadButton() {
    const hasUrls = urlInput.value.split("\n").some(line => line.trim().length > 0);
    uploadBtn.disabled = selectedFiles.length === 0 && !hasUrls;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }

  // ── Ingest flow ──────────────────────────────────────────────────────
  uploadBtn.addEventListener("click", async () => {
    const urls = urlInput.value
      .split("\n").map(l => l.trim()).filter(Boolean);

    if (selectedFiles.length === 0 && urls.length === 0) return;

    uploadBtn.disabled = true;
    uploadStatus.hidden = false;
    const totalSources = selectedFiles.length + urls.length;
    ingestStep.textContent = `Reading ${totalSources} ${totalSources === 1 ? "source" : "sources"}…`;
    ingestDetail.textContent = "Extracting text, then identifying stories with an LLM.";

    const form = new FormData();
    for (const f of selectedFiles) form.append("files", f);
    if (urls.length) form.append("urls", urls.join("\n"));

    try {
      const resp = await fetch("/ingest", { method: "POST", body: form });
      const data = await resp.json();

      if (!resp.ok) {
        ingestStep.textContent = data.error || "Ingestion failed";
        ingestDetail.textContent = "";
        uploadBtn.disabled = false;
        return;
      }

      renderPreview(data);
      switchScreen("preview");
      window.scrollTo({ top: 0 });
    } catch (err) {
      ingestStep.textContent = `Ingestion failed: ${err.message}`;
      uploadBtn.disabled = false;
    } finally {
      uploadStatus.hidden = true;
    }
  });

  // ── Preview rendering ────────────────────────────────────────────────
  function renderPreview(data) {
    previewState = (data.sources || []).map(src => ({
      source_label: src.source_label,
      kind: src.kind,
      excluded: src.excluded,
      skip_reason: src.skip_reason,
      extract_error: src.extract_error,
      char_count: src.char_count,
      stories: (src.stories || []).map(s => ({
        title: s.title || "",
        date: s.date || "",
        author: s.author || "",
        link: s.link || "",
        content: s.content || "",
        confidence: s.confidence || "medium",
        reasoning: s.reasoning || "",
        included: true,
      })),
    }));

    const totalStories = previewState.reduce((sum, src) => sum + src.stories.length, 0);
    const includedSources = previewState.filter(src => !src.excluded);
    previewTitle.textContent = totalStories === 1
      ? "We found 1 story"
      : `We found ${totalStories} stories`;
    previewSummary.textContent =
      `From ${includedSources.length} ${includedSources.length === 1 ? "source" : "sources"}. ` +
      "Review the metadata below, deselect anything you don't want, then run the pipeline.";

    // Excluded sources
    const excluded = previewState.filter(src => src.excluded);
    if (excluded.length === 0) {
      previewExcluded.hidden = true;
      previewExcluded.innerHTML = "";
    } else {
      previewExcluded.hidden = false;
      previewExcluded.innerHTML = excluded.map(src => {
        // Show the generic skip_reason and the underlying error detail when
        // it adds new information — a bare "Normalization failed." with no
        // detail is useless when the real cause is e.g. an API timeout.
        const reason = src.skip_reason || "";
        const detail = src.extract_error || "";
        const showBoth = reason && detail && detail !== reason;
        const text = showBoth
          ? `${reason} — ${detail}`
          : (reason || detail || "Excluded.");
        return `
          <div class="preview-excluded-item">
            <span class="excluded-label">${escapeHtml(src.source_label)}</span>
            <span>${escapeHtml(text)}</span>
          </div>
        `;
      }).join("");
    }

    // Sources with stories
    previewSources.innerHTML = "";
    includedSources.forEach((src, srcIdx) => {
      const card = document.createElement("div");
      card.className = "preview-source";
      card.dataset.srcIdx = previewState.indexOf(src);

      const header = document.createElement("div");
      header.className = "preview-source-header";
      header.innerHTML = `
        <span class="preview-source-label">${escapeHtml(src.source_label)}</span>
        <span class="preview-source-meta">${src.stories.length} ${src.stories.length === 1 ? "story" : "stories"} · ${src.char_count.toLocaleString()} chars</span>
      `;
      card.appendChild(header);

      const list = document.createElement("div");
      list.className = "preview-stories";

      src.stories.forEach((story, storyIdx) => {
        list.appendChild(buildStoryRow(previewState.indexOf(src), storyIdx, story));
      });

      card.appendChild(list);
      previewSources.appendChild(card);
    });

    refreshIncludedCount();
    updateConfidenceCounts();
    applyConfidenceFilter();
  }

  function buildStoryRow(srcIdx, storyIdx, story) {
    const row = document.createElement("div");
    row.className = "preview-story";
    row.dataset.confidence = (story.confidence || "medium").toLowerCase();
    if (!story.included) row.classList.add("excluded");

    // Row 1: include checkbox + editable fields
    const topRow = document.createElement("div");
    topRow.className = "preview-story-row";

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "preview-story-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = story.included;
    toggle.addEventListener("change", () => {
      previewState[srcIdx].stories[storyIdx].included = toggle.checked;
      row.classList.toggle("excluded", !toggle.checked);
      refreshIncludedCount();
    });
    toggleLabel.appendChild(toggle);
    toggleLabel.appendChild(document.createTextNode("Include"));
    topRow.appendChild(toggleLabel);

    const fields = document.createElement("div");
    fields.className = "preview-story-fields";
    fields.appendChild(buildField("Title", "title", story.title, "title-input", srcIdx, storyIdx));
    fields.appendChild(buildField("Date", "date", story.date, "", srcIdx, storyIdx, "YYYY-MM-DD"));
    fields.appendChild(buildField("Author", "author", story.author, "", srcIdx, storyIdx, "Byline"));
    topRow.appendChild(fields);

    row.appendChild(topRow);

    // Row 2: collapsed content preview (click to expand)
    const content = document.createElement("div");
    content.className = "preview-story-content";
    content.textContent = story.content;
    content.title = "Click to expand";
    content.addEventListener("click", () => content.classList.toggle("expanded"));
    row.appendChild(content);

    // Row 3: confidence + reasoning
    const foot = document.createElement("div");
    foot.className = "preview-story-foot";
    const chip = document.createElement("span");
    chip.className = `confidence-chip ${story.confidence}`;
    chip.textContent = `${story.confidence} confidence`;
    foot.appendChild(chip);
    if (story.reasoning) {
      const reasoning = document.createElement("span");
      reasoning.className = "preview-reasoning";
      reasoning.textContent = story.reasoning;
      foot.appendChild(reasoning);
    }
    row.appendChild(foot);

    return row;
  }

  function buildField(_label, key, value, extraClass, srcIdx, storyIdx, placeholder) {
    const input = document.createElement("input");
    input.type = "text";
    input.value = value;
    if (extraClass) input.classList.add(extraClass);
    if (placeholder) input.placeholder = placeholder;
    input.addEventListener("input", () => {
      previewState[srcIdx].stories[storyIdx][key] = input.value;
    });
    return input;
  }

  function refreshIncludedCount() {
    const total = previewState.reduce((sum, src) =>
      sum + src.stories.filter(s => s.included).length, 0);
    previewIncluded.textContent = total === 0
      ? "Nothing selected"
      : `${total} ${total === 1 ? "story" : "stories"} selected`;
    previewRunBtn.disabled = total === 0;
  }

  function updateConfidenceCounts() {
    const counts = { high: 0, medium: 0, low: 0 };
    for (const src of previewState) {
      if (src.excluded) continue;
      for (const s of src.stories) {
        const level = (s.confidence || "medium").toLowerCase();
        if (level in counts) counts[level]++;
      }
    }
    for (const level of Object.keys(counts)) {
      if (filterCountEls[level]) filterCountEls[level].textContent = counts[level];
    }
  }

  function applyConfidenceFilter() {
    // Hide individual story rows whose confidence is filtered out.
    document.querySelectorAll(".preview-story").forEach(row => {
      const level = row.dataset.confidence || "medium";
      row.classList.toggle("filtered-out", !confidenceFilter[level]);
    });
    // Hide source cards whose stories are all filtered out, so the screen
    // doesn't end up with a sea of empty source headers.
    document.querySelectorAll(".preview-source").forEach(card => {
      const anyVisible = !!card.querySelector(".preview-story:not(.filtered-out)");
      card.classList.toggle("filtered-out", !anyVisible);
    });
  }

  filterChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const level = chip.dataset.level;
      confidenceFilter[level] = !confidenceFilter[level];
      chip.classList.toggle("active", confidenceFilter[level]);
      chip.setAttribute("aria-pressed", String(confidenceFilter[level]));
      applyConfidenceFilter();
    });
  });

  previewBackBtn.addEventListener("click", () => {
    switchScreen("upload");
    uploadBtn.disabled = selectedFiles.length === 0 && !urlInput.value.trim();
  });

  // ── Run pipeline ─────────────────────────────────────────────────────
  const STEP_LABELS = {
    embedding: "Generating embeddings",
    reducing:  "Reducing dimensions",
    clustering: "Clustering stories",
    labeling:  "Labeling topics",
  };
  const STEP_WEIGHTS = { embedding: 0.30, reducing: 0.10, clustering: 0.10, labeling: 0.50 };
  const STEP_ORDER   = ["embedding", "reducing", "clustering", "labeling"];

  function calcOverall(step, fraction) {
    let total = 0;
    for (const s of STEP_ORDER) {
      if (s === step) { total += STEP_WEIGHTS[s] * fraction; break; }
      total += STEP_WEIGHTS[s];
    }
    return Math.min(total, 1);
  }

  previewRunBtn.addEventListener("click", async () => {
    const stories = [];
    for (const src of previewState) {
      for (const s of src.stories) {
        if (!s.included) continue;
        stories.push({
          title: s.title,
          content: s.content,
          date: s.date,
          author: s.author,
          link: s.link,
        });
      }
    }
    if (stories.length === 0) return;

    previewRunBtn.disabled = true;
    previewBackBtn.disabled = true;
    previewStatus.hidden = false;
    previewProgressStep.textContent = "Preparing…";
    previewProgressBar.style.width = "0%";
    previewProgressDetail.textContent = "";

    try {
      const resp = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stories }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        previewProgressStep.textContent = err.error || "Pipeline failed";
        previewRunBtn.disabled = false;
        previewBackBtn.disabled = false;
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const msg = JSON.parse(line.slice(6));

          if (msg.type === "progress") {
            const label = STEP_LABELS[msg.step] || msg.step;
            previewProgressStep.textContent = label;
            previewProgressDetail.textContent = msg.detail || "";
            previewProgressBar.style.width =
              `${Math.round(calcOverall(msg.step, msg.fraction) * 100)}%`;
          }

          if (msg.type === "done") {
            previewProgressStep.textContent = "Done.";
            previewProgressBar.style.width = "100%";
            previewProgressDetail.textContent = `${msg.num_stories} stories · ${msg.num_topics} topics`;
            setTimeout(() => startSession(msg), 500);
          }

          if (msg.type === "error") {
            previewProgressStep.textContent = "Pipeline failed";
            previewProgressDetail.textContent = msg.error || "";
            previewProgressBar.style.width = "0%";
            previewRunBtn.disabled = false;
            previewBackBtn.disabled = false;
          }
        }
      }
    } catch (err) {
      previewProgressStep.textContent = `Pipeline failed: ${err.message}`;
      previewRunBtn.disabled = false;
      previewBackBtn.disabled = false;
    }
  });

  // ── Elapsed time ticker ──────────────────────────────────────────────
  function startElapsed() {
    if (elapsedTimer) return;
    elapsedStart = Date.now();
    updateElapsed();
    elapsedTimer = setInterval(updateElapsed, 1000);
  }

  function stopElapsed() {
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }

  function updateElapsed() {
    if (!generatingElapsed || !elapsedStart) return;
    const secs = Math.floor((Date.now() - elapsedStart) / 1000);
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    generatingElapsed.textContent = `${m}:${s.toString().padStart(2, "0")}`;
    // Tick the active stage chip so its timer keeps moving.
    renderStatsChips();
  }

  // ── Stepper state ────────────────────────────────────────────────────
  const STAGE_ORDER = ["review", "write", "research", "cite"];
  const STAGE_LABELS = {
    review: "Review",
    write: "Write",
    research: "Research",
    cite: "Cite",
  };

  // Per-stage timing (for the testing/debug HUD). Tracks the wall-clock time
  // spent in each pipeline stage. The active stage ticks live; completed
  // stages freeze.
  let activeStage = null;
  const stageStartTs = {};      // stage -> Date.now() ms
  const stageDurationsMs = {};  // stage -> committed duration (ms)

  function commitStage(stage) {
    if (!stage || !stageStartTs[stage]) return;
    if (stageDurationsMs[stage] == null) {
      stageDurationsMs[stage] = Date.now() - stageStartTs[stage];
    }
  }

  function formatStageMs(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${r.toString().padStart(2, "0")}`;
  }

  function setStage(stage) {
    if (!stepperEl) return;
    // Commit timing on the prior stage; start timing for the new one.
    if (activeStage && activeStage !== stage) commitStage(activeStage);
    if (stage && !stageStartTs[stage]) stageStartTs[stage] = Date.now();
    activeStage = stage;

    const idx = STAGE_ORDER.indexOf(stage);
    stepperEl.querySelectorAll(".step").forEach(el => {
      const s = el.getAttribute("data-step");
      const sIdx = STAGE_ORDER.indexOf(s);
      el.classList.remove("active", "done");
      if (sIdx < idx) el.classList.add("done");
      else if (sIdx === idx) el.classList.add("active");
    });
    renderStatsChips();
  }

  function markAllStagesDone() {
    if (!stepperEl) return;
    stepperEl.querySelectorAll(".step").forEach(el => {
      el.classList.remove("active");
      el.classList.add("done");
    });
  }

  // ── Shimmer bar control ──────────────────────────────────────────────
  function setShimmerDeterminate(fraction) {
    if (!shimmerBar || !shimmerFill) return;
    shimmerBar.classList.add("determinate");
    shimmerFill.style.width = `${Math.min(Math.max(fraction, 0), 1) * 100}%`;
  }

  function setShimmerIndeterminate() {
    if (!shimmerBar || !shimmerFill) return;
    shimmerBar.classList.remove("determinate");
    shimmerFill.style.width = "";
  }

  // ── Start session: go to generating, open WebSocket ──────────────────
  function startSession(uploadData) {
    const sessionText = `${uploadData.num_stories} stories · ${uploadData.num_topics} topics`;
    sessionInfoEls.forEach(el => { el.textContent = sessionText; });

    setGenerating("Generating your beat book", "Reviewing your coverage…");
    setStage("review");
    setShimmerIndeterminate();
    startElapsed();
    switchScreen("generating");
    startWebSocket(uploadData.session_id);
  }

  // ── Generating screen helpers ────────────────────────────────────────
  function plural(n, single, multi) { return `${n} ${n === 1 ? single : multi}`; }

  function renderStatsChips() {
    if (!generatingStats) return;
    const parts = [];

    // Per-stage timing chips: completed stages first, active stage last.
    STAGE_ORDER.forEach(stage => {
      if (!stageStartTs[stage]) return;
      let ms = stageDurationsMs[stage];
      let isActive = false;
      if (ms == null) {
        if (stage !== activeStage) return;
        ms = Date.now() - stageStartTs[stage];
        isActive = true;
      }
      parts.push({
        label: `${STAGE_LABELS[stage]} <span class="chip-num">${formatStageMs(ms)}</span>`,
        active: isActive,
      });
    });

    if (stats.storiesRead)  parts.push({ label: plural(stats.storiesRead, "story", "stories") + " read" });
    if (stats.searches)     parts.push({ label: plural(stats.searches, "search", "searches") + " run" });
    if (stats.topicsListed) parts.push({ label: plural(stats.topicsListed, "topic", "topics") + " explored" });

    generatingStats.innerHTML = parts
      .map(p => `<span class="chip${p.active ? " chip-active" : ""}">${p.label}</span>`)
      .join("");
  }

  function bumpStats(toolName) {
    if (toolName === "read_story") stats.storiesRead++;
    else if (toolName === "search_stories") stats.searches++;
    else if (toolName === "list_stories_in_topic") stats.topicsListed++;
    renderStatsChips();
  }

  function setGenerating(label, detail) {
    if (label) generatingLabel.textContent = label;
    generatingDetail.textContent = detail || "";
  }

  // ── WebSocket ────────────────────────────────────────────────────────
  function startWebSocket(sessionId) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);

    ws.onopen = () => {
      setGenerating("Generating your beat book", "Reviewing your coverage…");
      setStage("review");
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);

      switch (msg.type) {
        case "message":
          break;

        case "tool_status":
          bumpStats(msg.tool_name);
          setGenerating("Generating your beat book", formatToolDetail(msg));
          break;

        case "questions":
          showInterview(msg);
          break;

        case "research_started":
          setGenerating("Researching context", "Opening the sandbox for the research agent…");
          setStage("research");
          setShimmerIndeterminate();
          break;

        case "research_tool_status":
          setGenerating("Researching context", formatToolDetail(msg));
          break;

        case "research_progress":
          setGenerating("Researching context", msg.detail || msg.stage || "");
          break;

        case "research_message":
          break;

        case "research_complete":
          setGenerating("Research complete", "Handing off to citation matcher…");
          break;

        case "beat_book_markdown_saved":
          setGenerating("Matching citations", "Embedding source sentences…");
          setStage("cite");
          setShimmerDeterminate(0.02);
          break;

        case "citation_progress": {
          const detail = msg.detail || msg.stage || "";
          setGenerating("Matching citations", detail);
          if (typeof msg.fraction === "number") {
            setShimmerDeterminate(msg.fraction);
          }
          break;
        }

        case "beat_book":
          showDone(msg);
          break;

        case "error":
          setGenerating("Something went wrong", msg.text || "Please try again.");
          setShimmerIndeterminate();
          break;
      }
    };

    ws.onclose = () => { /* no-op */ };
  }

  function formatToolDetail(msg) {
    if (msg.detail) return `${msg.tool} — ${msg.detail}`;
    return msg.tool || "";
  }

  // ── Interview rendering ──────────────────────────────────────────────
  function showInterview(msg) {
    activeInterview = {
      intro: msg.intro || "",
      questions: msg.questions || [],
    };

    interviewFormHost.innerHTML = "";

    const form = document.createElement("div");
    form.className = "interview-form";

    const collectors = [];

    activeInterview.questions.forEach((q, i) => {
      const block = document.createElement("div");
      block.className = "question-block";

      const num = document.createElement("div");
      num.className = "question-num";
      num.textContent = `Question ${i + 1} of ${activeInterview.questions.length}`;
      block.appendChild(num);

      const qText = document.createElement("div");
      qText.className = "question-text";
      qText.textContent = q.question;
      block.appendChild(qText);

      const type = q.question_type;
      const options = q.options || [];

      if (type === "free_response") {
        const ta = document.createElement("textarea");
        ta.className = "free-text";
        ta.placeholder = "Type your answer…";
        block.appendChild(ta);
        collectors.push(() => ta.value.trim());
      } else {
        const inputType = type === "single_choice" ? "radio" : "checkbox";
        const list = document.createElement("div");
        list.className = "option-list";

        options.forEach((opt, j) => {
          const item = document.createElement("label");
          item.className = "option-item";
          const id = `q${i}-opt${j}`;
          item.htmlFor = id;

          const input = document.createElement("input");
          input.type = inputType;
          input.name = `q${i}-option`;
          input.value = opt;
          input.id = id;

          const span = document.createElement("span");
          span.textContent = opt;

          item.appendChild(input);
          item.appendChild(span);

          input.addEventListener("change", () => {
            if (inputType === "radio") {
              list.querySelectorAll(".option-item").forEach(el => el.classList.remove("checked"));
            }
            item.classList.toggle("checked", input.checked);
          });

          list.appendChild(item);
        });

        block.appendChild(list);
        collectors.push(() =>
          [...list.querySelectorAll("input:checked")].map(el => el.value)
        );
      }

      form.appendChild(block);
    });

    const row = document.createElement("div");
    row.className = "submit-row";

    const hint = document.createElement("span");
    hint.className = "submit-hint";
    hint.textContent = `${activeInterview.questions.length} question${activeInterview.questions.length === 1 ? "" : "s"}`;
    row.appendChild(hint);

    const submitBtn = document.createElement("button");
    submitBtn.className = "btn primary";
    submitBtn.textContent = "Submit answers";
    submitBtn.addEventListener("click", () => {
      const answers = activeInterview.questions.map((q, i) => ({
        question: q.question,
        answer: collectors[i](),
      }));
      submitInterview(answers);
    });
    row.appendChild(submitBtn);

    form.appendChild(row);
    interviewFormHost.appendChild(form);

    switchScreen("interview");
    window.scrollTo({ top: 0 });
  }

  function submitInterview(answers) {
    activeInterview = null;
    interviewFormHost.innerHTML = "";

    setGenerating("Writing your beat book", "Processing your answers…");
    setStage("write");
    setShimmerIndeterminate();
    switchScreen("generating");

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ answers }));
    }
  }

  // ── Done screen ──────────────────────────────────────────────────────
  function showDone(msg) {
    const viewerUrl = msg.viewer_url || `/static/viewer/viewer.html?book=${encodeURIComponent(msg.stem || "")}`;
    const markdownPath = msg.markdown_path || `/output/${encodeURIComponent(msg.filename)}`;

    doneViewerLink.href = viewerUrl;
    doneMarkdownLink.href = markdownPath;
    doneMarkdownLink.textContent = `Download raw Markdown (${msg.filename})`;

    // Freeze the final stage timing before stopping the elapsed ticker.
    commitStage(activeStage);
    activeStage = null;
    markAllStagesDone();
    setShimmerDeterminate(1);
    stopElapsed();

    const parts = [];
    if (stats.storiesRead)  parts.push(plural(stats.storiesRead, "story", "stories") + " read");
    if (stats.searches)     parts.push(plural(stats.searches, "search", "searches") + " run");
    if (stats.topicsListed) parts.push(plural(stats.topicsListed, "topic", "topics") + " explored");

    const totalMs = elapsedStart ? Date.now() - elapsedStart : 0;
    const stageBits = STAGE_ORDER
      .filter(s => stageDurationsMs[s] != null)
      .map(s => `${STAGE_LABELS[s]} ${formatStageMs(stageDurationsMs[s])}`);

    const sentences = [];
    if (parts.length) sentences.push(`Built from ${parts.join(" · ")}.`);
    if (totalMs > 0) {
      const totalLabel = `Took ${formatStageMs(totalMs)} total`;
      sentences.push(stageBits.length ? `${totalLabel} (${stageBits.join(" · ")}).` : `${totalLabel}.`);
    }
    doneSubtitle.textContent = sentences.join(" ");

    switchScreen("done");
  }

})();
