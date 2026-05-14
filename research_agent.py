"""
research_agent.py
-----------------
Second-stage agent that deepens the beat book with live web research.

It runs *between* the first beat-book agent (agent.py) and the citation matcher
(citation_matcher.py). The first agent produces a Markdown file from the
uploaded source stories; this agent is handed that file inside a private
sandbox directory, does its own research on the open internet, and rewrites the
file in place with additional contextual material a reporter would find
useful (history, key figures, related policy, adjacent coverage, recent news).

Model: Claude Opus 4.7 with adaptive thinking + high effort.

Tools given to the model:
  - bash_20250124            (client-executed, CWD pinned to sandbox)
  - text_editor_20250728     (client-executed, paths pinned to sandbox)
  - web_search_20260209      (server-executed, with dynamic filtering)
  - web_fetch_20260209       (server-executed, with dynamic filtering)
  - finalize_beat_book       (our signal that the markdown is final)

The loop terminates when the model either calls `finalize_beat_book` or ends
its turn naturally. The final revised markdown is read from the sandbox and
returned to the caller, which hands it to the citation matcher.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from anthropic import Anthropic

from claude_client import thinking_enabled, thinking_param

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-7"
MAX_TOKENS_PER_TURN = 16000
MAX_TURNS = 12
BASH_TIMEOUT_SECONDS = 30
WEB_SEARCH_MAX_USES = 8
WEB_FETCH_MAX_USES = 8
WEB_FETCH_MAX_CONTENT_TOKENS = 15_000

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

FINALIZE_TOOL_NAME = "finalize_beat_book"

def build_tools() -> List[Dict[str, Any]]:
    """Tool list passed to the Opus 4.7 API.

    Do NOT add a standalone code_execution tool — the `_20260209` web tools
    run their dynamic filtering inside a managed code-execution sandbox on
    Anthropic's side, and including our own would create two conflicting
    execution environments (per the server-tools docs).
    """
    return [
        {
            "type": "bash_20250124",
            "name": "bash",
        },
        {
            "type": "text_editor_20250728",
            "name": "str_replace_based_edit_tool",
        },
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": WEB_SEARCH_MAX_USES,
        },
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": WEB_FETCH_MAX_USES,
            "max_content_tokens": WEB_FETCH_MAX_CONTENT_TOKENS,
            "citations": {"enabled": True},
        },
        {
            "name": FINALIZE_TOOL_NAME,
            "description": (
                "Call this exactly once, after you have finished editing the "
                "beat book Markdown file. This signals that the file in your "
                "sandbox is the final version and the application should now "
                "hand it to the citation-matching step. After calling this "
                "tool, stop responding."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Filename (not path) of the finalized Markdown "
                            "file inside the sandbox, e.g. 'beat_book.md'."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "One short paragraph summarizing what you added "
                            "or revised and which sources you drew on."
                        ),
                    },
                },
                "required": ["filename", "summary"],
            },
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# Reference list of vetted primary-data sources, injected into the system
# prompt under a `<suggested_sources>` XML tag. Kept as a separate constant
# (rather than inlined into the f-string template) so the literal `{...}`
# placeholders in the URL/API patterns don't conflict with str.format().
SUGGESTED_SOURCES = """\
<suggested_sources>
The following are vetted primary-data sources. Most are scoped to Chicago, \
Cook County, or Illinois — match these against the beat's geography and \
skip any that don't fit. The federal and institutional sources at the \
bottom of the list apply nationally. These are starting points, not \
requirements: use them when they actually serve the story, find others \
when they don't. Many of the data portals listed here also make excellent \
scraper targets — Socrata-based Chicago and Cook County datasets in \
particular return JSON from a plain `requests.get`, no HTML parsing \
required.

## City of Chicago — Open Data Chicago

- Portal: https://data.cityofchicago.org
- API pattern: `https://data.cityofchicago.org/resource/{DATASET_ID}.json?{SoQL_params}`
- Catalog search (when the dataset ID is unknown): `https://data.cityofchicago.org/browse?q={keyword}`
- SoQL params: `$where`, `$order`, `$limit`, `$select`. Date fields are typically `date` or `arrest_date`. Geographic fields: `community_area` (int 1–77), `ward` (int 1–50), `beat`, `district`.

### Crime & public safety
- **Crimes — 2001 to Present** (`ijzp-q8t2`): query `community_area`, `ward`, `primary_type`, `date`. Filter by community area or ward; order by `date` DESC; filter `primary_type` (HOMICIDE, ROBBERY) when the story specifies a crime type.
- **Victims of Homicides & Non-Fatal Shootings** (`gumc-mgzr`): query `community_area`, `ward`, `date`. Includes victim age, race, gender, location.
- **OEMC 311 Service Calls** (`n7dm-b26x`): query `sr_type`, `community_area`. Filter `sr_type` (Abandoned Vehicle, Street Light) for quality-of-life beat context.
- **Arrests** (`dpt3-jri9`): query `community_area`, `charge_description`, `arrest_date`. Match `charge_description` to the story topic (e.g. WEAPONS VIOLATION).
- **Speed / Red Light Camera Violations** (`hhkd-xvj4` / `spqx-js37`): query `address`, `violation_date`. Pull by intersection or address — useful for traffic-enforcement stories.

### Housing & development
- **Building Permits** (`ydr8-5enu`): query `community_area`, `ward`, `work_description`, `issue_date`. Search `work_description` for type (new construction, demolition, renovation); order by `issue_date` DESC.
- **Building Violations** (`22u3-xenr`): query `address`, `community_area`, `violation_date`. `violation_description` carries the detail.
- **Affordable Rental Housing Developments** (`s6ha-ppgi`): query `community_area`, `ward`. Includes unit counts, funding source, developer.
- **TIF District Revenues & Expenditures** (`qnki-7a4z`): query `tif_district_name`. Use when the beatbook names a TIF district or known TIF-active neighborhood.
- **Homeless Shelter Locations** (`7fnd-mgm3`): static lookup. Pull the full list and join to the beat area by address.

### Transit & transportation
- **CTA Daily Ridership — L Stations** (`6iiy-9s97`): query `stationname`, `date`. Pull last ~90 days for a station near the beat — good for service-cut or transit-development angles.
- **CTA Daily Ridership — Bus Routes** (`bynn-gwxy`): query `route`, `date`. Filter by route number when the story mentions a specific bus line.
- **Chicago Traffic Tracker — Congestion** (`n4j6-wkkf`): query `_traffic`, `street`. Filter by street name for infrastructure / accident stories.
- **Transportation Network Providers (Uber/Lyft)** (`m6dm-c72p`): query `pickup_community_area`, `dropoff_community_area`. Aggregate trip volume by community area for transportation-equity stories.

### Education
- **CPS School Profile Information** (`kh4r-387c`): query `school_nm`, `community_area_number`, `ward`. Returns enrollment, demographics, school type.
- **CPS School Progress Reports** (`cp7s-7gxg`): query `school_name`. Returns SQRP rating, attendance, growth scores.
- **CPS School Locations** (`3fhj-xtn5`): query `community_area_name`. Spatial lookup — all schools in a named area.
- **CPS School Budgets** (`7e8t-hmrc`): query `school_name`, `fiscal_year`. Useful for funding-cut or equity stories.

### Health
- **Public Health Statistics — Selected Indicators** (`iqnk-2tcu`): query `community_area_name`. All indicators for the area: birth rate, infant mortality, lead exposure, poverty, cancer.
- **CDPH Environmental Records** (`um2n-yweb`): query `address`, `community_area`. Lead paint, asbestos, environmental complaints.
- **Food Inspections** (`4ijn-s7e5`): query `dba_name`, `address`, `community_area`. Filter to `Fail` for story context; includes violation descriptions.

### City government & politics
- **City Council Voting Records** (`fg6s-gzvg`): query `alderman_name`, `ward`, `agenda_item_title`. Search agenda titles for keywords or filter by alderman.
- **Lobbyist Activity** (`g6zi-3tx5`): query `action_sought`, `client_name`. Reveals who is lobbying for what.
- **Lobbyist Compensation** (`fvf5-veis`): query `client_name`, `lobbyist_name`. Cross-reference with lobbying activity for dollar amounts.
- **City Contracts** (`rsxa-ify5`): query `vendor_name`, `department`, `award_date`. Search by vendor or company from the beatbook.
- **Employee Payroll** (`xzkq-xp2w`): query `name`, `department_description`, `title`. Public-employee compensation lookups.
- **FOIA Requests Log** (`ixfu-8ru6`): query `requestor_name`, `department`, `date_received`. Reveals what other reporters/orgs are investigating.
- **City Budget Appropriations** (`25uj-qe7m`): query `department_name`, `appropriation_authority`. Filter by department; compare year-over-year.

### Chicago City Clerk — meeting records
- URL: https://chicityclerk.com/city-council/council-meetings
- No public API. Navigate to the meeting record for the relevant date or ordinance number; search the full-text agenda PDFs for organization names, addresses, or topics from the beatbook.

## Cook County

- Data catalog: https://datacatalog.cookcountyil.gov · search at `https://datacatalog.cookcountyil.gov/browse?q={keyword}`. Most datasets are Socrata and accept the same `$where` / `$order` SoQL pattern as Open Data Chicago.

### Cook County datasets
- **Medical Examiner Case Archive** (`cjeq-bs86`): query `manner_of_death`, `primary_cause`, `incident_city`, `death_date`. Filter `incident_city = 'Chicago'` + date range; `primary_cause` for cause-specific stories (gun, fentanyl, etc.).
- **Criminal Court Date Dispositions** (`apwk-dzx8`): query `charge_description`, `disposition_date`, `court_facility_name`. Reveals case outcomes, judge, sentence.
- **Property Tax Assessment (Residential)** (`tx2p-k2g9`): query `address`, `mail_address`. Returns assessed value, property class, owner name.
- **Residential Sales** (`wvhk-k5uv`): query `address`, `sale_date`, `nbhd`. Recent sales for an address or neighborhood — gentrification stories.
- **Cook County Budget** (`sd3e-tys6`): query `fund_name`, `department_name`. Filter by department.

### Cook County Assessor (manual lookup)
- URL: https://www.cookcountyassessor.com/address-search
- Submit an address from the beatbook to get the PIN, owner name, assessed value, exemptions. PIN can be cross-referenced with sales history, tax payments, and building permits.

### Cook County Circuit Court (Clerk)
- URL: https://courtclerk.org/case-search/
- Search by party name (person/org from beatbook) or case number. Filter by case type (civil, criminal, eviction/forcible entry, domestic relations). Returns case status, filings, judgment amounts. No bulk API — individual lookups only.

## State of Illinois

- **Illinois Comptroller — Ledger**: https://ledger.illinoiscomptroller.gov · expenditure search at `/expenditures`. Search by vendor or agency; export CSV for volume queries. Filter agency to IDOT, DCFS, IDHS for Chicago-relevant state spending.
- **Illinois Secretary of State — Corporate Filings**: https://www.ilsos.gov/corporatellc/ · search by org name; returns registered agent, incorporation date, principal address, officers/directors. Cross-reference officer names against other beatbook entities. Status (dissolved/inactive) appears in results.
- **Illinois Campaign Finance (ILCAMPAIGN)**: committee search at https://www.elections.il.gov/CampaignDisclosure/SearchByCommittee.aspx · contribution search at https://www.elections.il.gov/CampaignDisclosure/ContributionSearchByAllContributions.aspx · bulk downloads at https://www.elections.il.gov/downloads/CampaignDisclosure/CDfiles.aspx.
- **Illinois Department of Public Health (IDPH)**: https://dph.illinois.gov/data-statistics · most datasets are bulk CSVs. Opioid dashboard: https://dph.illinois.gov/topics-services/prevention-wellness/opioid. Use the Vital Statistics Query System for births/deaths.
- **Illinois Dept. of Corrections — Inmate Search**: https://www.illinoisdoc.com/roster · search by name; returns current facility, sentence start/end dates, offense.
- **Illinois Courts — Odyssey / Supreme & Appellate**: Cook County Circuit Court is on `courtclerk.org` (above). Statewide opinions: https://www.illinoiscourts.gov/courts/supreme-court/opinions. Other circuits often use Tyler Odyssey portals — search by case number or party name via the circuit's portal. Not all courts are online.

## Federal sources

- **U.S. District Court — Northern District of Illinois**: use CourtListener first (free): `https://www.courtlistener.com/?q={query}&court=ilnd`. PACER (account required): https://ecf.ilnd.uscourts.gov. Primary source for civil cases against Chicago city/county entities and federal criminal cases (gun trafficking, corruption, immigration).
- **U.S. Census — American Community Survey**: `https://api.census.gov/data/{year}/acs/acs5?get={variables}&for=tract:*&in=state:17+county:031` (Cook County FIPS = 17/031). Useful variables: `B01003_001E` total population, `B19013_001E` median household income, `B25070_010E` rent burden >50%, `B03002_003E` non-Hispanic white, `B23025_005E` unemployed. https://censusreporter.org is a friendly front-end.
- **Federal Election Commission**: https://api.open.fec.gov/v1/ · candidates: `/v1/candidates/?state=IL&office=H` (or `S`). Contributions: `/v1/schedules/schedule_b/?contributor_name={org_name}`. Free API key from https://api.data.gov.
- **USAspending.gov**: https://api.usaspending.gov/api/v2/ · awards to a Chicago vendor: `/v2/search/spending_by_award/` with `filters.recipient_search_text={org_name}` + `filters.place_of_performance_locations.city=Chicago`. Web UI: https://www.usaspending.gov/search.
- **OSHA Inspections**: establishment search at https://www.osha.gov/enforcement/establishment-search · bulk data at https://enforcedata.dol.gov/views/data_summary.php (filter state=IL, city=Chicago).
- **EPA ECHO — Environmental Compliance**: facility search at https://echo.epa.gov/facilities/facility-search. For neighborhood-level environmental burden, use EJSCREEN: https://ejscreen.epa.gov/mapper/ (enter a Chicago address).
- **Bureau of Labor Statistics — Chicago Metro**: https://api.bls.gov/publicAPI/v2/timeseries/data/. Series IDs: `LAUMT171698` (Chicago MSA unemployment rate), `SMU17169800000000001` (total nonfarm employment). Find more series via https://beta.bls.gov/dataQuery/find?fq=survey:[sm]&q=chicago.
- **HUD — Affordable Housing**: https://hudgis-hud.opendata.arcgis.com · search "LIHTC", "Section 8", "public housing"; filter to IL/Chicago. Fair Market Rents: https://www.huduser.gov/portal/datasets/fmr.html.
- **Home Mortgage Disclosure Act (HMDA)**: https://ffiec.cfpb.gov/data-download · download the Illinois loan-level file. Filter `lei` (lender) by bank name or `census_tract` to the beatbook's community area. Key fields: `action_taken`, `loan_purpose`, `applicant_race`, `income`. Used for redlining / lending-equity stories.

## Established institutional sources

- **ProPublica Nonprofit Explorer**: https://projects.propublica.org/nonprofits/ · search by org name for IRS 990 filings: revenue, expenses, executive compensation, board members. API: https://projects.propublica.org/nonprofits/api.
- **EvictionLab (Princeton)**: https://evictionlab.org/data-downloads/ · download IL census-tract or ZIP file; filter to Cook County FIPS 17031 and the tracts/ZIPs for the beat's community area. Fields: eviction filings, eviction rate, eviction-judgment rate by year.
</suggested_sources>
"""


SYSTEM_PROMPT_TEMPLATE = """\
You are a research assistant for a reporter. A prior agent has produced a \
Markdown beat book — a reporting guide for a specific beat — based on past \
coverage in the reporter's newsroom. Your job is to deepen it with live \
research from the open internet so the reporter has richer context.

# Your sandbox

You are operating inside a private working directory. All files you need to \
read or write live here. Use the `bash` tool or the text editor with \
relative paths (or paths under this directory). Do not try to read or write \
files outside this directory — those attempts will be rejected.

Python 3 is available through `bash` — run ad-hoc logic with \
`python3 -c "..."` or by writing a helper script to the sandbox and \
running it with `python3 script.py`. Use it whenever it beats shell \
plumbing: parsing JSON / HTML, computing stats, rewriting the Markdown \
programmatically, deduping links, etc.

The beat book Markdown file is already in your sandbox. Its filename is:

    {markdown_filename}

Start by reading it (the text editor's `view` command is the fastest way) \
before planning your research.

# What to research

Cast a slightly wider net than the explicit topic. A reporter picking up a \
beat benefits from background that a prior agent working only from old \
stories cannot provide:

- Current state of the beat as of today: major recent developments, ongoing \
  legal proceedings, new legislation, leadership changes.
- Key people, organizations, and institutions — their history, mandates, \
  funding sources, controversies. Confirm spellings, titles, and roles.
- Statistical / demographic context relevant to the beat (population, \
  budgets, crime rates, historical trends — whatever fits).
- Adjacent beats and storylines that a reporter should know exist even if \
  they aren't the focus (e.g. for a sports beat, note ownership disputes, \
  stadium-financing debates, labor issues).
- Authoritative primary sources the reporter should bookmark: agency \
  dashboards, court dockets, public records portals, budget documents, \
  research reports, watchdog orgs.
- Recurring events, deadlines, and meeting cadences not already in the file.

Use `web_search` to find candidates and `web_fetch` to read the most \
promising pages in depth. Web fetch can only retrieve URLs that have \
already appeared in the conversation (including from prior search results), \
so you must search before fetching an unfamiliar URL.

# Build at least one scraper

Before you finalize, you MUST write and run at least one small Python \
scraper that pulls structured data from a live web page relevant to this \
beat. This is non-negotiable — shell `curl` or `web_fetch` alone do not \
count. The scraper should:

- Live in the sandbox as a `.py` file you create (e.g. `scrape_meetings.py`).
- Use the standard library (`urllib.request`, `html.parser`, `json`, `re`, \
  `csv`) or any of `requests`, `beautifulsoup4` (import `bs4`), or `lxml` \
  — all three are installed.
- Target something the reporter would actually want in structured form: \
  an upcoming meetings / hearings calendar, a roster of officials, a \
  court docket listing, press-release archive, budget line items, an \
  RSS/Atom feed, a JSON API response, etc. Many of the portals listed \
  in the `<suggested_sources>` block (below) are good candidates when \
  the beat falls in their geographic scope.
- Write its parsed output to a file in the sandbox (JSON or Markdown \
  works well) and print a short summary to stdout.
- Be polite: a reasonable `User-Agent` header, no tight loops, respect \
  obvious `robots.txt` hints. One page fetch is enough to count.

Run the scraper with `python3 scraper_name.py`, inspect the output, then \
fold the most useful rows into the beat book (e.g. a "Calendar" or "Key \
People" section). Keep the scraper file in the sandbox — the reporter may \
reuse it.

If your first target 4xx / 5xxs or returns unexpected HTML, try a \
different source rather than scraping something useless; one working \
scraper beats three broken ones.

# How to revise the file

Your revisions should feel native to the document, not bolted on. \
Guidelines:

- Prefer *integrating* new material into existing sections over appending a \
  new "Web research" section at the bottom.
- When you add a fact from the web, include a brief inline attribution with \
  the publication and date (e.g. "(Chicago Tribune, Mar 2026)"). The next \
  pipeline stage will add formal citations from the reporter's own source \
  stories — so you do not need to insert Markdown footnotes, but do keep \
  the inline attribution text short and natural.
- Add new bullet points, sub-sections, or short paragraphs where the \
  existing document thins out (e.g. a "Key Sources & Players" section \
  missing notable figures, or a "Calendar" section missing regular \
  meetings).
- Prefer specific, verifiable facts (dates, dollar amounts, names with \
  titles, case numbers) over generic color.
- Do not remove content from the file unless it is demonstrably wrong and \
  you have a replacement.
- Do not add a table of contents. The viewer builds its own.
- Do not invent facts. If you can't verify something, leave it out.

# Suggested sources

The `<suggested_sources>` block below lists vetted primary-data sources. \
Consult it whenever the beat's geography overlaps Chicago, Cook County, \
Illinois, or one of the federal/national sources at the bottom of the \
list — and skip the rest. Many of these portals are also good targets \
for the scraper requirement above.

{suggested_sources}

# Workflow

1. View the Markdown file.
2. Plan 2–3 research threads based on the beat and the file's current gaps. Be selective — prioritize the most impactful gaps. You have a limited turn budget so move efficiently.
3. Search + fetch authoritative sources. Prefer primary sources, major \
   newspapers, and government/NGO publications. The `<suggested_sources>` \
   block has vetted starting points when the beat overlaps Chicago, Cook \
   County, Illinois, or the listed federal sources.
4. Write and run at least one Python scraper (see "Build at least one \
   scraper" above) and fold its output into the beat book.
5. Edit the file incrementally with the text editor's `str_replace` and \
   `insert` commands. Use `bash` for larger operations (e.g. `cat` to \
   re-read, `wc -w` to track length).
6. When the file is meaningfully improved and your scraper has run, call \
   `finalize_beat_book` immediately — do not keep searching once you have \
   solid additions.

Keep your running text messages brief — your real work is in the tools. \
Do not narrate every step; progress updates are enough.\
"""



# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX-AWARE TOOL HANDLERS (client-executed)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_inside_sandbox(sandbox_dir: Path, raw_path: str) -> Optional[Path]:
    """Resolve `raw_path` against the sandbox and confirm it stays inside.

    Returns the resolved Path on success, or None if the path escapes the
    sandbox. Symlinks are followed during resolution, so a symlink pointing
    outside will also be rejected.
    """
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = sandbox_dir / candidate
    try:
        resolved = candidate.resolve(strict=False)
        sandbox_resolved = sandbox_dir.resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    if resolved != sandbox_resolved and sandbox_resolved not in resolved.parents:
        return None
    return resolved


def _run_bash(command: Optional[str], restart: bool, sandbox_dir: Path) -> str:
    """Execute a bash command inside the sandbox. Returns combined
    stdout/stderr (possibly prefixed with an error note)."""
    if restart:
        # We don't maintain a persistent shell — every call is a fresh
        # subprocess — so "restart" is a no-op. Return a friendly note.
        return "Bash session reset. (Each command runs in a fresh shell.)"
    if not command:
        return "Error: bash requires a `command` or `restart: true`."

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(sandbox_dir),
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {BASH_TIMEOUT_SECONDS}s."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        out = f"(exit {proc.returncode})\n{out}"
    # Cap output so a runaway command can't blow out the context window.
    if len(out) > 20_000:
        out = out[:20_000] + "\n\n[... output truncated ...]"
    return out or "(no output)"


def _run_text_editor(tool_input: Dict[str, Any], sandbox_dir: Path) -> str:
    """Execute a text_editor_20250728 command against a file inside the sandbox."""
    command = tool_input.get("command")
    raw_path = tool_input.get("path", "")
    resolved = _resolve_inside_sandbox(sandbox_dir, raw_path)
    if resolved is None:
        return f"Error: path '{raw_path}' is outside the sandbox and cannot be accessed."

    try:
        if command == "view":
            view_range = tool_input.get("view_range")
            if resolved.is_dir():
                entries = sorted(p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir())
                return "\n".join(entries) if entries else "(empty directory)"
            if not resolved.exists():
                return f"Error: file not found: {raw_path}"
            text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if view_range and isinstance(view_range, list) and len(view_range) == 2:
                start, end = view_range
                start = max(1, int(start))
                end_i = len(lines) if int(end) == -1 else min(len(lines), int(end))
                lines = lines[start - 1 : end_i]
                offset = start
            else:
                offset = 1
            numbered = [f"{offset + i}: {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered) if numbered else "(empty file)"

        if command == "create":
            file_text = tool_input.get("file_text", "")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(file_text, encoding="utf-8")
            return f"File created at {raw_path} ({len(file_text)} chars)."

        if command == "str_replace":
            if not resolved.exists():
                return f"Error: file not found: {raw_path}"
            old_str = tool_input.get("old_str", "")
            new_str = tool_input.get("new_str", "")
            content = resolved.read_text(encoding="utf-8", errors="replace")
            count = content.count(old_str)
            if count == 0:
                return "Error: No match found for replacement. Please check your text and try again."
            if count > 1:
                return f"Error: Found {count} matches for replacement text. Please provide more context to make a unique match."
            resolved.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
            return "Successfully replaced text at exactly one location."

        if command == "insert":
            if not resolved.exists():
                return f"Error: file not found: {raw_path}"
            insert_line = int(tool_input.get("insert_line", 0))
            insert_text = tool_input.get("insert_text", "")
            content = resolved.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)
            if insert_line < 0 or insert_line > len(lines):
                return f"Error: insert_line {insert_line} out of range (file has {len(lines)} lines)."
            if insert_text and not insert_text.endswith("\n"):
                insert_text += "\n"
            lines.insert(insert_line, insert_text)
            resolved.write_text("".join(lines), encoding="utf-8")
            return f"Inserted text after line {insert_line}."

        return f"Error: unknown text_editor command '{command}'."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str, str], Awaitable[None]]        # (stage, detail)
ToolStatusCallback = Callable[[str, str, str], Awaitable[None]]  # (tool_name, desc, detail)
TextCallback = Callable[[str], Awaitable[None]]                   # (assistant text)


TOOL_DESCRIPTIONS = {
    "bash": "Running shell command",
    "str_replace_based_edit_tool": "Editing beat book",
    "web_search": "Searching the web",
    "web_fetch": "Fetching a web page",
    FINALIZE_TOOL_NAME: "Finalizing beat book",
}


def _short_detail_for(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Pick a short human-readable detail string for a tool status event."""
    if tool_name == "bash":
        cmd = tool_input.get("command") or ("restart" if tool_input.get("restart") else "")
        return (cmd or "")[:120]
    if tool_name == "str_replace_based_edit_tool":
        return f"{tool_input.get('command', '')} {tool_input.get('path', '')}".strip()[:120]
    if tool_name == "web_search":
        return str(tool_input.get("query", ""))[:120]
    if tool_name == "web_fetch":
        return str(tool_input.get("url", ""))[:120]
    if tool_name == FINALIZE_TOOL_NAME:
        return str(tool_input.get("filename", ""))[:120]
    return ""


async def _emit(cb: Optional[Callable], *args) -> None:
    if cb is None:
        return
    result = cb(*args)
    if asyncio.iscoroutine(result):
        await result


async def run_research_agent(
    sandbox_dir: Path,
    markdown_filename: str,
    anthropic_api_key: str,
    on_progress: Optional[ProgressCallback] = None,
    on_tool_status: Optional[ToolStatusCallback] = None,
    on_text: Optional[TextCallback] = None,
    initial_content: Optional[str] = None,
) -> str:
    """Run the research agent and return the final Markdown content.

    If `initial_content` is provided the file is (re)written with that content
    before the agent starts — useful when the caller wants to start research
    before the full draft is ready (concurrent mode).

    The caller is responsible for creating `sandbox_dir` before calling.
    """
    sandbox_dir = Path(sandbox_dir)
    if not sandbox_dir.is_dir():
        raise FileNotFoundError(f"Sandbox directory does not exist: {sandbox_dir}")
    markdown_path = sandbox_dir / markdown_filename
    if initial_content is not None:
        markdown_path.write_text(initial_content, encoding="utf-8")
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Markdown file not found in sandbox: {markdown_path}")

    client = Anthropic(api_key=anthropic_api_key)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        markdown_filename=markdown_filename,
        suggested_sources=SUGGESTED_SOURCES,
    )

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Your sandbox is ready. The beat book Markdown is at "
                f"`{markdown_filename}`. Read it, plan your research, revise "
                "it with additional contextual material, and call "
                "`finalize_beat_book` when done."
            ),
        }
    ]

    tools = build_tools()
    finalized = False
    container_id: Optional[str] = None

    await _emit(on_progress, "starting", f"Research agent initializing in sandbox {sandbox_dir.name}")

    for turn in range(MAX_TURNS):
        await _emit(on_progress, "thinking", f"Turn {turn + 1}/{MAX_TURNS}")

        # Server-executed tools (web_search / web_fetch with dynamic filtering)
        # run inside an Anthropic-managed code-execution container. Once one is
        # allocated we must thread its id back on every subsequent request or
        # the API returns: "container_id is required when there are pending
        # tool uses generated by code execution with tools."
        # Opus 4.7 only supports adaptive thinking (the legacy
        # type=enabled+budget_tokens shape returns 400). When the flag is off
        # we send type=disabled — equivalent to omitting the field but makes
        # the intent explicit in the request.
        request_kwargs: Dict[str, Any] = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS_PER_TURN,
            "system": [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            "tools": tools,
            "messages": messages,
            "thinking": thinking_param(),
        }
        if container_id is not None:
            request_kwargs["container"] = container_id

        print(
            f"[research_agent] turn={turn + 1} sending container={container_id!r}",
            flush=True,
        )

        def _stream_once():
            # Streaming is required by the SDK for requests whose expected
            # duration exceeds ~10 minutes. Opus 4.7 with adaptive thinking +
            # medium effort + 32k max_tokens lands in that regime, so we stream
            # and collect the final message. Run inside a thread so the async
            # event loop (FastAPI's WebSocket handler) is never blocked.
            #
            # The server-tool container_id is delivered through mid-stream
            # message_start / message_delta events, not the consolidated
            # final Message — so we iterate the stream to capture it.
            streamed_container_id: Optional[str] = None
            with client.messages.stream(**request_kwargs) as stream:
                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "message_start":
                        msg = getattr(event, "message", None)
                        c = getattr(msg, "container", None) if msg is not None else None
                        if c is not None:
                            streamed_container_id = c.id
                    elif etype == "message_delta":
                        delta = getattr(event, "delta", None)
                        c = getattr(delta, "container", None) if delta is not None else None
                        if c is not None:
                            streamed_container_id = c.id
                return stream.get_final_message(), streamed_container_id

        try:
            response, streamed_cid = await asyncio.to_thread(_stream_once)
        except Exception as e:
            raise RuntimeError(f"Opus 4.7 request failed on turn {turn + 1}: {e}") from e

        if streamed_cid is not None:
            container_id = streamed_cid
            print(
                f"[research_agent] turn={turn + 1} captured container_id={container_id!r}",
                flush=True,
            )
        else:
            # Fall back to the final-message field (usually None in this flow)
            # but don't wipe out a previously cached id.
            container_obj = getattr(response, "container", None)
            if container_obj is not None:
                container_id = container_obj.id
                print(
                    f"[research_agent] turn={turn + 1} final container_id={container_id!r}",
                    flush=True,
                )
            else:
                print(
                    f"[research_agent] turn={turn + 1} no container event "
                    f"(cached container_id={container_id!r})",
                    flush=True,
                )

        # What tools did this turn actually use? Useful when diagnosing
        # container-lifecycle issues.
        turn_block_types: List[str] = []
        for b in response.content:
            bt = getattr(b, "type", "?")
            if bt in ("server_tool_use", "tool_use", "web_search_tool_result", "web_fetch_tool_result"):
                turn_block_types.append(f"{bt}:{getattr(b, 'name', '')}".rstrip(":"))
        if turn_block_types:
            print(
                f"[research_agent] turn={turn + 1} stop_reason={response.stop_reason} "
                f"blocks={turn_block_types}",
                flush=True,
            )

        # Preserve the full assistant content (including any thinking blocks)
        # in the running transcript so interleaved thinking stays coherent.
        messages.append({"role": "assistant", "content": response.content})

        # Forward any plain-text narration to the caller.
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "").strip()
                if text:
                    await _emit(on_text, text)

        stop_reason = response.stop_reason

        if stop_reason == "end_turn":
            break

        if stop_reason == "pause_turn":
            # The API paused a long server-tool turn. Continue by re-sending
            # the transcript as-is; the server resumes where it left off.
            await _emit(on_progress, "paused", "Server-side tools paused; resuming")
            continue

        if stop_reason == "max_tokens":
            # Ran out of output budget mid-turn — nudge the model to continue.
            messages.append(
                {
                    "role": "user",
                    "content": "Your previous response hit the token limit. Please continue.",
                }
            )
            continue

        if stop_reason == "tool_use":
            tool_results: List[Dict[str, Any]] = []

            for block in response.content:
                block_type = getattr(block, "type", None)
                # server_tool_use blocks (web_search / web_fetch) are executed
                # by Anthropic; the API also returns their results in the same
                # assistant turn. We don't synthesize tool_result for them.
                if block_type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input or {}

                await _emit(
                    on_tool_status,
                    tool_name,
                    TOOL_DESCRIPTIONS.get(tool_name, tool_name),
                    _short_detail_for(tool_name, tool_input),
                )

                if tool_name == "bash":
                    result = _run_bash(
                        tool_input.get("command"),
                        bool(tool_input.get("restart")),
                        sandbox_dir,
                    )
                elif tool_name == "str_replace_based_edit_tool":
                    result = _run_text_editor(tool_input, sandbox_dir)
                elif tool_name == FINALIZE_TOOL_NAME:
                    final_filename = tool_input.get("filename") or markdown_filename
                    summary = tool_input.get("summary", "").strip()
                    await _emit(on_progress, "finalizing", summary or "Finalized.")
                    # Verify the claimed final file exists inside the sandbox
                    # and update the path we'll read back at the end.
                    candidate = _resolve_inside_sandbox(sandbox_dir, final_filename)
                    if candidate is not None and candidate.is_file():
                        markdown_path = candidate
                        result = (
                            f"Beat book finalized. The application will now "
                            f"read `{final_filename}` and hand it to the "
                            f"citation-matching step."
                        )
                        finalized = True
                    else:
                        result = (
                            f"Error: the file '{final_filename}' you named was "
                            "not found in the sandbox. Create or rename it, "
                            "then call finalize_beat_book again."
                        )
                else:
                    result = f"Error: unknown tool '{tool_name}'."

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if finalized:
                break
            continue

        # Any other stop reason: bail out rather than loop forever.
        await _emit(on_progress, "unexpected_stop", f"Unexpected stop_reason: {stop_reason}")
        break

    await _emit(on_progress, "done", "Research agent finished")
    return markdown_path.read_text(encoding="utf-8")
