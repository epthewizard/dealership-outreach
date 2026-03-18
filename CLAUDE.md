# CLAUDE.md — Dealership Contact Agent v5

> **Only file to use: `main_update.py` (browser-use 0.12.2, `venv_new/`)**
> `main.py` and `test.py` are deprecated — never edit or run them.

---

## Project Overview

Python automation agent that:
1. Discovers car dealership websites in the Minneapolis area via Google Maps scraping (`discover.py`)
2. For each dealership, launches an AI-controlled browser that navigates to the contact form, fills it out, and submits it (`main_update.py`)

**Campaign**: Eric Paulson (ACES Inc. marketing) pitching a vehicle partnership on behalf of Victor Caratini, catcher for the Minnesota Twins.

---

## Stack

| Component | Version / Detail |
|-----------|-----------------|
| Python | 3.11+ |
| browser-use | 0.12.2 (venv_new/) |
| Playwright | Chromium, headless=False |
| LLM | qwen3:14b via Ollama (local, Windows) |
| HTTP probe | httpx (browser-use transitive dep) |
| GPU | RTX 4080 Super, 16 GB VRAM |

---

## File Structure

```
dealership-outreach-browseruse/
├── config.py           # All user settings — edit before running
├── main_update.py      # ONLY MAINTAINED VERSION (browser-use 0.12.2)
├── main.py             # DEPRECATED (browser-use 0.1.40) — do not use
├── test.py             # DEPRECATED — imports from main.py, do not use
├── discover.py         # Google Maps scraper
├── requirements.txt
├── results.json        # Auto-created — submitted/failed/skipped/tested lists
├── leads.json          # Auto-created — email addresses when no form found
├── logs/               # Auto-created — run_v5_YYYYMMDD_HHMMSS.log files
└── CLAUDE.md           # This file
```

---

## Running the Agent

```powershell
# Always use venv_new
.\venv_new\Scripts\Activate.ps1

# Full production run (discovery + form submissions)
python main_update.py

# Discovery only (see what URLs would be scraped, no submissions)
python discover.py
```

**To retry a failed/skipped URL**: remove it from the appropriate list in `results.json`, then re-run.  
**To wipe all history**: delete `results.json`.

---

## config.py — All User Settings

```python
USER_INFO = {
    "name":  "Eric Paulson",
    "email": "epaulson@acesinc1.com",
    "phone": "347-633-2712",
    "message": "...(full Victor Caratini pitch)...",
    "zip": "11201",
}
EXTRA_DEALERSHIPS = []          # Manually add URLs to process before auto-discovery
SEARCH_QUERIES = [...]           # Google Maps queries — one Maps search per entry
MAX_PER_QUERY = 10               # Max dealerships to pull per query
DELAY_BETWEEN_SUBMISSIONS = 3    # Seconds between form submissions
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_BASE_URL = "http://localhost:11434"
USE_VISION = False               # DOM-only mode — saves tokens, works well
```

> The message body contains `https://acesincbaseball.com`. Some spam filters flag messages with URLs. The agent prompt handles this: if a form rejects the submission citing "link/URL/spam", the agent is instructed to retry without the URL.

---

## Architecture Deep Dive

### Phase 1: Discovery (`discover.py`)

Uses Playwright against **Google Maps** (not Google Search — Maps avoids bot/captcha detection).

- Selector `[role="feed"] a[href*="/maps/place/"]` matches listing links by URL pattern, stable across Maps layout changes
- Navigates to each place page, grabs website URL from `a[data-item-id="authority"]`
- Always normalizes to root domain (`https://dealership.com`) — Maps sometimes links to `/service` or `/parts` subpages
- Deduplication enforced globally across all queries and `already_seen` (from results.json)
- `is_likely_dealership(url, place_name)` filters against:
  - `_DEALERSHIP_BRANDS` — Toyota, Ford, BMW, Mercedes, etc. (franchise brands)
  - `_SKIP_KEYWORDS` — AutoZone, body shop, tire shop, rental, etc.

### Phase 2: HTTP Probe (`probe_contact_url`)

Before launching the AI agent, all 11 contact URL paths are probed **in parallel** via httpx:

```
/contact-us  /contact  /contactus  /about/contact-us  /about-us/contact
/connect  /reach-us  /get-in-touch  /inquiry  /customer-care  /dealership/contact
```

**Probe logic per URL:**
1. `HEAD` request (follow_redirects=True, 7s timeout)
2. If `HEAD` returns 200: check for redirect-to-root (path empty after strip → reject)
3. Fetch body via `GET`, check first 4000 chars for `_SOFT_404_PHRASES`
4. If server rejects HEAD (405/501): fallback to `GET` directly with same checks
5. Returns first valid URL in priority order (index 0 = `/contact-us` wins over index 5 = `/connect`)

**Total timeout**: 15 seconds via `asyncio.wait_for` — never blocks the run for more than that.

**Benefit**: Saves 5–10 agent navigation steps per dealership. If probe finds the contact page, the agent starts directly on it with the context "you are already on the contact page."

### Phase 3: AI Agent (`contact_dealership` → `Agent.run`)

**LLM**: `_ChatOllamaNoThink` (subclass of browser-use's native `ChatOllama`)
- Overrides `ainvoke()` to inject `think=False` into every Ollama API call
- **Why mandatory**: qwen3:14b with thinking enabled burns ALL `num_predict` tokens on its `<think>` monologue and returns `content=''` — empty response, crashes agent every step
- `_clean_response()` as defense-in-depth: strips `<tool_call>` XML wrappers that qwen3 sometimes emits from conversation history. In 0.12.2, Ollama's `format=schema` prevents these at API level so this rarely fires.

**Browser session**: `BrowserSession(BrowserProfile(headless=False))` — visible browser for debugging. Session is killed in the `finally` block regardless of outcome.

**Agent config**:
- `use_vision=False` — DOM-only (tokens saved, equivalent results for form filling)
- `use_thinking=False` — removes thinking field from output schema
- `max_actions_per_step=3` — 3 actions per LLM call max (prevents cascading failures when early action in a batch fails)
- `max_failures=5` — if LLM fails to parse 5 consecutive times, abort
- `max_steps=30` — 30 LLM calls total per dealership

**Controller** (custom actions registered via `@controller.action`):

| Action | Purpose |
|--------|---------|
| `dismiss_popups` | Runs `DISMISS_JS` (5-phase overlay remover). Call at start and any time a new overlay blocks the form. |
| `scrape_emails` | Fallback when no contact form found — scrapes `mailto:` links and inline email addresses |
| `select_by_label(label_text, option_text)` | Finds `<select>` by label text proximity (checks `<label for>`, `aria-labelledby`, and DOM parent text up to 5 levels). Fires `change`+`input` events. Use when `select_dropdown` by index fails or indices are stale. |
| `get_form_errors()` | Scans for Gravity Forms errors (`.gfield_error`), generic error classes, `aria-invalid` fields, red-border fields. Returns list of "Field needs value: X" entries. |
| `check_submission_state()` | Fast body-text scan for confirmation/error phrases. Returns one of: `submitted - confirmation visible`, `submitted - error visible`, `form_visible`, `form_not_visible - likely submitted or navigated away`, `unknown`. |

---

## DISMISS_JS — Popup/Overlay Remover (5 Phases)

Injected as a JavaScript function, returns count of removed elements.

| Phase | What it removes |
|-------|----------------|
| 0 | Cookie consent banners (accept/deny/dismiss phrases on buttons) |
| 1 | Modal close buttons (`[aria-label="Close"]`, `.close`, chat widget close buttons, proactive invite dismissals) |
| 2 | Fixed/absolute/sticky/z-index>100 overlays matching ~60 CSS selectors: generic overlays, cookie/consent badges, chat widgets (Gubagoo, ActivEngage, Conversica, Podium, Fuze, Tawk, LiveChat, Drift, Intercom), dealer CRM modals (DealerSocket, VinSolutions, DealerInspire, ePrice, lead-capture, appraisal) |
| 3 | Full-screen overlays by size: `position:fixed/absolute`, width >50% viewport, height >50% viewport, `z-index>10` |
| 4 | Sticky bottom bars that cover Submit buttons: `position:fixed/sticky`, bottom 30% of screen, height <30% viewport, `z-index>10`, does NOT contain a form/input (so chat forms are preserved) |

Also resets `document.body.style.overflow = 'auto'` to un-freeze scroll.

---

## Task Prompt Structure (`build_task`)

The prompt is constructed fresh per dealership run. Key sections in order:

1. **CONTACT INFO** — name, email, phone, zip, message (injected from config.py)
2. **FIELD MAPPING** — exact rules for every field type:
   - Name split (First/Last vs single)
   - Email in Email field only
   - Phone format `(347) 633-2712`
   - Message = LARGEST textarea (covers: Message, Comments, Question, Notes, Additional Information)
   - Department/Subject/Enquiry dropdown → "Sales" → "General Inquiry" → first non-placeholder
   - Vehicle of Interest → skip if optional; pick any if required
   - **Preferred Method of Contact** (BMW/luxury requirement) → always "Email"
   - Consent checkboxes → always check
   - Honeypot fields (hidden/invisible) → never fill
3. **HOW TO USE DROPDOWNS** — never use `input` on `<select>`. Use `select_dropdown` by index; fall back to `select_by_label` if stale.
4. **FINDING THE CONTACT PAGE** — branches:
   - Probe found URL → "You're already on the contact page"
   - Probe missed → "Try /contact-us, then /contact, then nav bar"
5. **NAVIGATION RULES** — never click privacy/legal/terms links
6. **IFRAMES** — CRM-embedded forms (DealerSocket, VinSolutions) look like bordered sections; interact normally
7. **MULTI-STEP/WIZARD FORMS** — fill → Next → fill → Next → final Submit; only call `done()` after last Submit
8. **CAPTCHA** — `done(captcha blocked)` immediately before Submit; `done(captcha blocked - may have submitted)` after
9. **POPUPS & OVERLAYS** — dismiss_popups, then Escape, then scroll past
10. **BLANK/BROKEN PAGES** — scroll, wait, dismiss_popups, then navigate back to root
11. **FALLBACKS** — (1) test drive/service form if no general form; (2) scrape_emails if no form at all; (3) captcha blocked
12. **STEPS** — 8 explicit numbered steps ending with:
    - Step 7: Pre-submit scroll validation (check all `*` fields, fix dropdowns showing "Please choose...")
    - Step 8: Submit line (see below)
13. **AFTER A FAILED SUBMISSION** — don't navigate; call `get_form_errors`; fix flagged fields; retry up to 3×; if error mentions "link/URL/spam", retype message without URL
14. **ABSOLUTE RULES** — enforced at highest priority:
    - `check_submission_state()` is the FIRST action after every Submit click
    - Wait ONE time only
    - Then `done()` — no exceptions
    - NEVER click Submit more than once
    - NEVER navigate/reload/dismiss after Submit
    - `form_not_visible` or uncertain → `done('submitted - no confirmation page', True)`

### Submit Line Outcomes (what agent passes to `done()`)

| Page state | `done()` call |
|-----------|--------------|
| Blank/empty page, form gone, `form_not_visible` | `done('submitted - no confirmation page', success=True)` |
| Red fields, "required" text, `error visible` | `done('form error: [exact text]', success=False)` |
| Thank-you / confirmation message, `confirmation visible` | `done('submitted successfully', success=True)` |

---

## Result Classification (`contact_dealership`)

After `agent.run()`, `result.final_result()` is checked against keyword lists:

**SUCCESS keywords** (→ logged as `submitted`):
- Explicit: `submitted successfully`, `test ok`, `no confirmation page`
- Confirmation phrases: `thank you`, `message sent`, `received your`, `has been submitted`, etc.
- CRM-specific: `we'll be in touch`, `a team member`, `our team will`, `one of our`, etc.
- Appointment fallbacks: `appointment request`, `test drive request`

**FORM_ERROR keywords** (→ logged as `failed`):
- `form error:`, `email rejected`, `invalid email`, `validation error`, `required field`

**SKIPPED keywords** (→ logged as `skipped`):
- `no contact form`, `captcha blocked`, `no form found`, `blank page`

**No email fallback** (→ logged as `skipped`):
- Checked first: if result contains `no form` + email regex match → saved to `leads.json`

**Fallthrough** (→ logged as `submitted`):
- Any outcome not matching above is treated as submitted to prevent double-sending on retry
- Full history is dumped to log at WARNING level for review

---

## results.json

```json
{ "submitted": [...], "failed": [...], "skipped": [...], "tested": [...] }
```

- Written after every single dealership — safe to stop and resume
- `tested` list contains test-mode runs (no real submission) — NOT skipped on next production run
- `submitted + failed + skipped` are skipped on next run

---

## LLM Config — `_ChatOllamaNoThink`

```python
@dataclass
class _ChatOllamaNoThink(_BUChatOllama):
    async def ainvoke(self, messages, output_format=None, **kwargs):
        ollama_msgs = OllamaMessageSerializer.serialize_messages(messages)
        resp = await client.chat(
            model=self.model,
            messages=ollama_msgs,
            options=self.ollama_options,   # num_ctx=32768, num_predict=2048
            think=False,                   # MANDATORY — prevents empty responses
            format=output_format.model_json_schema() if output_format else None,
        )
        content = self._clean_response(resp.message.content or "")
        ...
```

`_clean_response()` pipeline (defense-in-depth only — Ollama's `format=schema` does the real work):
1. Strip `<tool_call>...</tool_call>` XML wrapper
2. Strip stray opening/closing `<tool_call>` tags
3. `json.JSONDecoder().raw_decode()` — parse first valid JSON, discard trailing garbage
4. If result is `{"name": "AgentOutput", "arguments": {...}}` envelope, extract `arguments`

---

## browser-use 0.12.2 vs 0.1.40 — Why This Matters

| Feature | 0.1.40 (deprecated) | 0.12.2 (active) |
|---------|---------------------|----------------|
| LLM call | Sync `invoke()` | Async `ainvoke()` |
| JSON parsing | `json.loads()` — zero recovery | `format=schema` at Ollama API level — never fails |
| qwen3 `<tool_call>` XML | Crashes without manual `_clean_response` | Prevented at API level; `_clean_response` is defense only |
| Retry logic | Naive — single retry | Smart: empty-action detection, message compaction, fallback LLM |
| Message history | Unlimited, grows forever | Auto-compaction for long conversations |
| Loop detection | None | Nudges model if same action repeated 5× |
| Thinking param | Must hack `_chat_params` | Native `use_thinking=False` flag |

---

## Common Failures and Diagnostics

### Agent can't select a dropdown
**Symptom**: Agent uses `input` action on a `<select>` — nothing gets selected.  
**Fix built in**: `HOW TO USE DROPDOWNS` section in prompt. `select_by_label` custom action as fallback.  
**If still failing**: Check log for the label text the agent is searching for — add it to FIELD MAPPING if it's a new site-specific label.

### Form submits but no confirmation shown → reload loop
**Symptom**: Blank page after submit, agent calls `navigate` or `reload`, re-fills form, submits again.  
**Fix built in**: ABSOLUTE RULES section. "Blank/empty = submitted." `check_submission_state` catches `form_not_visible`.

### Double-click on Submit button
**Symptom**: Agent clicks Submit at step N, then clicks Submit again at step N+1 because DOM shifted.  
**Fix built in**: `check_submission_state()` is mandatory first action after Submit. "Click only ONE time." NEVER re-click after Submit.

### reCAPTCHA blocking
**Symptom**: Agent tries to click captcha or loops waiting for it to disappear.  
**Fix built in**: CAPTCHA section. Agent calls `done(captcha blocked)` immediately. Never interacts with captcha elements.

### Email rejected by form (+ in address)
**Symptom**: Log shows form error after submit, agent loops retrying with same email.  
**Diagnosis**: Some dealership CRMs incorrectly reject email addresses with `+` addressing.  
**Fix**: Change `email` in config.py to a plain address without `+`. The current email `epaulson@acesinc1.com` does not have this issue.

### "Could not parse response" / blank LLM outputs
**Root cause**: qwen3 with thinking enabled burns all `num_predict` tokens on `<think>` monologue, returns `content=''`.  
**Fix built in**: `think=False` in `_ChatOllamaNoThink.ainvoke()`.

### Agent stuck navigating instead of filling form
**Symptom**: Agent keeps clicking nav links instead of filling visible form.  
**Fix built in**: NAVIGATION RULES section. "STOP navigating the moment you find a page with a form."

### Probe returns wrong URL (redirected to homepage)
**Symptom**: Agent told it's on contact page but page has no form.  
**Fix built in**: `probe_contact_url` checks if final redirect URL has empty path (= root/homepage) and rejects it.

### Sticky bottom bar covers Submit button
**Symptom**: Agent clicks Submit coordinates but hits a "Chat Now" sticky bar instead.  
**Fix built in**: DISMISS_JS Phase 4 removes sticky bottom bars that don't contain a form.

---

## Ollama Setup (Windows)

1. Install from ollama.com — runs as a background Windows service
2. Verify: visit `http://localhost:11434` → should show "Ollama is running"
3. Model: `qwen3:14b` (already pulled for this project)
4. Settings injected in code: `num_ctx=32768`, `num_predict=2048`

---

## Python Environments

```
venv/       → browser-use 0.1.40 — DEPRECATED (main.py, test.py)
venv_new/   → browser-use 0.12.2 — ACTIVE (main_update.py)
```

Always activate `venv_new` before running `main_update.py`.

---

## Logs

- `logs/run_v5_YYYYMMDD_HHMMSS.log` — full DEBUG output per run
- Console shows INFO level only
- Feed log files to Claude or another AI for debugging failed runs

---

## Campaign Context

**Agent**: Eric Paulson, Marketing Dept., ACES Inc. (Brooklyn-based MLB sports agency, 35+ years, 50+ active MLB players)  
**Pitch**: Victor Caratini (Minnesota Twins catcher, 9-year MLB vet, bilingual Puerto Rican, signed 2-year contract Jan 2025) wants a vehicle for the season  
**Exchange offered**: Autographed memorabilia, personal appearances, social media features, game tickets, meet & greet events, signage, co-branded content — flexible  
**Contact**: epaulson@acesinc1.com | (347) 633-2712  
**Website**: https://acesincbaseball.com

