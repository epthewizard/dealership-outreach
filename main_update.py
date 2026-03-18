"""
main_update.py — Dealership Contact Agent v5 (browser-use 0.12.x)

REQUIRES browser-use 0.12.x — NOT compatible with 0.1.40.

To set up a fresh venv for this version:
    python -m venv venv_new
    venv_new\\Scripts\\activate
    pip install "browser-use>=0.12.0" playwright
    playwright install chromium

Key differences from main.py (0.1.40):
  - browser_use.llm.ollama.chat.ChatOllama  (no langchain-ollama)
  - BrowserProfile + BrowserSession  (replaces BrowserConfig + Browser)
  - tool_calling_method / max_input_tokens removed (not available)
  - use_thinking=False  (reduces agent output schema, fewer tokens)
  - think=False injected via ChatOllama subclass (same reason as before)
  - No JS fill_form_fields — the AI fills fields natively via click/type
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from browser_use import Agent, BrowserProfile, BrowserSession, Controller
from browser_use.agent.views import ActionResult

try:
    from browser_use.llm.ollama.chat import ChatOllama as _BUChatOllama
    from browser_use.llm.ollama.serializer import OllamaMessageSerializer
    from browser_use.llm.views import ChatInvokeCompletion
    from browser_use.llm.exceptions import ModelProviderError
except ImportError as e:
    raise ImportError(
        "browser-use 0.12.x not installed. Run: pip install 'browser-use>=0.12.0'"
    ) from e

from config import (
    USER_INFO,
    DELAY_BETWEEN_SUBMISSIONS,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    USE_VISION,
)
from discover import get_dealership_urls


# ── Logging ──────────────────────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_log_file = LOG_DIR / f"run_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
))

logging.getLogger().addHandler(_file_handler)
logging.getLogger().setLevel(logging.DEBUG)

logging.getLogger("browser_use").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class _NoBBoxFilter(logging.Filter):
    """Suppress the very noisy 'BBox filtering excluded N nodes' debug spam."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "BBox filtering" not in record.getMessage()

logging.getLogger().addFilter(_NoBBoxFilter())

_LOG = logging.getLogger("agent")


# ── Files ──────────────────────────────────────────────────────────────────────

LOG_FILE   = Path("results.json")
LEADS_FILE = Path("leads.json")


def load_log() -> dict:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            data = json.load(f)
        data.setdefault("tested", [])
        return data
    return {"submitted": [], "failed": [], "skipped": [], "tested": []}


def save_log(log: dict):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def save_leads(url: str, emails: list[str]):
    data = []
    if LEADS_FILE.exists():
        with open(LEADS_FILE) as f:
            data = json.load(f)
    data.append({
        "url": url,
        "emails": emails,
        "scraped_at": datetime.now().isoformat(),
    })
    with open(LEADS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def strip_utm(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url.split("?")[0].rstrip("/")


# ── LLM — ChatOllama subclass that injects think=False ────────────────────────
#
# browser-use 0.12.x ships its own ChatOllama (browser_use.llm.ollama.chat).
# It calls ollama.AsyncClient.chat() without the think= parameter, so qwen3
# would still burn all tokens on its internal monologue and return content=''.
# We override ainvoke() to add think=False on every call.

@dataclass
class _ChatOllamaNoThink(_BUChatOllama):
    """ChatOllama that disables thinking and cleans qwen3's tool_call wrappers.

    Two problems fixed here:
    1. think=False — passed to ollama.AsyncClient.chat() so qwen3 skips its
       internal thinking and outputs the JSON action immediately.
    2. _clean_response — after step 1 the conversation history causes qwen3 to
       slip into XML tool_call format:
           <tool_call>{"name":"AgentOutput","arguments":{...}}</tool_call>
       or leave a stray </tool_call> suffix. We strip these before pydantic
       validation so the JSON can be parsed correctly.
    """

    @staticmethod
    def _clean_response(content: str) -> str:
        text = content.strip()
        if not text:
            return text
        # Strip <tool_call>...</tool_call> wrapper
        m = re.search(r'<tool_call>\s*([\s\S]*?)\s*</tool_call>', text)
        if m:
            text = m.group(1).strip()
        else:
            # Strip stray opening <tool_call> or closing </tool_call>
            text = re.sub(r'^\s*<tool_call>\s*', '', text)
            text = re.sub(r'\s*</tool_call>\s*$', '', text).strip()
        # Use raw_decode to parse the first valid JSON object, ignoring
        # trailing garbage like extra }} that qwen3 sometimes emits.
        if text.startswith('{'):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text)
                if (isinstance(obj, dict)
                        and obj.get('name') == 'AgentOutput'
                        and 'arguments' in obj):
                    text = json.dumps(obj['arguments'])
                else:
                    text = json.dumps(obj)
            except (json.JSONDecodeError, ValueError):
                pass
        return text

    async def ainvoke(
        self,
        messages: list,
        output_format: type | None = None,
        **kwargs: Any,
    ) -> "ChatInvokeCompletion":
        ollama_msgs = OllamaMessageSerializer.serialize_messages(messages)
        try:
            client = self.get_client()
            resp = await client.chat(
                model=self.model,
                messages=ollama_msgs,
                options=self.ollama_options,
                think=False,
                format=output_format.model_json_schema() if output_format else None,
            )
            # Clean response before parsing to remove qwen3's tool_call wrappers
            content = self._clean_response(resp.message.content or "")
            if output_format is None:
                return ChatInvokeCompletion(completion=content, usage=None)
            else:
                completion = output_format.model_validate_json(content)
                return ChatInvokeCompletion(completion=completion, usage=None)
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e


def get_llm() -> _ChatOllamaNoThink:
    try:
        from ollama import Options
        opts = Options(num_ctx=32768, num_predict=2048)
    except ImportError:
        opts = {"num_ctx": 32768, "num_predict": 2048}

    return _ChatOllamaNoThink(
        model=OLLAMA_MODEL,
        host=OLLAMA_BASE_URL,
        ollama_options=opts,
    )


# ── JS popup remover (injected on page load) ──────────────────────────────────

DISMISS_JS = """() => {
    let count = 0;

    // Phase 0: Accept / deny cookie consent banners so they stop blocking
    const cookieAcceptPhrases = [
        'allow all cookies', 'accept all cookies', 'accept all', 'accept cookies',
        'allow all', 'agree', 'i agree', 'got it', 'ok', 'okay',
        'deny marketing cookies', 'deny', 'reject all', 'decline',
        'close and continue', 'continue to site', 'continue without accepting',
        'dismiss', 'no thanks', 'not now', 'skip',
    ];
    document.querySelectorAll('button, a[role="button"], [role="button"]').forEach(btn => {
        const text = (btn.textContent || '').trim().toLowerCase();
        if (cookieAcceptPhrases.some(phrase => text === phrase || text.includes(phrase))) {
            try { btn.click(); count++; } catch(e) {}
        }
    });

    // Phase 1: Click close/dismiss buttons on modals and popups
    const closeBtnSelectors = [
        '[aria-label="Close"]', '[aria-label="close"]',
        '[aria-label="Dismiss"]', '[aria-label="dismiss"]',
        '[aria-label="No thanks"]', '[aria-label="Skip"]',
        'button.close', 'button.modal-close', 'button.popup-close',
        '.modal .close', '.modal [class*="close"]',
        '.popup [class*="close"]', '[class*="modal"] [class*="close"]',
        '[role="dialog"] button', '[aria-modal="true"] button',
        'button[class*="dismiss"]', 'button[class*="decline"]',
        'button[class*="no-thanks"]', 'button[class*="nothanks"]',
        'button[class*="skip"]',
        '.needsclick.kl-private-reset-css-Xuajs1',
        '[class*="chat"] [class*="close"]',
        '[id*="chat"] [class*="close"]',
        '[class*="proactive"] button',
        '[class*="invite"] button',
        '[class*="widget"] [aria-label="Close"]',
    ];
    for (const sel of closeBtnSelectors) {
        document.querySelectorAll(sel).forEach(btn => {
            try {
                const text = (btn.textContent || '').trim().toLowerCase();
                const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                if (text.length < 20 || text.includes('close') || text.includes('no') ||
                    text === 'x' || text === '\u00d7' || text === '\u2715' ||
                    ariaLabel.includes('close') || ariaLabel.includes('dismiss')) {
                    btn.click();
                    count++;
                }
            } catch(e) {}
        });
    }

    // Phase 2: Remove fixed/absolute overlay, popup, and chat widget elements
    const selectors = [
        // Generic overlays
        '[class*="overlay"]', '[class*="modal"]', '[class*="popup"]',
        '[class*="backdrop"]', '[class*="mask"]', '[class*="lightbox"]',
        '[class*="takeover"]',
        '[aria-modal="true"]', '[role="dialog"]',
        // Cookie / consent
        '[class*="cookie"]', '[id*="cookie"]', '[class*="consent"]',
        '[class*="privacy"]', '[class*="gdpr"]', '[id*="gdpr"]',
        '[class*="banner"]',
        // Chat widgets (all major platforms)
        '[class*="chat-widget"]', '[class*="chatbot"]',
        '[id*="chat"]', '[class*="chat-launcher"]',
        '[class*="drift"]', '[id*="drift"]',
        '[class*="intercom"]', '[id*="intercom"]',
        '[class*="gubagoo"]', '[id*="gubagoo"]',
        '[class*="activengage"]', '[id*="activengage"]',
        '[class*="conversica"]', '[id*="conversica"]',
        '[class*="smartwindow"]', '[class*="proactive"]',
        '[class*="fuze"]', '[id*="fuze"]',
        '[class*="podium"]', '[id*="podium"]',
        '[class*="livechat"]', '[id*="livechat"]',
        '[class*="tawk"]', '[id*="tawk"]',
        'iframe[src*="chat"]', 'iframe[src*="widget"]',
        'iframe[src*="proactive"]', 'iframe[src*="livechat"]',
        // Dealer CRM / lead capture modals
        '[class*="eprice"]', '[class*="e-price"]', '[class*="epricing"]',
        '[class*="promo-modal"]', '[class*="specials-modal"]',
        '[class*="specials-popup"]', '[class*="hero-cta-modal"]',
        '[class*="email-capture"]', '[class*="emailcapture"]',
        '[class*="lead-form-modal"]', '[class*="lead-capture"]',
        '[id*="LeadDelivery"]', '[class*="LeadDelivery"]',
        '[class*="appraisal"]',
        // Dealer platform-specific
        '[class*="dealer-inspire"]', '[class*="di-modal"]',
        '[class*="dealersocket"]', '[id*="dealersocket"]',
        '[class*="vinsolutions"]',
    ];
    for (const sel of selectors) {
        document.querySelectorAll(sel).forEach(el => {
            const style = window.getComputedStyle(el);
            if (
                style.position === 'fixed' ||
                style.position === 'absolute' ||
                style.position === 'sticky' ||
                parseInt(style.zIndex) > 100
            ) {
                el.remove();
                count++;
            }
        });
    }

    // Phase 3: Kill any remaining full-screen overlays by size
    document.querySelectorAll('div, section, aside').forEach(el => {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        if ((style.position === 'fixed' || style.position === 'absolute') &&
            rect.width > window.innerWidth * 0.5 &&
            rect.height > window.innerHeight * 0.5 &&
            parseInt(style.zIndex) > 10) {
            el.remove();
            count++;
        }
    });

    // Phase 4: Remove sticky bottom bars that cover form buttons
    // (e.g. "Chat Now", "Text Us", sticky CTA strips at screen bottom)
    // Skip any element that itself contains a form or input (could be a chat form).
    document.querySelectorAll('div, nav, section, aside, footer').forEach(el => {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        if (
            (style.position === 'fixed' || style.position === 'sticky') &&
            rect.top > window.innerHeight * 0.7 &&      // bottom 30% of screen
            rect.height < window.innerHeight * 0.3 &&   // not a full overlay
            parseInt(style.zIndex) > 10 &&
            !el.querySelector('form, input[type="text"], textarea') // not a form
        ) {
            el.remove();
            count++;
        }
    });

    document.body.style.overflow = 'auto';
    document.documentElement.style.overflow = 'auto';
    return count;
}"""


# ── Custom Controller ──────────────────────────────────────────────────────────

def make_controller() -> Controller:
    controller = Controller()

    @controller.action(
        "Dismiss popups, overlays, chat widgets, and cookie banners on the current page. "
        "Call this at the start. Returns how many elements were removed.",
    )
    async def dismiss_popups(browser_session: BrowserSession) -> ActionResult:
        page = await browser_session.get_current_page()
        count = await page.evaluate(DISMISS_JS)
        return ActionResult(extracted_content=f"Dismissed {count} popups/overlays")

    @controller.action(
        "Scrape all email addresses visible on the current page. "
        "Use this only when no contact form can be found anywhere on the site.",
    )
    async def scrape_emails(browser_session: BrowserSession) -> ActionResult:
        page = await browser_session.get_current_page()
        emails = await page.evaluate("""() => {
            const found = new Set();
            document.querySelectorAll('a[href^="mailto:"]').forEach(a => {
                const email = a.href.replace('mailto:', '').split('?')[0].trim();
                if (email) found.add(email.toLowerCase());
            });
            const text = document.body.innerText || '';
            const matches = text.match(
                /[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}/g
            );
            if (matches) matches.forEach(m => found.add(m.toLowerCase()));
            return [...found];
        }""")
        if emails:
            return ActionResult(extracted_content=f"Found emails: {', '.join(emails)}")
        return ActionResult(extracted_content="No email addresses found on page")

    @controller.action(
        "Select an option in a <select> dropdown by searching for it near a label. "
        "Use this when select_dropdown by index keeps failing or indices changed after a page update. "
        "label_text: visible label text near the dropdown (e.g. 'Preferred Method of Contact'). "
        "option_text: the option to select (e.g. 'Email'). Falls back to first non-blank option.",
    )
    async def select_by_label(
        label_text: str, option_text: str, browser_session: BrowserSession
    ) -> ActionResult:
        page = await browser_session.get_current_page()
        result = await page.evaluate(
            """([labelText, optionText]) => {
                const selects = Array.from(document.querySelectorAll('select'));
                for (const sel of selects) {
                    // Check <label for="id"> association
                    let match = false;
                    if (sel.id) {
                        const lbl = document.querySelector('label[for="' + sel.id + '"]');
                        if (lbl && lbl.textContent.toLowerCase().includes(labelText.toLowerCase()))
                            match = true;
                    }
                    // Check aria-labelledby association
                    if (!match && sel.getAttribute('aria-labelledby')) {
                        const lblId = sel.getAttribute('aria-labelledby');
                        const lbl = document.getElementById(lblId);
                        if (lbl && lbl.textContent.toLowerCase().includes(labelText.toLowerCase()))
                            match = true;
                    }
                    // Walk up DOM to find wrapping text containing the label
                    if (!match) {
                        let parent = sel.parentElement;
                        for (let i = 0; i < 5; i++) {
                            if (!parent) break;
                            if ((parent.textContent || '').toLowerCase().includes(labelText.toLowerCase())) {
                                match = true; break;
                            }
                            parent = parent.parentElement;
                        }
                    }
                    if (!match) continue;
                    // Find matching option
                    const opt = Array.from(sel.options).find(o =>
                        o.text.toLowerCase().includes(optionText.toLowerCase()) ||
                        o.value.toLowerCase().includes(optionText.toLowerCase())
                    );
                    const chosen = opt || Array.from(sel.options).find(o => o.value && o.value !== '');
                    if (chosen) {
                        sel.value = chosen.value;
                        sel.dispatchEvent(new Event('change', {bubbles: true}));
                        sel.dispatchEvent(new Event('input', {bubbles: true}));
                        return 'Selected "' + chosen.text + '" in dropdown near label "' + labelText + '"';
                    }
                    return 'Dropdown found near "' + labelText + '" but no matching option for "' + optionText + '"';
                }
                return 'No dropdown found near label "' + labelText + '"';
            }""",
            [label_text, option_text],
        )
        return ActionResult(extracted_content=str(result))

    @controller.action(
        "Scan the current page for form validation errors — red-highlighted fields, "
        "error messages, required-field warnings. Call this immediately after a failed "
        "form submission to get a concrete list of what needs to be fixed.",
    )
    async def get_form_errors(browser_session: BrowserSession) -> ActionResult:
        page = await browser_session.get_current_page()
        errors = await page.evaluate("""() => {
            const msgs = new Set();
            // Gravity Forms / generic CRM error classes
            const selectors = [
                '.gfield_error .gfield_label',
                '.gfield_error .validation_message',
                '.validation_error',
                '.field_error', '.error-message', '.alert-danger',
                '[class*="error"] label', '[class*="invalid"] label',
                '[aria-invalid="true"]',
            ];
            for (const sel of selectors) {
                document.querySelectorAll(sel).forEach(el => {
                    const t = (el.textContent || '').trim().replace(/\\s+/g, ' ');
                    if (t && t.length < 200) msgs.add(t);
                });
            }
            // Fields with red borders or aria-invalid
            document.querySelectorAll('input, select, textarea').forEach(el => {
                const style = window.getComputedStyle(el);
                const isRed = style.borderColor && (
                    style.borderColor.includes('rgb(255') ||
                    style.borderColor.includes('rgb(220') ||
                    style.outlineColor && style.outlineColor.includes('rgb(255')
                );
                const isInvalid = el.getAttribute('aria-invalid') === 'true';
                if (isRed || isInvalid) {
                    const lbl = el.labels && el.labels[0]
                        ? el.labels[0].textContent.trim()
                        : (el.placeholder || el.name || el.id || 'unknown field');
                    msgs.add('Field needs value: ' + lbl);
                }
            });
            return [...msgs].slice(0, 20);
        }""")
        if errors:
            msg = "Form errors: " + " | ".join(errors)
        else:
            msg = "No form errors detected (form may have submitted successfully)"
        return ActionResult(extracted_content=msg)

    @controller.action(
        "Check if the form was already submitted (prevents accidental double-submission). "
        "Returns 'submitted' if a confirmation/error message is visible, 'form_visible' if the form "
        "is still present and editable, or 'unknown' if page state is unclear.",
    )
    async def check_submission_state(browser_session: BrowserSession) -> ActionResult:
        page = await browser_session.get_current_page()
        state = await page.evaluate("""() => {
            // Fast body-text scan — avoids iterating all DOM elements which can be slow
            const bodyText = (document.body
                ? (document.body.innerText || document.body.textContent || '')
                : '').toLowerCase();

            const confirmPhrases = [
                'thank you', 'thank-you', 'successfully submitted', 'message sent',
                'received your', 'we received', 'submitted successfully',
                'thank you for', 'will be in touch', 'appreciate your',
                'get back to you', "we'll contact", 'someone will contact',
                'we will contact',
            ];
            if (confirmPhrases.some(p => bodyText.includes(p)))
                return 'submitted - confirmation visible';

            const errorPhrases = [
                'problem with your submission', 'errors have been highlighted',
                'validation error', 'form error', 'please fix the following',
                'highlighted in red', 'fields are required',
            ];
            if (errorPhrases.some(p => bodyText.includes(p)))
                return 'submitted - error visible';

            // Check if form is still present and interactive
            const form = document.querySelector('form');
            const submit = form
                ? form.querySelector('[type="submit"], button[type="submit"]')
                : null;
            if (form && submit && form.offsetParent !== null) return 'form_visible';
            if (!form) return 'form_not_visible - likely submitted or navigated away';
            return 'unknown';
        }""")
        return ActionResult(extracted_content=str(state))

    return controller


# ── Task builder ───────────────────────────────────────────────────────────────

# Contact page paths — tried by the HTTP probe AND as browser fallbacks.
_CONTACT_PATHS = [
    "/contact-us",
    "/contact",
    "/contactus",
    "/about/contact-us",
    "/about-us/contact",
    "/connect",
    "/reach-us",
    "/get-in-touch",
    "/inquiry",
    "/customer-care",
    "/dealership/contact",
]

_SOFT_404_PHRASES = [
    "page not found", "404 error", "error 404",
    "page doesn't exist", "page no longer exists",
    "couldn't find this page", "can't find this page",
    "this page does not exist",
]


async def probe_contact_url(root: str) -> str | None:
    """Probe all contact URL paths in parallel via lightweight HTTP requests.

    Returns the first valid contact URL (in priority order) or None.
    Runs in ~2-4 seconds — avoids spending 5-10 AI steps just navigating.
    """
    try:
        import httpx
    except ImportError:
        _LOG.warning("PROBE  httpx not available — skipping probe")
        return None

    candidates = [root.rstrip("/") + p for p in _CONTACT_PATHS]
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    root_base = root.rstrip("/")

    async def _check(client: "httpx.AsyncClient", idx: int, url: str) -> tuple[int, str | None]:
        try:
            r = await client.head(url)
            if r.status_code == 200:
                # Reject if the server redirected us back to the root/homepage
                final_path = urlparse(str(r.url)).path.strip("/")
                if not final_path:
                    return (idx, None)
                r2 = await client.get(url)
                body = r2.text[:4000].lower()
                if not any(p in body for p in _SOFT_404_PHRASES):
                    return (idx, url)
            elif r.status_code in (405, 501):  # server rejects HEAD — try GET
                r2 = await client.get(url)
                if r2.status_code == 200:
                    final_path = urlparse(str(r2.url)).path.strip("/")
                    if not final_path:
                        return (idx, None)
                    body = r2.text[:4000].lower()
                    if not any(p in body for p in _SOFT_404_PHRASES):
                        return (idx, url)
        except Exception:
            pass
        return (idx, None)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(7.0),
            headers={"User-Agent": ua},
        ) as client:
            results = await asyncio.wait_for(
                asyncio.gather(*[_check(client, i, u) for i, u in enumerate(candidates)]),
                timeout=15.0,
            )
    except asyncio.TimeoutError:
        _LOG.warning("PROBE  timed out after 15 s — AI will navigate from root")
        return None
    except Exception as exc:
        _LOG.warning("PROBE  error: %s — AI will navigate from root", exc)
        return None

    valid = sorted([(i, u) for i, u in results if u is not None], key=lambda x: x[0])
    if valid:
        _LOG.info("PROBE  → %s", valid[0][1])
        return valid[0][1]

    _LOG.info("PROBE  no contact URL found — AI navigates from root")
    return None


def build_task(url: str, contact_url: str | None = None, test_mode: bool = False) -> str:
    email   = USER_INFO["email"]
    message = USER_INFO["message"]
    zip_    = USER_INFO["zip"]

    root = url.rstrip("/")

    submit_line = (
        "Do NOT click Submit — test run only. "
        "Call done(text='TEST OK', success=True) once all fields are filled."
        if test_mode else
        "Click the Submit / Send / Send Message button (click only ONE time). "
        "Immediately after clicking Submit, call check_submission_state() to verify the submit worked. "
        "Then wait ONE time (5 seconds max), then look at the page and call done() with one of:\n"
        "  a) Page is blank/empty OR minimal content with no form visible, OR check_submission_state said form_not_visible: "
        "done(text='submitted - no confirmation page', success=True).\n"
        "  b) Red or highlighted fields visible, or text like 'please select', 'required', 'invalid', OR check_submission_state said 'error visible': "
        "done(text='form error: [exact error text]', success=False).\n"
        "  c) Confirmation, thank-you, or success message, OR check_submission_state said 'confirmation visible': "
        "done(text='submitted successfully', success=True).\n"
        "CRITICAL: The very next action after clicking Submit must be check_submission_state(). "
        "Only then wait. Only then call done(). Do NOT navigate, reload, or re-click submit."
    )

    if contact_url:
        nav_section = (
            "\u2501\u2501\u2501 FINDING THE CONTACT PAGE \u2501\u2501\u2501\n"
            f"The contact page was pre-located at: {contact_url}\n"
            f"You are starting there. After dismiss_popups, fill and submit the form.\n"
            f"If this page has no visible form, click any 'Message Us', 'Contact', "
            f"or 'Send Us a Message' tab or button on the page.\n"
            f"If truly no form exists here, go to {root} and find a Contact link in the nav/footer."
        )
        step2 = "2. You are already on the contact page. Proceed directly to step 3."
    else:
        nav_section = (
            "\u2501\u2501\u2501 FINDING THE CONTACT PAGE \u2501\u2501\u2501\n"
            f"Navigate from {root} to find the contact/inquiry form:\n"
            f"- Try {root}/contact-us in the address bar first. If 404, try {root}/contact.\n"
            f"- If both are 404, look for a Contact link in the top nav bar, "
            f"dropdown menus (hover 'About', 'About Us', 'Dealership', 'More'), "
            f"footer, or hamburger/mobile menu.\n"
            f"- STOP navigating the moment you find a page with a contact/inquiry form.\n"
            f"- Do NOT navigate away from a page that already has a form."
        )
        step2 = (
            f"2. Try {root}/contact-us — if 404 try {root}/contact — "
            f"if 404 go to {root} and find the Contact link in the nav/footer."
        )

    return f"""You are submitting a contact/inquiry form at this car dealership: {url}

CONTACT INFO:
  First Name: Eric  |  Last Name: Paulson  |  Full Name: Eric Paulson
  Email: {email}
  Phone: (347) 633-2712
  Zip: {zip_}
  Message: {message}

━━━ FIELD MAPPING ━━━
- Name: split into First=Eric / Last=Paulson if separate; single field = "Eric Paulson".
- Email: field labeled Email or E-mail only. Never put email in the Message box.
- Phone: use format (347) 633-2712. Field labeled Phone, Cell, Mobile, or similar.
- Message/Comments/Question/Notes/Additional Information: always the LARGEST text box (textarea).
  Never use a small one-line input. If the textarea has a character counter or maxlength,
  the browser will auto-truncate at the limit — proceed normally, do not modify the message.
- Zip/Postal Code: use {zip_}.
- Department / Subject / Enquiry Type / Reason dropdown: select "Sales" if available, else "General Inquiry", else the FIRST non-placeholder option.
- Vehicle of Interest / Year / Make / Model: if REQUIRED, select any available option (e.g. current year, first make, "N/A", or "General Inquiry"). If optional, leave blank.
- How did you hear about us / Lead Source: select any option if required, leave blank if optional.
- Preferred Method of Contact / Preferred Contact Method / How Would You Like to Be Contacted: 
  this is a REQUIRED dropdown on many BMW and luxury brand forms. ALWAYS select "Email" if that 
  option exists. If "Email" is not an option, select the first non-placeholder option.
  Look for this dropdown BEFORE clicking Submit — do not skip it.
- Best time to contact / Best Time to Reach You: select any option if required.
- Consent/Authorization checkboxes: ALWAYS check any box labeled "I agree", "I consent", "I authorize", "I accept terms", "I agree to be contacted" — these are mandatory.
- Hidden/honeypot fields: do NOT fill fields that are not visible on screen (display:none or visibility:hidden). These are bot traps.

━━━ HOW TO USE DROPDOWNS (<select> elements) ━━━
- NEVER use the `input` action on a dropdown — it does not work and will not select anything.
- To select a dropdown option, use the `select_dropdown` action with the element index.
- If `select_dropdown` fails because the index is stale or unavailable, use `select_by_label` instead:
  Example: select_by_label(label_text='Preferred Method of Contact', option_text='Email')
  This finds the dropdown by its label text and selects the matching option — even if indices changed.
- To see what options are available in a dropdown: use `dropdown_options(index=...)` first.

{nav_section}

━━━ NAVIGATION RULES ━━━
- NEVER click links containing: privacy, cookie, legal, terms, accessibility, opt-out, sitemap.
- ONLY click links that plausibly lead to a contact/inquiry/message form.
- If the contact page has tabs or sections (Message / Call / Email / Sales / Service), click "Message Us", "Send a Message", "Email Us", or "General Inquiry" — avoid "Schedule Service" tabs unless that is the only option.
- If there are multiple separate forms (Sales Inquiry, Service Inquiry, Parts), use the Sales or General Inquiry form.

━━━ IFRAMES ━━━
Many dealership forms are loaded inside an <iframe> (embedded sub-page from a CRM like DealerSocket or VinSolutions).
- If the form appears inside a bordered/embedded section, it is likely in an iframe.
- Interact with it normally — click its fields and type into them exactly as you would a normal form.
- If clicking a visible field has no effect, try scrolling to it and clicking again.

━━━ MULTI-STEP / WIZARD FORMS ━━━
Some forms span multiple pages (e.g. "Step 1 of 3") or show a "Next" / "Continue" button instead of Submit.
- Fill all fields on the current step fully, then click "Next" or "Continue".
- Keep filling + clicking Next on each subsequent step until you reach the final Submit button.
- Only call done() AFTER clicking the final Submit on the last step.

━━━ CAPTCHA ━━━
- If you see a reCAPTCHA checkbox ("I'm not a robot"), hCaptcha, or image-selection challenge
  BEFORE clicking Submit: call done(text='captcha blocked', success=False) immediately.
- If you see a captcha AFTER clicking Submit: call done(text='captcha blocked - may have submitted', success=False) immediately.
- NEVER attempt to click, solve, or interact with any captcha element.
- Invisible reCAPTCHA v3 (no visible checkbox) handles itself — just submit normally.

━━━ POPUPS & OVERLAYS ━━━
- Call dismiss_popups at the start and again any time a new overlay appears.
- If after dismiss_popups an overlay still blocks the form, try pressing Escape.
- If still blocked, try scrolling past the overlay to reach the form underneath.

━━━ BLANK OR BROKEN PAGES ━━━
- If the contact page loads blank, scroll to the bottom and wait 3 seconds.
- If still blank, call dismiss_popups (sometimes a hidden overlay is blocking rendering).
- If still blank after that, go back to root and navigate manually to Contact.

━━━ FALLBACKS (in order) ━━━
1. No general inquiry form but there IS a "Schedule a Test Drive" or "Schedule Service Appointment" form:
   Use it. Fill the same contact info and put the full message in the Comments/Notes field.
2. No form of any kind found: call scrape_emails, then done(text='no form - emails: [list]', success=True).
3. Captcha blocking: done(text='captcha blocked', success=False).

━━━ STEPS ━━━
1. Call dismiss_popups to clear overlays and cookie banners.
{step2}
3. Call dismiss_popups again if a new overlay appeared on the contact page.
4. If blank, scroll/wait/retry as described above.
5. Click the correct tab/section if the page has Sales/Service/Message tabs.
6. Fill every field using FIELD MAPPING rules. Handle ALL dropdowns with select_dropdown or select_by_label
   (never use input on a dropdown). Check all consent checkboxes.
7. BEFORE clicking Submit: scroll the entire form once from top to bottom to verify every required
   field (*) has a value. Pay special attention to any dropdown still showing 'Please choose...' or
   'Please select' — fix it now using select_by_label before proceeding.
8. {submit_line}

━━━ AFTER A FAILED SUBMISSION ━━━
If the page shows 'There was a problem with your submission' or highlights fields in red:
- DO NOT navigate away, reload, or go back. Stay on the page — the fields are still filled.
- Call get_form_errors to get a precise list of what is wrong.
- Fix ONLY the flagged fields (scroll to them, use select_by_label for dropdowns, retype for text).
- If the error mentions 'link', 'URL', or 'spam': retype the message field without the URL
  (https://acesincbaseball.com) and retry — some dealership spam filters reject messages with links.
- Click Submit again.
- Repeat this fix-and-retry cycle up to 3 times total.
- If still failing after 3 attempts, call done(text='form error: [error list]', success=False).

ABSOLUTE RULES AFTER CLICKING SUBMIT (any submit attempt):
- IMMEDIATELY after clicking Submit: call check_submission_state(). This is mandatory.
- Then wait ONE time (5 seconds).
- Then call done() based on what check_submission_state and the page now show.
- NEVER click Submit more than once per attempt.
- NEVER navigate to any URL after clicking Submit.
- NEVER reload the page after clicking Submit.
- NEVER call dismiss_popups after clicking Submit.
- If check_submission_state says "form_not_visible": done(text='submitted - no confirmation page', success=True).
- If you cannot determine what happened: done(text='submitted - no confirmation page', success=True).
"""


# ── Single dealership runner ───────────────────────────────────────────────────

async def contact_dealership(url: str, llm, test_mode: bool = False) -> str:
    url = strip_utm(url)

    _LOG.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _LOG.info("START  %s", url)

    # Probe contact URL via HTTP before launching the AI agent.
    # Runs all candidate paths in parallel (~2-4 s) and returns the first
    # working one. If found, the agent starts directly on the contact page
    # instead of spending 5-10 steps navigating there.
    print(f"   Probing contact URL...")
    contact_url = await probe_contact_url(url)
    start_url = contact_url or url
    if contact_url:
        print(f"   Found: {contact_url}")
    else:
        print(f"   Not found via probe — AI navigates from root")

    session = BrowserSession(
        browser_profile=BrowserProfile(headless=False),
    )

    agent = Agent(
        task=build_task(url, contact_url=contact_url, test_mode=test_mode),
        llm=llm,
        browser=session,
        controller=make_controller(),
        initial_actions=[{"navigate": {"url": start_url, "new_tab": False}}],
        use_vision=USE_VISION,
        use_thinking=False,     # removes thinking field from output schema
        max_actions_per_step=3,
        max_failures=5,
    )

    try:
        label = "[TEST] " if test_mode else ""
        print(f"\n{label}Contacting: {url}")

        result = await agent.run(max_steps=20 if test_mode else 30)

        try:
            result_text = result.final_result() or ""
        except Exception:
            result_text = ""
            _LOG.warning("final_result() raised — treating outcome as unknown")

        _LOG.info("RESULT  %s", result_text[:400])

        result_lower = result_text.lower()

        if "no form" in result_lower or "emails scraped" in result_lower or "emails:" in result_lower:
            scraped = re.findall(
                r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
                result_text,
            )
            if scraped:
                save_leads(url, scraped)
                print(f"   No form — scraped {len(scraped)} email(s) -> leads.json")
            else:
                print(f"   No form and no emails found")
            _LOG.info("STATUS  skipped (no form)")
            return "skipped"

        SUCCESS = [
            # Agent explicit success
            "submitted successfully", "test ok", "no confirmation page",
            # Generic thank-you and confirmation
            "thank you", "thank-you", "thankyou",
            "message sent", "message received", "received your message",
            "received your inquiry", "confirmation", "we received",
            "has been received", "has been submitted", "successfully submitted",
            "form submitted", "form has been submitted", "sent successfully",
            "request received", "inquiry received",
            # CRM / dealership-specific confirmation phrases
            "we will contact you", "we'll contact you",
            "we'll be in touch", "we will be in touch",
            "someone will be in touch", "get back to you", "we'll get back",
            "your inquiry", "your request", "your message has been",
            "your submission", "your information has been",
            "someone will", "a team member", "our team will",
            "a representative", "a sales", "a specialist",
            "one of our", "we appreciate", "appreciate your",
            # Schedule/test drive form confirmations (fallback path)
            "appointment request", "appointment confirmed", "appointment received",
            "test drive request", "service request",
        ]
        SKIPPED = [
            "no contact form", "captcha blocked", "no form found",
            "blank page", "no form -",
        ]
        FORM_ERRORS = [
            "form error:", "email rejected", "invalid email",
            "validation error", "please enter a valid", "required field",
        ]

        if any(k in result_lower for k in SUCCESS):
            print(f"   {'Test passed' if test_mode else 'Submitted'}: {result_text[:200]}")
            _LOG.info("STATUS  submitted")
            return "submitted"
        elif any(k in result_lower for k in FORM_ERRORS):
            print(f"   Form error (likely email rejected by site): {result_text[:200]}")
            _LOG.warning("STATUS  failed (form error — check if + in email was rejected)")
            return "failed"
        elif any(k in result_lower for k in SKIPPED):
            print(f"   Skipped: {result_text[:120]}")
            _LOG.info("STATUS  skipped")
            return "skipped"
        else:
            # No error reported and no known-bad outcome — agent most likely submitted
            # but the site didn't show a recognizable confirmation. Treat as submitted
            # to avoid double-submission on retry. Logged as uncertain for review.
            print(f"   Uncertain (treating as submitted): {result_text[:200]}")
            _LOG.warning("STATUS  submitted-uncertain (review log to verify)")
            try:
                for i, h in enumerate(result.history, 1):
                    _LOG.warning("  step %02d: %s", i, str(h)[:300])
            except Exception as dump_err:
                _LOG.warning("  (history dump failed: %s)", dump_err)
                _LOG.warning("  raw result: %s", str(result)[:3000])
            return "submitted"

    except Exception as e:
        _LOG.error("EXCEPTION  %s", e, exc_info=True)
        print(f"   Error: {e}")
        return "failed"
    finally:
        try:
            await session.kill()
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  DEALERSHIP CONTACT AGENT  v5  (browser-use 0.12.x)")
    print(f"  Model  : {OLLAMA_MODEL}")
    print(f"  Vision : {USE_VISION}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    log = load_log()
    print(
        f"\nPrevious run: {len(log['submitted'])} submitted, "
        f"{len(log['failed'])} failed, {len(log['skipped'])} skipped"
    )

    print("\n--- PHASE 1: DISCOVERY ---")
    # Only skip URLs that were already submitted/failed/skipped in production.
    # "tested" URLs were test-only (no real submission) and must be retried in production.
    already_seen = set(log["submitted"] + log["failed"] + log["skipped"])
    raw_urls = await get_dealership_urls(already_seen=already_seen)
    new_urls = [strip_utm(u) for u in raw_urls]

    print(f"{len(new_urls)} new dealership(s) to contact")
    if not new_urls:
        print("\nNothing new to process.")
        return

    print(f"\n--- PHASE 2: LOADING {OLLAMA_MODEL} ---")
    llm = get_llm()
    print("Model ready")

    print(f"\n--- PHASE 3: CONTACTING {len(new_urls)} DEALERSHIP(S) ---")
    for i, url in enumerate(new_urls, 1):
        print(f"\n[{i}/{len(new_urls)}]")
        status = await contact_dealership(url, llm, test_mode=False)
        log[status].append(url)
        save_log(log)
        if i < len(new_urls):
            await asyncio.sleep(DELAY_BETWEEN_SUBMISSIONS)

    print("\n" + "=" * 60)
    print("  RUN COMPLETE")
    print(f"  Submitted : {len(log['submitted'])}")
    print(f"  Skipped   : {len(log['skipped'])}  (no form or captcha)")
    print(f"  Failed    : {len(log['failed'])}")
    print(f"\n  Logs      : results.json")
    print(f"  Leads     : leads.json  (email fallbacks)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
