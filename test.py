"""
test.py — Dry run mode. Fills forms, stops before Send.
Imports from main.py to stay in sync.
"""

import asyncio
import logging
from datetime import datetime

from config import SEARCH_QUERIES, OLLAMA_MODEL
from main import get_llm, contact_dealership, strip_utm, load_log, save_log


# ── Truncate noisy logs ──────────────────────────────────────────────────────

class _TrimLongLines(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if len(msg) > 400:
            record.msg = record.msg[:300] + " ...[truncated]"
            record.args = ()
        return True

_trim = _TrimLongLines()
for _name in ("agent", "controller", "browser_use", "browser_use.agent.service", "root"):
    logging.getLogger(_name).addFilter(_trim)


# ── Discovery: 1 per query ───────────────────────────────────────────────────

async def get_one_per_query(already_seen: set = None) -> list[dict]:
    from playwright.async_api import async_playwright

    results = []
    seen_urls = set(already_seen or [])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=80)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            geolocation={"latitude": 44.9778, "longitude": -93.2650},
            permissions=["geolocation"],
        )
        page = await context.new_page()

        for query in SEARCH_QUERIES:
            print(f'\nSearching: "{query}"')
            found = None

            try:
                await page.goto(
                    f"https://www.google.com/maps/search/{query.replace(' ', '+')}",
                    wait_until="load", timeout=20000,
                )
                await asyncio.sleep(2)

                # Dismiss consent banners
                for selector in [
                    'button:has-text("Accept all")',
                    'button:has-text("Reject all")',
                    'button:has-text("I agree")',
                    'form[action*="consent"] button',
                ]:
                    try:
                        btn = page.locator(selector).first
                        if await btn.is_visible(timeout=2000):
                            await btn.click()
                            await asyncio.sleep(1)
                            break
                    except Exception:
                        pass

                # Scroll to load more listings
                for _ in range(8):
                    try:
                        panel = page.locator('[role="feed"]')
                        await panel.evaluate("el => el.scrollBy(0, 600)")
                        await asyncio.sleep(0.8)
                    except Exception:
                        break

                listing_locator = page.locator('[role="feed"] a[href*="/maps/place/"]')
                count = await listing_locator.count()
                print(f"   {count} listings found")

                place_urls = []
                seen_hrefs: set = set()
                for i in range(count):
                    href = await listing_locator.nth(i).get_attribute("href")
                    if href and href not in seen_hrefs:
                        seen_hrefs.add(href)
                        if href.startswith("/"):
                            href = "https://www.google.com" + href
                        place_urls.append(href)

                for place_url in place_urls:
                    try:
                        await page.goto(place_url, wait_until="load", timeout=20000)
                        await asyncio.sleep(2)
                        website_btn = page.locator('a[data-item-id="authority"]')
                        if await website_btn.count() > 0:
                            href = await website_btn.first.get_attribute("href")
                            if href and href.startswith("http"):
                                clean = strip_utm(href)
                                if clean not in seen_urls:
                                    seen_urls.add(clean)
                                    found = clean
                                    print(f"   Found: {clean}")
                                    break
                    except Exception as e:
                        print(f"   Warning: {e}")
                        continue

            except Exception as e:
                print(f"   Error: {e}")

            if found:
                results.append({"query": query, "url": found})
            else:
                print(f"   No URL found for this query")

            await asyncio.sleep(2)

        await browser.close()

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  DEALERSHIP AGENT — TEST MODE (no submissions)")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    log = load_log()
    already_seen = set(log["submitted"] + log["failed"] + log["skipped"] + log.get("tested", []))
    print(f"\nPreviously seen: {len(already_seen)} URLs")

    print("\n--- DISCOVERY ---")
    targets = await get_one_per_query(already_seen=already_seen)

    if not targets:
        print("\nNo dealerships found.")
        return

    print(f"\n{len(targets)} to test:")
    for i, t in enumerate(targets, 1):
        print(f"  {i}. {t['url']}")

    print(f"\n--- LOADING {OLLAMA_MODEL} ---")
    llm = get_llm()
    print("Model ready")

    print(f"\n--- FILLING FORMS (no submit) ---\n")
    for i, target in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {target['url']}")
        await contact_dealership(target["url"], llm, test_mode=True)
        url = strip_utm(target["url"])
        if url not in log.get("tested", []):
            log.setdefault("tested", []).append(url)
            save_log(log)
        if i < len(targets):
            await asyncio.sleep(3)

    print("\n" + "=" * 60)
    print("  TEST COMPLETE — nothing submitted")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
