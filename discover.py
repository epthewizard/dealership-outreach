"""
discover.py — Finds car dealership websites from Google Maps.
Uses Playwright directly against Maps (avoids Google Search bot detection).
Deduplication enforced globally across all queries and previous runs.
"""

import asyncio
from urllib.parse import urlparse, urlunparse
from playwright.async_api import async_playwright
from config import SEARCH_QUERIES, MAX_PER_QUERY, EXTRA_DEALERSHIPS


def strip_utm(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url.split("?")[0].rstrip("/")


# Keywords that indicate an official franchise / new-car dealership
_DEALERSHIP_BRANDS = {
    "toyota", "ford", "honda", "chevy", "chevrolet", "gmc", "buick", "cadillac",
    "jeep", "dodge", "ram", "chrysler", "fiat", "alfa", "maserati",
    "bmw", "mercedes", "benz", "audi", "volkswagen", "vw", "porsche",
    "lexus", "acura", "infiniti", "genesis", "lincoln", "volvo",
    "subaru", "mazda", "nissan", "hyundai", "kia", "mitsubishi",
    "jaguar", "land rover", "landrover", "range rover",
    "tesla", "rivian", "lucid", "polestar",
    "cjdr", "cdjr", "chrysler dodge jeep ram",
}

# Keywords that indicate NOT an official dealership
_SKIP_KEYWORDS = {
    "autozone", "oreilly", "o'reilly", "napa", "advance auto",
    "jiffy lube", "valvoline", "meineke", "midas", "pep boys",
    "carmax", "carvana", "vroom", "shift",
    "u-pull", "pick-n-pull", "junkyard", "salvage",
    "tint", "detail", "wrap", "audio", "stereo",
    "insurance", "rental", "rent-a-car", "enterprise",
    "tire", "brake", "muffler", "transmission",
    "collision", "body shop",
}


def is_likely_dealership(url: str, place_name: str = "") -> bool:
    """Return True if the URL / place name looks like an official new-car dealership."""
    combined = (url + " " + place_name).lower()
    # If any brand keyword appears in the URL or place name, it's likely legit
    if any(brand in combined for brand in _DEALERSHIP_BRANDS):
        return True
    # If skip keywords match, reject
    if any(kw in combined for kw in _SKIP_KEYWORDS):
        return False
    # If the place name contains "dealer" or "motors", allow
    if any(w in combined for w in ("dealer", "motors", "auto group", "automotive group")):
        return True
    # Default: reject — avoids independent used-car lots and other non-franchise businesses
    return False


async def get_dealership_urls(already_seen: set = None) -> list[str]:
    """
    Searches Google Maps for each query in SEARCH_QUERIES.
    Caps results at MAX_PER_QUERY per query.
    Skips any URL already in already_seen (previous runs).
    Returns a deduplicated list of new URLs.
    """
    seen = set(already_seen or [])
    ordered_results = []

    # Seed with manually specified extras first
    for url in EXTRA_DEALERSHIPS:
        url = url.strip().rstrip("/")
        if url and url not in seen:
            seen.add(url)
            ordered_results.append(url)
            print(f"   📌  Extra: {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        for query in SEARCH_QUERIES:
            print(f"\n🔍  [{MAX_PER_QUERY} max] Searching: {query}")
            found_this_query = 0

            try:
                search_url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
                await page.goto(search_url, wait_until="load", timeout=20000)
                await asyncio.sleep(2)

                # Scroll the results panel to load more listings
                for _ in range(5):
                    try:
                        panel = page.locator('[role="feed"]')
                        await panel.evaluate("el => el.scrollBy(0, 600)")
                        await asyncio.sleep(1.2)
                    except Exception:
                        break

                # Collect all place page hrefs first — avoids stale-element issues
                listing_locator = page.locator('[role="feed"] a[href*="/maps/place/"]')
                count = await listing_locator.count()
                print(f"   Maps returned {count} listings")

                place_urls = []
                seen_hrefs: set = set()
                for i in range(count):
                    href = await listing_locator.nth(i).get_attribute("href")
                    if href and href not in seen_hrefs:
                        seen_hrefs.add(href)
                        if href.startswith("/"):
                            href = "https://www.google.com" + href
                        place_urls.append(href)

                # Navigate to each place page and grab the website URL
                for place_url in place_urls:
                    if found_this_query >= MAX_PER_QUERY:
                        print(f"   ✋  Hit cap of {MAX_PER_QUERY} for this query")
                        break

                    try:
                        await page.goto(place_url, wait_until="load", timeout=20000)
                        await asyncio.sleep(2)

                        # Grab the place name for dealership filtering
                        place_name = ""
                        try:
                            heading = page.locator('h1')
                            if await heading.count() > 0:
                                place_name = (await heading.first.inner_text()).strip()
                        except Exception:
                            pass

                        website_btn = page.locator('a[data-item-id="authority"]')
                        if await website_btn.count() > 0:
                            href = await website_btn.first.get_attribute("href")
                            if href and href.startswith("http"):
                                # Always normalize to root domain — Maps sometimes links to
                                # /service or /parts subpages which are useless for contact forms
                                p = urlparse(strip_utm(href))
                                clean = f"{p.scheme}://{p.netloc}"
                                if clean in seen:
                                    print(f"   ♻️  Duplicate skipped: {clean}")
                                elif not is_likely_dealership(clean, place_name):
                                    print(f"   🚫  Not a dealership, skipped: {place_name or clean}")
                                else:
                                    seen.add(clean)
                                    ordered_results.append(clean)
                                    found_this_query += 1
                                    print(f"   ✅  [{found_this_query}/{MAX_PER_QUERY}] {clean}")
                        else:
                            print(f"   –  No website button on this listing")

                    except Exception as e:
                        print(f"   ⚠️  Skipped a listing: {e}")
                        continue

            except Exception as e:
                print(f"   ❌  Query failed: {e}")
                continue

            print(f"   → {found_this_query} new URLs from this query")
            await asyncio.sleep(2)

        await browser.close()

    print(f"\n📋  Total unique dealerships discovered: {len(ordered_results)}")
    return ordered_results


if __name__ == "__main__":
    urls = asyncio.run(get_dealership_urls())
    print("\nAll discovered URLs:")
    for u in urls:
        print(f"  {u}")
