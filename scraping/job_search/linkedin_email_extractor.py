import re
import sys
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


SEARCH_KEYWORDS = [
    '"AI Developer Remote"'
]

BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
AUTH_DIR = BASE_DIR / "playwright" / ".auth"
AUTH_FILE = AUTH_DIR / "linkedin_user.json"
OUTPUT_FILE = OUTPUT_DIR / "emails.txt"

SCROLL_PAUSE_SECONDS = 2
PAGE_TIMEOUT_MS = 600000

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_DIR.mkdir(parents=True, exist_ok=True)


def normalize_email(email: str) -> str:
    email = email.strip()
    email = re.sub(r"[)\]}>.,;:!?]+$", "", email)
    return email


def save_emails_live(email_set):
    emails = sorted(email_set, key=lambda x: x.lower())
    OUTPUT_FILE.write_text(",".join(emails), encoding="utf-8")
    return emails


def add_email(email_set, email):
    email = normalize_email(email)
    if email and email not in email_set:
        email_set.add(email)
        save_emails_live(email_set)
        print(f"New email found: {email}")


def is_logged_in(page) -> bool:
    selectors = [
        'input[placeholder*="Search"]',
        'input[aria-label*="Search"]',
        '[data-test-global-nav-search-input]',
        'a[href*="/feed/"]',
    ]
    for selector in selectors:
        try:
            page.locator(selector).first.wait_for(state="visible", timeout=4000)
            return True
        except Exception:
            pass
    return False


def wait_for_manual_login(page, context):
    page.goto("https://www.linkedin.com/", wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)

    if is_logged_in(page):
        context.storage_state(path=str(AUTH_FILE))
        return

    print("\nPlease log in to LinkedIn manually in the opened Chrome window.")
    print("When login is complete and LinkedIn home page is visible, press ENTER here.\n")
    input()

    page.wait_for_load_state("domcontentloaded")
    if not is_logged_in(page):
        print("Login not detected. Please rerun and complete login.")
        sys.exit(1)

    context.storage_state(path=str(AUTH_FILE))


def try_click(page, selectors, timeout_ms=2500) -> bool:
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            loc.wait_for(state="visible", timeout=timeout_ms)
            loc.click(timeout=timeout_ms)
            return True
        except Exception:
            pass
    return False


def open_posts_filter_and_apply_latest(page) -> bool:
    page.wait_for_timeout(3000)

    filter_openers = [
        lambda: page.get_by_role("button", name=re.compile(r"all filters|filters", re.I)).first.click(timeout=4000),
        lambda: page.locator('button:has-text("All filters")').first.click(timeout=4000),
        lambda: page.locator('button:has-text("Filters")').first.click(timeout=4000),
    ]

    for action in filter_openers:
        try:
            action()
            page.wait_for_timeout(2000)
            break
        except Exception:
            pass

    posts_actions = [
        lambda: page.get_by_role("radio", name=re.compile(r"posts", re.I)).first.check(timeout=3000),
        lambda: page.get_by_label(re.compile(r"posts", re.I)).first.check(timeout=3000),
        lambda: page.locator('input[type="radio"]').locator("..").get_by_text(re.compile(r"posts", re.I)).click(timeout=3000),
    ]

    for action in posts_actions:
        try:
            action()
            page.wait_for_timeout(1000)
            break
        except Exception:
            pass

    latest_actions = [
        lambda: page.get_by_role("radio", name=re.compile(r"latest|recent", re.I)).first.check(timeout=3000),
        lambda: page.get_by_label(re.compile(r"latest|recent", re.I)).first.check(timeout=3000),
        lambda: page.locator('label:has-text("Latest")').first.click(timeout=3000),
        lambda: page.locator('label:has-text("Recent")').first.click(timeout=3000),
        lambda: page.locator('text=Latest').first.click(timeout=3000),
        lambda: page.locator('text=Recent').first.click(timeout=3000),
    ]

    latest_selected = False
    for action in latest_actions:
        try:
            action()
            latest_selected = True
            page.wait_for_timeout(1000)
            print("Latest/Recent selected.")
            break
        except Exception:
            pass

    show_results_actions = [
        lambda: page.get_by_role("button", name=re.compile(r"show results|apply", re.I)).first.click(timeout=4000),
        lambda: page.locator('button:has-text("Show results")').first.click(timeout=4000),
        lambda: page.locator('button:has-text("Apply")').first.click(timeout=4000),
    ]

    for action in show_results_actions:
        try:
            action()
            page.wait_for_timeout(4000)
            print("Show results clicked.")
            return latest_selected
        except Exception:
            pass

    print("Could not click Show results, continuing with current page state.")
    return latest_selected


def go_to_posts_search(page, keyword: str):
    url = f"https://www.linkedin.com/search/results/content/?keywords={quote(keyword)}"
    page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    page.wait_for_timeout(4000)

    post_view_actions = [
        lambda: page.get_by_role("button", name=re.compile(r"posts", re.I)).first.click(timeout=3000),
        lambda: page.get_by_role("link", name=re.compile(r"posts", re.I)).first.click(timeout=3000),
        lambda: page.locator('button:has-text("Posts")').first.click(timeout=3000),
        lambda: page.locator('a:has-text("Posts")').first.click(timeout=3000),
        lambda: page.locator('text=Posts').first.click(timeout=3000),
    ]

    for action in post_view_actions:
        try:
            action()
            page.wait_for_timeout(2500)
            break
        except Exception:
            pass

    open_posts_filter_and_apply_latest(page)
    page.wait_for_timeout(3000)


def extract_emails_from_text(text: str, email_set: set):
    for match in EMAIL_REGEX.findall(text or ""):
        add_email(email_set, match)


def expand_post_if_needed(post):
    more_selectors = [
        'button:has-text("see more")',
        'button:has-text("See more")',
        'button:has-text("more")',
        'button:has-text("…more")',
        'button:has-text("...more")',
        'span:has-text("see more")',
        'span:has-text("See more")',
        'span:has-text("more")',
        'div:has-text("see more")',
        'div:has-text("See more")',
    ]

    for selector in more_selectors:
        try:
            more_btn = post.locator(selector).first
            if more_btn.is_visible(timeout=1000):
                more_btn.scroll_into_view_if_needed(timeout=3000)
                more_btn.click(timeout=2000)
                post.page.wait_for_timeout(800)
                return True
        except Exception:
            pass
    return False


def extract_from_single_post(post, email_set: set):
    try:
        post.scroll_into_view_if_needed(timeout=3000)
    except Exception:
        pass

    try:
        expand_post_if_needed(post)
    except Exception:
        pass

    try:
        post_text = post.inner_text(timeout=3000)
        extract_emails_from_text(post_text, email_set)
    except Exception:
        pass

    try:
        hrefs = post.locator('a[href^="mailto:"]').evaluate_all(
            "(anchors) => anchors.map(a => a.getAttribute('href') || '')"
        )
        for href in hrefs:
            email = re.sub(r"^mailto:", "", href, flags=re.IGNORECASE).strip()
            if email:
                add_email(email_set, email)
    except Exception:
        pass


def expand_and_extract_from_posts(page, email_set: set):
    post_selectors = [
        "div.feed-shared-update-v2",
        "div.occludable-update",
        "div[data-urn]",
    ]

    post_locator = None

    for selector in post_selectors:
        try:
            locator = page.locator(selector)
            count = locator.count()
            if count > 0:
                post_locator = locator
                break
        except Exception:
            pass

    if not post_locator:
        try:
            body_text = page.locator("body").inner_text(timeout=5000)
            extract_emails_from_text(body_text, email_set)
        except Exception:
            pass
        return

    try:
        post_count = post_locator.count()
    except Exception:
        post_count = 0

    for i in range(post_count):
        try:
            post = post_locator.nth(i)
            extract_from_single_post(post, email_set)
        except Exception:
            pass


def click_show_more_results(page):
    actions = [
        lambda: page.get_by_role("button", name=re.compile(r"show more results|see more results|load more", re.I)).first.click(timeout=3000),
        lambda: page.locator('button:has-text("Show more results")').first.click(timeout=3000),
        lambda: page.locator('button:has-text("See more results")').first.click(timeout=3000),
        lambda: page.locator('button:has-text("Load more")').first.click(timeout=3000),
    ]

    for action in actions:
        try:
            action()
            page.wait_for_timeout(2500)
            return True
        except Exception:
            pass
    return False


def auto_scroll_and_collect_forever(page, email_set: set):
    scroll_round = 0

    try:
        page.locator("body").hover(timeout=3000)
    except Exception:
        pass

    print("\nStarted continuous scrolling. Press Ctrl+C in terminal to stop.\n")

    while True:
        scroll_round += 1

        expand_and_extract_from_posts(page, email_set)

        for _ in range(6):
            try:
                page.mouse.wheel(0, 1400)
            except Exception:
                pass

            page.wait_for_timeout(int(SCROLL_PAUSE_SECONDS * 1000))
            expand_and_extract_from_posts(page, email_set)

        clicked_more = click_show_more_results(page)
        if clicked_more:
            page.wait_for_timeout(2000)
            expand_and_extract_from_posts(page, email_set)

        try:
            page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass

        try:
            current_height = page.evaluate("document.body.scrollHeight")
        except Exception:
            current_height = 0

        save_emails_live(email_set)
        print(f"Round: {scroll_round}, page height: {current_height}, total emails: {len(email_set)}")


def run_keyword(page, keyword: str, email_set: set):
    print(f"Running search for: {keyword}")
    go_to_posts_search(page, keyword)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(2000)
    auto_scroll_and_collect_forever(page, email_set)


def create_context(browser):
    if AUTH_FILE.exists():
        return browser.new_context(storage_state=str(AUTH_FILE))
    return browser.new_context()


def main():
    ensure_dirs()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=False,
            slow_mo=200
        )

        context = create_context(browser)
        page = context.new_page()

        if not AUTH_FILE.exists():
            wait_for_manual_login(page, context)
            context.close()
            context = create_context(browser)
            page = context.new_page()
        else:
            page.goto(
                "https://www.linkedin.com/",
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT_MS
            )

        emails = set()

        try:
            for keyword in SEARCH_KEYWORDS:
                run_keyword(page, keyword, emails)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except PlaywrightTimeoutError as ex:
            print(f"Timeout error: {ex}")
        except Exception as ex:
            print(f"Error: {ex}")
        finally:
            saved_emails = save_emails_live(emails)
            print("\nDone.")
            print(f"Total unique emails found: {len(saved_emails)}")
            print(f"Saved to: {OUTPUT_FILE}")
            print(",".join(saved_emails))

            context.storage_state(path=str(AUTH_FILE))
            context.close()
            browser.close()


if __name__ == "__main__":
    main()