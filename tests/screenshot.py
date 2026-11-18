"""Headless UI check and screenshots. Run with the server up:
    .venv/bin/python tests/screenshot.py [base_url]
Writes shot-home.png, shot-pdf.png, shot-blog.png (1280 wide) and shot-*-narrow.png (390 wide).
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).resolve().parent.parent
BLOG = "https://lilianweng.github.io/posts/2023-06-23-agent/"


def launch(p):
    for kw in ({"executable_path": "/usr/bin/google-chrome"}, {}):
        try:
            return p.chromium.launch(args=["--no-sandbox"], **kw)
        except Exception:
            continue
    raise SystemExit("no chromium available")


def main():
    report = {}
    with sync_playwright() as p:
        b = launch(p)
        for width, height, suffix in ((1280, 860, ""), (390, 844, "-narrow")):
            pg = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=2 if suffix else 1)
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

            # home
            pg.goto(BASE + "/")
            pg.wait_for_selector("#home:not(.hidden)")
            pg.wait_for_function("document.fonts.status === 'loaded'")
            pg.wait_for_selector("#recent[data-loaded]", state="attached")
            pg.wait_for_timeout(400)
            pg.screenshot(path=str(OUT / f"shot-home{suffix}.png"), full_page=bool(suffix) or True)

            # empty submit shows an inline message, no navigation
            pg.click("#intake button[type=submit]")
            assert pg.locator("#formError").inner_text().strip(), "empty submit should explain itself"

            # sample PDF via the example button
            pg.click("[data-sample=attention]")
            pg.wait_for_selector(".page .hl", timeout=60000)
            pg.wait_for_function("[...document.images].slice(0,2).every(i => i.complete && i.naturalWidth > 0)")
            pg.wait_for_timeout(300)
            first = pg.locator(".page .hl.claim").first
            first.scroll_into_view_if_needed()
            pg.mouse.wheel(0, -160)
            pg.wait_for_timeout(300)
            if not suffix:
                first.hover()
                pg.wait_for_timeout(250)
            pg.screenshot(path=str(OUT / f"shot-pdf{suffix}.png"))
            report[f"pdf{suffix}"] = {"rects": pg.locator(".page .hl").count(), "labels": pg.locator(".page .badge").count()}

            # home link resets
            pg.click("#homeLink")
            pg.wait_for_selector("#home:not(.hidden)")
            assert pg.locator("#reader").inner_html() == "", "reader should be cleared"
            assert pg.url.rstrip("/") == BASE.rstrip("/"), pg.url
            pg.wait_for_selector("#recentList li", timeout=5000)

            # blog post through the URL box
            pg.fill("#urlInput", BLOG)
            pg.press("#urlInput", "Enter")
            pg.wait_for_selector("article.article mark.hl", timeout=90000)
            pg.wait_for_timeout(300)
            if not suffix:
                pg.locator("article .sh").nth(1).scroll_into_view_if_needed()
                pg.mouse.wheel(0, -300)
                pg.wait_for_timeout(300)
                m = pg.locator("article mark.hl.complex").first
                m.scroll_into_view_if_needed()
                pg.mouse.wheel(0, -200)
                pg.wait_for_timeout(250)
                m.hover()
                pg.wait_for_timeout(250)
            pg.screenshot(path=str(OUT / f"shot-blog{suffix}.png"))
            report[f"blog{suffix}"] = {"marks": pg.locator("article mark.hl").count(), "labels": pg.locator("article .badge").count(),
                                       "plan_items": pg.locator("#tab-plan .sec").count()}

            # plan click scrolls to a section; browser back returns home
            if suffix:
                pg.click("#panelToggle")
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(OUT / f"shot-panel{suffix}.png"))
            pg.locator("#tab-plan .sec").first.click()
            pg.go_back()
            pg.wait_for_selector("#home:not(.hidden)")
            report[f"errors{suffix}"] = errors
            pg.close()
        b.close()
    print(report)


if __name__ == "__main__":
    main()
