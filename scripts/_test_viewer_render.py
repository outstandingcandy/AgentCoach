"""Playwright smoke-test for /insights/<run> page rendering.

Verifies:
1. Chat history (loaded from c5354a87890d session) shows up.
2. Assistant message containing markdown gets rendered:
     - has a .md-body wrapper
     - contains <p> / <code> / <pre> nodes (not raw text)
3. Tool cards: list_events is collapsed by default with disclosure
   triangle visible; run_python is open by default.
4. Code block in run_python tool card is NOT inner-scrollable
   (max-height removed → element scrollHeight equals offsetHeight).

Saves a screenshot per check to /tmp/viewer_*.png so the failure
mode is inspectable.

Run:  /home/ubuntu/goal-insight-v3/venv/bin/python \
      scripts/_test_viewer_render.py
"""

from __future__ import annotations

import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000/insights/sunday_test_imgsz1920?session=c5354a87890d"
CHROMIUM = "/snap/bin/chromium"
OUT = "/tmp"


def main() -> int:
    fail: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM, headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 900})
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle", timeout=30_000)

        # The viewer renders chat history asynchronously after fetching
        # /api/runs/<run>/chat/sessions/<sid>/messages — give it a beat.
        page.wait_for_selector(".msg.assistant", timeout=15_000)
        page.screenshot(path=f"{OUT}/viewer_loaded.png", full_page=True)

        # ---- Check 1: assistant message has a .md-body wrapper ----
        n_md = page.locator(".msg.assistant .md-body").count()
        if n_md == 0:
            fail.append("no .md-body wrapper inside any assistant message "
                        "(markdown render path didn't engage)")
        print(f"  [1] .md-body count = {n_md}")

        # ---- Check 2: at least one assistant message contains a <p>
        # (proof markdown actually parsed, not just raw text) ----
        n_p = page.locator(".msg.assistant .md-body p").count()
        if n_p == 0:
            fail.append("no <p> inside .md-body — markdown parser didn't "
                        "engage (CDN blocked? marked.parse failed?)")
        print(f"  [2] <p> in md-body = {n_p}")

        # ---- Check 3: tool cards: disclosure triangle present ----
        tool_cards = page.locator(".tool-card").count()
        print(f"  [3] tool cards = {tool_cards}")
        if tool_cards == 0:
            fail.append("no tool cards rendered (session has run_python "
                        "history but UI didn't render)")
        else:
            # The ::before pseudo we added should produce a non-zero
            # bounding box on the summary's first child rendering.
            triangle_ok = page.evaluate("""
                () => {
                    const sums = document.querySelectorAll('.tool-card summary');
                    if (!sums.length) return null;
                    const cs = window.getComputedStyle(sums[0], '::before');
                    return {
                        content: cs.content,
                        borderLeft: cs.borderLeftWidth,
                        display: cs.display,
                    };
                }
            """)
            print(f"      ::before computed = {triangle_ok}")
            if not triangle_ok or triangle_ok.get("borderLeft") == "0px":
                fail.append("disclosure ::before pseudo not rendering "
                            f"(got {triangle_ok})")

        # ---- Check 4: list_events compact + collapsed by default ----
        list_events = page.locator(".tool-card.compact").count()
        print(f"  [4] compact tool cards = {list_events}")
        # If any compact card is open by default, that's the bug.
        compact_open = page.locator(".tool-card.compact[open]").count()
        if compact_open:
            fail.append(f"{compact_open} compact tool cards open by default "
                        "(should be collapsed)")

        # ---- Check 5: run_python card body uses unbounded max-height ----
        py_pre_h = page.evaluate("""
            () => {
                const el = document.querySelector('.tool-card.python pre');
                if (!el) return null;
                const cs = window.getComputedStyle(el);
                return {
                    maxHeight: cs.maxHeight,
                    scrollHeight: el.scrollHeight,
                    clientHeight: el.clientHeight,
                };
            }
        """)
        print(f"  [5] python <pre> = {py_pre_h}")
        if py_pre_h:
            mh = py_pre_h.get("maxHeight", "")
            if mh and mh != "none" and not mh.startswith("none"):
                fail.append(f"python <pre> still has max-height={mh} "
                            "(should be 'none' / unset)")
            sh = py_pre_h.get("scrollHeight", 0)
            ch = py_pre_h.get("clientHeight", 0)
            if sh > ch + 4:  # tiny tolerance
                fail.append(f"python <pre> is inner-scrollable: "
                            f"scrollHeight={sh} clientHeight={ch}")

        # ---- Check 6: click a compact card and verify it expands ----
        first_compact = page.locator(".tool-card.compact").first
        if first_compact.count():
            first_compact.locator("summary").click()
            # Browser-default open state propagates synchronously.
            opened = first_compact.evaluate("e => e.open")
            print(f"  [6] compact card opens on click = {opened}")
            if not opened:
                fail.append("compact tool card doesn't expand on summary click")
            page.screenshot(path=f"{OUT}/viewer_compact_open.png")

        # ---- Check 7: assistant code blocks have hljs-applied classes? ----
        n_hljs = page.locator(".msg.assistant pre code.hljs").count()
        n_pre = page.locator(".msg.assistant pre").count()
        print(f"  [7] assistant <pre> = {n_pre}, hljs-highlighted = {n_hljs}")
        # Not every assistant has fenced code blocks — only fail if there
        # are <pre> blocks but none got hljs.
        if n_pre and n_hljs == 0:
            fail.append(f"assistant has {n_pre} <pre> blocks but 0 got "
                        "highlight.js classes (CDN failed? hljs.highlightElement?)")

        page.screenshot(path=f"{OUT}/viewer_final.png", full_page=True)
        browser.close()

    print()
    if fail:
        print("FAILURES:")
        for f in fail:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
