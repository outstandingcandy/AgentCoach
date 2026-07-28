/* Global top-of-page navigation, injected into every static page.
 *
 * One source of truth for the module list + per-run link rewriting,
 * so pages don't drift apart over time. The script is no-op in
 * iframed contexts (shell.html is the only place that still uses an
 * iframe-of-pages layout, and showing a nested nav inside it would be
 * confusing).
 */
(function () {
  if (window.top !== window.self) return; // skip iframed renders
  // Don't double-inject if the page already has its own nav (shell.html).
  if (document.getElementById('goalinsight-nav')) return;
  if (document.querySelector('nav .brand')) return;

  // ---------------------------------------------------------------
  // Detect which module we're on + extract any run name in the URL.
  // ---------------------------------------------------------------
  const path = window.location.pathname;
  // /insights/<run>, /tracking/<run>, /match/<run> all carry a run.
  // /match/ (trailing slash) is the index, not a per-run route.
  const runMatch = path.match(/^\/(insights|tracking|match)\/([^/]+)\/?$/);
  const RUN = runMatch && runMatch[2] !== '' ? decodeURIComponent(runMatch[2]) : null;

  const queryRun = new URLSearchParams(window.location.search).get('run');
  // /pipeline?run=<run> uses a query string instead of a path segment.
  const RUN_NAME = RUN || queryRun || null;

  // Fallback for the run-scoped tabs when the current page URL has no
  // run in it (e.g. we're on /library). We fetch the most recently
  // finished run once and rewrite the Pipeline / Match / Tracking
  // links to point at it — otherwise clicking those tabs from Library
  // lands on an empty picker or a 404. Async: happens after the nav
  // is first rendered so the DOM is available.
  let FALLBACK_RUN = null;
  async function _loadFallbackRun() {
    if (RUN_NAME) return;
    try {
      const r = await fetch('/api/library/runs', {cache: 'no-store'});
      if (!r.ok) return;
      const runs = await r.json();
      // Prefer a run that has BOTH tracking+field_registration completed.
      // The list order matches list_runs() which returns them
      // most-recently-modified first.
      const picked = runs.find(x => x.stages
        && x.stages.field_registration && x.stages.tracking) || runs[0];
      if (picked) FALLBACK_RUN = picked.run_name;
    } catch (_) { /* silent — nav still works without it */ }
  }
  const _rewriteFallbackLinks = () => {
    if (!FALLBACK_RUN) return;
    // ``nav`` (the local variable, created below) holds the anchor
    // elements even before ``inject()`` attaches it to document.body.
    // Query directly through it so this fallback rewrite works
    // regardless of whether the nav has been inserted yet.
    const map = {
      pipeline: `/pipeline?run=${encodeURIComponent(FALLBACK_RUN)}`,
      match: `/match/${encodeURIComponent(FALLBACK_RUN)}`,
      tracking: `/tracking/${encodeURIComponent(FALLBACK_RUN)}`,
    };
    for (const [id, href] of Object.entries(map)) {
      const el = nav.querySelector(`a[data-nav-id="${id}"]`);
      if (el) el.href = href;
    }
    // Tracking is rendered as a disabled span when no run was known —
    // replace it with a real link now that we have one.
    const span = nav.querySelector('span.disabled');
    if (span && span.textContent === 'Tracking') {
      const a = document.createElement('a');
      a.href = map.tracking;
      a.textContent = 'Tracking';
      a.dataset.navId = 'tracking';
      span.replaceWith(a);
    }
  };
  // ``nav`` is created further down inside this IIFE; wrap the resolve
  // in a defer so it always fires AFTER the anchors have been added.
  _loadFallbackRun().then(() =>
    Promise.resolve().then(_rewriteFallbackLinks));

  function moduleId() {
    if (path.startsWith('/library')) return 'library';
    if (path.startsWith('/pipeline')) return 'pipeline';
    if (path.startsWith('/annotate')) return 'annotate';
    if (path.startsWith('/match')) return 'match';
    if (path.startsWith('/insights')) return 'insights';
    if (path.startsWith('/tracking')) return 'tracking';
    return null;
  }
  const CURRENT = moduleId();

  // Each item: where to go if we have a run vs not. Run-scoped items
  // (insights / tracking) are disabled when no run is in the URL —
  // we don't have anywhere meaningful to send the user.
  const ITEMS = [
    { id: 'library',  label: 'Library',
      href: () => '/library' },
    { id: 'pipeline', label: 'Pipeline',
      href: () => RUN_NAME ? `/pipeline?run=${encodeURIComponent(RUN_NAME)}` : '/pipeline' },
    { id: 'annotate', label: 'Annotate',
      href: () => '/annotate' },
    { id: 'match', label: 'Match',
      href: () => RUN_NAME ? `/match/${encodeURIComponent(RUN_NAME)}` : '/match/' },
    { id: 'insights', label: 'Insights',
      // Always clickable: with a run we go straight to the chat;
      // without one we land on /insights index, which is a run
      // picker. Same UX as Match.
      href: () => RUN_NAME ? `/insights/${encodeURIComponent(RUN_NAME)}` : '/insights' },
    { id: 'tracking', label: 'Tracking',
      requiresRun: true,
      href: () => RUN_NAME ? `/tracking/${encodeURIComponent(RUN_NAME)}` : null },
  ];

  // ---------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------
  const NAV_HEIGHT = 40;

  const style = document.createElement('style');
  // The pages use `body { height: 100vh }` with internal grid/flex
  // layouts, so we can't just add padding (it would push layouts past
  // the viewport). Instead we shrink body to "viewport minus nav" and
  // place the nav above it. ``html { height: 100% }` stays the same.
  //
  // The reserved offset is driven by the `--gi-nav-h` custom property,
  // NOT a hardcoded constant. `syncNavHeight()` (below) measures the
  // nav's *real* rendered height and writes it back, so a nav that
  // grows past its 40px baseline (larger base font, browser zoom
  // rounding, a late-loading web font, a wrapped run pill) still gets
  // exactly its own height reserved — the fixed bar can never overflow
  // down onto the page header / video. The 40px in the var is only a
  // pre-measurement fallback for the first paint.
  style.textContent = `
    html { height: 100%; }
    body { height: calc(100vh - var(--gi-nav-h, ${NAV_HEIGHT}px)) !important;
           min-height: calc(100vh - var(--gi-nav-h, ${NAV_HEIGHT}px)) !important;
           margin-top: var(--gi-nav-h, ${NAV_HEIGHT}px) !important; }
    #goalinsight-nav { position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
      min-height: ${NAV_HEIGHT}px; display: flex; align-items: center; gap: 4px;
      padding: 0 14px; background: #0f1419; border-bottom: 1px solid #262b33;
      font: 500 13px/1.2 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    #goalinsight-nav .brand { color: #fff; font-weight: 700; font-size: 14px;
      margin-right: 14px; letter-spacing: 0.02em; }
    #goalinsight-nav a { color: #cfe3ff; text-decoration: none;
      padding: 6px 12px; border-radius: 6px; transition: background 0.1s; }
    #goalinsight-nav a:hover { background: #1e242b; }
    #goalinsight-nav a.active { background: #3b82f6; color: #fff; }
    #goalinsight-nav span.disabled { color: #4a525c; padding: 6px 12px;
      cursor: not-allowed; }
    #goalinsight-nav .run-pill { margin-left: auto; color: #9aa3ad;
      font-size: 12px; padding: 4px 10px; background: #1a2027;
      border: 1px solid #232a32; border-radius: 999px; max-width: 32ch;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  `;
  document.head.appendChild(style);

  const nav = document.createElement('nav');
  nav.id = 'goalinsight-nav';

  const brand = document.createElement('span');
  brand.className = 'brand';
  brand.textContent = 'GoalInsight';
  nav.appendChild(brand);

  for (const item of ITEMS) {
    const href = item.href();
    if (!href && item.requiresRun) {
      const span = document.createElement('span');
      span.className = 'disabled';
      span.textContent = item.label;
      span.title = 'Open a run from Library or Match to enable this page';
      nav.appendChild(span);
    } else {
      const a = document.createElement('a');
      a.href = href;
      a.textContent = item.label;
      a.dataset.navId = item.id;
      if (item.id === CURRENT) a.classList.add('active');
      nav.appendChild(a);
    }
  }

  // Hide the Insights tab when the server tells us chat is disabled
  // (offline deployment). Asynchronous so the rest of the nav renders
  // immediately — the user only sees the Insights tab disappear if
  // they happen to be staring at the nav at <100ms latency.
  fetch('/api/features', { cache: 'no-store' })
    .then(r => r.ok ? r.json() : null)
    .then(features => {
      if (features && features.chat_enabled === false) {
        const el = nav.querySelector('a[data-nav-id="insights"]');
        if (el) el.remove();
      }
    })
    .catch(() => { /* feature probe unreachable — keep default nav */ });

  if (RUN_NAME) {
    const pill = document.createElement('span');
    pill.className = 'run-pill';
    pill.textContent = `run: ${RUN_NAME}`;
    pill.title = RUN_NAME;
    nav.appendChild(pill);
  }

  // Keep the reserved body offset exactly equal to the nav's real
  // rendered height. Without this, the fixed bar is reserved a hardcoded
  // 40px; if it renders taller (zoom/font/wrap) it spills over the page
  // header and the top of the video. Round up so a fractional height
  // never leaves a sub-pixel sliver of the bar overlapping the content.
  function syncNavHeight() {
    const h = Math.ceil(nav.getBoundingClientRect().height) || NAV_HEIGHT;
    document.documentElement.style.setProperty('--gi-nav-h', h + 'px');
  }

  // Insert at the very top so it sits above any page-level header.
  function inject() {
    document.body.insertBefore(nav, document.body.firstChild);
    syncNavHeight();
    // Re-measure after layout settles and on any reflow that can change
    // the bar's height (viewport resize, late web-font swap).
    window.addEventListener('resize', syncNavHeight);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(syncNavHeight).catch(() => {});
    }
    if (typeof ResizeObserver === 'function') {
      new ResizeObserver(syncNavHeight).observe(nav);
    }
  }
  if (document.body) inject();
  else document.addEventListener('DOMContentLoaded', inject);
})();
