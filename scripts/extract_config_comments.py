#!/usr/bin/env python3
"""One-shot tool: extract every comment from configs/*.yaml.

For each top-level pipeline config (not under configs/pitches/ — pitch
yamls are tiny and their comments are already key descriptions, not
tuning rationale), walk the parsed ruamel tree, pull every comment
fragment together with its dot-path key, dump everything to
``docs/config_notes.md``, and rewrite the yaml file stripped of all
comments except a 2-line header pointing back at the md.

Run once::

    python scripts/extract_config_comments.py

The extraction is idempotent — running it twice produces the same md
and leaves the already-stripped yamls alone (no comments left to find).
"""

from __future__ import annotations

import sys
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.tokens import CommentToken


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DOCS_PATH = REPO_ROOT / "docs" / "config_notes.md"

# Comment lines we treat as boilerplate to drop on round-trip.
# Mostly the first-line file titles ruamel won't attribute to a key.
_BOILERPLATE_PREFIXES = (
    "# SoccerMaster Data Curation Pipeline",
)


def _clean_comment(raw: str) -> str:
    """Strip leading ``# `` from a raw comment line and right-trim."""
    raw = raw.rstrip()
    if raw.startswith("# "):
        return raw[2:]
    if raw.startswith("#"):
        return raw[1:]
    return raw


def _join_lines(tokens: list[CommentToken]) -> list[str]:
    """Flatten ruamel CommentToken list to ordered comment-line strings.

    Each token's ``.value`` can contain multiple newline-separated
    lines (some are blank or just whitespace — we drop those). Comment
    lines have ``#`` near the start; non-comment whitespace separators
    between groups are silently dropped.
    """
    out: list[str] = []
    for tok in tokens or []:
        if tok is None:
            continue
        text = tok.value if isinstance(tok.value, str) else ""
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                continue
            out.append(_clean_comment(stripped))
    return out


def _collect_from_node(node, path: str, acc: dict[str, list[str]]) -> None:
    """Recurse through a CommentedMap / CommentedSeq, capturing comments.

    Comments live in ``node.ca`` (the comment attribute). For a mapping,
    ``ca.items[key]`` is a 4-tuple: [pre_key, post_key, eol, none]. We
    treat pre_key + eol as the comment(s) "for" that key. Empty entries
    are skipped.
    """
    if isinstance(node, CommentedMap):
        ca = node.ca
        for key, value in node.items():
            sub_path = f"{path}.{key}" if path else str(key)
            tokens = ca.items.get(key, [None, None, None, None])
            # tokens layout: (pre_key, post_key, eol_first, eol_subsequent)
            # We collect any of them — ruamel doesn't always populate the
            # same slot, depends on whether the comment is on the same
            # line as the key (eol) or above it (pre/post).
            grouped: list[CommentToken] = []
            for slot in tokens:
                if slot is None:
                    continue
                if isinstance(slot, list):
                    grouped.extend(slot)
                else:
                    grouped.append(slot)
            lines = _join_lines(grouped)
            if lines:
                acc.setdefault(sub_path, []).extend(lines)
            _collect_from_node(value, sub_path, acc)
    elif isinstance(node, CommentedSeq):
        ca = node.ca
        for idx, value in enumerate(node):
            sub_path = f"{path}[{idx}]"
            tokens = ca.items.get(idx, [None, None, None, None])
            grouped: list[CommentToken] = []
            for slot in tokens:
                if slot is None:
                    continue
                if isinstance(slot, list):
                    grouped.extend(slot)
                else:
                    grouped.append(slot)
            lines = _join_lines(grouped)
            if lines:
                acc.setdefault(sub_path, []).extend(lines)
            _collect_from_node(value, sub_path, acc)


def _collect_file_header(node) -> list[str]:
    """Pull any pre-document comment (file-level header) out of the tree."""
    if not isinstance(node, CommentedMap):
        return []
    comment = node.ca.comment  # (eol, [pre]) on the document
    if not comment:
        return []
    grouped: list[CommentToken] = []
    for slot in comment:
        if slot is None:
            continue
        if isinstance(slot, list):
            grouped.extend(slot)
        else:
            grouped.append(slot)
    lines = _join_lines(grouped)
    return [
        ln for ln in lines
        if not any(ln.startswith(p[2:]) for p in _BOILERPLATE_PREFIXES)
    ]


def _strip_comments(node) -> None:
    """Recursively wipe every CommentToken from a ruamel tree."""
    if isinstance(node, CommentedMap):
        node.ca.comment = None
        node.ca.items.clear()
        node.ca.end = []
        for v in node.values():
            _strip_comments(v)
    elif isinstance(node, CommentedSeq):
        node.ca.comment = None
        node.ca.items.clear()
        node.ca.end = []
        for v in node:
            _strip_comments(v)


def _build_header(yaml_name: str) -> str:
    return (
        f"# {yaml_name}\n"
        f"# Comments / rationale live in docs/config_notes.md (#{yaml_name}).\n"
    )


def _md_for_file(yaml_name: str, header_lines: list[str],
                 by_path: dict[str, list[str]]) -> str:
    parts: list[str] = []
    parts.append(f"## {yaml_name}\n")
    if header_lines:
        parts.append("### _header_\n")
        for ln in header_lines:
            parts.append(f"> {ln}\n")
        parts.append("\n")
    if not by_path:
        if not header_lines:
            parts.append("_(no inline comments)_\n\n")
        return "".join(parts)
    for key_path in by_path:                # preserve insertion (= source) order
        parts.append(f"### `{key_path}`\n\n")
        for ln in by_path[key_path]:
            parts.append(f"- {ln}\n")
        parts.append("\n")
    return "".join(parts)


def main(argv: list[str]) -> int:
    targets = sorted(p for p in CONFIGS_DIR.glob("*.yaml") if p.is_file())
    if not targets:
        print(f"No yaml files under {CONFIGS_DIR}", file=sys.stderr)
        return 1

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 200            # don't auto-wrap long values on the rewrite

    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)

    md_chunks: list[str] = [
        "# Config notes\n\n",
        "Auto-extracted from `configs/*.yaml` by "
        "`scripts/extract_config_comments.py`. Comments were moved here so the "
        "yaml files themselves stay short and diff-friendly; each section is "
        "keyed by the yaml file and the dotted key path the comment was "
        "originally attached to.\n\n",
    ]

    for yaml_path in targets:
        rel = yaml_path.name
        original_text = yaml_path.read_text()
        data = yaml.load(original_text)
        if data is None:
            print(f"  {rel}: empty / scalar yaml — skipping", file=sys.stderr)
            continue

        header_lines = _collect_file_header(data)
        by_path: dict[str, list[str]] = {}
        _collect_from_node(data, "", by_path)

        # Idempotency guard: if there are no comments at all, leave the
        # file alone (re-running wouldn't change anything anyway).
        if not header_lines and not by_path:
            print(f"  {rel}: no comments found — leaving as-is")
            md_chunks.append(_md_for_file(rel, header_lines, by_path))
            continue

        md_chunks.append(_md_for_file(rel, header_lines, by_path))

        # Strip and re-write. ruamel will keep formatting (indent, list
        # style) on the round-trip — we just lose the comments.
        _strip_comments(data)
        new_text_io: list[str] = []

        # Render via an io.StringIO to capture output.
        import io
        buf = io.StringIO()
        yaml.dump(data, buf)
        rewritten = _build_header(rel) + "\n" + buf.getvalue()

        # Cross-check: parsed values match the original.
        from yaml import safe_load
        if safe_load(original_text) != safe_load(rewritten):
            print(
                f"  {rel}: REFUSED — round-trip changed values, skipping",
                file=sys.stderr,
            )
            continue

        yaml_path.write_text(rewritten)
        total = sum(len(v) for v in by_path.values()) + len(header_lines)
        print(f"  {rel}: {total} comment lines moved to docs/config_notes.md")

    DOCS_PATH.write_text("".join(md_chunks))
    print(f"\nWrote {DOCS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
