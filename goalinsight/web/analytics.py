"""Built-in viewer analytics: stats JSON + pitch-overlay PNGs.

All endpoints are scoped to a run via ``/api/runs/{run_name}/analytics/...``
and rely on the ``RunRegistry`` to lazy-load per-run state.

Stats endpoints (player/team) proxy the existing pure-Python helpers in
``match_tools`` so the chat tools and the click-to-visualize panel stay
in sync.

PNG endpoints render matplotlib figures over the same pitch background
the annotator uses (``annotation/pitch_diagram.make_pitch_canvas`` +
``draw_pitch_structure``), so visualizations look consistent across the
product.
"""

from __future__ import annotations

import io
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

import matplotlib
matplotlib.use("Agg")  # no display in server processes
import matplotlib.pyplot as plt  # noqa: E402

from ..annotation import pitch_constants  # noqa: E402
from ..annotation.pitch_diagram import (  # noqa: E402
    draw_pitch_structure,
    make_pitch_canvas,
)
from ..highlights._context import MatchContext  # noqa: E402
from ._runs import RunRegistry  # noqa: E402

PITCH_SCALE = 12      # px per metre
PITCH_MARGIN = 30     # px around the pitch


def register_analytics_routes(app: FastAPI, runs: RunRegistry) -> None:
    def _get_ctx(run_name: str) -> MatchContext:
        try:
            return runs.get(run_name).ctx
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    # ---- JSON ------------------------------------------------------------

    @app.get("/api/runs/{run_name}/analytics/players")
    def list_players(run_name: str) -> JSONResponse:
        ctx = _get_ctx(run_name)
        # Aggregate observed counts per track id.
        counts: dict[str, int] = defaultdict(int)
        for items in ctx.player_tracks.values():
            for t in items:
                tid = t.get("track_id")
                if tid is not None:
                    counts[str(tid)] += 1
        out = []
        for tid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            out.append({
                "player_id": tid,
                "team_id": ctx.team_assignments.get(tid, "unknown"),
                "frames_observed": n,
            })
        return JSONResponse(out)

    @app.get("/api/runs/{run_name}/analytics/teams")
    def list_teams(run_name: str) -> JSONResponse:
        ctx = _get_ctx(run_name)
        # Set of distinct team labels referenced by tracks or events.
        teams: set[str] = set(ctx.team_assignments.values())
        for e in ctx.events:
            t = e.get("team_id")
            if t:
                teams.add(t)
        return JSONResponse(sorted(teams))

    @app.get("/api/runs/{run_name}/analytics/player_stats")
    def player_stats(run_name: str, player_id: str) -> JSONResponse:
        from .match_tools import get_player_stats  # local import: heavy module
        ctx = _get_ctx(run_name)
        return JSONResponse(get_player_stats(ctx, player_id=player_id))

    @app.get("/api/runs/{run_name}/analytics/team_stats")
    def team_stats(run_name: str, team_id: str) -> JSONResponse:
        from .match_tools import get_team_stats
        ctx = _get_ctx(run_name)
        return JSONResponse(get_team_stats(ctx, team_id=team_id))

    # ---- PNG -------------------------------------------------------------

    @app.get("/api/runs/{run_name}/analytics/heatmap.png")
    def heatmap(run_name: str, player_id: str | None = None,
                team_id: str | None = None, bins: int = 40) -> Response:
        ctx = _get_ctx(run_name)
        positions = _collect_positions(ctx, player_id=player_id, team_id=team_id)
        if not positions:
            raise HTTPException(404, "no pitch positions for that filter")
        png = _render_heatmap(positions, bins=bins,
                              title=_filter_title(player_id, team_id))
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/runs/{run_name}/analytics/shot_map.png")
    def shot_map(run_name: str, team_id: str | None = None) -> Response:
        ctx = _get_ctx(run_name)
        shots = _collect_shots(ctx, team_id=team_id)
        if not shots:
            raise HTTPException(404, "no shot events for that filter")
        png = _render_shot_map(shots, title=_filter_title(None, team_id))
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/runs/{run_name}/analytics/pass_network.png")
    def pass_network(run_name: str, team_id: str | None = None,
                     min_passes: int = 1) -> Response:
        ctx = _get_ctx(run_name)
        nodes, edges = _collect_pass_network(ctx, team_id=team_id,
                                             min_passes=min_passes)
        if not edges:
            raise HTTPException(404, "no pass events for that filter")
        png = _render_pass_network(nodes, edges,
                                   title=_filter_title(None, team_id))
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_title(player_id: str | None, team_id: str | None) -> str:
    if player_id:
        return f"player {player_id}"
    if team_id:
        return f"team {team_id}"
    return "all players"


def _collect_positions(
    ctx: MatchContext,
    *,
    player_id: str | None,
    team_id: str | None,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for items in ctx.player_tracks.values():
        for t in items:
            if player_id is not None and str(t.get("track_id")) != player_id:
                continue
            if team_id is not None:
                tid = ctx.team_assignments.get(str(t.get("track_id")), "unknown")
                if tid != team_id and t.get("team") != team_id:
                    continue
            pp = t.get("pitch_position") or [None, None]
            if pp[0] is None:
                continue
            out.append((float(pp[0]), float(pp[1])))
    return out


def _collect_shots(
    ctx: MatchContext, *, team_id: str | None,
) -> list[dict[str, Any]]:
    out = []
    for e in ctx.events:
        if e.get("type") != "shot":
            continue
        if team_id is not None and e.get("team_id") != team_id:
            continue
        sp = e.get("start_position")
        if sp is None or len(sp) < 2:
            continue
        out.append({
            "x": float(sp[0]),
            "y": float(sp[1]),
            "outcome": (e.get("metadata") or {}).get("outcome", "unknown"),
            "team_id": e.get("team_id"),
            "player_id": e.get("player_id"),
        })
    return out


def _collect_pass_network(
    ctx: MatchContext, *, team_id: str | None, min_passes: int,
) -> tuple[dict[str, tuple[float, float]],
           list[tuple[str, str, int, tuple[float, float], tuple[float, float]]]]:
    # avg position per player (touched by any pass) for nodes
    per_player_pos: dict[str, list[tuple[float, float]]] = defaultdict(list)
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    edge_endpoints: dict[tuple[str, str],
                         list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)

    for e in ctx.events:
        if e.get("type") != "pass":
            continue
        if team_id is not None and e.get("team_id") != team_id:
            continue
        if (e.get("metadata") or {}).get("outcome") == "failed":
            continue  # exclude failures from network — they have no receiver
        passer = str(e.get("player_id") or "")
        receiver = str((e.get("metadata") or {}).get("receiver_id") or "")
        if not passer or not receiver:
            continue
        sp = e.get("start_position")
        ep = e.get("end_position")
        if sp is None or ep is None:
            continue
        per_player_pos[passer].append((float(sp[0]), float(sp[1])))
        per_player_pos[receiver].append((float(ep[0]), float(ep[1])))
        edge_counts[(passer, receiver)] += 1
        edge_endpoints[(passer, receiver)].append(
            ((float(sp[0]), float(sp[1])), (float(ep[0]), float(ep[1])))
        )

    nodes = {pid: (sum(p[0] for p in pts) / len(pts),
                   sum(p[1] for p in pts) / len(pts))
             for pid, pts in per_player_pos.items()}
    edges: list[tuple[str, str, int, tuple[float, float], tuple[float, float]]] = []
    for (a, b), n in edge_counts.items():
        if n < min_passes or a not in nodes or b not in nodes:
            continue
        edges.append((a, b, n, nodes[a], nodes[b]))
    return nodes, edges


def _pitch_axes(title: str) -> tuple[plt.Figure, plt.Axes, "callable", int, int]:
    img, to_px, w, h = make_pitch_canvas(PITCH_SCALE, PITCH_MARGIN)
    draw_pitch_structure(img, to_px, PITCH_SCALE, color=(255, 255, 255), thickness=2)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img[:, :, ::-1])  # BGR canvas → RGB
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    ax.set_title(title, color="#222")
    return fig, ax, to_px, w, h


def _save_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _render_heatmap(
    positions: list[tuple[float, float]], *, bins: int, title: str,
) -> bytes:
    fig, ax, to_px, w, h = _pitch_axes(f"Heatmap — {title}")
    pitch = pitch_constants.get_active_pitch()
    L = pitch.PITCH_LENGTH / 2.0
    W = pitch.PITCH_WIDTH / 2.0
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    H, xedges, yedges = np.histogram2d(
        xs, ys, bins=bins, range=[[-L, L], [-W, W]],
    )
    # Map histogram bin centres into pixel coords using to_px so the
    # overlay aligns with the pitch we just drew.
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    pxs = (xx + L) * PITCH_SCALE + PITCH_MARGIN
    pys = (W - yy) * PITCH_SCALE + PITCH_MARGIN
    ax.pcolormesh(
        pxs, pys, H, alpha=0.55, shading="auto", cmap="hot",
    )
    return _save_png(fig)


_OUTCOME_COLORS = {
    "Goal": "#2ecc71",
    "Saved": "#3498db",
    "Off_Target": "#e67e22",
    "Blocked": "#e74c3c",
    "unknown": "#bdc3c7",
}


def _render_shot_map(shots: list[dict[str, Any]], *, title: str) -> bytes:
    fig, ax, to_px, w, h = _pitch_axes(f"Shot map — {title}")
    seen_outcomes: set[str] = set()
    for s in shots:
        px, py = to_px(s["x"], s["y"])
        color = _OUTCOME_COLORS.get(s["outcome"], _OUTCOME_COLORS["unknown"])
        ax.plot(px, py, "o", color=color,
                markeredgecolor="black", markersize=10,
                label=s["outcome"] if s["outcome"] not in seen_outcomes else None)
        seen_outcomes.add(s["outcome"])
    if seen_outcomes:
        ax.legend(loc="lower center", ncol=len(seen_outcomes),
                  bbox_to_anchor=(0.5, -0.02), fontsize=9)
    return _save_png(fig)


def _render_pass_network(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str, int, tuple[float, float], tuple[float, float]]],
    *,
    title: str,
) -> bytes:
    fig, ax, to_px, w, h = _pitch_axes(f"Pass network — {title}")
    if not edges:
        return _save_png(fig)
    max_w = max(n for _, _, n, _, _ in edges)
    for a, b, n, (ax_, ay_), (bx_, by_) in edges:
        p1 = to_px(ax_, ay_)
        p2 = to_px(bx_, by_)
        lw = 0.8 + 4.0 * (n / max_w)
        alpha = 0.35 + 0.55 * (n / max_w)
        ax.annotate(
            "",
            xy=p2, xycoords="data",
            xytext=p1, textcoords="data",
            arrowprops=dict(arrowstyle="->", color="#3498db",
                            lw=lw, alpha=alpha),
        )
    for pid, (x, y) in nodes.items():
        px, py = to_px(x, y)
        ax.plot(px, py, "o", color="white",
                markeredgecolor="black", markersize=10)
        ax.text(px + 6, py - 6, str(pid), color="black", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.7))
    return _save_png(fig)
