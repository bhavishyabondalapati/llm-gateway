"""Dashboard: an HTML page plus the JSON stats API it polls.

If DASHBOARD_TOKEN is set, the API requires header `X-Dashboard-Token`.
"""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()
HTML = Path(__file__).parent / "static" / "dashboard.html"


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page():
    return HTML.read_text()


@router.get("/dashboard/api/stats")
async def dashboard_stats(request: Request, window: int = Query(3600, ge=60, le=30 * 86400)):
    state = request.app.state.gw
    expected = state.settings.dashboard_token
    if expected and not hmac.compare_digest(request.headers.get("x-dashboard-token", ""), expected):
        return JSONResponse({"error": "invalid dashboard token"}, status_code=401)
    stats = await state.store.stats(window)
    stats["cache_entries"] = len(state.cache) if state.cache is not None else 0
    stats["budgets"] = [
        {"team": t.name, "used": await state.store.tokens_this_month(t.name), "budget": t.monthly_token_budget}
        for t in state.config.teams
    ]
    return stats
