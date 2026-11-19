"""Persistent watchlist management and quantitative portfolio sizing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"
WATCHLIST_FILE = DATA_DIR / "watchlists.json"


def load_watchlists() -> dict[str, dict[str, Any]]:
    """Load saved watchlists and theses from disk."""
    if not WATCHLIST_FILE.exists():
        return {}
    try:
        with WATCHLIST_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_watchlist(name: str, tickers: list[str], notes: str = "") -> None:
    """Save or update a named watchlist on disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_wl = load_watchlists()
    all_wl[name.strip()] = {
        "tickers": list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip())),
        "notes": notes.strip(),
    }
    with WATCHLIST_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_wl, f, indent=2, ensure_ascii=False)


def delete_watchlist(name: str) -> None:
    """Remove a saved watchlist by name."""
    all_wl = load_watchlists()
    if name in all_wl:
        del all_wl[name]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with WATCHLIST_FILE.open("w", encoding="utf-8") as f:
            json.dump(all_wl, f, indent=2, ensure_ascii=False)


def compute_portfolio_allocation(
    scored_df: pd.DataFrame,
    bundles: dict[str, dict[str, Any]],
    top_n: int = 10,
    method: str = "Quant Score Weighted",
    capital: float = 10000.0,
    max_sector_pct: float = 0.35,
) -> pd.DataFrame:
    """Compute suggested portfolio weights, dollar allocation, and shares with optional sector cap."""
    candidates = scored_df.head(top_n).copy()
    if candidates.empty:
        return pd.DataFrame()

    raw_weights: list[float] = []
    for _, row in candidates.iterrows():
        sym = row["ticker"]
        if method == "Equal Weight":
            raw_weights.append(1.0)
        elif method == "Inverse Volatility":
            hist = bundles.get(sym, {}).get("history", pd.DataFrame())
            if hist is not None and not hist.empty and len(hist) > 30:
                vol = float(hist["Close"].pct_change().dropna().std() * np.sqrt(252.0))
                raw_weights.append(1.0 / max(vol, 0.05))
            else:
                raw_weights.append(1.0)
        else:  # Quant Score Weighted
            raw_weights.append(max(float(row["quant_score"]), 1.0))

    w = np.array(raw_weights, dtype=float)
    w = w / w.sum()

    # Iterative sector cap enforcement
    sectors = candidates["sector"].fillna("Unknown").tolist()
    for _ in range(5):
        sector_totals: dict[str, float] = {}
        for idx, sec in enumerate(sectors):
            sector_totals[sec] = sector_totals.get(sec, 0.0) + w[idx]
        excess_found = False
        for sec, tot in sector_totals.items():
            if tot > max_sector_pct + 1e-6:
                excess_found = True
                scale = max_sector_pct / tot
                for idx, s in enumerate(sectors):
                    if s == sec:
                        w[idx] *= scale
        if not excess_found:
            break
        w = w / w.sum()

    out = candidates[["rank", "ticker", "name", "sector", "price", "quant_score", "f_score", "margin_of_safety"]].copy()
    out["weight_pct"] = w * 100.0
    out["allocation_usd"] = w * capital
    out["est_shares"] = out.apply(
        lambda r: (r["allocation_usd"] / r["price"]) if (pd.notna(r["price"]) and r["price"] > 0) else np.nan,
        axis=1,
    )
    return out
