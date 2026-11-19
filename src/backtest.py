"""Lookback portfolio simulation for top-ranked Quant Value stocks vs. SPY benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.cache import get_cached, set_cached


def _fetch_spy_history(use_cache: bool = True) -> pd.Series:
    cache_key = "yf_spy_2y_v1"
    if use_cache:
        cached = get_cached(cache_key, ttl_seconds=21600)
        if cached is not None:
            return cached
    try:
        hist = yf.Ticker("SPY").history(period="2y", auto_adjust=True)
        close = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
    except Exception:
        close = pd.Series(dtype=float)
    if use_cache and not close.empty:
        set_cached(cache_key, close)
    return close


def _perf_stats(returns: pd.Series) -> dict[str, float]:
    if returns.empty or len(returns) < 20:
        return {"total_return": 0.0, "cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    cum = (1.0 + returns).cumprod()
    total_ret = float(cum.iloc[-1] - 1.0)
    n_years = len(returns) / 252.0
    cagr = float((1.0 + total_ret) ** (1.0 / n_years) - 1.0) if n_years > 0 and total_ret > -1.0 else 0.0
    vol = float(returns.std() * np.sqrt(252.0))
    sharpe = float((returns.mean() * 252.0) / vol) if vol > 0 else 0.0
    roll_max = cum.cummax()
    dd = (cum / roll_max) - 1.0
    max_dd = float(dd.min())
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def run_top_n_backtest(
    scored_df: pd.DataFrame,
    bundles: dict[str, dict[str, Any]],
    top_n: int = 10,
    weighting: str = "Equal Weight",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Simulate cumulative performance of the Top-N ranked stocks vs. SPY over available 2Y history."""
    top_df = scored_df.head(top_n).copy()
    if top_df.empty:
        return {"error": "No hay acciones disponibles para simular."}

    price_series: dict[str, pd.Series] = {}
    scores_map: dict[str, float] = {}
    for _, row in top_df.iterrows():
        sym = row["ticker"]
        hist = bundles.get(sym, {}).get("history", pd.DataFrame())
        if hist is not None and not hist.empty and "Close" in hist.columns:
            s = hist["Close"].dropna()
            if len(s) > 30:
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                price_series[sym] = s
                scores_map[sym] = max(float(row["quant_score"]), 1.0)

    if not price_series:
        return {"error": "No hay series de precios suficientes para los tickers seleccionados."}

    prices_df = pd.DataFrame(price_series).dropna(how="all").ffill().dropna()
    if prices_df.shape[0] < 30:
        return {"error": "Historial común insuficiente para simular."}

    rets_df = prices_df.pct_change().dropna()

    if weighting == "Quant Score Weighted":
        raw_w = np.array([scores_map[c] for c in rets_df.columns], dtype=float)
        weights = raw_w / raw_w.sum()
    else:
        weights = np.ones(len(rets_df.columns), dtype=float) / len(rets_df.columns)

    port_rets = pd.Series(rets_df.values @ weights, index=rets_df.index, name="Quant Top-N")

    spy_close = _fetch_spy_history(use_cache=use_cache)
    curves = pd.DataFrame({"Quant Top-N": (1.0 + port_rets).cumprod() - 1.0})
    spy_stats = None

    if not spy_close.empty:
        spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None).normalize()
        spy_rets = spy_close.pct_change().dropna().reindex(port_rets.index).dropna()
        if len(spy_rets) > 20:
            aligned_port = port_rets.reindex(spy_rets.index).dropna()
            curves = pd.DataFrame(
                {
                    "Quant Top-N": (1.0 + aligned_port).cumprod() - 1.0,
                    "SPY (Benchmark)": (1.0 + spy_rets).cumprod() - 1.0,
                }
            )
            spy_stats = _perf_stats(spy_rets)
            port_stats = _perf_stats(aligned_port)
        else:
            port_stats = _perf_stats(port_rets)
    else:
        port_stats = _perf_stats(port_rets)

    return {
        "curves": curves,
        "port_stats": port_stats,
        "spy_stats": spy_stats,
        "tickers_used": list(rets_df.columns),
        "weights": dict(zip(rets_df.columns, weights)),
    }
