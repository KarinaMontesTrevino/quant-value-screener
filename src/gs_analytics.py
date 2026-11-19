"""Local quantitative risk & econometrics powered by Goldman Sachs `gs_quant.timeseries` (no GsSession required)."""

from __future__ import annotations

from typing import Any

import gs_quant.timeseries as ts
import numpy as np
import pandas as pd
import yfinance as yf

from src.cache import get_cached, set_cached


def _get_spy_close() -> pd.Series:
    cache_key = "yf_spy_2y_v1"
    cached = get_cached(cache_key, ttl_seconds=21600)
    if cached is not None:
        return cached
    try:
        hist = yf.Ticker("SPY").history(period="2y", auto_adjust=True)
        close = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
    except Exception:
        close = pd.Series(dtype=float)
    if not close.empty:
        set_cached(cache_key, close)
    return close


def compute_gs_risk_metrics(hist: pd.DataFrame, rf_rate: float = 0.045) -> dict[str, Any]:
    """Compute local GS-Quant timeseries risk & econometric metrics for a ticker."""
    if hist is None or hist.empty or len(hist) < 65:
        return {}

    close = hist["Close"].dropna().astype(float).copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()

    vol_30d = ts.volatility(close, 30)
    max_dd = ts.max_drawdown(close)
    zscore_60 = ts.zscores(close, 60)
    rsi_14 = ts.relative_strength_index(close, 14)

    rets = ts.returns(close).dropna()
    tail_rets = rets.tail(126)
    sharpe_6m = None
    if len(tail_rets) >= 30 and tail_rets.std() > 0:
        ann_ret = float(tail_rets.mean() * 252.0)
        ann_vol = float(tail_rets.std() * np.sqrt(252.0))
        sharpe_6m = (ann_ret - rf_rate) / ann_vol

    spy_close = _get_spy_close()
    beta_60 = pd.Series(dtype=float)
    if not spy_close.empty:
        spy_s = spy_close.astype(float).copy()
        spy_s.index = pd.to_datetime(spy_s.index).tz_localize(None).normalize()
        common = close.index.intersection(spy_s.index)
        if len(common) >= 65:
            beta_60 = ts.beta(close.loc[common], spy_s.loc[common], 60)

    return {
        "vol_30d_last": float(vol_30d.dropna().iloc[-1]) if not vol_30d.dropna().empty else None,
        "max_dd_2y": float(max_dd.dropna().iloc[-1]) if not max_dd.dropna().empty else None,
        "zscore_60_last": float(zscore_60.dropna().iloc[-1]) if not zscore_60.dropna().empty else None,
        "rsi_14_last": float(rsi_14.dropna().iloc[-1]) if not rsi_14.dropna().empty else None,
        "sharpe_6m_last": sharpe_6m,
        "beta_60_last": float(beta_60.dropna().iloc[-1]) if not beta_60.dropna().empty else None,
        "vol_series": vol_30d.dropna(),
        "beta_series": beta_60.dropna(),
    }
