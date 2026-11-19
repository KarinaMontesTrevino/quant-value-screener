"""Data providers for fundamental, market, and enrichment data (yfinance, Finnhub, Finviz)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd
import requests
import yfinance as yf

from src.cache import get_cached, set_cached

PRESET_UNIVERSES: dict[str, list[str]] = {
    "Value & Quality Watchlist (20)": [
        "BRK-B", "JPM", "UNH", "GOOGL", "META", "MSFT", "V", "MA",
        "JNJ", "PG", "KO", "PEP", "CVX", "XOM", "HD", "LOW",
        "COST", "ABBV", "MRK", "CSCO",
    ],
    "Dow 30": [
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V",
        "VZ", "WMT",
    ],
    "Deep Value & Cyclicals (15)": [
        "C", "BAC", "WFC", "F", "GM", "VALE", "PBR", "BABA", "PYPL", "INTC",
        "VZ", "T", "MO", "OXY", "DVN",
    ],
    "LatAm ADRs (12)": [
        "MELI", "NU", "PBR", "VALE", "ITUB", "BBD", "ABEV", "AMX",
        "FMX", "KOF", "SQM", "EC",
    ],
}

FINVIZ_SCREENER_PRESETS: dict[str, dict[str, str]] = {
    "Finviz: Deep Value (P/E < 15, P/B < 2, Mid+)": {
        "Market Cap.": "+Mid (over $2bln)",
        "P/E": "Under 15",
        "P/B": "Under 2",
    },
    "Finviz: Quality Value (ROE > 15%, P/E < 20, Low Debt)": {
        "Market Cap.": "+Mid (over $2bln)",
        "P/E": "Under 20",
        "Return on Equity": "Over +15%",
        "Debt/Equity": "Under 1",
    },
    "Finviz: High Dividend Value (Div > 3%, P/E < 18)": {
        "Market Cap.": "+Mid (over $2bln)",
        "Dividend Yield": "Over 3%",
        "P/E": "Under 18",
    },
}


def run_finviz_screener(
    preset_name: str, limit: int = 25, use_cache: bool = True
) -> list[str]:
    """Run a Finviz market screener preset and return matching ticker symbols."""
    filters = FINVIZ_SCREENER_PRESETS.get(preset_name)
    if not filters:
        return []
    cache_key = f"fv_screener_v1:{preset_name}:{limit}"
    if use_cache:
        cached = get_cached(cache_key, ttl_seconds=21600)
        if cached is not None:
            return cached

    try:
        from finvizfinance.screener.overview import Overview

        foverview = Overview()
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view(order="Market Cap.", ascend=False)
        if df is not None and not df.empty and "Ticker" in df.columns:
            tickers = [str(t).strip().upper() for t in df["Ticker"].head(limit).tolist()]
            if use_cache and tickers:
                set_cached(cache_key, tickers)
            return tickers
    except Exception:
        pass
    return []


def fetch_yfinance_bundle(ticker: str, use_cache: bool = True) -> dict[str, Any]:
    """Fetch info, financial statements, and 2Y price history from Yahoo Finance."""
    symbol = ticker.strip().upper()
    cache_key = f"yf_bundle_v2:{symbol}"
    if use_cache:
        cached = get_cached(cache_key, ttl_seconds=21600)
        if cached is not None:
            return cached

    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    def _safe_df(getter: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        try:
            df = getter()
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    inc = _safe_df(lambda: t.financials)
    bal = _safe_df(lambda: t.balance_sheet)
    cf = _safe_df(lambda: t.cashflow)
    hist = _safe_df(lambda: t.history(period="2y", auto_adjust=True))

    news_items: list[dict[str, str]] = []
    try:
        raw_news = t.news or []
        for item in raw_news[:5]:
            if isinstance(item, dict):
                content = item.get("content") if isinstance(item.get("content"), dict) else item
                title = content.get("title") or item.get("title") or ""
                provider = (
                    (content.get("provider") or {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else item.get("publisher", "")
                )
                if title:
                    news_items.append({"title": str(title), "publisher": str(provider or "")})
    except Exception:
        pass

    bundle = {
        "ticker": symbol,
        "info": info,
        "income_stmt": inc,
        "balance_sheet": bal,
        "cashflow": cf,
        "history": hist,
        "news": news_items,
    }
    if use_cache and (info or not hist.empty):
        set_cached(cache_key, bundle)
    return bundle


def fetch_finnhub_enrichment(
    ticker: str, api_key: str, use_cache: bool = True
) -> dict[str, Any]:
    """Fetch supplementary metrics, recommendations, and price targets from Finnhub."""
    if not api_key:
        return {}
    symbol = ticker.strip().upper()
    cache_key = f"finnhub_v1:{symbol}"
    if use_cache:
        cached = get_cached(cache_key, ttl_seconds=21600)
        if cached is not None:
            return cached

    base_url = "https://finnhub.io/api/v1"
    out: dict[str, Any] = {}
    try:
        r_metric = requests.get(
            f"{base_url}/stock/metric",
            params={"symbol": symbol, "metric": "all", "token": api_key},
            timeout=6,
        )
        if r_metric.status_code == 200:
            metric = r_metric.json().get("metric", {}) or {}
            out["fh_pe_ttm"] = metric.get("peBasicExclExtraTTM")
            out["fh_roic_ttm"] = metric.get("roicTTM")
            out["fh_roe_ttm"] = metric.get("roeTTM")
            out["fh_52w_high"] = metric.get("52WeekHigh")
            out["fh_52w_low"] = metric.get("52WeekLow")
            out["fh_beta"] = metric.get("beta")

        r_rec = requests.get(
            f"{base_url}/stock/recommendation",
            params={"symbol": symbol, "token": api_key},
            timeout=6,
        )
        if r_rec.status_code == 200 and isinstance(r_rec.json(), list) and r_rec.json():
            latest = r_rec.json()[0]
            out["fh_rec_buy"] = (latest.get("strongBuy", 0) or 0) + (latest.get("buy", 0) or 0)
            out["fh_rec_hold"] = latest.get("hold", 0) or 0
            out["fh_rec_sell"] = (latest.get("sell", 0) or 0) + (latest.get("strongSell", 0) or 0)
            out["fh_rec_period"] = latest.get("period")

        r_pt = requests.get(
            f"{base_url}/stock/price-target",
            params={"symbol": symbol, "token": api_key},
            timeout=6,
        )
        if r_pt.status_code == 200 and isinstance(r_pt.json(), dict):
            pt = r_pt.json()
            out["fh_target_mean"] = pt.get("targetMean")
            out["fh_target_high"] = pt.get("targetHigh")
            out["fh_target_low"] = pt.get("targetLow")
    except Exception:
        pass

    if use_cache and out:
        set_cached(cache_key, out)
    return out


def fetch_finviz_enrichment(ticker: str, use_cache: bool = True) -> dict[str, Any]:
    """Fetch snapshot fundamentals from Finviz via finvizfinance (best-effort)."""
    symbol = ticker.strip().upper()
    cache_key = f"finviz_v1:{symbol}"
    if use_cache:
        cached = get_cached(cache_key, ttl_seconds=21600)
        if cached is not None:
            return cached

    out: dict[str, Any] = {}
    try:
        from finvizfinance.quote import finvizfinance

        stock = finvizfinance(symbol)
        fund = stock.ticker_fundament() or {}
        for key_src, key_dst in [
            ("Insider Own", "fv_insider_own"),
            ("Inst Own", "fv_inst_own"),
            ("Short Float", "fv_short_float"),
            ("Target Price", "fv_target_price"),
            ("Recom", "fv_recom"),
            ("PEG", "fv_peg"),
            ("RSI (14)", "fv_rsi14"),
        ]:
            if key_src in fund and fund[key_src] not in (None, "-", ""):
                out[key_dst] = fund[key_src]
    except Exception:
        pass

    if use_cache and out:
        set_cached(cache_key, out)
    return out


def load_universe_data(
    tickers: list[str],
    use_cache: bool = True,
    finnhub_key: str | None = None,
    use_finviz: bool = False,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch data bundles for a list of tickers concurrently."""
    clean_tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    results: dict[str, dict[str, Any]] = {}
    total = len(clean_tickers)
    if total == 0:
        return results

    def _worker(sym: str) -> tuple[str, dict[str, Any]]:
        bundle = fetch_yfinance_bundle(sym, use_cache=use_cache)
        if finnhub_key:
            bundle["finnhub"] = fetch_finnhub_enrichment(sym, finnhub_key, use_cache=use_cache)
        else:
            bundle["finnhub"] = {}
        if use_finviz:
            bundle["finviz"] = fetch_finviz_enrichment(sym, use_cache=use_cache)
        else:
            bundle["finviz"] = {}
        return sym, bundle

    completed = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_worker, sym): sym for sym in clean_tickers}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                s, bundle = fut.result()
                results[s] = bundle
            except Exception:
                results[sym] = {"ticker": sym, "info": {}, "error": True}
            completed += 1
            if progress_cb is not None:
                progress_cb(completed, total, sym)

    return results
