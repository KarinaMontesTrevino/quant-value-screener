"""Quant Value, Quality, Safety (Piotroski F-Score, Altman Z, Beneish M), Momentum, Consistency, and DCF."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_div(num: float | None, den: float | None) -> float | None:
    """Safely divide two numbers, returning None if invalid or zero denominator."""
    if num is None or den is None:
        return None
    if pd.isna(num) or pd.isna(den) or den == 0:
        return None
    return float(num) / float(den)


def _get_row(df: pd.DataFrame, names: list[str], col_idx: int = 0) -> float | None:
    """Extract a scalar float from a financial statement DataFrame by candidate row names."""
    if df is None or df.empty or df.shape[1] <= col_idx:
        return None
    index_lower = {str(idx).strip().lower(): idx for idx in df.index}
    for name in names:
        key = name.strip().lower()
        if key in index_lower:
            val = df.loc[index_lower[key]].iloc[col_idx]
            if pd.notna(val):
                return float(val)
    return None


def compute_piotroski_f_score(
    inc: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame
) -> tuple[int | None, list[dict[str, Any]]]:
    """Compute Piotroski F-Score (0-9) and return per-criterion breakdown."""
    criteria: list[dict[str, Any]] = []

    ni_0 = _get_row(inc, ["Net Income", "Net Income Common Stockholders"], 0)
    ni_1 = _get_row(inc, ["Net Income", "Net Income Common Stockholders"], 1)
    ta_0 = _get_row(bal, ["Total Assets"], 0)
    ta_1 = _get_row(bal, ["Total Assets"], 1)
    cfo_0 = _get_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0)

    roa_0 = safe_div(ni_0, ta_0)
    roa_1 = safe_div(ni_1, ta_1)

    ltd_0 = _get_row(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
    ltd_1 = _get_row(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1)
    lev_0 = safe_div(ltd_0, ta_0)
    lev_1 = safe_div(ltd_1, ta_1)

    ca_0 = _get_row(bal, ["Current Assets", "Total Current Assets"], 0)
    ca_1 = _get_row(bal, ["Current Assets", "Total Current Assets"], 1)
    cl_0 = _get_row(bal, ["Current Liabilities", "Total Current Liabilities"], 0)
    cl_1 = _get_row(bal, ["Current Liabilities", "Total Current Liabilities"], 1)
    cr_0 = safe_div(ca_0, cl_0)
    cr_1 = safe_div(ca_1, cl_1)

    sh_0 = _get_row(bal, ["Ordinary Shares Number", "Share Issued"], 0)
    sh_1 = _get_row(bal, ["Ordinary Shares Number", "Share Issued"], 1)

    gp_0 = _get_row(inc, ["Gross Profit"], 0)
    gp_1 = _get_row(inc, ["Gross Profit"], 1)
    rev_0 = _get_row(inc, ["Total Revenue", "Operating Revenue"], 0)
    rev_1 = _get_row(inc, ["Total Revenue", "Operating Revenue"], 1)

    gm_0 = safe_div(gp_0, rev_0)
    gm_1 = safe_div(gp_1, rev_1)
    at_0 = safe_div(rev_0, ta_0)
    at_1 = safe_div(rev_1, ta_1)

    def _add(name: str, category: str, passed: bool | None, detail: str) -> None:
        criteria.append(
            {"criterion": name, "category": category, "passed": passed, "detail": detail}
        )

    _add("1. ROA > 0", "Rentabilidad", roa_0 > 0 if roa_0 is not None else None, f"ROA: {roa_0:.2%}" if roa_0 is not None else "N/A")
    _add("2. Operating Cash Flow > 0", "Rentabilidad", cfo_0 > 0 if cfo_0 is not None else None, f"CFO: {cfo_0:,.0f}" if cfo_0 is not None else "N/A")
    _add("3. ΔROA > 0", "Rentabilidad", (roa_0 > roa_1) if (roa_0 is not None and roa_1 is not None) else None, f"{roa_1:.2%} → {roa_0:.2%}" if (roa_0 is not None and roa_1 is not None) else "N/A")
    _add("4. Accruals (CFO > Net Income)", "Rentabilidad", (cfo_0 > ni_0) if (cfo_0 is not None and ni_0 is not None) else None, "Calidad de utilidades")
    _add("5. ΔApalancamiento ≤ 0", "Solidez", (lev_0 <= lev_1) if (lev_0 is not None and lev_1 is not None) else None, f"{lev_1:.2f} → {lev_0:.2f}" if (lev_0 is not None and lev_1 is not None) else "N/A")
    _add("6. ΔLiquidez (Current Ratio) > 0", "Solidez", (cr_0 > cr_1) if (cr_0 is not None and cr_1 is not None) else None, f"{cr_1:.2f} → {cr_0:.2f}" if (cr_0 is not None and cr_1 is not None) else "N/A")
    _add("7. Sin dilución de acciones", "Solidez", (sh_0 <= sh_1 * 1.005) if (sh_0 is not None and sh_1 is not None) else None, "Acciones emitidas estables o menores")
    _add("8. ΔMargen Bruto > 0", "Eficiencia", (gm_0 > gm_1) if (gm_0 is not None and gm_1 is not None) else None, f"{gm_1:.2%} → {gm_0:.2%}" if (gm_0 is not None and gm_1 is not None) else "N/A")
    _add("9. ΔRotación de Activos > 0", "Eficiencia", (at_0 > at_1) if (at_0 is not None and at_1 is not None) else None, f"{at_1:.2f} → {at_0:.2f}" if (at_0 is not None and at_1 is not None) else "N/A")

    valid_checks = [c["passed"] for c in criteria if c["passed"] is not None]
    if len(valid_checks) < 5:
        return None, criteria
    score = sum(1 for v in valid_checks if v)
    return score, criteria


def compute_altman_z_score(
    info: dict[str, Any], inc: pd.DataFrame, bal: pd.DataFrame
) -> float | None:
    """Compute Altman Z-Score for non-financial companies."""
    sector = str(info.get("sector", "")).lower()
    if "financial" in sector:
        return None

    ta = _get_row(bal, ["Total Assets"], 0)
    ca = _get_row(bal, ["Current Assets", "Total Current Assets"], 0)
    cl = _get_row(bal, ["Current Liabilities", "Total Current Liabilities"], 0)
    re = _get_row(bal, ["Retained Earnings"], 0)
    ebit = _get_row(inc, ["EBIT", "Operating Income"], 0)
    tl = _get_row(bal, ["Total Liabilities Net Minority Interest", "Total Liabilities"], 0)
    rev = _get_row(inc, ["Total Revenue", "Operating Revenue"], 0)
    mcap = info.get("marketCap")

    if any(v is None for v in [ta, ca, cl, re, ebit, tl, rev, mcap]) or ta <= 0 or tl <= 0:
        return None

    wc = ca - cl
    x1 = wc / ta
    x2 = re / ta
    x3 = ebit / ta
    x4 = float(mcap) / tl
    x5 = rev / ta
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def compute_beneish_m_score(
    info: dict[str, Any], inc: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame
) -> float | None:
    """Compute Beneish M-Score (> -1.78 indicates higher probability of earnings manipulation)."""
    sector = str(info.get("sector", "")).lower()
    if "financial" in sector:
        return None

    rec_0 = _get_row(bal, ["Accounts Receivable", "Receivables"], 0)
    rec_1 = _get_row(bal, ["Accounts Receivable", "Receivables"], 1)
    rev_0 = _get_row(inc, ["Total Revenue", "Operating Revenue"], 0)
    rev_1 = _get_row(inc, ["Total Revenue", "Operating Revenue"], 1)
    gp_0 = _get_row(inc, ["Gross Profit"], 0)
    gp_1 = _get_row(inc, ["Gross Profit"], 1)
    ta_0 = _get_row(bal, ["Total Assets"], 0)
    ta_1 = _get_row(bal, ["Total Assets"], 1)
    ca_0 = _get_row(bal, ["Current Assets", "Total Current Assets"], 0)
    ca_1 = _get_row(bal, ["Current Assets", "Total Current Assets"], 1)
    ppe_0 = _get_row(bal, ["Net PPE", "Property Plant Equipment Net"], 0)
    ppe_1 = _get_row(bal, ["Net PPE", "Property Plant Equipment Net"], 1)
    dep_0 = _get_row(cf, ["Depreciation And Amortization", "Depreciation"], 0)
    dep_1 = _get_row(cf, ["Depreciation And Amortization", "Depreciation"], 1)
    sga_0 = _get_row(inc, ["Selling General And Administration"], 0)
    sga_1 = _get_row(inc, ["Selling General And Administration"], 1)
    ltd_0 = _get_row(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0) or 0.0
    ltd_1 = _get_row(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1) or 0.0
    cl_0 = _get_row(bal, ["Current Liabilities", "Total Current Liabilities"], 0) or 0.0
    cl_1 = _get_row(bal, ["Current Liabilities", "Total Current Liabilities"], 1) or 0.0
    ni_0 = _get_row(inc, ["Net Income", "Net Income Common Stockholders"], 0)
    cfo_0 = _get_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0)

    if any(v is None or v <= 0 for v in [rev_0, rev_1, ta_0, ta_1]):
        return None

    dsri = safe_div(safe_div(rec_0, rev_0), safe_div(rec_1, rev_1)) if (rec_0 is not None and rec_1 is not None) else 1.0
    gmi = safe_div(safe_div(gp_1, rev_1), safe_div(gp_0, rev_0)) if (gp_0 is not None and gp_1 is not None) else 1.0

    aqi_0 = safe_div(ta_0 - (ca_0 or 0.0) - (ppe_0 or 0.0), ta_0) if (ca_0 is not None and ppe_0 is not None) else None
    aqi_1 = safe_div(ta_1 - (ca_1 or 0.0) - (ppe_1 or 0.0), ta_1) if (ca_1 is not None and ppe_1 is not None) else None
    aqi = safe_div(aqi_0, aqi_1) if (aqi_0 is not None and aqi_1 is not None and aqi_1 > 0) else 1.0

    sgi = safe_div(rev_0, rev_1) or 1.0

    depi_0 = safe_div(dep_0, (dep_0 or 0.0) + (ppe_0 or 0.0)) if (dep_0 is not None and ppe_0 is not None) else None
    depi_1 = safe_div(dep_1, (dep_1 or 0.0) + (ppe_1 or 0.0)) if (dep_1 is not None and ppe_1 is not None) else None
    depi = safe_div(depi_1, depi_0) if (depi_0 is not None and depi_1 is not None and depi_0 > 0) else 1.0

    sgai = safe_div(safe_div(sga_0, rev_0), safe_div(sga_1, rev_1)) if (sga_0 is not None and sga_1 is not None) else 1.0
    lvgi = safe_div((ltd_0 + cl_0) / ta_0, (ltd_1 + cl_1) / ta_1) if ((ltd_1 + cl_1) > 0) else 1.0
    tata = safe_div((ni_0 - cfo_0), ta_0) if (ni_0 is not None and cfo_0 is not None) else 0.0

    vals = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
    if any(v is None or pd.isna(v) for v in vals):
        return None

    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )
    return float(m_score)


def compute_multiyear_consistency(
    inc: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame
) -> dict[str, Any]:
    """Compute 3-4 year consistency metrics: positive FCF years, average ROIC, and margin stability."""
    n_cols = min(
        4,
        inc.shape[1] if inc is not None and not inc.empty else 0,
        bal.shape[1] if bal is not None and not bal.empty else 0,
        cf.shape[1] if cf is not None and not cf.empty else 0,
    )
    if n_cols == 0:
        return {"fcf_pos_years": None, "fcf_total_years": 0, "roic_3y_avg": None, "op_margin_std": None}

    fcf_pos = 0
    fcf_count = 0
    roics: list[float] = []
    op_margins: list[float] = []

    for i in range(n_cols):
        cfo = _get_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], i)
        capex = _get_row(cf, ["Capital Expenditure"], i)
        if cfo is not None and capex is not None:
            fcf_i = cfo - abs(capex)
            fcf_count += 1
            if fcf_i > 0:
                fcf_pos += 1

        ebit = _get_row(inc, ["EBIT", "Operating Income"], i)
        rev = _get_row(inc, ["Total Revenue", "Operating Revenue"], i)
        eq = _get_row(bal, ["Stockholders Equity", "Total Equity Gross Minority Interest"], i)
        debt = _get_row(bal, ["Total Debt"], i) or 0.0
        cash = _get_row(bal, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], i) or 0.0
        if ebit is not None and eq is not None and (eq + debt - cash) > 0:
            roics.append((ebit * 0.79) / (eq + debt - cash))
        if ebit is not None and rev is not None and rev > 0:
            op_margins.append(ebit / rev)

    return {
        "fcf_pos_years": fcf_pos if fcf_count > 0 else None,
        "fcf_total_years": fcf_count,
        "roic_3y_avg": float(np.mean(roics)) if roics else None,
        "op_margin_std": float(np.std(op_margins)) if len(op_margins) >= 2 else None,
    }


def compute_dcf(
    fcf: float | None,
    net_debt: float | None,
    shares: float | None,
    current_price: float | None,
    discount_rate: float = 0.10,
    growth_rate: float = 0.05,
    terminal_growth: float = 0.025,
    years: int = 5,
) -> tuple[float | None, float | None]:
    """Compute conservative 5Y DCF intrinsic value per share and Margin of Safety (%)."""
    if (
        fcf is None
        or fcf <= 0
        or shares is None
        or shares <= 0
        or current_price is None
        or current_price <= 0
        or discount_rate <= terminal_growth
    ):
        return None, None

    nd = net_debt or 0.0
    pv_fcf = 0.0
    projected_fcf = fcf
    for yr in range(1, years + 1):
        projected_fcf *= 1.0 + growth_rate
        pv_fcf += projected_fcf / ((1.0 + discount_rate) ** yr)

    terminal_val = (projected_fcf * (1.0 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_val / ((1.0 + discount_rate) ** years)
    equity_value = pv_fcf + pv_terminal - nd
    intrinsic_price = equity_value / shares
    if intrinsic_price <= 0:
        return None, None
    mos = (intrinsic_price - current_price) / intrinsic_price
    return intrinsic_price, mos


def compute_ticker_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute full Quant Value, Quality, Safety, Momentum, Consistency, and DCF metrics for a ticker."""
    ticker = bundle.get("ticker", "")
    info = bundle.get("info", {}) or {}
    inc = bundle.get("income_stmt", pd.DataFrame())
    bal = bundle.get("balance_sheet", pd.DataFrame())
    cf = bundle.get("cashflow", pd.DataFrame())
    hist = bundle.get("history", pd.DataFrame())
    fh = bundle.get("finnhub", {}) or {}
    fv = bundle.get("finviz", {}) or {}
    news = bundle.get("news", []) or []

    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"
    is_financial = "financial" in sector.lower()
    is_reit = "reit" in industry.lower() or "real estate" in sector.lower()

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None and not hist.empty:
        price = float(hist["Close"].iloc[-1])

    mcap = info.get("marketCap")
    ev = info.get("enterpriseValue")
    shares = info.get("sharesOutstanding") or _get_row(bal, ["Ordinary Shares Number", "Share Issued"], 0)

    # Income / Cash Flow / Balance Sheet items
    ebit = _get_row(inc, ["EBIT", "Operating Income"], 0)
    ebitda = info.get("ebitda") or _get_row(inc, ["EBITDA", "Normalized EBITDA"], 0)
    ni = _get_row(inc, ["Net Income", "Net Income Common Stockholders"], 0)
    gp = _get_row(inc, ["Gross Profit"], 0)
    rev_0 = _get_row(inc, ["Total Revenue", "Operating Revenue"], 0)
    rev_1 = _get_row(inc, ["Total Revenue", "Operating Revenue"], 1)
    int_exp = _get_row(inc, ["Interest Expense", "Interest Expense Non Operating"], 0)
    da = _get_row(cf, ["Depreciation And Amortization", "Depreciation"], 0)

    cfo = _get_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0)
    capex = _get_row(cf, ["Capital Expenditure"], 0)
    fcf = info.get("freeCashflow")
    if fcf is None and cfo is not None and capex is not None:
        fcf = cfo - abs(capex)

    total_debt = info.get("totalDebt") or _get_row(bal, ["Total Debt"], 0) or 0.0
    cash = info.get("totalCash") or _get_row(bal, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], 0) or 0.0
    net_debt = total_debt - cash
    equity = _get_row(bal, ["Stockholders Equity", "Total Equity Gross Minority Interest"], 0)
    ta = _get_row(bal, ["Total Assets"], 0)
    ca = _get_row(bal, ["Current Assets", "Total Current Assets"], 0)
    cl = _get_row(bal, ["Current Liabilities", "Total Current Liabilities"], 0)

    # Value Multiples
    ev_ebit = None if is_financial else safe_div(ev, ebit)
    ev_ebitda = None if is_financial else (info.get("enterpriseToEbitda") or safe_div(ev, ebitda))
    fcf_yield = safe_div(fcf, mcap)
    pe_trailing = info.get("trailingPE")
    pe_forward = info.get("forwardPE")
    pb = info.get("priceToBook")
    ps = info.get("priceToSalesTrailing12Months")
    p_ffo = safe_div(mcap, (ni or 0.0) + (da or 0.0)) if (is_reit and ni is not None and da is not None and (ni + da) > 0) else None

    div_yield = info.get("dividendYield")
    if div_yield is not None and div_yield > 0.25:
        div_yield = div_yield / 100.0

    # Net Buyback Yield
    repurchase = _get_row(cf, ["Repurchase Of Capital Stock", "Common Stock Payments"], 0)
    issuance = _get_row(cf, ["Issuance Of Capital Stock", "Common Stock Issuance"], 0)
    net_buyback = 0.0
    if repurchase is not None:
        net_buyback += abs(repurchase)
    if issuance is not None:
        net_buyback -= abs(issuance)
    buyback_yield = safe_div(net_buyback, mcap) if (repurchase is not None or issuance is not None) else 0.0
    shareholder_yield = (div_yield or 0.0) + (buyback_yield or 0.0) if (div_yield is not None or buyback_yield is not None) else None

    # Quality Metrics
    invested_capital = (equity + total_debt - cash) if equity is not None else None
    nopat = ebit * 0.79 if ebit is not None else None
    roic = None if is_financial else (safe_div(nopat, invested_capital) if (invested_capital is not None and invested_capital > 0) else None)
    roe = info.get("returnOnEquity") or safe_div(ni, equity)
    gross_profitability = safe_div(gp, ta)
    op_margin = info.get("operatingMargins") or safe_div(ebit, rev_0)
    fcf_margin = safe_div(fcf, rev_0)
    fcf_conversion = safe_div(fcf, ni) if (ni is not None and ni > 0) else None
    rev_growth = safe_div(rev_0 - rev_1, abs(rev_1)) if (rev_0 is not None and rev_1 is not None and rev_1 != 0) else None

    # Multi-Year Consistency
    consistency = compute_multiyear_consistency(inc, bal, cf)

    # Safety & Financial Health
    f_score, f_details = compute_piotroski_f_score(inc, bal, cf)
    z_score = compute_altman_z_score(info, inc, bal)
    m_score = compute_beneish_m_score(info, inc, bal, cf)
    net_debt_ebitda = None if is_financial else (safe_div(net_debt, ebitda) if (ebitda is not None and ebitda > 0) else None)
    interest_cov = None if is_financial else (safe_div(ebit, abs(int_exp)) if (int_exp is not None and int_exp != 0) else None)
    current_ratio = info.get("currentRatio") or safe_div(ca, cl)

    # Momentum & Own-History Valuation Band (2Y Price percentile: lower = cheaper vs own 2Y range)
    ret_6m = None
    ret_12m_1m = None
    dist_52w_high = None
    price_vs_sma200 = None
    price_2y_pct = None
    if not hist.empty and len(hist) > 20:
        close = hist["Close"].dropna()
        last_p = float(close.iloc[-1])
        price_2y_pct = float((close <= last_p).mean())
        if len(close) >= 126:
            ret_6m = (last_p / float(close.iloc[-126])) - 1.0
        if len(close) >= 252 and len(close) > 21:
            ret_12m_1m = (float(close.iloc[-21]) / float(close.iloc[-252])) - 1.0
        high_52w = float(close.tail(252).max())
        if high_52w > 0:
            dist_52w_high = (last_p / high_52w) - 1.0
        if len(close) >= 200:
            sma200 = float(close.tail(200).mean())
            if sma200 > 0:
                price_vs_sma200 = (last_p / sma200) - 1.0

    # Conservative DCF
    dcf_price, mos = compute_dcf(fcf, net_debt, shares, price)

    # Red Flags (Value Trap Detector)
    red_flags: list[str] = []
    if f_score is not None and f_score <= 3:
        red_flags.append(f"F-Score bajo ({f_score}/9)")
    if fcf is not None and fcf < 0 and not is_financial:
        red_flags.append("FCF negativo")
    if net_debt_ebitda is not None and net_debt_ebitda > 4.0 and not is_reit:
        red_flags.append(f"Deuda alta ({net_debt_ebitda:.1f}x EBITDA)")
    if rev_growth is not None and rev_growth < -0.10:
        red_flags.append(f"Caída de ingresos ({rev_growth:.1%})")
    if interest_cov is not None and interest_cov < 2.0:
        red_flags.append(f"Cobertura intereses baja ({interest_cov:.1f}x)")
    if m_score is not None and m_score > -1.78:
        red_flags.append(f"Beneish M-Score alerta ({m_score:.2f} > -1.78)")

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": sector,
        "industry": industry,
        "is_financial": is_financial,
        "is_reit": is_reit,
        "price": price,
        "market_cap": mcap,
        "ev": ev,
        "fcf": fcf,
        "net_debt": net_debt,
        "shares": shares,
        # Value
        "ev_ebit": ev_ebit,
        "ev_ebitda": ev_ebitda,
        "fcf_yield": fcf_yield,
        "pe_trailing": pe_trailing,
        "pe_forward": pe_forward,
        "pb": pb,
        "ps": ps,
        "p_ffo": p_ffo,
        "div_yield": div_yield,
        "shareholder_yield": shareholder_yield,
        "price_2y_pct": price_2y_pct,
        # Quality & Consistency
        "roic": roic,
        "roic_3y_avg": consistency["roic_3y_avg"],
        "roe": roe,
        "gross_profitability": gross_profitability,
        "op_margin": op_margin,
        "op_margin_std": consistency["op_margin_std"],
        "fcf_margin": fcf_margin,
        "fcf_conversion": fcf_conversion,
        "fcf_pos_years": consistency["fcf_pos_years"],
        "fcf_total_years": consistency["fcf_total_years"],
        "rev_growth": rev_growth,
        # Safety
        "f_score": f_score,
        "f_details": f_details,
        "z_score": z_score,
        "m_score": m_score,
        "net_debt_ebitda": net_debt_ebitda,
        "interest_cov": interest_cov,
        "current_ratio": current_ratio,
        # Momentum
        "ret_6m": ret_6m,
        "ret_12m_1m": ret_12m_1m,
        "dist_52w_high": dist_52w_high,
        "price_vs_sma200": price_vs_sma200,
        # DCF & Red Flags
        "dcf_price": dcf_price,
        "margin_of_safety": mos,
        "red_flags": red_flags,
        "red_flag_count": len(red_flags),
        # Enrichment & News
        "fh_target_mean": fh.get("fh_target_mean"),
        "fh_rec_buy": fh.get("fh_rec_buy"),
        "fh_rec_hold": fh.get("fh_rec_hold"),
        "fh_rec_sell": fh.get("fh_rec_sell"),
        "fv_insider_own": fv.get("fv_insider_own"),
        "fv_short_float": fv.get("fv_short_float"),
        "fv_target_price": fv.get("fv_target_price"),
        "fv_recom": fv.get("fv_recom"),
        "news": news,
    }
