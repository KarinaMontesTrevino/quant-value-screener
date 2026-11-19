"""Cross-sectional quantitative scoring, winsorization, and factor ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

# (column_name, higher_is_better, must_be_positive_to_score)
VALUE_FACTORS: list[tuple[str, bool, bool]] = [
    ("ev_ebit", False, True),
    ("ev_ebitda", False, True),
    ("fcf_yield", True, False),
    ("pe_forward", False, True),
    ("pb", False, True),
    ("shareholder_yield", True, False),
    ("price_2y_pct", False, False),
]

QUALITY_FACTORS: list[tuple[str, bool, bool]] = [
    ("roic", True, False),
    ("roic_3y_avg", True, False),
    ("roe", True, False),
    ("gross_profitability", True, False),
    ("op_margin", True, False),
    ("fcf_margin", True, False),
]

SAFETY_FACTORS: list[tuple[str, bool, bool]] = [
    ("f_score", True, False),
    ("net_debt_ebitda", False, False),
    ("interest_cov", True, False),
    ("current_ratio", True, False),
    ("z_score", True, False),
]

MOMENTUM_FACTORS: list[tuple[str, bool, bool]] = [
    ("ret_6m", True, False),
    ("ret_12m_1m", True, False),
    ("price_vs_sma200", True, False),
]


def _winsorize(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 5:
        return series
    lo, hi = valid.quantile(lower_q), valid.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def _percentile_score(
    series: pd.Series,
    higher_is_better: bool,
    must_be_positive: bool = False,
) -> pd.Series:
    """Convert raw metric to 0-100 percentile score. Negative multiples get 0 when must_be_positive=True."""
    s = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=s.index, dtype=float)

    if must_be_positive:
        pos_mask = s > 0
        neg_mask = s <= 0
        valid = _winsorize(s[pos_mask])
        if len(valid) >= 1:
            pct = valid.rank(pct=True, ascending=higher_is_better) * 100.0
            result.loc[pos_mask] = pct
        result.loc[neg_mask] = 0.0
    else:
        valid = _winsorize(s.dropna())
        if len(valid) >= 1:
            pct = valid.rank(pct=True, ascending=higher_is_better) * 100.0
            result.loc[valid.index] = pct

    return result


def _compute_pillar_score(
    df: pd.DataFrame,
    factors: list[tuple[str, bool, bool]],
    sector_neutral: bool = False,
) -> pd.Series:
    """Compute average 0-100 percentile score across a pillar's factors."""
    factor_scores: list[pd.Series] = []
    for col, hib, pos_only in factors:
        if col not in df.columns:
            continue
        if sector_neutral and "sector" in df.columns:
            col_score = pd.Series(np.nan, index=df.index, dtype=float)
            for _, idx in df.groupby("sector").groups.items():
                if len(idx) >= 3:
                    col_score.loc[idx] = _percentile_score(df.loc[idx, col], hib, pos_only)
                else:
                    # Fallback to universe-wide for tiny sectors
                    col_score.loc[idx] = _percentile_score(df[col], hib, pos_only).loc[idx]
        else:
            col_score = _percentile_score(df[col], hib, pos_only)
        factor_scores.append(col_score)

    if not factor_scores:
        return pd.Series(50.0, index=df.index)
    mat = pd.concat(factor_scores, axis=1)
    return mat.mean(axis=1, skipna=True).fillna(50.0)


def score_universe(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    sector_neutral: bool = False,
) -> pd.DataFrame:
    """Add Value, Quality, Safety, Momentum, and Composite Quant Scores (0-100) to DataFrame."""
    if df.empty:
        return df

    out = df.copy()
    w = weights or {"value": 0.40, "quality": 0.30, "safety": 0.20, "momentum": 0.10}
    total_w = sum(w.values()) or 1.0
    wn = {k: v / total_w for k, v in w.items()}

    out["value_score"] = _compute_pillar_score(out, VALUE_FACTORS, sector_neutral)
    out["quality_score"] = _compute_pillar_score(out, QUALITY_FACTORS, sector_neutral)
    out["safety_score"] = _compute_pillar_score(out, SAFETY_FACTORS, sector_neutral)
    out["momentum_score"] = _compute_pillar_score(out, MOMENTUM_FACTORS, sector_neutral)

    out["quant_score"] = (
        out["value_score"] * wn["value"]
        + out["quality_score"] * wn["quality"]
        + out["safety_score"] * wn["safety"]
        + out["momentum_score"] * wn["momentum"]
    )

    out = out.sort_values("quant_score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
