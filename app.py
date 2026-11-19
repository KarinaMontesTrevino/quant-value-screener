"""Streamlit Quant Value Screener, Backtester, Portfolio Builder & AI Analyst."""

from __future__ import annotations

import io
import os
import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ai_analyst import INVESTOR_PERSONAS, analyze_stock_with_ai, compare_stocks_with_ai
from src.backtest import run_top_n_backtest
from src.cache import clear_cache
from src.gs_analytics import compute_gs_risk_metrics
from src.data_providers import (
    FINVIZ_SCREENER_PRESETS,
    PRESET_UNIVERSES,
    load_universe_data,
    run_finviz_screener,
)
from src.metrics import compute_dcf, compute_ticker_metrics
from src.portfolio import (
    compute_portfolio_allocation,
    delete_watchlist,
    load_watchlists,
    save_watchlist,
)
from src.scoring import score_universe

st.set_page_config(
    page_title="Quant Value Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Quant Value Screener & Stock Evaluator")
st.caption(
    "Evaluación sistemática de acciones: **Value**, **Quality**, **Safety (Piotroski F-Score / Altman Z / Beneish M)**, "
    "**Momentum**, **Backtest** y **Comité IA**."
)

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Universo & Datos")
    saved_watchlists = load_watchlists()
    saved_wl_names = [f"⭐ Watchlist: {k}" for k in saved_watchlists.keys()]
    universe_options = (
        ["Personalizado"]
        + list(PRESET_UNIVERSES.keys())
        + list(FINVIZ_SCREENER_PRESETS.keys())
        + saved_wl_names
    )
    preset_name = st.selectbox("Fuente de Universo", options=universe_options, index=1)

    if preset_name in PRESET_UNIVERSES:
        default_tickers = ", ".join(PRESET_UNIVERSES[preset_name])
    elif preset_name in FINVIZ_SCREENER_PRESETS:
        if st.button("🔎 Buscar candidatos en Finviz", use_container_width=True):
            with st.spinner("Consultando Finviz Screener..."):
                found = run_finviz_screener(preset_name, limit=25, use_cache=True)
                if found:
                    st.session_state["finviz_found"] = ", ".join(found)
                else:
                    st.warning("Finviz no devolvió resultados o limitó la petición.")
        default_tickers = st.session_state.get("finviz_found", "BRK-B, JPM, C, BAC, XOM, CVX")
    elif preset_name.startswith("⭐ Watchlist: "):
        wl_key = preset_name.replace("⭐ Watchlist: ", "", 1)
        default_tickers = ", ".join(saved_watchlists.get(wl_key, {}).get("tickers", []))
    else:
        default_tickers = "BRK-B, JPM, GOOGL, MSFT, UNH, V, XOM, CVX"

    tickers_input = st.text_area(
        "Tickers (separados por coma, espacio o salto de línea)",
        value=default_tickers,
        height=95,
        help="Acepta cualquier formato: AAPL, MSFT o $AAPL $MSFT. Bolsas internacionales: WALMEX.MX, SAN.MC, HSBA.L.",
    )
    uploaded_file = st.file_uploader(
        "📄 Importar lista (.csv o .txt)",
        type=["csv", "txt"],
        help="En CSV detecta la columna 'Ticker'/'Symbol' o usa la primera columna.",
    )

    def _parse_tickers(raw: str) -> list[str]:
        tokens = re.split(r"[\s,;|\t\n]+", raw or "")
        return list(dict.fromkeys(tok.strip().lstrip("$").upper() for tok in tokens if tok.strip()))

    tickers = _parse_tickers(tickers_input)
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            if uploaded_file.name.lower().endswith(".csv"):
                up_df = pd.read_csv(io.StringIO(raw_bytes))
                col_match = next(
                    (c for c in up_df.columns if str(c).strip().lower() in {"ticker", "symbol", "tickers", "symbols"}),
                    up_df.columns[0] if len(up_df.columns) > 0 else None,
                )
                if col_match is not None:
                    file_tickers = _parse_tickers(" ".join(up_df[col_match].dropna().astype(str).tolist()))
                    tickers = list(dict.fromkeys(file_tickers))
            else:
                tickers = _parse_tickers(raw_bytes)
            st.caption(f"✅ {len(tickers)} tickers cargados desde `{uploaded_file.name}`.")
        except Exception as exc:
            st.warning(f"No se pudo leer el archivo: {exc}")

    with st.expander("⭐ Guardar / Gestionar Watchlist", expanded=False):
        wl_name = st.text_input("Nombre de Watchlist", placeholder="Ej. Mi Cartera Value")
        wl_notes = st.text_input("Notas / Tesis breve", placeholder="Ej. Compounders con ROIC > 15%")
        wc1, wc2 = st.columns(2)
        with wc1:
            if st.button("💾 Guardar", use_container_width=True):
                if wl_name.strip() and tickers:
                    save_watchlist(wl_name, tickers, wl_notes)
                    st.toast(f"Watchlist '{wl_name}' guardada.")
                    st.rerun()
        with wc2:
            if preset_name.startswith("⭐ Watchlist: ") and st.button("🗑️ Borrar", use_container_width=True):
                delete_watchlist(preset_name.replace("⭐ Watchlist: ", "", 1))
                st.toast("Watchlist eliminada.")
                st.rerun()

    with st.expander("🔌 Fuentes adicionales & IA", expanded=False):
        gemini_key = st.text_input(
            "Gemini API Key (para Analista IA)",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Habilita el Comité de Value Investing y Comparador con IA.",
        )
        finnhub_key = st.text_input(
            "Finnhub API Key (opcional)",
            value=os.environ.get("FINNHUB_API_KEY", ""),
            type="password",
            help="Enriquece con consenso de analistas y Price Targets de Finnhub.",
        )
        use_finviz = st.checkbox(
            "Enriquecer con Finviz (Short Float, Insider Own, Target)",
            value=False,
        )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_cache = st.checkbox("Usar caché (6h)", value=True)
    with col_c2:
        if st.button("🗑️ Limpiar caché"):
            n = clear_cache()
            st.toast(f"Caché limpiado ({n} archivos).")

    run_btn = st.button("🚀 Analizar Universo", type="primary", use_container_width=True)

    st.divider()
    st.header("⚖️ Ponderación Quant")
    w_val = st.slider("Value (%)", 0, 100, 40, 5)
    w_qual = st.slider("Quality (%)", 0, 100, 30, 5)
    w_safe = st.slider("Safety (%)", 0, 100, 20, 5)
    w_mom = st.slider("Momentum (%)", 0, 100, 10, 5)
    sector_neutral = st.checkbox(
        "Normalizar por sector (Sector-Neutral)",
        value=False,
        help="Compara múltiplos y calidad dentro de cada sector cuando hay ≥3 empresas.",
    )

    st.divider()
    st.header("🔍 Filtros Screener")
    min_fscore = st.slider("Piotroski F-Score mínimo", 0, 9, 0)
    max_ev_ebit = st.number_input("EV/EBIT máximo (0 = sin límite)", min_value=0.0, value=0.0, step=1.0)
    min_roic = st.number_input("ROIC mínimo (%)", min_value=-100.0, value=-100.0, step=2.0)
    exclude_red_flags = st.checkbox("Excluir acciones con Banderas Rojas", value=False)


# ---------------------------------------------------------------------------
# Data Loading & Caching in Session State
# ---------------------------------------------------------------------------
if "bundles" not in st.session_state:
    st.session_state["bundles"] = {}

if run_btn or (not st.session_state["bundles"] and tickers):
    if not tickers:
        st.warning("Ingresa al menos un ticker válido.")
        st.stop()
    progress_bar = st.progress(0, text="Descargando datos...")

    def _on_progress(done: int, total: int, sym: str) -> None:
        progress_bar.progress(done / total, text=f"Procesando {sym} ({done}/{total})...")

    bundles = load_universe_data(
        tickers=tickers,
        use_cache=use_cache,
        finnhub_key=finnhub_key or None,
        use_finviz=use_finviz,
        progress_cb=_on_progress,
    )
    progress_bar.empty()
    st.session_state["bundles"] = bundles

bundles: dict[str, dict[str, Any]] = st.session_state["bundles"]
if not bundles:
    st.info("Haz clic en **🚀 Analizar Universo** en la barra lateral para comenzar.")
    st.stop()

rows = [
    compute_ticker_metrics(b)
    for b in bundles.values()
    if b.get("info") or not b.get("history", pd.DataFrame()).empty
]
valid_symbols = {r["ticker"] for r in rows}
missing_symbols = [sym for sym in bundles.keys() if sym not in valid_symbols]
if missing_symbols:
    st.warning(
        f"⚠️ Sin datos para: **{', '.join(missing_symbols)}**. "
        "Verifica el símbolo o agrega el sufijo de bolsa (ej. `.MX`, `.MC`, `.L`)."
    )
if not rows:
    st.error("No se pudieron obtener datos válidos para los tickers ingresados.")
    st.stop()

raw_df = pd.DataFrame(rows)
scored_df = score_universe(
    raw_df,
    weights={"value": w_val, "quality": w_qual, "safety": w_safe, "momentum": w_mom},
    sector_neutral=sector_neutral,
)

# Apply Filters
filtered_df = scored_df.copy()
if min_fscore > 0:
    filtered_df = filtered_df[filtered_df["f_score"].fillna(0) >= min_fscore]
if max_ev_ebit > 0:
    filtered_df = filtered_df[
        (filtered_df["ev_ebit"].notna()) & (filtered_df["ev_ebit"] > 0) & (filtered_df["ev_ebit"] <= max_ev_ebit)
    ]
if min_roic > -100.0:
    filtered_df = filtered_df[filtered_df["roic"].fillna(-999) * 100.0 >= min_roic]
if exclude_red_flags:
    filtered_df = filtered_df[filtered_df["red_flag_count"] == 0]


# ---------------------------------------------------------------------------
# KPI Header
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    top_row = filtered_df.iloc[0] if not filtered_df.empty else None
    st.metric(
        "🏆 #1 Quant Rank",
        f"{top_row['ticker']} ({top_row['quant_score']:.1f})" if top_row is not None else "N/A",
    )
with k2:
    med_ev = filtered_df["ev_ebit"].dropna()
    med_ev_pos = med_ev[med_ev > 0]
    st.metric("Mediana EV/EBIT", f"{med_ev_pos.median():.1f}x" if not med_ev_pos.empty else "N/A")
with k3:
    avg_fs = filtered_df["f_score"].dropna()
    st.metric("F-Score Promedio", f"{avg_fs.mean():.1f} / 9" if not avg_fs.empty else "N/A")
with k4:
    st.metric("Acciones Filtradas", f"{len(filtered_df)} / {len(scored_df)}")


# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Ranking & Screener",
        "🎯 Matriz Value vs Quality",
        "🔬 Deep Dive & IA",
        "📈 Backtest & Portafolio",
    ]
)

with tab1:
    display_cols = [
        "rank", "ticker", "name", "sector", "price",
        "quant_score", "value_score", "quality_score", "safety_score", "momentum_score",
        "ev_ebit", "ev_ebitda", "fcf_yield", "pe_forward", "pb", "p_ffo", "price_2y_pct",
        "shareholder_yield", "roic", "roic_3y_avg", "gross_profitability", "op_margin",
        "f_score", "m_score", "net_debt_ebitda", "ret_6m", "margin_of_safety", "red_flag_count",
    ]
    disp = filtered_df[[c for c in display_cols if c in filtered_df.columns]].copy()

    pct_cols = [
        "fcf_yield", "price_2y_pct", "shareholder_yield", "roic", "roic_3y_avg",
        "gross_profitability", "op_margin", "ret_6m", "margin_of_safety",
    ]
    for c in pct_cols:
        if c in disp.columns:
            disp[c] = disp[c] * 100.0

    col_config = {
        "rank": st.column_config.NumberColumn("Rank", format="%d"),
        "ticker": "Ticker",
        "name": "Empresa",
        "sector": "Sector",
        "price": st.column_config.NumberColumn("Precio", format="$%.2f"),
        "quant_score": st.column_config.ProgressColumn("Quant Score", min_value=0, max_value=100, format="%.1f"),
        "value_score": st.column_config.NumberColumn("Value", format="%.1f"),
        "quality_score": st.column_config.NumberColumn("Quality", format="%.1f"),
        "safety_score": st.column_config.NumberColumn("Safety", format="%.1f"),
        "momentum_score": st.column_config.NumberColumn("Mom.", format="%.1f"),
        "ev_ebit": st.column_config.NumberColumn("EV/EBIT", format="%.1fx"),
        "ev_ebitda": st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
        "fcf_yield": st.column_config.NumberColumn("FCF Yield %", format="%.1f%%"),
        "pe_forward": st.column_config.NumberColumn("Fwd P/E", format="%.1fx"),
        "pb": st.column_config.NumberColumn("P/B", format="%.1fx"),
        "p_ffo": st.column_config.NumberColumn("P/FFO (REIT)", format="%.1fx"),
        "price_2y_pct": st.column_config.NumberColumn("Pctil Precio 2Y", format="%.0f%%"),
        "shareholder_yield": st.column_config.NumberColumn("Sh. Yield %", format="%.1f%%"),
        "roic": st.column_config.NumberColumn("ROIC %", format="%.1f%%"),
        "roic_3y_avg": st.column_config.NumberColumn("ROIC 3Y Avg %", format="%.1f%%"),
        "gross_profitability": st.column_config.NumberColumn("GP/Assets %", format="%.1f%%"),
        "op_margin": st.column_config.NumberColumn("Op. Margin %", format="%.1f%%"),
        "f_score": st.column_config.NumberColumn("F-Score", format="%d/9"),
        "m_score": st.column_config.NumberColumn("Beneish M", format="%.2f"),
        "net_debt_ebitda": st.column_config.NumberColumn("NetDebt/EBITDA", format="%.1fx"),
        "ret_6m": st.column_config.NumberColumn("Ret 6M %", format="%.1f%%"),
        "margin_of_safety": st.column_config.NumberColumn("DCF MoS %", format="%.1f%%"),
        "red_flag_count": st.column_config.NumberColumn("🚩 Flags", format="%d"),
    }

    st.dataframe(
        disp,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    csv_bytes = filtered_df.drop(columns=["f_details", "red_flags", "news"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV completo",
        data=csv_bytes,
        file_name="quant_value_ranking.csv",
        mime="text/csv",
    )

with tab2:
    if filtered_df.empty:
        st.info("No hay acciones que cumplan los filtros actuales.")
    else:
        plot_df = filtered_df.copy()
        plot_df["mcap_b"] = (plot_df["market_cap"].fillna(1e9) / 1e9).clip(lower=1.0)
        plot_df["estado"] = plot_df["red_flag_count"].apply(
            lambda x: "🚩 Con Alertas" if x > 0 else "✅ Limpia"
        )
        fig = px.scatter(
            plot_df,
            x="value_score",
            y="quality_score",
            size="mcap_b",
            color="quant_score",
            hover_name="ticker",
            hover_data={
                "name": True,
                "sector": True,
                "quant_score": ":.1f",
                "ev_ebit": ":.1f",
                "roic": ":.2%",
                "f_score": True,
                "estado": True,
                "mcap_b": False,
            },
            text="ticker",
            color_continuous_scale="Viridis",
            title="Cuadrante Quant: Value Score vs. Quality Score",
            labels={"value_score": "Value Score (0-100)", "quality_score": "Quality Score (0-100)", "quant_score": "Quant Score"},
        )
        fig.update_traces(textposition="top center")
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.6)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.6)
        fig.add_annotation(x=85, y=92, text="💎 Sweet Spot (Barata + Alta Calidad)", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=85, y=10, text="⚠️ Posible Value Trap", showarrow=False, font=dict(color="orange"))
        fig.add_annotation(x=15, y=92, text="📈 Calidad Cara", showarrow=False, font=dict(color="steelblue"))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    ticker_options = scored_df["ticker"].tolist()
    selected_ticker = st.selectbox("Selecciona una empresa para análisis profundo", options=ticker_options)
    row = scored_df[scored_df["ticker"] == selected_ticker].iloc[0]
    bundle = bundles[selected_ticker]

    st.subheader(f"{row['name']} ({row['ticker']}) — {row['sector']} | {row['industry']}")
    if row["red_flags"]:
        st.warning("🚩 **Banderas Rojas detectadas:** " + " • ".join(row["red_flags"]))
    else:
        st.success("✅ Sin banderas rojas cuantitativas detectadas.")

    # Quick Consistency & Accounting Strip
    q1, q2, q3, q4 = st.columns(4)
    fcf_yrs_str = f"{int(row['fcf_pos_years'])}/{int(row['fcf_total_years'])}" if pd.notna(row.get("fcf_pos_years")) else "N/A"
    q1.metric("Años FCF+", fcf_yrs_str)
    q2.metric("ROIC Prom. 3-4Y", f"{row['roic_3y_avg']:.1%}" if pd.notna(row.get("roic_3y_avg")) else "N/A")
    q3.metric("Beneish M-Score", f"{row['m_score']:.2f}" if pd.notna(row.get("m_score")) else "N/A", help="> -1.78 sugiere posible manipulación contable.")
    q4.metric("Percentil Precio 2Y", f"{row['price_2y_pct']:.0%}" if pd.notna(row.get("price_2y_pct")) else "N/A", help="Posición del precio actual vs su propio rango de 2 años.")

    c_left, c_right = st.columns([1, 1])

    with c_left:
        categories = ["Value", "Quality", "Safety", "Momentum"]
        vals = [row["value_score"], row["quality_score"], row["safety_score"], row["momentum_score"]]
        radar = go.Figure(
            data=go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=selected_ticker,
            )
        )
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title=f"Perfil Quant (Composite: {row['quant_score']:.1f}/100)",
            height=360,
        )
        st.plotly_chart(radar, use_container_width=True)

    with c_right:
        st.markdown("#### 🧮 Simulador DCF Conservador")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            wacc = st.number_input("Tasa Descuento (%)", 5.0, 20.0, 10.0, 0.5, key=f"wacc_{selected_ticker}") / 100.0
        with d_col2:
            g_5y = st.number_input("Crecimiento FCF 5Y (%)", -5.0, 25.0, 5.0, 0.5, key=f"g_{selected_ticker}") / 100.0
        with d_col3:
            g_term = st.number_input("Crecimiento Terminal (%)", 0.0, 5.0, 2.5, 0.25, key=f"gt_{selected_ticker}") / 100.0

        dcf_val, dcf_mos = compute_dcf(
            fcf=row["fcf"],
            net_debt=row["net_debt"],
            shares=row["shares"],
            current_price=row["price"],
            discount_rate=wacc,
            growth_rate=g_5y,
            terminal_growth=g_term,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Precio Actual", f"${row['price']:.2f}" if pd.notna(row["price"]) else "N/A")
        m2.metric("Valor Intrínseco DCF", f"${dcf_val:.2f}" if dcf_val else "N/A (FCF ≤ 0)")
        m3.metric("Margen de Seguridad", f"{dcf_mos:.1%}" if dcf_mos is not None else "N/A")

        if any(pd.notna(row.get(k)) for k in ["fh_target_mean", "fv_target_price", "fv_short_float", "fv_insider_own"]):
            st.markdown("#### 📡 Datos Externos (Finnhub / Finviz)")
            e1, e2, e3 = st.columns(3)
            if pd.notna(row.get("fh_target_mean")):
                e1.metric("Finnhub Target", f"${row['fh_target_mean']:.2f}")
            if pd.notna(row.get("fh_rec_buy")):
                e2.metric("Consenso (Buy/Hold/Sell)", f"{int(row['fh_rec_buy'])}/{int(row['fh_rec_hold'])}/{int(row['fh_rec_sell'])}")
            if pd.notna(row.get("fv_short_float")):
                e3.metric("Finviz Short Float", str(row["fv_short_float"]))

    # Piotroski F-Score Checklist
    st.markdown(f"#### 🛡️ Piotroski F-Score: **{row['f_score']}/9**" if pd.notna(row["f_score"]) else "#### 🛡️ Piotroski F-Score: N/A")
    f_df = pd.DataFrame(row["f_details"])
    if not f_df.empty:
        f_df["Estado"] = f_df["passed"].map({True: "✅ Pasa", False: "❌ Falla", None: "⚪ N/A"})
        st.dataframe(
            f_df[["criterion", "category", "Estado", "detail"]].rename(
                columns={"criterion": "Criterio", "category": "Categoría", "detail": "Detalle"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    # Price Chart + SMAs
    hist = bundle.get("history", pd.DataFrame())
    if not hist.empty:
        hist_plot = hist.copy()
        hist_plot["SMA50"] = hist_plot["Close"].rolling(50).mean()
        hist_plot["SMA200"] = hist_plot["Close"].rolling(200).mean()
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot["Close"], name="Precio", line=dict(width=2)))
        pfig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot["SMA50"], name="SMA 50", line=dict(dash="dot")))
        pfig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot["SMA200"], name="SMA 200", line=dict(dash="dash")))
        pfig.update_layout(title=f"{selected_ticker} — Precio 2Y y Medias Móviles", height=380, xaxis_title="Fecha", yaxis_title="USD")
        st.plotly_chart(pfig, use_container_width=True)

        # GS-Quant Open-Source Risk & Econometrics
        gs_res = compute_gs_risk_metrics(hist)
        if gs_res:
            st.markdown("#### 📉 Riesgo & Econometría (`gs-quant` Open-Source)")
            g1, g2, g3, g4, g5, g6 = st.columns(6)
            g1.metric("Volatilidad 30D", f"{gs_res['vol_30d_last']:.1f}%" if gs_res.get("vol_30d_last") is not None else "N/A")
            g2.metric("Beta 60D vs SPY", f"{gs_res['beta_60_last']:.2f}" if gs_res.get("beta_60_last") is not None else "N/A")
            g3.metric("Max DD (2Y)", f"{gs_res['max_dd_2y']:.1%}" if gs_res.get("max_dd_2y") is not None else "N/A")
            g4.metric("Sharpe 6M", f"{gs_res['sharpe_6m_last']:.2f}" if gs_res.get("sharpe_6m_last") is not None else "N/A")
            g5.metric("Z-Score 60D", f"{gs_res['zscore_60_last']:.2f}" if gs_res.get("zscore_60_last") is not None else "N/A")
            g6.metric("RSI (14)", f"{gs_res['rsi_14_last']:.1f}" if gs_res.get("rsi_14_last") is not None else "N/A")

            beta_ser = gs_res.get("beta_series")
            if beta_ser is not None and not beta_ser.empty:
                bfig = px.line(
                    beta_ser,
                    title=f"{selected_ticker} — Beta Móvil 60D vs. SPY (gs_quant.timeseries.beta)",
                    labels={"value": "Beta", "index": "Fecha"},
                )
                bfig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.6)
                bfig.update_layout(height=280, showlegend=False)
                st.plotly_chart(bfig, use_container_width=True)

    # Recent News
    if row.get("news"):
        with st.expander("📰 Titulares Recientes (contexto para el análisis)", expanded=False):
            for n in row["news"]:
                st.markdown(f"- **{n.get('title', '')}** *({n.get('publisher', '')})*")

    # AI Value Investing Committee
    st.divider()
    st.markdown("#### 🤖 Comité de Value Investing (IA — Gemini)")
    ai_col1, ai_col2 = st.columns([2, 1])
    with ai_col1:
        persona = st.selectbox(
            "Enfoque / Referente de Value Investing",
            options=list(INVESTOR_PERSONAS.keys()),
            key=f"persona_{selected_ticker}",
        )
    with ai_col2:
        st.write("")
        st.write("")
        gen_ai = st.button("✨ Generar Memo Value con IA", use_container_width=True, key=f"btn_ai_{selected_ticker}")

    if "ai_memos" not in st.session_state:
        st.session_state["ai_memos"] = {}

    memo_key = f"{selected_ticker}:{persona}"
    if gen_ai:
        if not gemini_key:
            st.warning("Ingresa tu **Gemini API Key** en la barra lateral (`🔌 Fuentes adicionales & IA`).")
        else:
            with st.spinner(f"Analizando {selected_ticker} bajo el lente de {persona}..."):
                try:
                    memo = analyze_stock_with_ai(row.to_dict(), persona=persona, api_key=gemini_key)
                    st.session_state["ai_memos"][memo_key] = memo
                except Exception as exc:
                    st.error(f"Error al consultar Gemini: {exc}")

    if memo_key in st.session_state["ai_memos"]:
        st.markdown(st.session_state["ai_memos"][memo_key])

    # Head-to-Head AI Comparator
    st.divider()
    st.markdown("#### ⚔️ Comparador Head-to-Head con IA")
    comp_tickers = st.multiselect(
        "Elige 2 o 3 empresas para comparar asignación de capital",
        options=ticker_options,
        default=ticker_options[:2] if len(ticker_options) >= 2 else ticker_options,
        max_selections=3,
    )
    if st.button("⚖️ Comparar con IA", key="btn_compare_ai"):
        if not gemini_key:
            st.warning("Ingresa tu **Gemini API Key** en la barra lateral (`🔌 Fuentes adicionales & IA`).")
        elif len(comp_tickers) < 2:
            st.warning("Selecciona al menos 2 empresas.")
        else:
            comp_rows = [scored_df[scored_df["ticker"] == t].iloc[0].to_dict() for t in comp_tickers]
            with st.spinner("Comparando candidatas con Gemini..."):
                try:
                    st.session_state["ai_comparison"] = compare_stocks_with_ai(comp_rows, api_key=gemini_key)
                except Exception as exc:
                    st.error(f"Error al comparar con Gemini: {exc}")

    if "ai_comparison" in st.session_state:
        st.markdown(st.session_state["ai_comparison"])

with tab4:
    st.subheader("📈 Simulación Histórica (Top-N vs. SPY)")
    st.caption(
        "Simulación *lookback* de 2 años sobre el universo actual (nota: no incluye empresas deslistadas históricas; "
        "útil para evaluar volatilidad, drawdown y comportamiento relativo del Top-N actual frente a `SPY`)."
    )
    bt_c1, bt_c2 = st.columns(2)
    with bt_c1:
        bt_top_n = st.slider("Top N acciones del ranking filtrado", 3, max(3, min(25, len(filtered_df) or 5)), min(10, max(3, len(filtered_df))))
    with bt_c2:
        bt_weight = st.selectbox("Ponderación del Backtest", ["Equal Weight", "Quant Score Weighted"])

    if not filtered_df.empty:
        bt_res = run_top_n_backtest(filtered_df, bundles, top_n=bt_top_n, weighting=bt_weight, use_cache=use_cache)
        if "error" in bt_res:
            st.warning(bt_res["error"])
        else:
            curves = bt_res["curves"] * 100.0
            bt_fig = px.line(
                curves,
                title=f"Retorno Acumulado (%) — Top {len(bt_res['tickers_used'])} Quant vs. SPY",
                labels={"value": "Retorno (%)", "index": "Fecha", "variable": "Estrategia"},
            )
            bt_fig.update_layout(height=400)
            st.plotly_chart(bt_fig, use_container_width=True)

            p_st = bt_res["port_stats"]
            s_st = bt_res["spy_stats"]
            stats_rows = [
                {
                    "Estrategia": f"Quant Top-{len(bt_res['tickers_used'])}",
                    "Retorno Total": f"{p_st['total_return']:.1%}",
                    "CAGR": f"{p_st['cagr']:.1%}",
                    "Volatilidad": f"{p_st['vol']:.1%}",
                    "Sharpe": f"{p_st['sharpe']:.2f}",
                    "Max Drawdown": f"{p_st['max_dd']:.1%}",
                }
            ]
            if s_st:
                stats_rows.append(
                    {
                        "Estrategia": "SPY (Benchmark)",
                        "Retorno Total": f"{s_st['total_return']:.1%}",
                        "CAGR": f"{s_st['cagr']:.1%}",
                        "Volatilidad": f"{s_st['vol']:.1%}",
                        "Sharpe": f"{s_st['sharpe']:.2f}",
                        "Max Drawdown": f"{s_st['max_dd']:.1%}",
                    }
                )
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("💼 Construcción de Portafolio & Sizing Cuantitativo")
    pf_c1, pf_c2, pf_c3, pf_c4 = st.columns(4)
    with pf_c1:
        pf_capital = st.number_input("Capital a invertir (USD)", min_value=500.0, value=10000.0, step=500.0)
    with pf_c2:
        pf_top_n = st.slider("Número de posiciones", 3, max(3, min(20, len(filtered_df) or 5)), min(10, max(3, len(filtered_df))), key="pf_top_n")
    with pf_c3:
        pf_method = st.selectbox("Método de Sizing", ["Quant Score Weighted", "Equal Weight", "Inverse Volatility"])
    with pf_c4:
        pf_sector_cap = st.slider("Límite máximo por sector (%)", 15, 100, 35, 5) / 100.0

    if not filtered_df.empty:
        alloc_df = compute_portfolio_allocation(
            filtered_df,
            bundles,
            top_n=pf_top_n,
            method=pf_method,
            capital=pf_capital,
            max_sector_pct=pf_sector_cap,
        )
        if not alloc_df.empty:
            alloc_disp = alloc_df.copy()
            alloc_disp["margin_of_safety"] = alloc_disp["margin_of_safety"] * 100.0
            st.dataframe(
                alloc_disp,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "ticker": "Ticker",
                    "name": "Empresa",
                    "sector": "Sector",
                    "price": st.column_config.NumberColumn("Precio", format="$%.2f"),
                    "quant_score": st.column_config.NumberColumn("Quant Score", format="%.1f"),
                    "f_score": st.column_config.NumberColumn("F-Score", format="%d/9"),
                    "margin_of_safety": st.column_config.NumberColumn("DCF MoS %", format="%.1f%%"),
                    "weight_pct": st.column_config.NumberColumn("Peso %", format="%.1f%%"),
                    "allocation_usd": st.column_config.NumberColumn("Monto USD", format="$%,.2f"),
                    "est_shares": st.column_config.NumberColumn("Acciones est.", format="%.2f"),
                },
                use_container_width=True,
                hide_index=True,
            )
            alloc_csv = alloc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar Portafolio (CSV)",
                data=alloc_csv,
                file_name="quant_portfolio_allocation.csv",
                mime="text/csv",
            )
