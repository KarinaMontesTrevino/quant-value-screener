"""AI Value Investing Committee powered by Google Gemini (google-genai SDK)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from google import genai

INVESTOR_PERSONAS: dict[str, str] = {
    "Comité Completo (Multi-Enfoque)": (
        "Evalúa la empresa desde 3 lentes complementarios: "
        "(1) Benjamin Graham (margen de seguridad y balance), "
        "(2) Warren Buffett & Charlie Munger (calidad del negocio, ROIC y foso competitivo), y "
        "(3) Joel Greenblatt / Tobias Carlisle (Quant Value: EV/EBIT, FCF Yield y eficiencia de capital)."
    ),
    "Benjamin Graham (Deep Value & Balance)": (
        "Adopta el criterio estricto de Benjamin Graham: prioriza precio bajo frente a activos/beneficios, "
        "solidez del balance (Current Ratio, deuda controlada), protección contra pérdida permanente de capital "
        "y un Margen de Seguridad amplio."
    ),
    "Warren Buffett & Charlie Munger (Quality Value)": (
        "Adopta el criterio de Buffett y Munger: busca negocios de alta calidad con ROIC elevado y sostenible, "
        "márgenes operativos fuertes, conversión a Free Cash Flow, baja necesidad de apalancamiento y un precio razonable."
    ),
    "Joel Greenblatt & Tobias Carlisle (Quant Value)": (
        "Adopta el criterio puramente cuantitativo de Greenblatt (Magic Formula) y Carlisle (Acquirer's Multiple): "
        "enfócate en EV/EBIT, FCF Yield, ROIC, Gross Profitability (Novy-Marx) y el Piotroski F-Score para evitar value traps."
    ),
    "Howard Marks & Seth Klarman (Riesgo & Asimetría)": (
        "Adopta el criterio de Marks y Klarman: enfócate primero en qué puede salir mal (downside risk, deuda, "
        "Altman Z-Score, banderas rojas, deterioro de ingresos) antes de estimar el potencial alcista."
    ),
}


def _fmt(val: Any, kind: str = "num") -> str:
    if val is None or pd.isna(val):
        return "N/A"
    if kind == "pct":
        return f"{float(val):.2%}"
    if kind == "x":
        return f"{float(val):.2f}x"
    if kind == "usd":
        return f"${float(val):,.2f}"
    if kind == "big_usd":
        return f"${float(val) / 1e9:,.2f}B"
    return f"{val}"


def build_stock_context(row: dict[str, Any]) -> str:
    """Serialize quantitative metrics and recent news of a stock into a structured prompt context."""
    red_flags = ", ".join(row.get("red_flags") or []) or "Ninguna"
    news_list = row.get("news") or []
    news_str = "\n".join(
        f"- {n.get('title', '')} ({n.get('publisher', '')})" for n in news_list[:5]
    ) or "Sin titulares recientes disponibles."

    fcf_yrs = (
        f"{row.get('fcf_pos_years')}/{row.get('fcf_total_years')}"
        if row.get("fcf_pos_years") is not None
        else "N/A"
    )
    return f"""
Empresa: {row.get('name')} ({row.get('ticker')})
Sector / Industria: {row.get('sector')} | {row.get('industry')}
Precio actual: {_fmt(row.get('price'), 'usd')} | Market Cap: {_fmt(row.get('market_cap'), 'big_usd')}

[SCORES QUANT (0-100)]
- Composite Quant Score: {_fmt(row.get('quant_score'))}
- Value Score: {_fmt(row.get('value_score'))}
- Quality Score: {_fmt(row.get('quality_score'))}
- Safety Score: {_fmt(row.get('safety_score'))}
- Momentum Score: {_fmt(row.get('momentum_score'))}

[VALORACIÓN (VALUE)]
- EV/EBIT (Acquirer's Multiple): {_fmt(row.get('ev_ebit'), 'x')}
- EV/EBITDA: {_fmt(row.get('ev_ebitda'), 'x')}
- FCF Yield: {_fmt(row.get('fcf_yield'), 'pct')}
- P/E Trailing: {_fmt(row.get('pe_trailing'), 'x')} | P/E Forward: {_fmt(row.get('pe_forward'), 'x')}
- P/B: {_fmt(row.get('pb'), 'x')} | P/S: {_fmt(row.get('ps'), 'x')}
- Percentil precio propio (2Y): {_fmt(row.get('price_2y_pct'), 'pct')}
- Shareholder Yield (Dividendos + Recompras netas): {_fmt(row.get('shareholder_yield'), 'pct')}
- DCF Conservador: {_fmt(row.get('dcf_price'), 'usd')} | Margen de Seguridad DCF: {_fmt(row.get('margin_of_safety'), 'pct')}

[CALIDAD Y CONSISTENCIA (QUALITY)]
- ROIC actual: {_fmt(row.get('roic'), 'pct')} | ROIC promedio 3-4Y: {_fmt(row.get('roic_3y_avg'), 'pct')}
- ROE: {_fmt(row.get('roe'), 'pct')}
- Gross Profitability (Novy-Marx GP/Assets): {_fmt(row.get('gross_profitability'), 'pct')}
- Margen Operativo: {_fmt(row.get('op_margin'), 'pct')} | Margen FCF: {_fmt(row.get('fcf_margin'), 'pct')}
- Años con FCF positivo: {fcf_yrs}
- Crecimiento de Ingresos YoY: {_fmt(row.get('rev_growth'), 'pct')}

[SOLIDEZ Y RIESGO (SAFETY)]
- Piotroski F-Score: {row.get('f_score')}/9
- Altman Z-Score: {_fmt(row.get('z_score'))}
- Beneish M-Score (> -1.78 alerta): {_fmt(row.get('m_score'))}
- Net Debt / EBITDA: {_fmt(row.get('net_debt_ebitda'), 'x')}
- Cobertura de Intereses: {_fmt(row.get('interest_cov'), 'x')}
- Current Ratio: {_fmt(row.get('current_ratio'), 'x')}
- Banderas Rojas automáticas: {red_flags}

[MOMENTUM]
- Retorno 6M: {_fmt(row.get('ret_6m'), 'pct')} | Retorno 12M-1M: {_fmt(row.get('ret_12m_1m'), 'pct')}
- Distancia al máximo 52W: {_fmt(row.get('dist_52w_high'), 'pct')} | Precio vs SMA200: {_fmt(row.get('price_vs_sma200'), 'pct')}

[TITULARES RECIENTES]
{news_str}
""".strip()


def analyze_stock_with_ai(
    row: dict[str, Any],
    persona: str,
    api_key: str,
    model: str = "gemini-flash-latest",
) -> str:
    """Generate a structured Value Investing memo using Gemini."""
    if not api_key:
        raise ValueError("Se requiere una Gemini API Key para ejecutar el análisis con IA.")

    persona_instruction = INVESTOR_PERSONAS.get(
        persona, INVESTOR_PERSONAS["Comité Completo (Multi-Enfoque)"]
    )
    context = build_stock_context(row)

    prompt = f"""Eres un analista institucional de Quant Value Investing.
{persona_instruction}

Analiza EXCLUSIVAMENTE con base en los datos cuantitativos provistos abajo (no inventes cifras que no estén aquí; si algún dato es N/A, señálalo). Responde en español, de forma concisa y estructurada con estas secciones:

1. **Veredicto Quant-Value** (Atractiva / Watchlist / Posible Value Trap / Cara) y por qué en 2-3 líneas.
2. **Tesis Alcista (Bull Case)** apoyada en los múltiplos y métricas de calidad.
3. **Riesgos y Posibles Value Traps (Bear Case)** considerando F-Score, deuda, márgenes, momentum y banderas rojas.
4. **Checklist de Due Diligence** (3 preguntas concretas que un inversor debería verificar en el último 10-K o llamada de resultados).

Datos de la empresa:
{context}
"""
    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
    return response.text or "No se recibió respuesta del modelo."


def compare_stocks_with_ai(
    rows: list[dict[str, Any]],
    api_key: str,
    model: str = "gemini-flash-latest",
) -> str:
    """Compare 2-3 candidate stocks head-to-head for capital allocation using Gemini."""
    if not api_key:
        raise ValueError("Se requiere una Gemini API Key para comparar acciones con IA.")
    if len(rows) < 2:
        raise ValueError("Selecciona al menos 2 acciones para comparar.")

    contexts = "\n\n---\n\n".join(build_stock_context(r) for r in rows[:3])
    prompt = f"""Eres el Director de Inversiones (CIO) de un fondo Quant Value.
Compara estas acciones cara a cara basándote EXCLUSIVAMENTE en los datos cuantitativos y titulares provistos. Responde en español con:

1. **Tabla Comparativa de Fortalezas y Debilidades** (Value, Calidad, Seguridad/Riesgo).
2. **Ganador de Asignación de Capital** (cuál ofrece mejor relación riesgo/retorno hoy y por qué).
3. **Cuándo elegir la alternativa** (qué tendría que cambiar o qué perfil de inversor preferiría la otra opción).

Datos:
{contexts}
"""
    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
    return response.text or "No se recibió respuesta del modelo."
