"""Visual building blocks for the Streamlit dashboard (styles and small HTML components).

Kept separate from app.py so the page code reads as layout, not markup.
"""
from html import escape

INK, MUTED, LINE = "#1B2430", "#5B6675", "#E3E7ED"
ACCENT, RLS_COLOR, NO_RLS_COLOR = "#2F4B7C", "#C0703A", "#5B7A99"
MODEL_COLORS = {"Logistic Regression": "#2F4B7C", "Decision Tree": "#C0703A",
                "Random Forest": "#3C8D5A", "Baseline (most frequent)": "#9AA3AE"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
}}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 8px 12px; border-radius: 8px; width: 100%; margin: 0; cursor: pointer;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: #F1F4F8; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: #E7EDF6; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: {ACCENT}; font-weight: 600; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
.brand {{ padding: 4px 4px 18px 4px; border-bottom: 1px solid {LINE}; margin-bottom: 14px; }}
.brand .mark {{ display:inline-block; width: 40px; height: 32px; border-radius: 8px; background: {ACCENT};
  color: #fff; font-weight: 600; font-size: 13px; line-height: 30px; text-align: center; margin-bottom: 10px; }}
.brand .name {{ font-weight: 600; font-size: 1.02rem; color: {INK}; line-height: 1.3; }}
.brand .sub {{ color: {MUTED}; font-size: 0.82rem; margin-top: 2px; }}
.pipeline {{ font-size: 0.78rem; color: {MUTED}; line-height: 1.9; padding: 4px; }}
.pipeline b {{ color: {INK}; font-weight: 500; }}

/* page header */
.page-head {{ margin-bottom: 1.4rem; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.74rem; font-weight: 600; color: {ACCENT}; }}
.page-head h1 {{ font-size: 2rem; font-weight: 600; margin: 0.2rem 0 0.35rem 0; padding: 0; }}
.page-head p {{ color: {MUTED}; font-size: 1rem; max-width: 70ch; margin: 0; }}

/* cards */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 0.2rem 0 1.4rem 0; }}
.kpi {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; }}
.kpi .label {{ color: {MUTED}; font-size: 0.8rem; }}
.kpi .value {{ font-size: 1.55rem; font-weight: 600; color: {INK}; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.kpi .note {{ color: {MUTED}; font-size: 0.76rem; margin-top: 2px; }}
.kpi .value.text {{ font-size: 1.12rem; line-height: 1.35; margin-top: 6px; }}
.kpi.accent {{ border-color: {ACCENT}; box-shadow: inset 3px 0 0 {ACCENT}; }}
.section-title {{ font-weight: 600; font-size: 1.05rem; color: {INK}; margin: 0 0 0.2rem 0; }}
.section-sub {{ color: {MUTED}; font-size: 0.86rem; margin: 0 0 0.8rem 0; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFFFFF; border-radius: 12px; }}

/* confusion matrix */
.cm {{ display: grid; grid-template-columns: 70px 1fr 1fr; gap: 6px; font-size: 0.8rem; }}
.cm .h {{ color: {MUTED}; text-align: center; align-self: end; }}
.cm .r {{ color: {MUTED}; align-self: center; }}
.cm .c {{ border-radius: 8px; padding: 14px 6px; text-align: center; }}
.cm .c b {{ display: block; font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.cm .ok {{ background: #E7EDF6; color: {ACCENT}; }}
.cm .bad {{ background: #F7ECE4; color: #9A5226; }}

/* prediction result */
.result {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; padding: 22px; }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; }}
.badge.rls {{ background: #F7ECE4; color: #9A5226; }}
.badge.norls {{ background: #E5F0EA; color: #2E6B4A; }}
.prob {{ font-size: 2.6rem; font-weight: 600; color: {INK}; line-height: 1.1; margin: 10px 0 2px 0; font-variant-numeric: tabular-nums; }}
.meter {{ position: relative; height: 10px; border-radius: 6px; background: linear-gradient(90deg, #DDEBE3, #F3E3D8); margin: 14px 0 6px 0; }}
.meter i {{ position: absolute; top: -5px; width: 4px; height: 20px; border-radius: 2px; background: {INK}; }}
.meter b {{ position: absolute; top: -3px; left: 50%; width: 1px; height: 16px; background: {MUTED}; }}
.scale {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.74rem; }}
.muted {{ color: {MUTED}; font-size: 0.84rem; }}
.contrib {{ display: grid; grid-template-columns: minmax(0, 11rem) 1fr 3.4rem; gap: 6px 10px; align-items: center; font-size: 0.82rem; margin-top: 8px; }}
.contrib .bar {{ position: relative; height: 8px; background: #EEF1F5; border-radius: 4px; }}
.contrib .bar span {{ position: absolute; top: 0; height: 100%; border-radius: 4px; }}
.contrib .bar em {{ position: absolute; left: 50%; top: -3px; width: 1px; height: 14px; background: #B7BFCA; }}
.contrib .v {{ text-align: right; color: {MUTED}; font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; }}
.stButton button[kind="primary"], .stFormSubmitButton button {{ border-radius: 8px; font-weight: 600; padding: 0.5rem 1.4rem; }}
</style>
"""


def page_header(eyebrow: str, title: str, lede: str) -> str:
    return (f"<div class='page-head'><div class='eyebrow'>{escape(eyebrow)}</div>"
            f"<h1>{escape(title)}</h1><p>{escape(lede)}</p></div>")


def kpi_cards(items) -> str:
    """items: list of (label, value, note, accent: bool)."""
    cards = "".join(
        f"<div class='kpi{' accent' if acc else ''}'><div class='label'>{escape(label)}</div>"
        f"<div class='value{' text' if len(str(value)) > 9 else ''}'>{escape(str(value))}</div><div class='note'>{escape(note)}</div></div>"
        for label, value, note, acc in items)
    return f"<div class='kpis'>{cards}</div>"


def section(title: str, sub: str = "") -> str:
    return f"<div class='section-title'>{escape(title)}</div>" + (f"<div class='section-sub'>{escape(sub)}</div>" if sub else "")


def confusion_matrix(tn, fp, fn, tp) -> str:
    total = tn + fp + fn + tp
    cell = lambda v, tag, ok: (f"<div class='c {'ok' if ok else 'bad'}'><b>{v}</b>{tag} · {v / total:.0%}</div>")
    return ("<div class='cm'><span></span><span class='h'>Predicted No RLS</span><span class='h'>Predicted RLS</span>"
            f"<span class='r'>Actual No RLS</span>{cell(tn, 'TN', True)}{cell(fp, 'FP', False)}"
            f"<span class='r'>Actual RLS</span>{cell(fn, 'FN', False)}{cell(tp, 'TP', True)}</div>")


def prediction_card(label: str, prob: float, model_name: str, recorded: str = "") -> str:
    cls = "rls" if label == "RLS" else "norls"
    text = "Likely RLS" if label == "RLS" else "RLS unlikely"
    pos = max(0.0, min(1.0, prob)) * 100
    rec = f"<div class='muted' style='margin-top:10px'>Recorded label for this example: <b>{escape(recorded)}</b></div>" if recorded else ""
    return (f"<div class='result'><span class='badge {cls}'>{text}</span>"
            f"<div class='prob'>{prob:.0%}</div><div class='muted'>estimated probability of RLS · {escape(model_name)}</div>"
            f"<div class='meter'><b></b><i style='left:calc({pos:.1f}% - 2px)'></i></div>"
            "<div class='scale'><span>0%</span><span>decision threshold 50%</span><span>100%</span></div>"
            f"{rec}</div>")


def contributions(rows, scale: float) -> str:
    """rows: list of (feature label, contribution in log-odds)."""
    out = []
    for name, v in rows:
        w = min(abs(v) / scale, 1) * 50
        left = 50 if v >= 0 else 50 - w
        color = RLS_COLOR if v >= 0 else "#3C8D5A"
        out.append(f"<span>{escape(name)}</span><div class='bar'><span style='left:{left:.1f}%;width:{w:.1f}%;"
                   f"background:{color}'></span><em></em></div><span class='v'>{v:+.2f}</span>")
    return f"<div class='contrib'>{''.join(out)}</div>"


# =====================================================================================
# Charts drawn as inline SVG / HTML. These replace Streamlit's built-in charts, which
# depend on the Altair library (Altair 5 fails to import on Python 3.14).
# Every function returns one line of markup for st.markdown(..., unsafe_allow_html=True).
# =====================================================================================
import math

GRID = "#E8ECF1"
TEXT_STYLE = "font-family:'IBM Plex Sans','Segoe UI',sans-serif"

CHART_CSS = f"""
<style>
.chart svg {{ width: 100%; height: auto; display: block; }}
.chart svg text {{ fill: {MUTED}; font-size: 11px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 0 0 6px 0; font-size: 0.8rem; color: {MUTED}; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
.hbars {{ display: grid; grid-template-columns: minmax(0, 15rem) 1fr; gap: 7px 14px; align-items: center; font-size: 0.84rem; }}
.hbars .lab {{ color: {INK}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.hbars .track {{ display: grid; gap: 3px; }}
.hbars .row {{ display: grid; grid-template-columns: 1fr 3.4rem; gap: 8px; align-items: center; }}
.hbars .bar {{ height: 11px; background: #F1F4F8; border-radius: 3px; overflow: hidden; }}
.hbars .bar span {{ display: block; height: 100%; border-radius: 3px; }}
.hbars .v {{ color: {MUTED}; font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; text-align: right; }}
</style>
"""


def _esc(s):
    return escape(str(s))


def legend(items) -> str:
    """items: list of (name, color)."""
    return "<div class='legend'>" + "".join(f"<span><i style='background:{c}'></i>{_esc(n)}</span>" for n, c in items) + "</div>"


def _nice_max(v):
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= v:
            return m * mag
    return 10 * mag


def _ticks(ymax, n=5):
    return [ymax * i / n for i in range(n + 1)]


def _fmt(v, fmt):
    return fmt.format(v)


def grouped_bars(categories, series, ymax=None, fmt="{:.2f}", tick_fmt=None, height=300, y_label="",
                 show_values=True, bar_colors=None, width=640) -> str:
    """Vertical grouped bar chart.

    categories: x-axis labels (drawn in the given order).
    series: list of (name, values, color); one bar per series in each category.
    bar_colors: optional per-category colours (only used with a single series).
    """
    W, H = width, height
    left, right, top, bottom = 48, 10, 14, 30
    pw, ph = W - left - right, H - top - bottom
    vmax = max((v for _, vals, _ in series for v in vals if v == v), default=1)
    ymax = ymax or _nice_max(vmax * 1.08)
    tick_fmt = tick_fmt or ("{:.1f}" if ymax <= 1.5 else "{:,.0f}")
    y = lambda v: top + ph * (1 - min(v, ymax) / ymax)
    parts = [f"<svg viewBox='0 0 {W} {H}' role='img' style='{TEXT_STYLE};max-width:{int(W * 1.3)}px'>"]
    for t in _ticks(ymax):
        parts.append(f"<line x1='{left}' x2='{W - right}' y1='{y(t):.1f}' y2='{y(t):.1f}' stroke='{GRID}'/>"
                     f"<text x='{left - 8}' y='{y(t) + 4:.1f}' text-anchor='end'>{_esc(_fmt(t, tick_fmt))}</text>")
    n_cat, n_ser = len(categories), len(series)
    gw = pw / max(n_cat, 1)
    bw = min(gw * 0.78 / max(n_ser, 1), 46)
    for i, cat in enumerate(categories):
        gx = left + gw * i + (gw - bw * n_ser) / 2
        for j, (name, vals, color) in enumerate(series):
            v = vals[i]
            if v != v:  # NaN
                continue
            c = bar_colors[i] if (bar_colors and n_ser == 1) else color
            x0, y0 = gx + j * bw, y(v)
            parts.append(f"<rect x='{x0 + 1:.1f}' y='{y0:.1f}' width='{max(bw - 2, 1):.1f}' height='{top + ph - y0:.1f}' "
                         f"rx='2' fill='{c}'><title>{_esc(name)} · {_esc(cat)}: {_esc(_fmt(v, fmt))}</title></rect>")
            if show_values and n_cat * n_ser <= 16:
                parts.append(f"<text x='{x0 + bw / 2:.1f}' y='{y0 - 4:.1f}' text-anchor='middle' "
                             f"style='font-size:10px;fill:{INK}'>{_esc(_fmt(v, fmt))}</text>")
        parts.append(f"<text x='{left + gw * i + gw / 2:.1f}' y='{H - 10}' text-anchor='middle' "
                     f"style='font-size:11.5px;fill:{INK}'>{_esc(cat)}</text>")
    parts.append(f"<line x1='{left}' x2='{W - right}' y1='{top + ph}' y2='{top + ph}' stroke='#B7BFCA'/>")
    if y_label:
        parts.append(f"<text transform='translate(12 {top + ph / 2:.0f}) rotate(-90)' text-anchor='middle'>{_esc(y_label)}</text>")
    parts.append("</svg>")
    leg = legend([(n, c) for n, _, c in series]) if n_ser > 1 else ""
    return f"<div class='chart'>{leg}{''.join(parts)}</div>"


def line_chart(series, x_label="", y_label="", diagonal=True, height=420, vline=None) -> str:
    """Line chart on [0,1] x [0,1]. series: list of (name, xs, ys, color, dashed)."""
    W, H = 560, height
    left, right, top, bottom = 50, 14, 12, 42
    pw, ph = W - left - right, H - top - bottom
    X = lambda v: left + pw * v
    Y = lambda v: top + ph * (1 - v)
    parts = [f"<svg viewBox='0 0 {W} {H}' role='img' style='{TEXT_STYLE};max-width:{int(W * 1.3)}px'>"]
    for t in _ticks(1.0):
        parts.append(f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top}' y2='{top + ph}' stroke='{GRID}'/>"
                     f"<line x1='{left}' x2='{left + pw}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='{GRID}'/>"
                     f"<text x='{X(t):.1f}' y='{top + ph + 16}' text-anchor='middle'>{t:.1f}</text>"
                     f"<text x='{left - 8}' y='{Y(t) + 4:.1f}' text-anchor='end'>{t:.1f}</text>")
    if diagonal:
        parts.append(f"<line x1='{X(0)}' y1='{Y(0)}' x2='{X(1)}' y2='{Y(1)}' stroke='#B7BFCA' stroke-dasharray='4 4'/>")
    if vline is not None:
        parts.append(f"<line x1='{X(vline):.1f}' x2='{X(vline):.1f}' y1='{top}' y2='{top + ph}' stroke='{INK}' "
                     f"stroke-width='1.2' stroke-dasharray='3 3'/><text x='{X(vline) + 5:.1f}' y='{top + 12}' "
                     f"style='font-size:11px;fill:{INK}'>threshold {vline:.2f}</text>")
    for name, xs, ys, color, dashed in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys))
        dash = " stroke-dasharray='6 4'" if dashed else ""
        parts.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2.2' "
                     f"stroke-linejoin='round'{dash}><title>{_esc(name)}</title></polyline>")
    parts.append(f"<text x='{left + pw / 2:.0f}' y='{H - 6}' text-anchor='middle'>{_esc(x_label)}</text>"
                 f"<text transform='translate(13 {top + ph / 2:.0f}) rotate(-90)' text-anchor='middle'>{_esc(y_label)}</text>")
    parts.append("</svg>")
    return f"<div class='chart'>{legend([(s[0], s[3]) for s in series])}{''.join(parts)}</div>"


def hbars(rows, series, xmax=None, fmt="{:.3f}") -> str:
    """Horizontal bars. rows: list of (label, [value per series]); series: list of (name, color)."""
    vmax = max((v for _, vals in rows for v in vals if v == v), default=1)
    xmax = xmax or vmax
    out = []
    for label, vals in rows:
        bars = "".join(
            f"<div class='row'><div class='bar' title='{_esc(name)}: {_esc(_fmt(v, fmt))}'><span style='width:"
            f"{max(0.0, min(v / xmax, 1)) * 100:.1f}%;background:{color}'></span></div>"
            f"<span class='v'>{_esc(_fmt(v, fmt))}</span></div>"
            for v, (name, color) in zip(vals, series))
        out.append(f"<span class='lab' title='{_esc(label)}'>{_esc(label)}</span><div class='track'>{bars}</div>")
    leg = legend(series) if len(series) > 1 else ""
    return f"<div class='chart'>{leg}<div class='hbars'>{''.join(out)}</div></div>"

STEPS_CSS = f"""
<style>
.steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 4px 0 18px 0; }}
.step {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; }}
.step .n {{ display: inline-block; width: 26px; height: 26px; border-radius: 50%; background: #E7EDF6; color: {ACCENT};
  font-weight: 600; font-size: 0.8rem; text-align: center; line-height: 26px; margin-bottom: 8px; }}
.step .t {{ font-weight: 600; color: {INK}; margin-bottom: 4px; }}
.step .d {{ color: {MUTED}; font-size: 0.84rem; line-height: 1.45; }}
.formula {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; background: #F4F6F9; border-radius: 6px;
  padding: 2px 6px; color: {INK}; }}
</style>
"""


def steps(items) -> str:
    """items: list of (title, description). Numbered cards in a row."""
    return "<div class='steps'>" + "".join(
        f"<div class='step'><div class='n'>{i}</div><div class='t'>{_esc(t)}</div><div class='d'>{d}</div></div>"
        for i, (t, d) in enumerate(items, start=1)) + "</div>"


# Blank lines inside an HTML block end it early in Markdown, so remove them from the styles.
CSS = "\n".join(line for line in CSS.splitlines() if line.strip())
CHART_CSS = "\n".join(line for line in CHART_CSS.splitlines() if line.strip())
STEPS_CSS = "\n".join(line for line in STEPS_CSS.splitlines() if line.strip())
