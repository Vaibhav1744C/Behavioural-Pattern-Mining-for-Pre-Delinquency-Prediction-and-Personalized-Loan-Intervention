"""Shared visual tokens and Plotly styling for the research dashboard."""

NAVY = "#1E2761"
TEAL = "#028090"
PAGE = "#F6F7FB"
WHITE = "#FFFFFF"
INK = "#273252"
MUTED = "#66708A"
BORDER = "#E3E6F0"
GRID = "#E8EBF3"
AMBER_BG = "#FFF7E4"
AMBER_BORDER = "#E6B654"
GREY = "#AAB1C0"
RISK_COLORS = {"Very Low": "#2E9E83", "Low": "#8CC084", "Medium": "#F2C14E", "High": "#F28C4E", "Very High": "#D64550"}
RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
FONT_DISPLAY = "DM Serif Display"
FONT_BODY = "Inter"
SPACING = {"xs": 6, "sm": 10, "md": 16, "lg": 22, "xl": 30}
CARD_STYLE = {"background": WHITE, "border": f"1px solid {BORDER}", "border-radius": "12px", "padding": "20px 22px", "box-shadow": "0 3px 14px rgba(30,39,97,.045)"}

PLOTLY_TEMPLATE = {"layout": {
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, sans-serif", "color": NAVY, "size": 12},
    "title": {"font": {"family": "DM Serif Display, serif", "size": 19, "color": NAVY}, "x": 0.01},
    "margin": {"l": 42, "r": 18, "t": 54, "b": 42},
    "xaxis": {"gridcolor": GRID, "linecolor": BORDER, "tickfont": {"size": 12}, "zerolinecolor": GRID},
    "yaxis": {"gridcolor": GRID, "linecolor": BORDER, "tickfont": {"size": 12}, "zerolinecolor": GRID},
    "legend": {"font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
    "colorway": [TEAL, NAVY, "#8CC084", "#F2C14E", "#F28C4E", "#D64550"],
}}

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&display=swap');
:root {{ --navy:{NAVY}; --teal:{TEAL}; --page:{PAGE}; --border:{BORDER}; --muted:{MUTED}; }}
html, body, [class*="css"] {{ font-family:'{FONT_BODY}',sans-serif; color:{INK}; }}
.stApp {{ background:{PAGE}; }}
[data-testid="stHeader"], [data-testid="stToolbar"], footer {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:1.15rem; padding-bottom:2.4rem; max-width:1560px; }}
h1,h2,h3,h4 {{ color:{NAVY}; font-family:'{FONT_DISPLAY}',serif; letter-spacing:-.015em; }}
h1 {{ font-size:2.25rem !important; line-height:1.08 !important; margin-bottom:.2rem !important; }}
h2 {{ font-size:1.55rem !important; }} h3 {{ font-size:1.2rem !important; }}
p,label,.stMarkdown,.stCaption {{ color:{INK}; }}
[data-testid="stMetric"] {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:12px; padding:16px 18px; box-shadow:0 3px 14px rgba(30,39,97,.045); min-height:118px; }}
[data-testid="stMetricValue"] {{ color:{NAVY}; font-variant-numeric:tabular-nums; }}
[data-testid="stMetricLabel"] {{ color:{MUTED}; font-size:.74rem; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }}
.page-head {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin:0 0 20px; }}
.page-desc {{ color:{MUTED}; font-size:.94rem; margin-top:3px; }}
.context-chip {{ white-space:nowrap; color:{TEAL}; border:1px solid #B9D9DE; border-radius:999px; padding:7px 12px; font-size:.78rem; font-weight:700; background:#F2FAFB; }}
.card {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:12px; padding:20px 22px; box-shadow:0 3px 14px rgba(30,39,97,.045); margin-bottom:16px; }}
.card-title {{ color:{NAVY}; font-weight:700; font-size:1.04rem; margin:14px 0 2px; padding:14px 18px 0; }}
.card-subtitle {{ color:{MUTED}; font-size:.8rem; padding:0 18px 6px; }}
.stPlotlyChart {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:12px; padding:8px 10px; box-shadow:0 3px 14px rgba(30,39,97,.045); overflow:hidden; }}
[data-testid="stDataFrame"] {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}
.sim-banner {{ background:{AMBER_BG}; color:#604A16; padding:11px 15px; border:1px solid #F0D99F; border-left:4px solid {AMBER_BORDER}; border-radius:9px; margin:8px 0 18px; font-weight:600; }}
.notice-card {{ color:{MUTED}; background:#F5F6F9; padding:22px; border:1px dashed #B9C0CE; border-radius:12px; text-align:center; }}
.quote {{ background:#F3FAFB; border-left:3px solid {TEAL}; padding:16px 19px; border-radius:0 10px 10px 0; color:{NAVY}; font-size:1.02rem; line-height:1.55; }}
.facts-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; margin:12px 0 16px; }}
.fact-cell {{ min-width:0; border:1px solid {BORDER}; border-radius:9px; background:#FBFCFE; padding:11px 12px; }}
.fact-cell span {{ display:block; color:{MUTED}; font-size:.76rem; margin-bottom:4px; }}
.fact-cell strong {{ display:block; color:{NAVY}; font-size:1.22rem; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
.risk-pill {{ display:inline-block; padding:5px 10px; border-radius:999px; color:white; font-size:.76rem; font-weight:700; }}
.mono-chip {{ display:inline-block; font-family:ui-monospace,monospace; font-size:.72rem; background:#F1F3F8; color:{NAVY}; border:1px solid {BORDER}; border-radius:6px; padding:3px 7px; }}
.sidebar-brand {{ padding:10px 3px 20px; border-bottom:1px solid rgba(255,255,255,.18); margin-bottom:16px; }}
.sidebar-brand strong {{ display:block; font-family:'{FONT_DISPLAY}',serif; font-size:1.32rem; line-height:1.15; color:white; }}
.sidebar-brand span {{ display:block; color:#D6DBEF; font-size:.77rem; margin-top:7px; line-height:1.4; }}
section[data-testid="stSidebar"] {{ background:{NAVY}; }} section[data-testid="stSidebar"] * {{ color:#F7F8FC; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {{ border-radius:8px; padding:7px 10px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:rgba(255,255,255,.09); }}
.sidebar-foot {{ position:relative; margin-top:24px; padding-top:14px; border-top:1px solid rgba(255,255,255,.18); color:#D6DBEF; font-size:.72rem; line-height:1.55; }}
.stepper {{ display:flex; align-items:center; margin:12px 0 22px; }} .step-node {{ width:34px; height:34px; display:flex; align-items:center; justify-content:center; border-radius:50%; background:{TEAL}; color:white; font-weight:700; flex:0 0 auto; }} .step-line {{ height:2px; background:#BCD8DC; flex:1 1 auto; }}
.flow-box {{ border:1px solid {BORDER}; background:white; border-radius:10px; padding:13px 12px; text-align:center; color:{NAVY}; font-weight:600; min-height:54px; }}
.limitation {{ background:white; border:1px solid {BORDER}; border-radius:10px; padding:15px 17px; margin:9px 0; }}
div[data-testid="stExpander"] {{ background:white; border:1px solid {BORDER}; border-radius:10px; }} button[kind="primary"] {{ background:{TEAL}; border-color:{TEAL}; }}
@media(max-width:1100px) {{ .block-container {{ padding-left:1.2rem; padding-right:1.2rem; }} .card {{ padding:16px; }} h1 {{font-size:1.9rem !important;}} }}
"""
