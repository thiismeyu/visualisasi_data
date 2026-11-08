import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# 0. KONFIGURASI & TEMA
# ------------------------------------------------------------------
st.set_page_config(page_title="Insurance Cost Analytics", page_icon="🏥", layout="wide")

NAVY, TEAL, ORANGE, SLATE, GRID = "#0B2545", "#0FA3B1", "#F77F00", "#64748B", "#E8EDF4"
URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
NUM, CAT = ["age", "bmi", "children"], ["sex", "smoker", "region"]
LABEL = {"age": "Usia", "bmi": "BMI", "children": "Jumlah Anak", "sex": "Jenis Kelamin",
         "smoker": "Perokok", "region": "Wilayah", "kelompok_bmi": "Kategori BMI"}
SMOKER_COLOR = {"yes": ORANGE, "no": TEAL}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family:'Inter',sans-serif;}
.stApp {background:#F4F7FB;}
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1400px;}
h1,h2,h3,h4 {color:#0B2545; letter-spacing:-0.02em;}

/* Hero */
.hero {background:linear-gradient(135deg,#0B2545 0%,#13315C 55%,#0FA3B1 130%);
  border-radius:18px; padding:28px 32px; color:#fff; height:100%;
  box-shadow:0 8px 24px rgba(11,37,69,.22);}
.hero h1 {color:#fff; font-size:1.9rem; font-weight:800; margin:6px 0 8px 0; line-height:1.15;}
.hero p {color:#CFE3F1; font-size:.92rem; margin:0 0 14px 0; line-height:1.5;}
.badge {display:inline-block; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
  color:#fff; padding:3px 11px; border-radius:999px; font-size:.72rem; font-weight:600; margin-right:6px;}
.eyebrow {font-size:.72rem; font-weight:700; letter-spacing:.14em; color:#7FD8E0;}

/* Section label */
.sec {display:flex; align-items:center; gap:10px; margin:18px 0 8px 0;}
.sec .num {background:#0B2545; color:#fff; width:26px; height:26px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; font-size:.8rem; font-weight:700;}
.sec .ttl {font-size:1.05rem; font-weight:700; color:#0B2545;}
.sec .sub {font-size:.82rem; color:#64748B; margin-left:4px;}

/* Card container */
div[class*="st-key-card"] {background:#fff; border:1px solid #E5EAF2 !important;
  border-radius:16px !important; padding:6px 10px; box-shadow:0 2px 12px rgba(15,37,69,.05);}

/* KPI */
.kpi {background:#fff; border-radius:14px; padding:16px 18px; border:1px solid #E5EAF2;
  border-left:5px solid var(--accent); box-shadow:0 2px 12px rgba(15,37,69,.05);}
.kpi-label {font-size:.72rem; font-weight:700; letter-spacing:.08em; color:#64748B; text-transform:uppercase;}
.kpi-value {font-size:1.75rem; font-weight:800; color:#0B2545; margin:2px 0; line-height:1.15;}
.kpi-sub {font-size:.78rem; color:#64748B;}

/* Insight */
.insight {background:#ECFEFF; border:1px solid #BDEFF4; border-radius:12px; padding:12px 16px;
  color:#0B2545; font-size:.9rem; margin:12px 0 4px 0;}

/* Result */
.result {background:linear-gradient(135deg,#0B2545,#0FA3B1); border-radius:16px; padding:22px 24px;
  color:#fff; box-shadow:0 8px 24px rgba(15,163,177,.30);}
.r-label {font-size:.72rem; font-weight:700; letter-spacing:.14em; color:#BFEFF3;}
.r-value {font-size:2.7rem; font-weight:800; line-height:1.1; margin:6px 0 2px 0;}
.r-idr {font-size:1rem; color:#E0F7FA; margin-bottom:12px;}
.r-badge {display:inline-block; background:rgba(255,255,255,.18); padding:5px 12px;
  border-radius:999px; font-size:.8rem; font-weight:600;}

/* ===== Perbaikan keterbacaan (aman untuk mode gelap & terang) ===== */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] em {color:#1E3A8A;}
.stApp .hero p {color:#CFE3F1 !important;}
.stApp .hero h1 {color:#fff !important;}
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] * {color:#1E3A8A !important; font-weight:600;}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * {color:#475569 !important;}
.stApp [data-testid="stRadio"] label, .stApp [data-testid="stRadio"] label * {color:#1E3A8A !important;}
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * {color:#1E3A8A !important; font-weight:600;}
.stApp [data-testid="stExpander"] details {background:#fff; border:1px solid #E5EAF2; border-radius:12px;}
/* Tab */
.stApp [data-baseweb="tab-list"] button p {color:#475569 !important; font-weight:600;}
.stApp [data-baseweb="tab-list"] button[aria-selected="true"] p {color:#1D4ED8 !important;}
.stApp [data-baseweb="tab-highlight"] {background:#1D4ED8 !important;}
/* Dropdown / multiselect / input jadi terang */
.stApp div[data-baseweb="select"] > div, .stApp div[data-baseweb="input"] > div, .stApp div[data-baseweb="base-input"] {
  background:#fff !important; border:1px solid #CBD5E1 !important; border-radius:10px !important;}
.stApp div[data-baseweb="select"] *, .stApp input {color:#0B2545 !important;}
.stApp div[data-baseweb="select"] svg {fill:#1E3A8A !important;}
.stApp span[data-baseweb="tag"] {background:#1D4ED8 !important;}
.stApp span[data-baseweb="tag"] * {color:#fff !important;}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {background:#fff !important; color:#0B2545 !important;}
div[data-baseweb="popover"] li:hover {background:#E6F0FF !important;}
/* Slider */
.stApp [data-testid="stSlider"] [data-testid="stThumbValue"], .stApp [data-testid="stSlider"] [data-testid="stTickBarMin"],
.stApp [data-testid="stSlider"] [data-testid="stTickBarMax"] {color:#1D4ED8 !important; font-weight:700;}
.stApp [data-testid="stSlider"] div[role="slider"] {background:#1D4ED8 !important; border-color:#1D4ED8 !important;}

/* Tabs & widgets */
button[data-baseweb="tab"] {font-weight:600;}
.foot {text-align:center; color:#94A3B8; font-size:.78rem; margin-top:24px;}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub, color):
    return (f'<div class="kpi" style="--accent:{color}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def section(num, title, sub=""):
    st.markdown(f'<div class="sec"><div class="num">{num}</div><div class="ttl">{title}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def style(fig, h=360, legend=True):
    fig.update_layout(
        height=h, margin=dict(t=30, b=10, l=10, r=10),
        font=dict(family="Inter, sans-serif", size=12, color=NAVY),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        showlegend=legend, legend=dict(orientation="h", y=1.12, x=0, title_text=""),
        title_font=dict(size=14, color=NAVY))
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def rupiah(x):
    return "Rp " + f"{x:,.0f}".replace(",", ".")

# ------------------------------------------------------------------
# 1. DATA & MODEL
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(URL)
    df["kelompok_bmi"] = pd.cut(df["bmi"], [0, 18.5, 25, 30, 100],
                                labels=["Kurus", "Normal", "Gemuk", "Obesitas"])
    df["kelompok_usia"] = pd.cut(df["age"], [17, 25, 35, 45, 55, 64],
                                 labels=["18-25", "26-35", "36-45", "46-55", "56-64"])
    return df


@st.cache_resource
def train_model(n_estimators, max_depth, learning_rate):
    df = load_data()
    X, y = df[NUM + CAT], df["charges"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                            remainder="passthrough")
    model = Pipeline([("pre", pre),
                      ("gb", GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                       learning_rate=learning_rate, random_state=42))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    names = model.named_steps["pre"].get_feature_names_out()
    imp = model.named_steps["gb"].feature_importances_
    agg = {}
    for n, v in zip(names, imp):
        col = n.split("__")[1]
        for c in CAT:
            if col.startswith(c + "_"):
                col = c
        agg[LABEL[col]] = agg.get(LABEL[col], 0) + v
    imp_df = pd.DataFrame({"Fitur": list(agg), "Pengaruh": list(agg.values())}).sort_values("Pengaruh")
    return model, r2_score(y_te, pred), mean_absolute_error(y_te, pred), imp_df


df = load_data()

# ==================================================================
# LAYOUT POLA Z
#  Kiri-atas (hero) ──► Kanan-atas (filter)
#        ╲  diagonal: KPI + grafik eksplorasi
#  Kiri-bawah (model) ──► Kanan-bawah (PREDIKSI)
# ==================================================================

# ---------- BARIS 1 ----------
c_hero, c_filter = st.columns([1.25, 1.75], gap="large")
with c_hero:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">DATA ANALYTICS DASHBOARD</div>
      <h1>Insurance Cost Analytics &amp; Prediction</h1>
      <p>Eksplorasi faktor yang memengaruhi biaya asuransi kesehatan dan simulasikan
      prediksinya secara interaktif. Mulai dari filter ➜ baca insight ➜ coba prediksi.</p>
      <span class="badge">1.338 peserta</span><span class="badge">Gradient Boosting</span>
      <span class="badge">Plotly + Streamlit</span>
    </div>""", unsafe_allow_html=True)
with c_filter:
    with st.container(border=True, key="card_filter"):
        st.markdown("**🎛️ Filter Data** · berlaku untuk KPI & grafik di bawah")
        f1, f2, f3 = st.columns(3)
        sm = f1.multiselect("Status perokok", ["yes", "no"], default=["yes", "no"])
        rg = f2.multiselect("Wilayah", sorted(df.region.unique()), default=sorted(df.region.unique()))
        ag = f3.slider("Rentang usia", 18, 64, (18, 64))
        st.caption("Tips: pilih hanya *yes* pada perokok untuk melihat segmen biaya tertinggi.")

d = df[df.smoker.isin(sm) & df.region.isin(rg) & df.age.between(*ag)]
if d.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Silakan ubah filter.")
    st.stop()

# ---------- DIAGONAL : KPI ----------
section(1, "Ringkasan", "gambaran umum data terfilter")
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Jumlah Peserta", f"{len(d):,}", f"{len(d)/len(df)*100:.0f}% dari total data", NAVY), unsafe_allow_html=True)
k2.markdown(kpi("Rata-rata Biaya", f"${d.charges.mean():,.0f}", f"Median ${d.charges.median():,.0f}", TEAL), unsafe_allow_html=True)
k3.markdown(kpi("Biaya Tertinggi", f"${d.charges.max():,.0f}", f"Terendah ${d.charges.min():,.0f}", ORANGE), unsafe_allow_html=True)
k4.markdown(kpi("Proporsi Perokok", f"{(d.smoker == 'yes').mean()*100:.0f}%", f"Rata-rata BMI {d.bmi.mean():.1f}", SLATE), unsafe_allow_html=True)

# Insight otomatis
if d.smoker.nunique() == 2:
    ratio = d[d.smoker == "yes"].charges.mean() / d[d.smoker == "no"].charges.mean()
    st.markdown(f'<div class="insight">💡 <b>Insight:</b> pada data terfilter, peserta perokok membayar rata-rata '
                f'<b>{ratio:.1f}× lebih tinggi</b> dibanding non-perokok.</div>', unsafe_allow_html=True)

# ---------- DIAGONAL : GRAFIK ----------
section(2, "Eksplorasi", "temukan pola dengan mengubah pilihan di tiap grafik")
g1, g2 = st.columns(2, gap="large")
with g1:
    with st.container(border=True, key="card_g1"):
        st.markdown("**Hubungan variabel dengan biaya**")
        x_var = st.radio("Sumbu X", ["age", "bmi"], format_func=lambda v: LABEL[v], horizontal=True,
                         label_visibility="collapsed")
        fig = px.scatter(d, x=x_var, y="charges", color="smoker", opacity=0.75,
                         hover_data=["sex", "region", "children"], color_discrete_map=SMOKER_COLOR,
                         labels={x_var: LABEL[x_var], "charges": "Biaya (USD)", "smoker": "Perokok"})
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
        st.plotly_chart(style(fig, 380), width="stretch")
with g2:
    with st.container(border=True, key="card_g2"):
        t1, t2 = st.tabs(["Rata-rata per kelompok", "Heatmap usia × BMI"])
        with t1:
            grp = st.selectbox("Kelompokkan berdasarkan", ["region", "sex", "smoker", "children", "kelompok_bmi"],
                               format_func=lambda v: LABEL[v])
            a = d.groupby(grp, observed=True)["charges"].mean().reset_index()
            fig2 = px.bar(a, x=grp, y="charges", text_auto=".2s", labels={grp: LABEL[grp], "charges": "Rata-rata biaya (USD)"})
            fig2.update_traces(marker_color=TEAL, marker_line_width=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(style(fig2, 320, legend=False), width="stretch")
        with t2:
            hm = d.pivot_table(index="kelompok_usia", columns="kelompok_bmi", values="charges",
                               aggfunc="mean", observed=False)
            fig4 = px.imshow(hm, text_auto=".0f", aspect="auto", color_continuous_scale=["#E6F7F9", TEAL, NAVY],
                             labels=dict(x="Kategori BMI", y="Kelompok usia", color="USD"))
            fig4.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style(fig4, 365, legend=False), width="stretch")

# ---------- BARIS 3 : MODEL (kiri) ──► PREDIKSI (kanan) ----------
section(3, "Model & Prediksi", "atur parameter lalu lihat hasilnya langsung berubah")
left, right = st.columns([1, 1.7], gap="large")

with left:
    with st.container(border=True, key="card_model"):
        st.markdown("**🤖 Performa model**")
        with st.expander("⚙️ Atur parameter model"):
            n_est = st.slider("Jumlah pohon (n_estimators)", 50, 400, 150, 50)
            depth = st.slider("Kedalaman pohon (max_depth)", 1, 6, 3)
            lr = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
        model, r2, mae, imp_df = train_model(n_est, depth, lr)
        m1, m2 = st.columns(2)
        m1.markdown(kpi("Akurasi R²", f"{r2:.2f}", "mendekati 1 = bagus", TEAL), unsafe_allow_html=True)
        m2.markdown(kpi("Rata-rata galat", f"${mae:,.0f}", "MAE pada data uji", ORANGE), unsafe_allow_html=True)
        fi = px.bar(imp_df, x="Pengaruh", y="Fitur", orientation="h", title="Faktor paling berpengaruh")
        fi.update_traces(marker_color=[TEAL] * (len(imp_df) - 1) + [ORANGE], marker_line_width=0)
        fi.update_yaxes(showgrid=False)
        fi.update_xaxes(showgrid=True, gridcolor=GRID)
        st.plotly_chart(style(fi, 300, legend=False), width="stretch")

with right:
    with st.container(border=True, key="card_pred"):
        st.markdown("**🔮 Simulasi prediksi biaya**")
        in_col, out_col = st.columns([1.1, 1], gap="large")
        with in_col:
            age = st.slider("Usia", 18, 64, 35)
            bmi = st.slider("BMI", 15.0, 50.0, 27.0, 0.5)
            kids = st.slider("Jumlah anak", 0, 5, 1)
            q1, q2 = st.columns(2)
            smoker = q1.radio("Perokok?", ["no", "yes"], horizontal=True)
            sex = q2.radio("Jenis kelamin", ["female", "male"], horizontal=True)
            region = st.selectbox("Wilayah", sorted(df.region.unique()))
        row = pd.DataFrame([{"age": age, "bmi": bmi, "children": kids, "sex": sex, "smoker": smoker, "region": region}])
        hasil = model.predict(row)[0]
        pct = (df.charges < hasil).mean() * 100
        with out_col:
            kurs = st.number_input("Kurs USD → IDR", 10000, 20000, 16000, 100)
            st.markdown(f'<div class="result"><div class="r-label">PREDIKSI BIAYA PER TAHUN</div>'
                        f'<div class="r-value">${hasil:,.0f}</div><div class="r-idr">≈ {rupiah(hasil * kurs)}</div>'
                        f'<div class="r-badge">Lebih mahal dari {pct:.0f}% peserta</div></div>', unsafe_allow_html=True)
            sel = (hasil - df.charges.mean()) / df.charges.mean() * 100
            st.caption(f"{abs(sel):.0f}% {'di atas' if sel > 0 else 'di bawah'} rata-rata seluruh peserta (${df.charges.mean():,.0f}).")

        p1, p2 = st.tabs(["Posisi di distribusi biaya", "Simulasi biaya vs usia"])
        with p1:
            fig5 = px.histogram(df, x="charges", nbins=40, labels={"charges": "Biaya (USD)"})
            fig5.update_traces(marker_color="#BFE9ED", marker_line_width=0)
            fig5.add_vline(x=hasil, line_color=ORANGE, line_width=3, annotation_text="Prediksi Anda",
                           annotation_font_color=ORANGE)
            fig5.add_vline(x=df.charges.mean(), line_dash="dash", line_color=NAVY, annotation_text="Rata-rata",
                           annotation_position="top left", annotation_font_color=NAVY)
            fig5.update_yaxes(title="Jumlah peserta")
            st.plotly_chart(style(fig5, 270, legend=False), width="stretch")
        with p2:
            ages = list(range(18, 65, 2))
            fig3 = go.Figure()
            for s, col, nm in [("no", TEAL, "Non-perokok"), ("yes", ORANGE, "Perokok")]:
                grid = pd.DataFrame({"age": ages, "bmi": bmi, "children": kids, "sex": sex, "smoker": s, "region": region})
                fig3.add_scatter(x=ages, y=model.predict(grid), mode="lines", name=nm, line=dict(color=col, width=3))
            fig3.add_scatter(x=[age], y=[hasil], mode="markers", name="Input Anda",
                             marker=dict(size=15, color=NAVY, symbol="star", line=dict(width=2, color="white")))
            fig3.update_layout(xaxis_title="Usia", yaxis_title="Prediksi biaya (USD)")
            st.plotly_chart(style(fig3, 270), width="stretch")

st.markdown('<div class="foot">Sumber data: Medical Cost Personal Dataset (Brett Lantz, <i>Machine Learning with R</i>) · '
            'github.com/stedy/Machine-Learning-with-R-datasets</div>', unsafe_allow_html=True)