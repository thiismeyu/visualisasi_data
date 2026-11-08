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
# 0. KONFIGURASI HALAMAN
# ------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Biaya Asuransi", page_icon="🏥", layout="wide")

URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
NUM = ["age", "bmi", "children"]
CAT = ["sex", "smoker", "region"]
LABEL = {"age": "Usia", "bmi": "BMI", "children": "Jumlah Anak", "sex": "Jenis Kelamin",
         "smoker": "Perokok", "region": "Wilayah", "kelompok_bmi": "Kategori BMI"}

# ------------------------------------------------------------------
# 1. LOAD & OLAH DATA (sederhana, data sudah bersih)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(URL)
    df["kelompok_bmi"] = pd.cut(df["bmi"], [0, 18.5, 25, 30, 100],
                                labels=["Kurus", "Normal", "Gemuk", "Obesitas"])
    return df

# ------------------------------------------------------------------
# 2. MODEL ML RINGAN: Gradient Boosting Regressor
# ------------------------------------------------------------------
@st.cache_resource
def train_model(n_estimators, max_depth, learning_rate):
    df = load_data()
    X, y = df[NUM + CAT], df["charges"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                            remainder="passthrough")
    model = Pipeline([("pre", pre),
                      ("gb", GradientBoostingRegressor(n_estimators=n_estimators,
                                                       max_depth=max_depth,
                                                       learning_rate=learning_rate,
                                                       random_state=42))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    # pentingnya fitur (digabung per kolom asli)
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
#   Baris 1 : kiri-atas (judul)  ------------->  kanan-atas (filter)
#   Diagonal: KPI + grafik eksplorasi (mata turun ke kiri-bawah)
#   Baris 3 : kiri-bawah (akurasi model) ---->  kanan-bawah (PREDIKSI)
# ==================================================================

# ---------- BARIS 1 : JUDUL (kiri) -> FILTER (kanan) ----------
c_title, c_filter = st.columns([1.2, 1.8], gap="large")
with c_title:
    st.title("🏥 Dashboard Biaya Asuransi Kesehatan")
    st.caption("Panduan: ① atur filter di kanan → ② lihat ringkasan & grafik → ③ coba prediksi biaya di bagian paling bawah.")
with c_filter:
    st.markdown("**🎛️ Filter Data**")
    f1, f2, f3 = st.columns(3)
    sm = f1.multiselect("Perokok", ["yes", "no"], default=["yes", "no"])
    rg = f2.multiselect("Wilayah", sorted(df.region.unique()), default=sorted(df.region.unique()))
    ag = f3.slider("Rentang usia", int(df.age.min()), int(df.age.max()),
                   (int(df.age.min()), int(df.age.max())))

d = df[df.smoker.isin(sm) & df.region.isin(rg) & df.age.between(*ag)]
if d.empty:
    st.warning("Tidak ada data untuk filter ini. Silakan ubah filter.")
    st.stop()

st.divider()

# ---------- DIAGONAL : KPI ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah Peserta", f"{len(d):,}")
k2.metric("Rata-rata Biaya", f"${d.charges.mean():,.0f}")
k3.metric("Biaya Tertinggi", f"${d.charges.max():,.0f}")
k4.metric("% Perokok", f"{(d.smoker == 'yes').mean() * 100:.0f}%")

# ---------- DIAGONAL : GRAFIK EKSPLORASI ----------
g1, g2 = st.columns(2, gap="large")
with g1:
    st.subheader("📈 Apa yang memengaruhi biaya?")
    x_var = st.radio("Sumbu X", ["age", "bmi"], format_func=lambda v: LABEL[v], horizontal=True)
    fig = px.scatter(d, x=x_var, y="charges", color="smoker", opacity=0.7,
                     hover_data=["sex", "region", "children"],
                     labels={x_var: LABEL[x_var], "charges": "Biaya (USD)", "smoker": "Perokok"},
                     color_discrete_map={"yes": "#E4572E", "no": "#2E86AB"})
    fig.update_layout(height=380, margin=dict(t=10, b=10), legend_title_text="Perokok")
    st.plotly_chart(fig)
with g2:
    st.subheader("📊 Bandingkan rata-rata biaya per kelompok")
    grp = st.selectbox("Kelompokkan berdasarkan", ["region", "sex", "smoker", "children", "kelompok_bmi"],
                       format_func=lambda v: LABEL[v])
    agg = d.groupby(grp, observed=True)["charges"].mean().reset_index()
    fig2 = px.bar(agg, x=grp, y="charges", text_auto=".0f", color="charges",
                  color_continuous_scale="Blues",
                  labels={grp: LABEL[grp], "charges": "Rata-rata biaya (USD)"})
    fig2.update_layout(height=380, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig2)

st.divider()

# ---------- BARIS 3 : KIRI (AKURASI MODEL) -> KANAN (PREDIKSI) ----------
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("🤖 Seberapa akurat modelnya?")
    with st.expander("⚙️ Atur parameter model (opsional)"):
        n_est = st.slider("Jumlah pohon (n_estimators)", 50, 400, 150, 50)
        depth = st.slider("Kedalaman pohon (max_depth)", 1, 6, 3)
        lr = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    model, r2, mae, imp_df = train_model(n_est, depth, lr)
    m1, m2 = st.columns(2)
    m1.metric("Akurasi (R²)", f"{r2:.2f}", help="Semakin mendekati 1 semakin akurat")
    m2.metric("Rata-rata Selisih (MAE)", f"${mae:,.0f}", help="Rata-rata selisih prediksi vs data asli")
    fi = px.bar(imp_df, x="Pengaruh", y="Fitur", orientation="h",
                title="Faktor paling berpengaruh pada biaya")
    fi.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fi)

with right:
    st.subheader("🔮 Coba Prediksi Biaya Asuransi")
    st.caption("Ubah nilai di bawah, hasil prediksi langsung berubah.")
    p1, p2, p3 = st.columns(3)
    age = p1.slider("Usia", 18, 64, 35)
    bmi = p2.slider("BMI", 15.0, 50.0, 27.0, 0.5)
    kids = p3.slider("Jumlah anak", 0, 5, 1)
    q1, q2, q3 = st.columns(3)
    smoker = q1.radio("Perokok?", ["no", "yes"], horizontal=True)
    sex = q2.radio("Jenis kelamin", ["female", "male"], horizontal=True)
    region = q3.selectbox("Wilayah", sorted(df.region.unique()))

    row = pd.DataFrame([{"age": age, "bmi": bmi, "children": kids,
                         "sex": sex, "smoker": smoker, "region": region}])
    hasil = model.predict(row)[0]
    kurs = st.number_input("Kurs USD → IDR", 10000, 20000, 16000, 100)

    r1, r2c = st.columns(2)
    r1.metric("Prediksi Biaya / tahun", f"${hasil:,.0f}")
    r2c.metric("Setara (IDR)", f"Rp {hasil * kurs:,.0f}")
    rata = df.charges.mean()
    selisih = (hasil - rata) / rata * 100
    st.info(f"Prediksi ini **{abs(selisih):.0f}% {'lebih tinggi' if selisih > 0 else 'lebih rendah'}** "
            f"dari rata-rata seluruh peserta (${rata:,.0f}).")

    # Grafik what-if: biaya vs usia untuk perokok & non-perokok
    ages = list(range(18, 65, 2))
    fig3 = go.Figure()
    for s, col in [("no", "#2E86AB"), ("yes", "#E4572E")]:
        grid = pd.DataFrame({"age": ages, "bmi": bmi, "children": kids,
                             "sex": sex, "smoker": s, "region": region})
        fig3.add_scatter(x=ages, y=model.predict(grid), mode="lines",
                         name=f"Perokok: {s}", line=dict(color=col, width=3))
    fig3.add_scatter(x=[age], y=[hasil], mode="markers", name="Input Anda",
                     marker=dict(size=14, color="black", symbol="star"))
    fig3.update_layout(height=300, margin=dict(t=30, b=10), title="Simulasi: biaya vs usia (BMI & lainnya tetap)",
                       xaxis_title="Usia", yaxis_title="Prediksi biaya (USD)")
    st.plotly_chart(fig3)

st.caption("Sumber data: Medical Cost Personal Dataset (Brett Lantz, *Machine Learning with R*) — "
           "github.com/stedy/Machine-Learning-with-R-datasets")
