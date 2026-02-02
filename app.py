import os
import requests

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard — Redevabilité Catégories", layout="wide")
st.title("Dashboard — Redevabilité (Catégories)")

PARQUET_URL = "https://huggingface.co/datasets/sabderma/dashboard-streamlit-data/resolve/main/gd_redevabilite_enrichi.parquet"
LOCAL_PARQUET = "gd_redevabilite_enrichi.parquet"

COL_CAT = "CATEGORIE_LIBELLE_2"
COL_SSCAT = "CATEGORIE_LIBELLE_SSCAT"
COL_DATE = "mois_annee"
COL_PRESENCE = "presence_type_coord_03"


# =========================
# DATA LOADER (download + cache)
# =========================
@st.cache_data(show_spinner=True)
def _download_file(url: str, local_path: str) -> str:
    """Download once (cached by Streamlit)."""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    with st.spinner("Téléchargement du fichier parquet (premier lancement)…"):
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)
    return local_path


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    path = _download_file(PARQUET_URL, LOCAL_PARQUET)

    # Lecture parquet
    df = pd.read_parquet(path)

    # Sécuriser colonnes attendues
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
        df["mois"] = df[COL_DATE].dt.to_period("M").dt.to_timestamp()
    else:
        df["mois"] = pd.NaT

    # présence_type_coord_03 => 0/1 int
    if COL_PRESENCE in df.columns:
        df[COL_PRESENCE] = pd.to_numeric(df[COL_PRESENCE], errors="coerce").fillna(0).astype(int)
    else:
        df[COL_PRESENCE] = 0

    # Categorical (perf)
    for col in [COL_CAT, COL_SSCAT]:
        if col in df.columns:
            df[col] = df[col].astype("category")
        else:
            df[col] = pd.Series(["Inconnu"] * len(df), dtype="category")

    return df


df = load_data()


# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filtres")

# Période
if df["mois"].notna().any():
    mois_min = df["mois"].min()
    mois_max = df["mois"].max()

    liste_mois = pd.date_range(mois_min, mois_max, freq="MS")
    if len(liste_mois) == 0:
        liste_mois = pd.date_range(pd.Timestamp.today().normalize(), periods=1, freq="MS")

    mois_range = st.sidebar.select_slider(
        "Période (mois)",
        options=liste_mois,
        value=(liste_mois[0], liste_mois[-1]),
        format_func=lambda d: d.strftime("%Y-%m")
    )
else:
    st.sidebar.warning("Colonne de date indisponible : filtre période désactivé.")
    mois_range = None

# Catégorie
cat_options = sorted(pd.Series(df[COL_CAT]).dropna().astype(str).unique().tolist())
if len(cat_options) == 0:
    cat_options = ["Inconnu"]

cat_filter = st.sidebar.multiselect(
    f"Catégorie ({COL_CAT})",
    options=cat_options,
    default=cat_options
)

st.sidebar.divider()
st.sidebar.subheader("Détail par sous-catégorie")

cat_detail = st.sidebar.selectbox(
    f"Choisir une catégorie ({COL_CAT})",
    options=["(Toutes)"] + cat_options
)

st.sidebar.divider()
st.sidebar.subheader("Pie chart")

cat_detail_pie = st.sidebar.selectbox(
    "Choisir une catégorie pour le pie chart",
    options=["(Choisir)"] + cat_options,
    key="cat_pie"
)


# =========================
# FILTERING
# =========================
mask = pd.Series(True, index=df.index)

if mois_range is not None:
    start_mois, end_mois = mois_range
    mask &= df["mois"].between(start_mois, end_mois)

if cat_filter:
    mask &= df[COL_CAT].astype(str).isin([str(x) for x in cat_filter])

filtered = df.loc[mask].copy()


# =========================
# KPI
# =========================
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total lignes (global)", f"{len(df):,}".replace(",", " "))
k2.metric("Nb catégories (global)", int(df[COL_CAT].nunique(dropna=True)))
k3.metric("Nb sous-catégories (global)", int(df[COL_SSCAT].nunique(dropna=True)))
k4.metric("Nb mois (global)", int(df["mois"].nunique(dropna=True)))
k5.metric("Nb avec type coordonnée 03 (OUI)", int(df[COL_PRESENCE].sum()))

st.divider()
st.info("✅ OUI : la redevabilité possède au moins une coordonnée de type 03 (adresse e-mail).")
st.info("❌ NON : la redevabilité ne possède aucune coordonnée de type 03 (adresse e-mail).")
st.divider()


# =========================
# GRAPH 1 — Stacked bars + totals + % inside OUI
# =========================
tmp = filtered.copy()

agg = (
    tmp.groupby(COL_CAT)[COL_PRESENCE]
    .agg(total="size", oui="sum")
    .reset_index()
)

agg["non"] = agg["total"] - agg["oui"]
agg["Catégorie"] = agg[COL_CAT].astype(str).replace("nan", "Inconnu")
agg = agg.sort_values("total", ascending=False)

agg["pct_oui"] = (agg["oui"] / agg["total"] * 100).round(1).fillna(0)
agg["label_total"] = agg["total"].astype(int).astype(str)

long_df = agg.melt(
    id_vars=["Catégorie", "total", "pct_oui", "label_total"],
    value_vars=["non", "oui"],
    var_name="Présence type 03",
    value_name="Nombre"
)
long_df["Présence type 03"] = long_df["Présence type 03"].map({"oui": "OUI", "non": "NON"})

long_df["text_in"] = long_df.apply(
    lambda r: f"{r['pct_oui']}%" if r["Présence type 03"] == "OUI" else "",
    axis=1
)

fig1 = go.Figure()

df_non = long_df[long_df["Présence type 03"] == "NON"]
fig1.add_trace(go.Bar(
    x=df_non["Catégorie"],
    y=df_non["Nombre"],
    name="NON",
    text=[""] * len(df_non),
    textposition="inside"
))

df_oui = long_df[long_df["Présence type 03"] == "OUI"]
fig1.add_trace(go.Bar(
    x=df_oui["Catégorie"],
    y=df_oui["Nombre"],
    name="OUI",
    text=df_oui["text_in"],
    textposition="inside"
))

fig1.add_trace(go.Scatter(
    x=agg["Catégorie"],
    y=agg["total"],
    mode="text",
    text=agg["label_total"],
    textposition="top center",
    showlegend=False
))

fig1.update_layout(
    barmode="stack",
    title="Répartition par catégorie (filtré) + présence type coordonnée 03",
    xaxis_title="Catégorie",
    yaxis_title="Nombre de lignes",
    xaxis=dict(
        categoryorder="array",
        categoryarray=agg["Catégorie"].tolist(),
        tickangle=-30
    ),
    legend_title="Type coordonnée 03",
    uniformtext_minsize=10,
    uniformtext_mode="hide"
)

st.plotly_chart(fig1, use_container_width=True)


# =========================
# GRAPH 2 — Drill-down sous-catégories (top 10)
# =========================
detail_df = filtered
if cat_detail != "(Toutes)":
    detail_df = detail_df[detail_df[COL_CAT].astype(str) == str(cat_detail)]

sscat_count = (
    detail_df[COL_SSCAT]
    .astype(str)
    .fillna("Inconnu")
    .value_counts(dropna=False)
    .head(10)
    .reset_index()
)
sscat_count.columns = ["Sous-catégorie", "Nombre"]

title_cat = cat_detail if cat_detail != "(Toutes)" else "Toutes catégories"

fig2 = px.bar(
    sscat_count,
    x="Nombre",
    y="Sous-catégorie",
    orientation="h",
    text="Nombre",
    title=f"Sous-catégories — {title_cat} (filtré)"
)
fig2.update_traces(textposition="outside")
fig2.update_layout(
    xaxis_title="Nombre de lignes",
    yaxis_title="Sous-catégorie",
    uniformtext_minsize=10,
    uniformtext_mode="hide"
)
st.plotly_chart(fig2, use_container_width=True)


# =========================
# GRAPH 3 — Pie chart OUI/NON pour une catégorie
# =========================
if cat_detail_pie == "(Choisir)":
    st.info("Choisis une catégorie dans le filtre 'Pie chart' pour afficher le graphique.")
else:
    pie_df = filtered[filtered[COL_CAT].astype(str) == str(cat_detail_pie)].copy()

    pie_counts = (
        pie_df[COL_PRESENCE]
        .value_counts()
        .reindex([1, 0], fill_value=0)
        .reset_index()
    )
    pie_counts.columns = ["Présence type 03", "Nombre"]
    pie_counts["Présence type 03"] = pie_counts["Présence type 03"].map({1: "OUI", 0: "NON"})

    fig3 = px.pie(
        pie_counts,
        names="Présence type 03",
        values="Nombre",
        title=f"Présence type coordonnée 03 — {cat_detail_pie} (filtré)"
    )
    fig3.update_traces(textinfo="percent+label", pull=[0.02, 0.02])
    fig3.update_layout(legend_title="Type coordonnée 03")

    st.plotly_chart(fig3, use_container_width=True)
