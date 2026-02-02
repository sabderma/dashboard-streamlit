import pandas as pd
import streamlit as st
import plotly.express as px



st.set_page_config(page_title="Dashboard — Redevabilité Catégories", layout="wide")
st.title("Dashboard — Redevabilité (Catégories)")

# -------------------------------------------------malek----
@st.cache_data
def load_data():
    df = pd.read_parquet("gd_redevabilite_enrichi.parquet")

    # Dates
    df["mois_annee"] = pd.to_datetime(df["mois_annee"], errors="coerce")

    # Mois pré-calculé (plus rapide)
    df["mois"] = df["mois_annee"].dt.to_period("M").dt.to_timestamp()

    # Colonnes catégorielles utiles uniquement
    for col in ["CATEGORIE_LIBELLE_2", "CATEGORIE_LIBELLE_SSCAT"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

df = load_data()

# -------------------------------------------------
# Sidebar filtres
# -------------------------------------------------
st.sidebar.header("Filtres")

# Période
mois_min = df["mois"].min()
mois_max = df["mois"].max()


liste_mois = pd.date_range(mois_min, mois_max, freq="MS")  

mois_range = st.sidebar.select_slider(
    "Période (mois)",
    options=liste_mois,
    value=(liste_mois[0], liste_mois[-1]),
    format_func=lambda d: d.strftime("%Y-%m")
)


# Catégorie
cat_options = sorted(df["CATEGORIE_LIBELLE_2"].dropna().unique())
cat_filter = st.sidebar.multiselect(
    "Catégorie (CATEGORIE_LIBELLE_2)",
    options=cat_options,
    default=cat_options
)

st.sidebar.divider()
st.sidebar.subheader("Détail par sous-catégorie")

cat_detail = st.sidebar.selectbox(
    "Choisir une catégorie (CATEGORIE_LIBELLE_2)",
    options=["(Toutes)"] + cat_options
)


st.sidebar.divider()
st.sidebar.subheader("pie chart")
cat_detail_pie = st.sidebar.selectbox(
    "Choisir une catégorie pour le pie chart",
    options=["il faut choisir une categorie"] + cat_options,
    key="cat_pie"
)

# -------------------------------------------------
# Filtrage  (mask)
# -------------------------------------------------
mask = pd.Series(True, index=df.index)

start_mois, end_mois = mois_range
mask &= df["mois"].between(start_mois, end_mois)

if cat_filter:
    mask &= df["CATEGORIE_LIBELLE_2"].isin(cat_filter)

filtered = df.loc[mask]


# -------------------------------------------------
# KPI 
# -------------------------------------------------
k1, k2, k3, k4 ,k5 =st.columns(5)    

k1.metric("Total lignes (global)", f"{len(df):,}".replace(",", " "))
k2.metric("Nb catégories (global)", df["CATEGORIE_LIBELLE_2"].nunique(dropna=True))
k3.metric("Nb sous-catégories (global)", df["CATEGORIE_LIBELLE_SSCAT"].nunique(dropna=True))
k4.metric("Nb mois (global)", df["mois"].nunique())

k5.metric(
    "Nb avec type coordonnée 03 (OUI)",
    int(df["presence_type_coord_03"].fillna(0).sum())
)

st.divider()

st.info(" OUI : la redevabilité possède au moins une coordonnée de type 03 (adresse e-mail).")
st.info( "NON : la redevabilité ne possède aucune coordonnée de type 03 (adresse e-mail).")

st.divider()

# -------------------------------------------------
# Graph 1 : Catégories triées + Total au-dessus + % OUI dans la barre OUI
# -------------------------------------------------
COL_PRESENCE = "presence_type_coord_03"  # <-- adapte si besoin

tmp = filtered.copy()

# Sécuriser en 0/1
tmp[COL_PRESENCE] = pd.to_numeric(tmp[COL_PRESENCE], errors="coerce").fillna(0).astype(int)

# Agrégation par catégorie
agg = (
    tmp.groupby("CATEGORIE_LIBELLE_2")[COL_PRESENCE]
    .agg(total="size", oui="sum")
    .reset_index()
)
agg["non"] = agg["total"] - agg["oui"]
agg["Catégorie"] = agg["CATEGORIE_LIBELLE_2"].astype(str).replace("nan", "Inconnu")

# Trier barres du + grand au + petit (total)
agg = agg.sort_values("total", ascending=False)

# % OUI
agg["pct_oui"] = (agg["oui"] / agg["total"] * 100).round(1).fillna(0)

# Labels
agg["label_total"] = agg["total"].astype(int).astype(str)

# Long format pour stacked
long_df = agg.melt(
    id_vars=["Catégorie", "total", "pct_oui", "label_total"],
    value_vars=["non", "oui"],
    var_name="Présence type 03",
    value_name="Nombre"
)
long_df["Présence type 03"] = long_df["Présence type 03"].map({"oui": "OUI", "non": "NON"})

# Texte à l’intérieur : % seulement sur OUI
long_df["text_in"] = long_df.apply(
    lambda r: f"{r['pct_oui']}%" if r["Présence type 03"] == "OUI" else "",
    axis=1
)

import plotly.graph_objects as go

fig1 = go.Figure()

# NON
df_non = long_df[long_df["Présence type 03"] == "NON"]
fig1.add_trace(go.Bar(
    x=df_non["Catégorie"],
    y=df_non["Nombre"],
    name="NON",
    text=[""] * len(df_non),
    textposition="inside"
))

# OUI (% à l'intérieur)
df_oui = long_df[long_df["Présence type 03"] == "OUI"]
fig1.add_trace(go.Bar(
    x=df_oui["Catégorie"],
    y=df_oui["Nombre"],
    name="OUI",
    text=df_oui["text_in"],
    textposition="inside"
))

# Total au-dessus (extérieur)
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


# -------------------------------------------------
# Graph 2 : Drill-down sous-catégories
# -------------------------------------------------
detail_df = filtered
if cat_detail != "(Toutes)":
    detail_df = detail_df[detail_df["CATEGORIE_LIBELLE_2"] == cat_detail]

sscat_count = (
    detail_df["CATEGORIE_LIBELLE_SSCAT"]
    .value_counts(dropna=False)
    .head(10)
    .reset_index()
)
sscat_count.columns = ["Sous-catégorie", "Nombre"]
sscat_count["Sous-catégorie"] = sscat_count["Sous-catégorie"].astype(str).replace("nan", "Inconnu")

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


# -------------------------------------------------
# Graph 3 : pie chart (OUI vs NON) pour UNE catégorie
# -------------------------------------------------

COL_PRESENCE = "presence_type_coord_03"

if cat_detail_pie == "il faut choisir une categorie":
    st.info(" Choisis une catégorie dans le filtre 'pie chart' pour afficher le graphique.")
else:
    pie_df = filtered[filtered["CATEGORIE_LIBELLE_2"] == cat_detail_pie].copy()

    # Sécuriser presence en 0/1
    pie_df[COL_PRESENCE] = pd.to_numeric(pie_df[COL_PRESENCE], errors="coerce").fillna(0).astype(int)
 

 
    # Compter OUI/NON
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
    fig3.update_layout(
        legend_title="Type coordonnée 03"
    )

    st.plotly_chart(fig3, use_container_width=True)
