import streamlit as st
import pandas as pd
import requests
from typing import Tuple

# Optional deps
try:
    import geopandas as gpd
    from shapely.geometry import shape
except Exception:
    gpd = None

try:
    import folium
    from streamlit_folium import st_folium
    from branca.colormap import linear as cm_linear
except Exception:
    folium = None
    st_folium = None

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Resultados Electorales – CABA", page_icon="🗳️", layout="wide")
VIOLETA = "#7030a0"
APP_BG = "#6c4c99ff"
APP_TEXT = "#ffffff"
BAR_COLOR = "#371859ff"
LINE_COLORS = {
    "LLA": "#5f497aff",
    "FUERZA PATRIA": "#00bfff",  # celeste
    "ALIANZA POTENCIA": "#800020",  # bordo
}

# ===== Estilos globales y “tarjetas” =====
st.markdown(f"""
<style>
  .stApp {{
    background-color: {APP_BG};
    color: {APP_TEXT};
    font-family: 'Montserrat', sans-serif;
  }}

  /* Títulos y textos en blanco */
  .stMarkdown, .stMarkdown p, h1, h2, h3, h4, h5, h6,
  .stCaption, label {{ color: {APP_TEXT} !important; }}

  /* Tabs y radios */
  .stTabs [role="tab"], .stTabs [role="tab"] p {{ color: {APP_TEXT} !important; }}
  [role="radiogroup"] label, [role="radiogroup"] label p {{ color: {APP_TEXT} !important; }}

  /* Select/Multiselect */
  div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.10);
    border: 1px solid #6c4c99ff !important;
    color: {APP_TEXT} !important;
    border-radius: 10px;
  }}
  div[data-baseweb="select"] span,
  div[data-baseweb="select"] input {{ color: {APP_TEXT} !important; }}
  div[data-baseweb="menu"] {{ background-color: rgba(0,0,0,0.35); color: {APP_TEXT}; }}
  div[data-baseweb="option"] {{ color: {APP_TEXT}; }}

  /* Contenedor tipo tarjeta */
  .rounded-box {{
    background: rgba(255, 255, 255, 0.07);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    margin-bottom: 24px;
  }}

  /* Borde redondeado y sombra en iframes (mapa/vega) */
  iframe, .folium-map, .vega-embed {{
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.25);
  }}
</style>
""", unsafe_allow_html=True)

# Defaults (editables desde la barra lateral)
DEFAULT_GEO_URL = "https://raw.githubusercontent.com/tartagalensis/circuitos_electorales_AR/main/geojson/CABA.geojson"
DEFAULT_DIP_URL = "https://raw.githubusercontent.com/Facunfer/elecciones_octubre_2025/refs/heads/main/CSV%20RESULTADOS%20-%20diputados.csv"
DEFAULT_SEN_URL = "https://raw.githubusercontent.com/Facunfer/elecciones_octubre_2025/refs/heads/main/CSV%20RESULTADOS%20-%20senadores.csv"

PARTY_LLA = "ALIANZA LA LIBERTAD AVANZA"
PARTY_FUERZA = "FUERZA PATRIA"
PARTY_POTENCIA = "ALIANZA POTENCIA"

# =============================
# HELPERS
# =============================
@st.cache_data(show_spinner=False)
def _github_to_raw(url: str) -> str:
    """Convierte URLs tipo https://github.com/user/repo/blob/branch/file.csv a RAW."""
    if not url:
        return url
    if "github.com" in url and "/blob/" in url:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return url

@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    url = _github_to_raw(url or "")
    if not url:
        return pd.DataFrame()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    from io import BytesIO
    df = pd.read_csv(BytesIO(resp.content))
    # normalizo columnas
    df.columns = [c.strip().upper() for c in df.columns]
    if "CIRCUITO" in df.columns:
        df["CIRCUITO"] = df["CIRCUITO"].astype(str)
    if "COMUNA" in df.columns:
        df["COMUNA"] = df["COMUNA"].astype(str)
    if "AGRUPACION_NOMBRE" in df.columns:
        df["AGRUPACION_NOMBRE"] = df["AGRUPACION_NOMBRE"].astype(str).str.upper().str.replace("\s+", " ", regex=True)
    if "VOTOS_CANTIDAD" in df.columns:
        df["VOTOS_CANTIDAD"] = pd.to_numeric(df["VOTOS_CANTIDAD"], errors="coerce").fillna(0)
    return df

@st.cache_data(show_spinner=False)
def load_geo(url: str) -> Tuple[pd.DataFrame, dict]:
    """Carga GeoJSON desde URL y devuelve (GeoDataFrame/DataFrame, raw)."""
    url = _github_to_raw(url or "")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    gj = resp.json()
    if gpd is not None:
        feats = []
        for f in gj.get("features", []):
            p = f.get("properties", {}).copy()
            try:
                p["geometry"] = shape(f["geometry"])  # type: ignore
            except Exception:
                pass
            feats.append(p)
        gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs="EPSG:4326")
    else:
        rows = [f.get("properties", {}) for f in gj.get("features", [])]
        gdf = pd.DataFrame(rows)
    for c in ["circuito", "coddepto"]:
        if c in gdf.columns:
            gdf[c] = gdf[c].astype(str)
    return gdf, gj

# ===== Métricas y validaciones =====
def assert_required(df: pd.DataFrame) -> None:
    req = {"SECCION_NOMBRE", "COMUNA", "CIRCUITO", "AGRUPACION_NOMBRE", "VOTOS_CANTIDAD"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        st.error("Faltan columnas requeridas en el CSV: " + ", ".join(missing))
        st.stop()

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Construye métricas por circuito: totales, votos LLA/FP/AP y porcentajes."""
    if df.empty:
        return df
    # Totales por circuito
    tot = (
        df.groupby(["COMUNA", "CIRCUITO"], as_index=False)["VOTOS_CANTIDAD"]
          .sum().rename(columns={"VOTOS_CANTIDAD": "TOTAL_VOTOS"})
    )

    def votes_for(party: str) -> pd.DataFrame:
        part = df[df["AGRUPACION_NOMBRE"] == party]
        if part.empty:
            return pd.DataFrame({"COMUNA": [], "CIRCUITO": [], f"VOTOS_{party}": []})
        return (
            part.groupby(["COMUNA", "CIRCUITO"], as_index=False)["VOTOS_CANTIDAD"].sum()
                .rename(columns={"VOTOS_CANTIDAD": f"VOTOS_{party}"})
        )

    lla = votes_for(PARTY_LLA)
    fp  = votes_for(PARTY_FUERZA)
    ap  = votes_for(PARTY_POTENCIA)

    out = (
        tot.merge(lla, on=["COMUNA", "CIRCUITO"], how="left")
           .merge(fp,  on=["COMUNA", "CIRCUITO"], how="left")
           .merge(ap,  on=["COMUNA", "CIRCUITO"], how="left")
    )

    for c in [f"VOTOS_{PARTY_LLA}", f"VOTOS_{PARTY_FUERZA}", f"VOTOS_{PARTY_POTENCIA}"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["TOTAL_VOTOS"] = pd.to_numeric(out["TOTAL_VOTOS"], errors="coerce").fillna(0)

    # Porcentajes
    out["PORC_LLA"]      = (out[f"VOTOS_{PARTY_LLA}"]     / out["TOTAL_VOTOS"].replace(0, pd.NA) * 100).fillna(0.0)
    out["PORC_FUERZA"]   = (out[f"VOTOS_{PARTY_FUERZA}"]  / out["TOTAL_VOTOS"].replace(0, pd.NA) * 100).fillna(0.0)
    out["PORC_POTENCIA"] = (out[f"VOTOS_{PARTY_POTENCIA}"] / out["TOTAL_VOTOS"].replace(0, pd.NA) * 100).fillna(0.0)

    return out

# ===== Utilidades GeoJSON =====
def _detect_pad_len(geo_df: pd.DataFrame) -> int:
    try:
        return int(geo_df["circuito"].astype(str).str.len().max())
    except Exception:
        return 5

def enrich_geojson_with_data(geojson_raw: dict, data: pd.DataFrame) -> dict:
    """Anexa métricas (TOTAL_VOTOS, PORC_*) a cada feature por circuito para tooltips."""
    by_circ = data.set_index("CIRCUITO").to_dict(orient="index")
    import copy
    gj = copy.deepcopy(geojson_raw)
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        circ = str(props.get("circuito"))
        row = by_circ.get(circ)
        if row:
            for key in [k for k in row.keys() if k in {"TOTAL_VOTOS","PORC_LLA","PORC_FUERZA","PORC_POTENCIA"} or k.startswith("VOTOS_")]:
                props[key] = row[key]
        feat["properties"] = props
    return gj

def _filter_geojson_by_circuits(geojson_raw: dict, circuits: set) -> dict:
    import copy
    if not circuits:
        return geojson_raw
    gj = copy.deepcopy(geojson_raw)
    feats = []
    for f in gj.get("features", []):
        props = f.get("properties", {})
        if str(props.get("circuito")) in circuits:
            feats.append(f)
    gj["features"] = feats
    return gj

# ===== Visualizaciones =====
def make_map(geojson_raw: dict, joined_df: pd.DataFrame, metric_col: str, legend: str):
    if folium is None or st_folium is None:
        st.warning("Instalá `folium` y `streamlit-folium` para ver el mapa. Se mostrarán tablas y gráficos igualmente.")
        return None

    # Enriquecemos GeoJSON para tooltips con métricas
    gj_enriched = enrich_geojson_with_data(geojson_raw, joined_df)

    # Centro en CABA y luego ajusto a bounds
    m = folium.Map(location=[-34.61, -58.44], tiles="cartodbpositron", zoom_start=11, control_scale=True)

    # Escala continua y suave: VERDE (bajo) → ROJO (alto)
    vals = pd.to_numeric(joined_df[metric_col], errors="coerce").fillna(0)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        vmax = vmin + (0.0001 if metric_col.startswith("PORC") else 1)
    cmap = cm_linear.RdYlGn_11.scale(vmin, vmax)  # rojo=bajo → verde=alto
    cmap.caption = legend

    # Diccionario valor por circuito
    v_by_circ = {str(r["CIRCUITO"]): float(r[metric_col]) for _, r in joined_df.iterrows()}

    def style_fn(feat):
        circ = str(feat.get("properties", {}).get("circuito"))
        val = v_by_circ.get(circ)
        if val is None or pd.isna(val):
            return {"fillColor": "#00000000", "color": "#555", "weight": 0.7, "fillOpacity": 0.0}
        color = cmap(val)
        return {"fillColor": color, "color": "#555", "weight": 0.7, "fillOpacity": 0.85}

    gj = folium.GeoJson(
        data=gj_enriched,
        style_function=style_fn,
        highlight_function=lambda f: {"weight": 2, "color": "#000"},
        tooltip=folium.GeoJsonTooltip(
            fields=["circuito", "coddepto", "TOTAL_VOTOS", "PORC_LLA"],
            aliases=["Circuito", "Comuna", "Total votos", "% LLA"],
            localize=True,
            sticky=True,
        ),
        name=legend,
    )
    gj.add_to(m)
    cmap.add_to(m)

    # Fit bounds a features visibles
    try:
        bounds = gj.get_bounds()  # type: ignore
        m.fit_bounds(bounds, padding=(10, 10))
    except Exception:
        pass

    folium.LayerControl(collapsed=True).add_to(m)
    return m

def top_bars(df: pd.DataFrame, by_col: str, title: str):
    try:
        import altair as alt
    except Exception:
        st.warning("Falta `altair` para los gráficos de barras. Instalá con: pip install altair")
        return

    if df.empty or by_col not in df.columns:
        st.info("Sin datos para el ranking.")
        return

    # Ordenar desc y concatenar etiqueta eje X: SECCION_NOMBRE - CIRCUITO
    df2 = df.copy()
    if "SECCION_NOMBRE" not in df2.columns:
        df2["SECCION_NOMBRE"] = ""
    df2["LABEL"] = (df2["SECCION_NOMBRE"].astype(str).fillna("") + " - " + df2["CIRCUITO"].astype(str))

    top10 = df2.sort_values(by_col, ascending=False).head(10)
    # Eje X en el orden ya ordenado:
    ordered_labels = top10["LABEL"].tolist()
    top10["LABEL"] = pd.Categorical(top10["LABEL"], categories=ordered_labels, ordered=True)

    chart = (
        alt.Chart(top10)
        .mark_bar(color=BAR_COLOR)
        .encode(
            x=alt.X("LABEL:N", sort=ordered_labels, title="Sección - Circuito"),
            y=alt.Y(f"{by_col}:Q", title=by_col),
            tooltip=["LABEL", alt.Tooltip(f"{by_col}:Q", format=".2f")],
        )
        .properties(height=320)
    )

    # Etiquetas en negro sobre cada barra
    text = (
        alt.Chart(top10)
        .mark_text(align='center', baseline='bottom', dy=-6, color='black')
        .encode(
            x=alt.X("LABEL:N", sort=ordered_labels),
            y=alt.Y(f"{by_col}:Q"),
            text=alt.Text(f"{by_col}:Q", format=".0f"),
        )
    )

    st.altair_chart(chart + text, use_container_width=True)
    st.caption(title)

def line_party(df: pd.DataFrame):
    try:
        import altair as alt
    except Exception:
        st.warning("Falta `altair` para el gráfico de líneas. Instalá con: pip install altair")
        return

    base = df.copy().sort_values("CIRCUITO")  # menor → mayor por circuito
    melted = base.melt(id_vars=["CIRCUITO"], value_vars=["PORC_LLA", "PORC_FUERZA", "PORC_POTENCIA"],
                       var_name="PARTIDO", value_name="PORC")
    partido_map = {"PORC_LLA": "LLA", "PORC_FUERZA": "FUERZA PATRIA", "PORC_POTENCIA": "ALIANZA POTENCIA"}
    melted["PARTIDO"] = melted["PARTIDO"].map(partido_map)

    line = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("CIRCUITO:N", sort=None, title="Circuito"),
            y=alt.Y("PORC:Q", title="% votos"),
            color=alt.Color("PARTIDO:N", title="Partido",
                            scale=alt.Scale(domain=["LLA","FUERZA PATRIA","ALIANZA POTENCIA"],
                                            range=[LINE_COLORS["LLA"], LINE_COLORS["FUERZA PATRIA"], LINE_COLORS["ALIANZA POTENCIA"]])),
            tooltip=["CIRCUITO", "PARTIDO", alt.Tooltip("PORC:Q", format=".2f")],
        )
        .properties(height=340)
    )

    # Etiquetas negras en cada punto (valor)
    text = (
        alt.Chart(melted)
        .mark_text(dy=-8, color='black')
        .encode(
            x=alt.X("CIRCUITO:N"),
            y=alt.Y("PORC:Q"),
            text=alt.Text("PORC:Q", format=".1f"),
            detail="PARTIDO:N",
            color=alt.value("black")
        )
    )

    # Reglas de promedio por cada serie
    layers = [line, text]
    for col, nombre, color_hex in [
        ("PORC_LLA", "Prom. LLA", LINE_COLORS["LLA"]),
        ("PORC_FUERZA", "Prom. Fuerza", LINE_COLORS["FUERZA PATRIA"]),
        ("PORC_POTENCIA", "Prom. Potencia", LINE_COLORS["ALIANZA POTENCIA"])
    ]:
        if col in df.columns:
            mean_val = float(pd.to_numeric(df[col], errors="coerce").mean())
            rule = alt.Chart(pd.DataFrame({"y": [mean_val]})).mark_rule(color=color_hex, strokeDash=[4,3]).encode(y="y:Q")
            label = alt.Chart(pd.DataFrame({"y": [mean_val], "txt":[nombre]})).mark_text(
                align="left", dx=5, dy=-4, color="black"
            ).encode(y="y:Q", text="txt:N")
            layers.extend([rule, label])

    st.altair_chart(alt.layer(*layers), use_container_width=True)

# =============================
# UI
# =============================
st.title("🗳️ Elecciones Octubre 2025")

with st.sidebar:
    st.subheader("Fuentes de datos (GitHub / RAW)")
    geo_url = st.text_input("URL GeoJSON", value=DEFAULT_GEO_URL, help="Pegá URL GitHub o RAW; se convierte a RAW automáticamente.")
    dip_url = st.text_input("CSV Diputados (URL)", value=DEFAULT_DIP_URL)
    sen_url = st.text_input("CSV Senadores (URL)", value=DEFAULT_SEN_URL)
    st.caption("Tip: usá URLs RAW de GitHub.")

# Cargar datos desde URLs
with st.spinner("Cargando datos…"):
    try:
        geo_gdf, geo_raw = load_geo(geo_url)
    except Exception as e:
        geo_gdf, geo_raw = pd.DataFrame(), {}
        st.error(f"No se pudo cargar GeoJSON: {e}")

    try:
        df_dip = read_csv_url(dip_url) if dip_url else pd.DataFrame()
        df_sen = read_csv_url(sen_url) if sen_url else pd.DataFrame()
    except Exception as e:
        st.error(f"No se pudieron cargar los CSV desde URL: {e}")
        df_dip, df_sen = pd.DataFrame(), pd.DataFrame()

# Validaciones mínimas
if not df_dip.empty:
    assert_required(df_dip)
else:
    st.warning("⚠️ Cargá la URL del CSV de Diputados en la barra lateral.")
if not df_sen.empty:
    assert_required(df_sen)
else:
    st.warning("⚠️ Cargá la URL del CSV de Senadores en la barra lateral.")

TAB_SEN, TAB_DIP = st.tabs(["Senadores", "Diputados"])

def tab_body(nombre: str, df_cat: pd.DataFrame):
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.subheader(nombre)
    if df_cat.empty:
        st.info("Subí un CSV válido para esta solapa.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Ajusto padding de CIRCUITO para que coincida con GeoJSON
    pad_len = _detect_pad_len(geo_gdf) if not geo_gdf.empty else 5
    df_cat = df_cat.copy()
    if "CIRCUITO" in df_cat.columns:
        df_cat["CIRCUITO"] = df_cat["CIRCUITO"].astype(str).str.zfill(pad_len)

    # Multiselección por sección (default: vacío) con opción "Todas"
    secciones = sorted(df_cat["SECCION_NOMBRE"].dropna().astype(str).unique()) if "SECCION_NOMBRE" in df_cat.columns else []
    sel_secciones = st.multiselect(
        "Filtrar por SECCIÓN (seccion_nombre)", options=["Todas"] + secciones, default=[], key=f"sec_multi_{nombre}"
    )

    if (not sel_secciones) or ("Todas" in sel_secciones):
        df_fil = df_cat
    else:
        df_fil = df_cat[df_cat["SECCION_NOMBRE"].isin(sel_secciones)]

    met = st.radio("Métrica para mapa y rankings", ["Cantidad de votos LLA", "% LLA"], horizontal=True, key=f"met_{nombre}")
    st.markdown('</div>', unsafe_allow_html=True)

    met_df = compute_metrics(df_fil)
    if met_df.empty:
        st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
        st.info("No hay datos para la selección actual.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Añadir SECCION_NOMBRE a met_df para etiquetas de barras
    if "SECCION_NOMBRE" in df_fil.columns:
        secc_map = df_fil[["CIRCUITO", "SECCION_NOMBRE"]].drop_duplicates().groupby("CIRCUITO").first().reset_index()
        met_df = met_df.merge(secc_map, on="CIRCUITO", how="left")
    else:
        met_df["SECCION_NOMBRE"] = ""

    # Aseguro padding en met_df también
    met_df["CIRCUITO"] = met_df["CIRCUITO"].astype(str).str.zfill(pad_len)

    # Join con geo y limitación de features a sólo los circuitos filtrados
    if not geo_gdf.empty:
        gjoin = geo_gdf[["circuito", "coddepto"]].copy()
        gjoin.rename(columns={"circuito": "CIRCUITO", "coddepto": "CODDEPTO"}, inplace=True)
        gjoin["CIRCUITO"] = gjoin["CIRCUITO"].astype(str).str.zfill(pad_len)
        gjoin = gjoin.merge(met_df, on="CIRCUITO", how="left")
    else:
        gjoin = met_df.copy()

    # ===== Mapa =====
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Mapa coroplético por circuito")
    metric_col, legend = (f"VOTOS_{PARTY_LLA}", "Votos LLA") if met == "Cantidad de votos LLA" else ("PORC_LLA", "% LLA")

    if geo_raw:
        circuits_selected = set(met_df["CIRCUITO"].astype(str).unique())
        filtered_geo = _filter_geojson_by_circuits(geo_raw, circuits_selected)
        m = make_map(filtered_geo, gjoin[gjoin["CIRCUITO"].isin(circuits_selected)], metric_col=metric_col, legend=legend)
        if m is not None and st_folium is not None:
            st_folium(m, width=None, height=580, key=f"map_{nombre}")
    else:
        st.warning("Sin GeoJSON cargado: se muestran solo tablas y gráficos.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== Barras =====
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Análisis por circuitos")
    top_bars(met_df, by_col=f"VOTOS_{PARTY_LLA}", title="Top 10 – Votos LLA")
    top_bars(met_df, by_col="PORC_LLA", title="Top 10 – % LLA")
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== Línea =====
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.markdown("### 📈 % por circuito – LLA / Fuerza Patria / Potencia (máx. 20 circuitos)")
    met_df_20 = met_df.sort_values(f"VOTOS_{PARTY_LLA}", ascending=False).head(20)
    try:
        line_party(met_df_20)
    except Exception as e:
        st.warning(f"No se pudo renderizar el gráfico de líneas (Altair): {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with TAB_SEN:
    tab_body("Senadores", df_sen)

with TAB_DIP:
    tab_body("Diputados", df_dip)

# =============================
# TESTS (modo script, no interfieren con Streamlit)
# =============================
if __name__ == "__main__":
    # Test de conversión de URL GitHub a RAW
    assert _github_to_raw("https://github.com/user/repo/blob/main/file.csv") == "https://raw.githubusercontent.com/user/repo/main/file.csv"

    # Tests mínimos para compute_metrics
    _df = pd.DataFrame({
        "SECCION_NOMBRE": ["Sec A"]*6,
        "COMUNA": ["1","1","1","1","1","1"],
        "CIRCUITO": ["1001","1001","1002","1002","1003","1003"],
        "AGRUPACION_NOMBRE": [PARTY_LLA, PARTY_FUERZA, PARTY_LLA, PARTY_POTENCIA, PARTY_FUERZA, PARTY_POTENCIA],
        "VOTOS_CANTIDAD": [100, 50, 200, 20, 10, 5],
    })
    _m = compute_metrics(_df)
    assert {"COMUNA","CIRCUITO","TOTAL_VOTOS","PORC_LLA","PORC_FUERZA","PORC_POTENCIA"}.issubset(_m.columns)
    row1001 = _m[_m["CIRCUITO"]=="1001"].iloc[0]
    assert int(row1001["TOTAL_VOTOS"]) == 150
    assert abs(float(row1001["PORC_LLA"]) - (100/150*100)) < 1e-6
    print("Tests OK")
