# -*- coding: utf-8 -*-
"""
Resultados Electorales CABA – Octubre 2025
Mapa coroplético por circuito electoral.

Cambios clave respecto de la versión anterior:
  1) El GeoJSON se puede cargar por ARCHIVO (uploader), RUTA LOCAL o URL.
  2) La clave de circuito se detecta automáticamente en el GeoJSON y se
     normaliza a entero, así "00001", "0001", "1", 1 y "CIR 1" matchean igual.
  3) Panel de diagnóstico que dice exactamente cuántos circuitos cruzaron.
  4) Se agregaron los rankings/tabla que antes estaban definidos pero nunca
     se llamaban.
"""

import io
import json
import re
import copy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ---- Dependencias opcionales -------------------------------------------------
try:
    import folium
    from folium.plugins import Fullscreen
    from branca.colormap import linear as cm_linear
except Exception:  # pragma: no cover
    folium = None
    Fullscreen = None
    cm_linear = None

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None


# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Resultados Electorales – CABA", page_icon="🗳️", layout="wide")

APP_BG = "#6c4c99"
APP_TEXT = "#ffffff"
BAR_COLOR = "#371859"

DEFAULT_GEO_URL = (
    "https://datar.info/dataset/fc99d6a8-9b47-4e9e-9436-58038735a5eb/resource/"
    "c836bb95-4e4e-4b33-8578-057175748d89/download/circuitoselectoralescaba.json"
)
DEFAULT_DIP_URL = (
    "https://raw.githubusercontent.com/Facunfer/elecciones_octubre_2025/"
    "refs/heads/main/CSV%20RESULTADOS%20-%20diputados.csv"
)
DEFAULT_SEN_URL = (
    "https://raw.githubusercontent.com/Facunfer/elecciones_octubre_2025/"
    "refs/heads/main/CSV%20RESULTADOS%20-%20senadores.csv"
)

PARTIES = {
    "LLA": "ALIANZA LA LIBERTAD AVANZA",
    "FUERZA": "FUERZA PATRIA",
    "POTENCIA": "ALIANZA POTENCIA",
}

# Nombres posibles de la propiedad "circuito" dentro del GeoJSON
CIRC_KEYS = [
    "circuito", "CIRCUITO", "circuito_id", "CIRCUITO_ID", "cir", "CIR",
    "circ", "CIRC", "codigo_circuito", "cod_circuito", "circuito_nro",
    "nro_circuito", "id_circuito",
]
COMUNA_KEYS = ["coddepto", "COMUNA", "comuna", "departamento", "depto", "cod_comuna", "barrio"]

# Fondos de mapa sin API key (CartoDB ya la exige en versiones nuevas de folium)
BASEMAPS = {
    "Gris claro (Esri)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/"
        "MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri",
    ),
    "Calles (OpenStreetMap)": ("openstreetmap", None),
    "Sin fondo": (None, None),
}


st.markdown(
    f"""
<style>
  .stApp {{ background-color: {APP_BG}; color: {APP_TEXT}; font-family: 'Montserrat', sans-serif; }}
  .stMarkdown, .stMarkdown p, h1, h2, h3, h4, h5, h6, .stCaption, label {{ color: {APP_TEXT} !important; }}
  .stTabs [role="tab"], .stTabs [role="tab"] p {{ color: {APP_TEXT} !important; }}
  [role="radiogroup"] label, [role="radiogroup"] label p {{ color: {APP_TEXT} !important; }}
  div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.10);
    border: 1px solid #6c4c99 !important;
    color: {APP_TEXT} !important;
    border-radius: 10px;
  }}
  div[data-baseweb="select"] span, div[data-baseweb="select"] input {{ color: {APP_TEXT} !important; }}
  div[data-baseweb="menu"] {{ background-color: rgba(0,0,0,0.35); color: {APP_TEXT}; }}
  div[data-baseweb="option"] {{ color: {APP_TEXT}; }}
  .rounded-box {{
    background: rgba(255,255,255,0.07);
    padding: 22px; border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    margin-bottom: 24px;
  }}
  iframe, .folium-map, .vega-embed {{
    border-radius: 16px !important; overflow: hidden !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.25);
  }}
  .stDataFrame {{ border-radius: 12px; overflow: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# NORMALIZACIÓN DE CLAVES
# =============================================================================
def norm_circ(value: Any) -> Optional[str]:
    """
    Convierte cualquier representación de circuito a una clave canónica.
    '00001' -> '1' ; '0010' -> '10' ; 1 -> '1' ; 'CIR 023' -> '23'
    Devuelve None si no hay dígitos.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return None
    try:
        return str(int(digits))
    except Exception:
        return None


def norm_circ_series(s: pd.Series) -> pd.Series:
    return s.map(norm_circ)


def pretty_circ(key: Optional[str]) -> str:
    """Muestra el circuito con padding de 5 (formato oficial)."""
    if key is None:
        return "-"
    try:
        return str(int(key)).zfill(5)
    except Exception:
        return str(key)


# =============================================================================
# CARGA DE DATOS
# =============================================================================
def _github_to_raw(url: str) -> str:
    if not url:
        return url
    if "github.com" in url and "/blob/" in url:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return url


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_bytes(url: str) -> bytes:
    url = _github_to_raw((url or "").strip())
    # datar.info a veces viene con doble barra en el path ("//dataset"); la limpiamos
    # sin tocar el "https://" del esquema.
    m = re.match(r"^(https?://)(.*)$", url, flags=re.I)
    if m:
        url = m.group(1) + re.sub(r"/{2,}", "/", m.group(2))
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content


@st.cache_data(show_spinner=False)
def parse_csv(content: bytes) -> pd.DataFrame:
    if not content:
        return pd.DataFrame()

    df = None
    for kwargs in (
        {},
        {"sep": ";"},
        {"encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ):
        try:
            cand = pd.read_csv(io.BytesIO(content), dtype=str, **kwargs)
            if cand.shape[1] > 1:
                df = cand
                break
        except Exception:
            continue
    if df is None:
        return pd.DataFrame()

    df.columns = [str(c).strip().upper() for c in df.columns]

    # Clave canónica de circuito: preferimos CIRCUITO_ID (viene limpio),
    # y si no está, usamos CIRCUITO.
    src_col = "CIRCUITO_ID" if "CIRCUITO_ID" in df.columns else ("CIRCUITO" if "CIRCUITO" in df.columns else None)
    if src_col:
        df["CIRC_KEY"] = norm_circ_series(df[src_col])

    if "AGRUPACION_NOMBRE" in df.columns:
        df["AGRUPACION_NOMBRE"] = (
            df["AGRUPACION_NOMBRE"].astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
        )
    if "VOTOS_TIPO" in df.columns:
        df["VOTOS_TIPO"] = df["VOTOS_TIPO"].astype(str).str.upper().str.strip()
    if "SECCION_NOMBRE" in df.columns:
        df["SECCION_NOMBRE"] = df["SECCION_NOMBRE"].astype(str).str.strip()
    if "VOTOS_CANTIDAD" in df.columns:
        df["VOTOS_CANTIDAD"] = pd.to_numeric(df["VOTOS_CANTIDAD"], errors="coerce").fillna(0)

    # Filas basura (sin circuito o sin votos válidos)
    if "CIRC_KEY" in df.columns:
        df = df[df["CIRC_KEY"].notna()].copy()

    return df


@st.cache_data(show_spinner=False)
def parse_geojson(content: bytes) -> Tuple[dict, Optional[str], Optional[str], List[str]]:
    """
    Devuelve (geojson_normalizado, circ_prop, comuna_prop, props_disponibles).
    Agrega a cada feature la propiedad 'CIRC_KEY' ya normalizada.
    """
    if not content:
        return {}, None, None, []

    text = content.decode("utf-8-sig", errors="replace")
    gj = json.loads(text)

    if isinstance(gj, list):  # lista suelta de features
        gj = {"type": "FeatureCollection", "features": gj}

    if gj.get("type") == "Topology":
        raise ValueError(
            "El archivo es TopoJSON, no GeoJSON. Convertilo primero "
            "(por ejemplo con mapshaper.org → Export → GeoJSON)."
        )

    if gj.get("type") == "Feature":
        gj = {"type": "FeatureCollection", "features": [gj]}

    feats = gj.get("features") or []
    if not feats:
        raise ValueError("El GeoJSON no tiene features.")

    # Propiedades disponibles (unión de las primeras features)
    available: List[str] = []
    for f in feats[:50]:
        for k in (f.get("properties") or {}).keys():
            if k not in available:
                available.append(k)

    # Detección de la columna de circuito
    circ_prop = next((k for k in CIRC_KEYS if k in available), None)
    if circ_prop is None:
        # heurística: la propiedad que más valores numéricos únicos aporta
        best, best_score = None, -1
        for k in available:
            vals = {norm_circ((f.get("properties") or {}).get(k)) for f in feats}
            vals.discard(None)
            score = len(vals)
            if score > best_score and 50 <= score <= 400:
                best, best_score = k, score
        circ_prop = best

    comuna_prop = next((k for k in COMUNA_KEYS if k in available), None)

    gj = copy.deepcopy(gj)
    for f in gj.get("features", []):
        props = f.get("properties") or {}
        props["CIRC_KEY"] = norm_circ(props.get(circ_prop)) if circ_prop else None
        props["CIRC_LABEL"] = pretty_circ(props["CIRC_KEY"])
        props["COMUNA_GEO"] = str(props.get(comuna_prop, "")) if comuna_prop else ""
        f["properties"] = props

    return gj, circ_prop, comuna_prop, available


# =============================================================================
# MÉTRICAS
# =============================================================================
def assert_required(df: pd.DataFrame, nombre: str) -> bool:
    req = {"AGRUPACION_NOMBRE", "VOTOS_CANTIDAD", "CIRC_KEY"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        st.error(f"[{nombre}] Faltan columnas requeridas en el CSV: {', '.join(missing)}")
        return False
    return True


def compute_metrics(df: pd.DataFrame, solo_positivos: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    base = df.copy()
    if solo_positivos and "VOTOS_TIPO" in base.columns:
        base = base[base["VOTOS_TIPO"] == "POSITIVO"]

    if base.empty:
        return pd.DataFrame()

    group_cols = ["CIRC_KEY"]
    extra = [c for c in ["SECCION_NOMBRE", "COMUNA"] if c in base.columns]

    tot = base.groupby(group_cols, as_index=False)["VOTOS_CANTIDAD"].sum()
    tot = tot.rename(columns={"VOTOS_CANTIDAD": "TOTAL_VOTOS"})

    if extra:
        labels = base.groupby(group_cols, as_index=False)[extra].first()
        tot = tot.merge(labels, on=group_cols, how="left")

    out = tot
    for short, full in PARTIES.items():
        part = base[base["AGRUPACION_NOMBRE"] == full]
        col = f"VOTOS_{short}"
        if part.empty:
            out[col] = 0.0
        else:
            agg = part.groupby(group_cols, as_index=False)["VOTOS_CANTIDAD"].sum().rename(
                columns={"VOTOS_CANTIDAD": col}
            )
            out = out.merge(agg, on=group_cols, how="left")
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["TOTAL_VOTOS"] = pd.to_numeric(out["TOTAL_VOTOS"], errors="coerce").fillna(0.0)
    denom = out["TOTAL_VOTOS"].replace(0, pd.NA)
    for short in PARTIES:
        out[f"PORC_{short}"] = (out[f"VOTOS_{short}"] / denom * 100).astype(float).fillna(0.0)

    out["CIRCUITO"] = out["CIRC_KEY"].map(pretty_circ)
    return out


# =============================================================================
# MAPA
# =============================================================================
def _centroid_from_geometry(geom: Dict[str, Any]) -> Optional[List[float]]:
    try:
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            return None

        def ring_centroid(ring: List[List[float]]) -> Optional[List[float]]:
            xs = ys = 0.0
            n = 0
            for pt in ring:
                if not pt or len(pt) < 2:
                    continue
                xs += float(pt[0]); ys += float(pt[1]); n += 1
            return [ys / n, xs / n] if n else None

        if gtype == "Polygon":
            return ring_centroid(coords[0])
        if gtype == "MultiPolygon":
            cents = [ring_centroid(p[0]) for p in coords if p and p[0]]
            cents = [c for c in cents if c]
            if not cents:
                return None
            return [sum(c[0] for c in cents) / len(cents), sum(c[1] for c in cents) / len(cents)]
        if gtype == "Point":
            lon, lat = coords[0], coords[1]
            return [float(lat), float(lon)]
    except Exception:
        return None
    return None


def _fmt(metric_col: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        if metric_col.startswith("PORC"):
            return f"{float(value):.1f}%"
        return f"{int(round(float(value))):,}".replace(",", ".")
    except Exception:
        return str(value)


def enrich_geojson(gj: dict, data: pd.DataFrame, keep_keys: Optional[set] = None) -> dict:
    """Inyecta las métricas en properties y opcionalmente filtra features."""
    by_circ = data.set_index("CIRC_KEY").to_dict(orient="index")
    metric_cols = ["TOTAL_VOTOS"] + [f"VOTOS_{s}" for s in PARTIES] + [f"PORC_{s}" for s in PARTIES]

    out = copy.deepcopy(gj)
    feats = []
    for feat in out.get("features", []):
        props = feat.get("properties") or {}
        key = props.get("CIRC_KEY")
        if keep_keys is not None and key not in keep_keys:
            continue
        row = by_circ.get(key, {})
        for col in metric_cols:
            val = row.get(col)
            props[col] = None if val is None or pd.isna(val) else float(val)
        props["SECCION"] = str(row.get("SECCION_NOMBRE", "") or "")
        props["MATCH"] = bool(row)
        feat["properties"] = props
        feats.append(feat)
    out["features"] = feats
    return out


def make_map(
    gj: dict,
    data: pd.DataFrame,
    metric_col: str,
    legend: str,
    show_labels: bool,
    max_labels: int,
    show_circuit_id: bool,
    basemap: str = "Gris claro (Esri)",
):
    if folium is None or cm_linear is None:
        st.warning("Falta `folium`. Instalá con: pip install folium branca")
        return None

    keep = set(data["CIRC_KEY"].dropna().astype(str))
    gj_e = enrich_geojson(gj, data, keep_keys=keep)

    if not gj_e.get("features"):
        st.error(
            "Ningún polígono del GeoJSON coincide con los circuitos del CSV. "
            "Revisá el panel de diagnóstico de la barra lateral."
        )
        return None

    tiles, attr = BASEMAPS.get(basemap, BASEMAPS["Gris claro (Esri)"])
    m = folium.Map(
        location=[-34.61, -58.44],
        tiles=tiles,
        attr=attr,
        zoom_start=11,
        control_scale=True,
    )
    if Fullscreen is not None:
        Fullscreen(position="topleft", title="Pantalla completa",
                   title_cancel="Salir", force_separate_button=True).add_to(m)

    vals = pd.to_numeric(data[metric_col], errors="coerce").dropna()
    vmin = float(vals.min()) if len(vals) else 0.0
    vmax = float(vals.max()) if len(vals) else 1.0
    if vmin == vmax:
        vmax = vmin + (0.0001 if metric_col.startswith("PORC") else 1.0)

    cmap = cm_linear.RdYlGn_11.scale(vmin, vmax)
    cmap.caption = legend

    v_by_circ = {str(r["CIRC_KEY"]): float(r[metric_col]) for _, r in data.iterrows()
                 if pd.notna(r[metric_col])}

    def style_fn(feat):
        val = v_by_circ.get(str((feat.get("properties") or {}).get("CIRC_KEY")))
        if val is None:
            return {"fillColor": "#cccccc", "color": "#555", "weight": 0.7, "fillOpacity": 0.25}
        return {"fillColor": cmap(val), "color": "#555", "weight": 0.7, "fillOpacity": 0.85}

    layer = folium.GeoJson(
        data=gj_e,
        style_function=style_fn,
        highlight_function=lambda f: {"weight": 2.5, "color": "#000"},
        tooltip=folium.GeoJsonTooltip(
            fields=["CIRC_LABEL", "SECCION", "TOTAL_VOTOS", "PORC_LLA", "PORC_FUERZA", "PORC_POTENCIA"],
            aliases=["Circuito", "Sección", "Total votos", "% LLA", "% Fuerza Patria", "% Potencia"],
            localize=True,
            sticky=True,
        ),
        name=legend,
    )
    layer.add_to(m)
    cmap.add_to(m)

    if show_labels:
        label_df = data.sort_values(metric_col, ascending=False)
        if max_labels and max_labels > 0:
            label_df = label_df.head(int(max_labels))
        allow = set(label_df["CIRC_KEY"].astype(str))
        rows = label_df.set_index("CIRC_KEY").to_dict(orient="index")

        for feat in gj_e["features"]:
            props = feat.get("properties") or {}
            key = str(props.get("CIRC_KEY"))
            if key not in allow:
                continue
            center = _centroid_from_geometry(feat.get("geometry") or {})
            if not center:
                continue
            txt = _fmt(metric_col, rows.get(key, {}).get(metric_col))

            # El wrapper con translate(-50%,-50%) centra la etiqueta en el
            # centroide del polígono (si no, Leaflet la ancla arriba a la izquierda).
            inner = (
                f"""<div style="font-size:10px;font-weight:900;letter-spacing:.6px;color:#111;
                                padding:1px 6px;border-radius:999px;background:rgba(0,0,0,0.08);
                                border:1px solid rgba(0,0,0,0.15);margin-bottom:3px;">
                      CIR {props.get('CIRC_LABEL')}
                    </div>
                    <div style="font-size:13px;font-weight:900;color:#000;">{txt}</div>"""
                if show_circuit_id
                else f"""<div style="font-size:12px;font-weight:800;color:#000;">{txt}</div>"""
            )
            html = f"""
            <div style="position:absolute;transform:translate(-50%,-50%);width:max-content;
                        font-family:Montserrat,Arial,sans-serif;text-align:center;line-height:1.1;
                        padding:4px 7px;border-radius:10px;background:rgba(255,255,255,0.88);
                        border:1px solid rgba(0,0,0,0.25);box-shadow:0 2px 6px rgba(0,0,0,0.2);
                        white-space:nowrap;pointer-events:none;">{inner}</div>"""

            folium.Marker(
                location=center,
                icon=folium.DivIcon(html=html, class_name="", icon_size=(0, 0), icon_anchor=(0, 0)),
                tooltip=f"Circuito {props.get('CIRC_LABEL')} | {legend}: {txt}",
            ).add_to(m)

    try:
        m.fit_bounds(layer.get_bounds(), padding=(10, 10))
    except Exception:
        pass

    folium.LayerControl(collapsed=True).add_to(m)
    return m


def top_bars(df: pd.DataFrame, col: str, title: str, ascending: bool = False):
    if alt is None:
        st.warning("Falta `altair`. Instalá con: pip install altair")
        return
    if df.empty or col not in df.columns:
        st.info("Sin datos para el ranking.")
        return

    d = df.copy()
    if "SECCION_NOMBRE" not in d.columns:
        d["SECCION_NOMBRE"] = ""
    d["LABEL"] = d["SECCION_NOMBRE"].astype(str) + " · " + d["CIRCUITO"].astype(str)

    top = d.sort_values(col, ascending=ascending).head(10)
    order = top["LABEL"].tolist()

    chart = (
        alt.Chart(top)
        .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4)
        .encode(
            x=alt.X("LABEL:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y(f"{col}:Q", title=col.replace("_", " ").title()),
            tooltip=["LABEL", alt.Tooltip(f"{col}:Q", format=",.2f"), "TOTAL_VOTOS"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(title)


# =============================================================================
# SIDEBAR – FUENTES
# =============================================================================
st.title("🗳️ Elecciones Octubre 2025 – CABA")

with st.sidebar:
    st.subheader("🗺️ GeoJSON de circuitos")
    geo_mode = st.radio(
        "Origen del GeoJSON",
        ["Subir archivo", "Ruta local", "URL"],
        index=0,
        key="geo_mode",
    )

    geo_bytes: bytes = b""
    geo_err: Optional[str] = None

    if geo_mode == "Subir archivo":
        up = st.file_uploader("Archivo .geojson / .json", type=["geojson", "json"], key="geo_up")
        if up is not None:
            geo_bytes = up.getvalue()
        else:
            st.caption(
                "Descargalo una vez desde datar.info (dataset *Circuitos electorales CABA*) "
                "y guardalo junto al script."
            )
    elif geo_mode == "Ruta local":
        path = st.text_input("Ruta del archivo", value="circuitos_electorales_caba.geojson", key="geo_path")
        if path:
            try:
                with open(path, "rb") as fh:
                    geo_bytes = fh.read()
            except Exception as e:
                geo_err = f"No se pudo abrir el archivo local: {e}"
    else:
        geo_url = st.text_input("URL GeoJSON", value=DEFAULT_GEO_URL, key="geo_url")
        if geo_url:
            try:
                geo_bytes = fetch_bytes(geo_url)
            except Exception as e:
                geo_err = f"No se pudo descargar el GeoJSON: {e}"

    st.divider()
    st.subheader("📊 Resultados")
    csv_mode = st.radio("Origen de los CSV", ["URL", "Subir archivos"], index=0, key="csv_mode")

    dip_bytes = sen_bytes = b""
    csv_err: Optional[str] = None

    if csv_mode == "URL":
        dip_url = st.text_input("CSV Diputados", value=DEFAULT_DIP_URL, key="dip_url")
        sen_url = st.text_input("CSV Senadores", value=DEFAULT_SEN_URL, key="sen_url")
        try:
            if dip_url:
                dip_bytes = fetch_bytes(dip_url)
            if sen_url:
                sen_bytes = fetch_bytes(sen_url)
        except Exception as e:
            csv_err = f"No se pudieron descargar los CSV: {e}"
    else:
        up_dip = st.file_uploader("CSV Diputados", type=["csv"], key="dip_up")
        up_sen = st.file_uploader("CSV Senadores", type=["csv"], key="sen_up")
        dip_bytes = up_dip.getvalue() if up_dip is not None else b""
        sen_bytes = up_sen.getvalue() if up_sen is not None else b""

    st.divider()
    basemap = st.selectbox("Fondo del mapa", list(BASEMAPS.keys()), index=0, key="basemap")
    solo_positivos = st.checkbox(
        "Calcular % solo sobre votos POSITIVOS",
        value=True,
        help="Si está desmarcado, el denominador incluye blancos, nulos, recurridos e impugnados.",
    )

# ---- Parseo ------------------------------------------------------------------
geo_gj: dict = {}
circ_prop = comuna_prop = None
geo_props: List[str] = []

if geo_err:
    st.sidebar.error(geo_err)
elif geo_bytes:
    try:
        geo_gj, circ_prop, comuna_prop, geo_props = parse_geojson(geo_bytes)
    except Exception as e:
        st.sidebar.error(f"GeoJSON inválido: {e}")

if csv_err:
    st.sidebar.error(csv_err)

df_dip = parse_csv(dip_bytes)
df_sen = parse_csv(sen_bytes)

# ---- Diagnóstico -------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("🔎 Diagnóstico")
    if not geo_gj:
        st.warning("Sin GeoJSON cargado. El mapa no se va a dibujar.")
    else:
        geo_keys = {(f.get("properties") or {}).get("CIRC_KEY") for f in geo_gj.get("features", [])}
        geo_keys.discard(None)
        st.write(f"**Features:** {len(geo_gj.get('features', []))}")
        st.write(f"**Propiedad circuito detectada:** `{circ_prop}`")
        st.write(f"**Circuitos únicos en el GeoJSON:** {len(geo_keys)}")

        csv_keys = set()
        for d in (df_dip, df_sen):
            if not d.empty and "CIRC_KEY" in d.columns:
                csv_keys |= set(d["CIRC_KEY"].dropna())
        if csv_keys:
            inter = geo_keys & csv_keys
            st.write(f"**Circuitos en los CSV:** {len(csv_keys)}")
            st.write(f"**✅ Coinciden:** {len(inter)}")
            faltan_geo = sorted(csv_keys - geo_keys, key=lambda x: int(x))[:25]
            if faltan_geo:
                st.caption("En CSV pero no en GeoJSON: " + ", ".join(pretty_circ(k) for k in faltan_geo))
        if circ_prop is None:
            st.error("No se detectó la propiedad de circuito.")
            st.caption("Propiedades disponibles: " + ", ".join(geo_props[:40]))

        with st.expander("Ver propiedades del GeoJSON"):
            st.code("\n".join(geo_props) or "(sin propiedades)")


# =============================================================================
# TABS
# =============================================================================
METRIC_OPTIONS = {
    "% LLA": ("PORC_LLA", "% LLA"),
    "Votos LLA": ("VOTOS_LLA", "Votos LLA"),
    "% Fuerza Patria": ("PORC_FUERZA", "% Fuerza Patria"),
    "Votos Fuerza Patria": ("VOTOS_FUERZA", "Votos Fuerza Patria"),
    "% Potencia": ("PORC_POTENCIA", "% Potencia"),
    "Total de votos": ("TOTAL_VOTOS", "Total de votos"),
}


def tab_body(nombre: str, df_cat: pd.DataFrame):
    if df_cat.empty:
        st.info(f"Cargá el CSV de {nombre} desde la barra lateral.")
        return
    if not assert_required(df_cat, nombre):
        return

    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.subheader(nombre)

    c1, c2 = st.columns([2, 1])
    with c1:
        secciones = (
            sorted(df_cat["SECCION_NOMBRE"].dropna().astype(str).unique())
            if "SECCION_NOMBRE" in df_cat.columns else []
        )
        sel = st.multiselect("Filtrar por sección / comuna", options=secciones, default=[], key=f"sec_{nombre}")
    with c2:
        met_label = st.selectbox("Métrica", list(METRIC_OPTIONS.keys()), index=0, key=f"met_{nombre}")

    df_fil = df_cat if not sel else df_cat[df_cat["SECCION_NOMBRE"].isin(sel)]

    c3, c4, c5 = st.columns(3)
    with c3:
        show_labels = st.checkbox("Etiquetas sobre el mapa", value=True, key=f"lbl_{nombre}")
    with c4:
        show_circuit_id = st.checkbox("Mostrar N° de circuito", value=True, key=f"lblc_{nombre}")
    with c5:
        max_labels = st.slider(
            "Máx. etiquetas", 0, 200, 50, 10, key=f"lblmax_{nombre}",
            help="Se muestran las de mayor valor. Con muchas etiquetas se pisan entre sí.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    met_df = compute_metrics(df_fil, solo_positivos)
    if met_df.empty:
        st.info("No hay datos para la selección actual.")
        return

    metric_col, legend = METRIC_OPTIONS[met_label]

    # --- KPIs ---
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    total = met_df["TOTAL_VOTOS"].sum()
    k1.metric("Circuitos", f"{met_df['CIRC_KEY'].nunique()}")
    k2.metric("Total de votos", f"{int(total):,}".replace(",", "."))
    for col, short, label in ((k3, "LLA", "LLA"), (k4, "FUERZA", "Fuerza Patria")):
        v = met_df[f"VOTOS_{short}"].sum()
        pct = (v / total * 100) if total else 0
        col.metric(label, f"{pct:.1f}%", f"{int(v):,}".replace(",", ".") + " votos")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Mapa ---
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Mapa coroplético por circuito")
    if geo_gj:
        m = make_map(geo_gj, met_df, metric_col, legend, show_labels, max_labels, show_circuit_id, basemap)
        if m is not None:
            components.html(m.get_root().render(), height=650, scrolling=False)
    else:
        st.warning("Sin GeoJSON cargado: se muestran solo tablas y gráficos.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Rankings ---
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.markdown("### 📈 Rankings")
    r1, r2 = st.columns(2)
    with r1:
        top_bars(met_df, metric_col, f"Top 10 circuitos por {legend}")
    with r2:
        top_bars(met_df, metric_col, f"Bottom 10 circuitos por {legend}", ascending=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Tabla ---
    st.markdown('<div class="rounded-box">', unsafe_allow_html=True)
    st.markdown("### 📋 Detalle por circuito")
    cols = [c for c in ["CIRCUITO", "SECCION_NOMBRE", "TOTAL_VOTOS",
                        "VOTOS_LLA", "PORC_LLA", "VOTOS_FUERZA", "PORC_FUERZA",
                        "VOTOS_POTENCIA", "PORC_POTENCIA"] if c in met_df.columns]
    tabla = met_df[cols].sort_values(metric_col if metric_col in cols else "TOTAL_VOTOS", ascending=False)
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar CSV",
        tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"resultados_{nombre.lower()}.csv",
        mime="text/csv",
        key=f"dl_{nombre}",
    )
    st.markdown("</div>", unsafe_allow_html=True)


TAB_SEN, TAB_DIP = st.tabs(["Senadores", "Diputados"])
with TAB_SEN:
    tab_body("Senadores", df_sen)
with TAB_DIP:
    tab_body("Diputados", df_dip)
