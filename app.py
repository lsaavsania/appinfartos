"""
App Streamlit para predecir riesgo de infarto con el modelo de Orange
(Random Forest) guardado en Modelo_Infarto.pkcls.

Ejecutar:
    streamlit run app.py
"""

import os
import pickle
import warnings
from pathlib import Path

# Orange importa Qt al cargar el modelo; en servidores sin pantalla esto evita errores.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import Orange  # noqa: F401  (necesario para deserializar el .pkcls)
from Orange.data import Domain, Table

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Modelo_Infarto.pkcls"
DATA_PATH = BASE_DIR / "Medicaldataset.xlsx"

# Nombres originales del Excel -> nombres usados por el modelo (Edit Domain en Orange)
COLUMN_MAP = {
    "Troponin": "Troponina",
    "CK-MB": "CK-MB",
    "Age": "Edad",
    "Gender": "Sexo",
    "Systolic blood pressure": "Presion_arterial_sistolica",
    "Diastolic blood pressure": "Presion_arterial_diastolica",
}

# Configuración de cada entrada: etiqueta, mín, máx, valor por defecto, paso, ayuda
FIELDS = {
    "Edad": ("Edad (años)", 1, 110, 56, 1, None),
    "Presion_arterial_sistolica": ("Presión arterial sistólica (mmHg)", 40, 250, 127, 1, None),
    "Presion_arterial_diastolica": ("Presión arterial diastólica (mmHg)", 20, 160, 72, 1, None),
    "CK-MB": ("CK-MB (ng/mL)", 0.0, 300.0, 3.0, 0.1, "Marcador cardíaco (creatina quinasa MB)."),
    "Troponina": ("Troponina (ng/mL)", 0.0, 11.0, 0.01, 0.001, "Marcador más específico de daño miocárdico."),
}

st.set_page_config(page_title="Predicción de Infarto", page_icon="❤️", layout="wide")


# ---------------------------------------------------------------- utilidades
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_excel(DATA_PATH)
    return None


def root_variables(model):
    """Variables crudas del Excel de las que deriva cada atributo del modelo.

    El flujo de Orange (Edit Domain -> Impute -> normalizar) queda guardado dentro
    del modelo. Pasándole los datos crudos, Orange aplica automáticamente el
    renombrado, la imputación y la normalización.
    """
    roots = []
    for a in model.domain.attributes:
        var, cv = a, a.compute_value
        while cv is not None and getattr(cv, "variable", None) is not None:
            var, cv = cv.variable, cv.variable.compute_value
        roots.append(var)
    return roots


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Recibe un DataFrame (columnas en español del modelo) y devuelve clase + probabilidades."""
    domain = model.original_domain
    inverse = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns={c: inverse[c] for c in df.columns if c in inverse})
    roots = root_variables(model)
    X = np.column_stack(
        [pd.to_numeric(df.get(v.name, np.nan), errors="coerce").to_numpy(dtype=float) for v in roots]
    )
    table = Table.from_numpy(Domain(roots), X)
    values, probs = model(table, model.ValueProbs)
    classes = list(domain.class_var.values)
    out = pd.DataFrame(probs, columns=[f"P({c})" for c in classes])
    out.insert(0, "Predicción", [classes[int(v)] for v in values])
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})


# ---------------------------------------------------------------- interfaz
st.title("❤️ Predicción de Infarto")
st.caption("Modelo Random Forest entrenado en Orange Data Mining con el dataset médico del curso.")

if not MODEL_PATH.exists():
    st.error(f"No se encontró el modelo en `{MODEL_PATH.name}`. Colócalo junto a app.py.")
    st.stop()

model = load_model()
class_values = list(model.original_domain.class_var.values)
pos_idx = class_values.index("positive") if "positive" in class_values else 1

with st.sidebar:
    st.header("Sobre el modelo")
    st.write(f"**Tipo:** {getattr(model, 'name', type(model).__name__)}")
    st.write("**Variables de entrada:**")
    for a in model.original_domain.attributes:
        st.write(f"- {a.name}")
    st.write(f"**Clases:** {', '.join(class_values)}")
    st.info("Herramienta académica. No reemplaza un diagnóstico médico.")

tab_ind, tab_lote, tab_datos = st.tabs(["🧍 Paciente individual", "📄 Predicción por lote", "📊 Dataset"])

# --- Paciente individual
with tab_ind:
    with st.form("form_paciente"):
        c1, c2 = st.columns(2)
        values = {}
        with c1:
            lbl, mn, mx, dv, stp, hlp = FIELDS["Edad"]
            values["Edad"] = st.number_input(lbl, mn, mx, dv, stp, help=hlp)
            sexo = st.radio("Sexo", ["Masculino", "Femenino"], horizontal=True)
            values["Sexo"] = 1 if sexo == "Masculino" else 0
            for key in ("Presion_arterial_sistolica", "Presion_arterial_diastolica"):
                lbl, mn, mx, dv, stp, hlp = FIELDS[key]
                values[key] = st.number_input(lbl, mn, mx, dv, stp, help=hlp)
        with c2:
            for key in ("CK-MB", "Troponina"):
                lbl, mn, mx, dv, stp, hlp = FIELDS[key]
                values[key] = st.number_input(lbl, mn, mx, dv, stp, format="%.3f", help=hlp)
        submitted = st.form_submit_button("Predecir", type="primary", width="stretch")

    if submitted:
        res = predict(model, pd.DataFrame([values]))
        p_pos = float(res.iloc[0, 1 + pos_idx])
        pred = res.iloc[0]["Predicción"]

        m1, m2 = st.columns(2)
        m1.metric("Resultado", "POSITIVO (riesgo de infarto)" if pred == "positive" else "NEGATIVO")
        m2.metric("Probabilidad de positivo", f"{p_pos:.1%}")
        st.progress(p_pos)
        if pred == "positive":
            st.error("El modelo indica un resultado **positivo**.")
        else:
            st.success("El modelo indica un resultado **negativo**.")

# --- Predicción por lote
with tab_lote:
    st.write(
        "Sube un archivo **.xlsx** o **.csv** con las columnas del dataset original "
        "(Age, Gender, Systolic blood pressure, Diastolic blood pressure, CK-MB, Troponin) "
        "o con los nombres en español del modelo."
    )
    file = st.file_uploader("Archivo de pacientes", type=["xlsx", "csv"])
    if file is not None:
        df_in = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df_norm = normalize_columns(df_in)
        missing = [a.name for a in model.original_domain.attributes if a.name not in df_norm.columns]
        if missing:
            st.warning(f"Columnas faltantes (se imputarán): {', '.join(missing)}")
        res = predict(model, df_norm)
        df_out = pd.concat([df_in.reset_index(drop=True), res], axis=1)
        st.dataframe(df_out, width="stretch")

        if "Result" in df_in.columns:
            acc = (df_in["Result"].astype(str).str.lower().values == res["Predicción"].values).mean()
            st.metric("Exactitud contra la columna Result", f"{acc:.1%}")

        st.download_button(
            "Descargar resultados (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            "predicciones_infarto.csv",
            "text/csv",
        )

# --- Dataset
with tab_datos:
    data = load_dataset()
    if data is None:
        st.info("Coloca Medicaldataset.xlsx junto a app.py para explorarlo aquí.")
    else:
        st.write(f"**{len(data)}** registros, **{data.shape[1]}** columnas.")
        st.dataframe(data.head(50), width="stretch")
        if "Result" in data.columns:
            st.subheader("Distribución de la clase")
            st.bar_chart(data["Result"].value_counts())
        st.subheader("Estadísticas descriptivas")
        st.dataframe(data.drop(columns=["DNI"], errors="ignore").describe().T, width="stretch")
