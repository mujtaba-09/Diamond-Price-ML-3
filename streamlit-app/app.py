"""
AIC354 - Assignment 3 (Task 2): Price Prediction web app
Student: Mujtaba Naveed | FA24-BSE-183

The app loads the trained model (model.pkl) and the description of the input
fields (meta.json), both produced by the Task 2 notebook, and builds the input
form automatically.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Price Predictor", page_icon="📈", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model.pkl")


@st.cache_data
def load_meta():
    return json.loads((BASE_DIR / "meta.json").read_text(encoding="utf-8"))


try:
    model = load_model()
    meta = load_meta()
except Exception as err:  # missing / incompatible files
    st.error(f"Could not load model.pkl / meta.json: {err}")
    st.stop()

features = meta["features"]
unit = meta.get("unit", "")

# ------------------------------------------------------------------ header
st.title(f"📈 {meta['problem_title']}")
st.write(
    f"Enter the details below and the model will estimate the **{meta['target']}** "
    f"({unit}) using the best algorithm found by {meta['n_folds']}-fold cross-validation."
)

with st.sidebar:
    st.header("About the model")
    st.write(f"**Algorithm:** {meta['best_model']}")
    st.write(f"**CV RMSE:** {meta['cv_rmse']:,.2f} {unit}")
    st.write(f"**CV R²:** {meta['cv_r2']:.3f}")
    st.write(f"**Training rows:** {meta['n_rows']:,}")
    st.caption("AIC354 - Assignment 3\n\nMujtaba Naveed | FA24-BSE-183")

# ------------------------------------------------------------------ input form
inputs = {}
with st.form("prediction_form"):
    left, right = st.columns(2)
    for i, (name, info) in enumerate(features.items()):
        column = left if i % 2 == 0 else right
        label = name.replace("_", " ").title()
        with column:
            if info["type"] == "category":
                inputs[name] = st.selectbox(label, info["options"], key=name)
            elif info["type"] == "int":
                inputs[name] = st.number_input(
                    label,
                    min_value=int(info["min"]),
                    max_value=int(info["max"]),
                    value=int(info["median"]),
                    step=1,
                    key=name,
                    help=f"Range seen in training data: {int(info['min'])} to {int(info['max'])}",
                )
            else:
                inputs[name] = st.number_input(
                    label,
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(info["median"]),
                    step=0.1,
                    format="%.2f",
                    key=name,
                    help=f"Range seen in training data: {info['min']:.2f} to {info['max']:.2f}",
                )
    submitted = st.form_submit_button("Predict")

# ------------------------------------------------------------------ prediction
if submitted:
    row = pd.DataFrame([inputs], columns=list(features.keys()))
    prediction = float(model.predict(row)[0])
    st.success("Prediction complete")
    st.metric(label=f"Estimated {meta['target']} ({unit})", value=f"{prediction:,.2f}")
    with st.expander("Show the inputs used"):
        st.dataframe(row, hide_index=True)
