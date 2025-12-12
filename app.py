# app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.joinpath("models", "best_model.pkl")
DATA_PATH = BASE_DIR.joinpath("data", "predictive_maintenance.csv")  # optional default dataset

# Helpers
def load_model(path: Path):
    if not path.exists():
        st.error(f"Model not found at {path}. Run training script first.")
        st.stop()
    return joblib.load(path)

def safe_show_df(df, caption=None):
    """Show DataFrame converting to strings to avoid Arrow errors in Streamlit."""
    st.write(caption or "")
    st.dataframe(df.astype(str))

def predict_df(pipeline, df_input):
    """Return predicted label and probability for each row."""
    # pipeline may accept raw df; some pipelines expect full feature set and column order
    try:
        probs = pipeline.predict_proba(df_input)[:, 1]
        preds = pipeline.predict(df_input)
    except Exception:
        # try transforming first (if pipeline contains only classifier saved as pipeline)
        try:
            clf = pipeline.named_steps["clf"]
            preproc = pipeline.named_steps.get("scaler", None)
            if preproc is not None:
                Xt = preproc.transform(df_input)
            else:
                Xt = df_input.values
            probs = clf.predict_proba(Xt)[:, 1]
            preds = clf.predict(Xt)
        except Exception as e:
            raise RuntimeError("Model prediction failed: " + str(e))
    return preds, probs

# --------------------------
# Load model
# --------------------------
st.sidebar.title("Model")
st.sidebar.write("Loading trained pipeline...")
model = load_model(MODEL_PATH)
st.sidebar.success("Model loaded")

# --------------------------
# Sidebar: single-record form
# --------------------------
st.sidebar.header("Single record input")
# Example features — adapt these to your dataset's columns exactly
# Use names that match training data columns (order not necessary if pipeline handles columns)
def single_input_form():
    # Replace or extend these fields based on your dataset feature names
    t_air = st.sidebar.number_input("Air temperature [K]", value=300.0, step=1.0)
    t_proc = st.sidebar.number_input("Process temperature [K]", value=310.0, step=1.0)
    rpm = st.sidebar.number_input("Rotational speed [rpm]", value=1600.0, step=10.0)
    torque = st.sidebar.number_input("Torque [Nm]", value=40.0, step=1.0)
    tool_wear = st.sidebar.number_input("Tool wear [min]", value=15.0, step=1.0)
    # If you encoded categorical 'Type' into Type_L / Type_M columns etc., add selects:
    # type_L = st.sidebar.selectbox("Type_L", [0,1], index=0)
    return {
        "Air temperature [K]": t_air,
        "Process temperature [K]": t_proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
        # include encoded categorical columns if used in training
    }

input_dict = single_input_form()
input_df = pd.DataFrame([input_dict])

# Main layout
st.title("Predictive Maintenance — Dashboard")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Data / EDA")
    uploaded = st.file_uploader("Upload CSV for batch predictions and EDA (optional)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.info("Uploaded dataset used for EDA and batch predictions")
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        st.info("Default dataset loaded for EDA (place your csv at data/predictive_maintenance.csv)")
    else:
        df = None
        st.warning("No dataset available for EDA. Upload a CSV or add a default dataset at data/")

    if df is not None:
        # Basic EDA
        st.markdown("**Dataset sample**")
        safe_show_df(df.head(10))
        st.markdown("**Numeric summary**")
        st.write(df.describe().T)
        # target distribution if exists
        if "Target" in df.columns:
            st.markdown("**Target distribution**")
            fig, ax = plt.subplots()
            df["Target"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("Target")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

with col2:
    st.subheader("Single record prediction")
    safe_show_df(input_df.T, caption="Input features (transposed)")
    if st.button("Predict single input"):
        try:
            pred, prob = predict_df(model, input_df)
            label = "Failure" if int(pred[0]) == 1 else "No Failure"
            st.metric("Prediction", label)
            st.metric("Failure probability", f"{prob[0]:.3f}")
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# --------------------------
# Batch prediction
# --------------------------
st.markdown("---")
st.header("Batch predictions (optional)")

if uploaded:
    if st.button("Run batch predictions on uploaded dataset"):
        # ensure required feature columns exist
        try:
            X_batch = df.copy()
            # drop label if present
            if "Target" in X_batch.columns:
                X_batch = X_batch.drop(columns=["Target"])
            preds, probs = predict_df(model, X_batch)
            df_out = df.copy()
            df_out["predicted_failure"] = preds
            df_out["failure_probability"] = probs
            safe_show_df(df_out.head(20), caption="Batch results (first 20 rows)")
            # allow download
            csv_out = df_out.to_csv(index=False)
            st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv")
        except Exception as e:
            st.error("Batch prediction failed: " + str(e))
else:
    st.info("Upload a CSV to enable batch predictions.")

# --------------------------
# Explainability section (SHAP)
# --------------------------
st.markdown("---")
st.header("Explainability")

with st.expander("SHAP explainability (may take a moment)"):
    try:
        # prepare background sample
        if df is not None and len(df) >= 10:
            X_full = df.copy()
            if "Target" in X_full.columns:
                X_full = X_full.drop(columns=["Target"])
            background = X_full.sample(n=min(50, len(X_full)), random_state=42)
        else:
            background = input_df

        # Select classifier (attempt to extract from pipeline)
        try:
            clf = model.named_steps["clf"]
        except Exception:
            clf = model

        # Transform data if pipeline contains transformer components:
        # If model is pipeline and expects raw df, pass raw df. We'll try both safe ways.
        try:
            # if pipeline supports transform, get transformed arrays for explainer background
            try:
                preproc = model.named_steps.get("scaler", None)
                if preproc is not None:
                    background_trans = preproc.transform(background)
                    input_trans = preproc.transform(input_df)
                else:
                    background_trans = background.values
                    input_trans = input_df.values
            except Exception:
                background_trans = background.values
                input_trans = input_df.values

            # Try TreeExplainer on booster if available
            use_kernel = False
            try:
                if hasattr(clf, "get_booster"):
                    booster = clf.get_booster()
                    explainer = shap.TreeExplainer(booster)
                    shap_vals = explainer.shap_values(input_trans)
                else:
                    explainer = shap.Explainer(clf.predict_proba, background_trans)
                    shap_result = explainer(input_trans)
                    # shap_result.values or .shap_values depending on newer versions
                    if hasattr(shap_result, "values"):
                        shap_vals = shap_result.values
                    else:
                        shap_vals = shap_result
            except Exception:
                use_kernel = True

            # Kernel fallback if Tree/Explainer failed
            if use_kernel:
                with st.spinner("Running KernelExplainer (model-agnostic) — may take >10s"):
                    explainer = shap.KernelExplainer(clf.predict_proba, background_trans)
                    shap_vals = explainer.shap_values(input_trans)

            # normalize to 2D and pick positive class if necessary
            sv = np.array(shap_vals)
            if isinstance(sv, np.ndarray) and sv.ndim == 3:
                # shape (n_classes, n_samples, n_features)
                # pick class=1 SHAP values if possible
                if sv.shape[0] > 1:
                    sv = sv[1]
                else:
                    sv = sv[0]
            if sv.ndim == 1:
                sv = sv.reshape(1, -1)

            # Plot SHAP summary (for small sample)
            fig = plt.figure(figsize=(8,4))
            shap.summary_plot(sv, input_trans, show=False)
            st.pyplot(fig)
            plt.close(fig)

            # Simple feature contribution table
            abs_vals = np.abs(sv).sum(axis=0)
            top_idx = np.argsort(abs_vals)[-10:][::-1]
            feat_names = None
            # attempt to get feature names from pipeline if available
            if hasattr(model, "named_steps") and hasattr(model.named_steps.get("scaler", None), "feature_names_in_"):
                feat_names = list(model.named_steps["scaler"].feature_names_in_)
            elif df is not None:
                feat_names = list((df.drop(columns=["Target"]) if "Target" in df.columns else df).columns)
            else:
                feat_names = [f"f_{i}" for i in range(sv.shape[1])]
            top_feats = [(feat_names[i] if i < len(feat_names) else f"f_{i}", float(abs_vals[i])) for i in top_idx]
            contrib_df = pd.DataFrame(top_feats, columns=["feature", "importance"])
            st.table(contrib_df.astype(str))
        except Exception as e:
            st.warning("SHAP explanation computation failed: " + str(e))
    except Exception as e:
        st.warning("SHAP not available in this environment: " + str(e))

st.markdown("---")
st.info("Notes: ensure pipeline's expected features match input fields. For production, add validation and stricter input checks.")
