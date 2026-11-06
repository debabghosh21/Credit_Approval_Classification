import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from scoring import FEATURES, TARGET, evaluate, load_models, prepare_upload

st.set_page_config(page_title="Credit Approval Model Bench", page_icon="💳", layout="wide")


@st.cache_resource
def get_models():
    return load_models()


models = get_models()

st.title("💳 Credit Approval — Classifier Bench")
st.caption(
    "UCI Credit Approval dataset (690 applicants, 15 anonymised attributes). "
    "Five pre-trained models are loaded from disk — nothing is retrained on your upload."
)

with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Upload test CSV", type="csv")
    st.markdown(
        f"The file needs columns **{FEATURES[0]}–{FEATURES[-1]}** plus the target **{TARGET}** "
        "(`+` approved / `-` rejected). Blank cells or `?` count as missing."
    )
    try:
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")
    except FileNotFoundError:
        pass
    st.header("2 · Model")
    choice = st.selectbox("Choose a classifier", list(models))

if up is None:
    st.info("Upload a test CSV in the sidebar to begin. You can grab the sample file there.")
    st.stop()

try:
    X, y = prepare_upload(pd.read_csv(up))
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # unreadable file etc.
    st.error(f"Could not read that file: {e}")
    st.stop()

st.subheader(f"Uploaded data · {len(X)} rows")
c1, c2 = st.columns([3, 1])
c1.dataframe(pd.concat([X, y.map({1: "+", 0: "-"}).rename(TARGET)], axis=1).head(10), use_container_width=True)
c2.metric("Approved (+)", int(y.sum()))
c2.metric("Rejected (-)", int((1 - y).sum()))

if y.nunique() < 2:
    st.warning("The target column has only one class, so AUC/MCC cannot be computed.")

metrics, cm, report, pred, proba = evaluate(models[choice], X, y)

st.subheader(f"Results · {choice}")
if metrics:
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    st.dataframe(report, use_container_width=True)

st.subheader("All models on this upload")
rows = []
for name, mdl in models.items():
    m, *_ = evaluate(mdl, X, y)
    if m:
        rows.append({"Model": name, **m})
if rows:
    comp = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comp.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
