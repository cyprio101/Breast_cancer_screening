import os

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from matplotlib import colormaps
from PIL import Image, ImageDraw

import mammo_cbm as M

st.set_page_config(page_title="Mammography second reader", layout="wide")
st.title("Mammography lesion second reader")
st.warning(
    "Research prototype. Not a medical device and not validated for clinical use. "
    "It supports, and never replaces, the radiologist's read."
)


WEIGHTS_REPO = "cyprio/breast-cancer-screening-cbm"  # Hugging Face model repo holding mammo_cbm_bundle.pt


@st.cache_resource
def get_bundle():
    path = hf_hub_download(repo_id=WEIGHTS_REPO, filename="mammo_cbm_bundle.pt", token=os.environ.get("HF_TOKEN"))
    return M.load_bundle(path)


try:
    b = get_bundle()
except Exception as e:
    st.error(f"Could not load the model weights from '{WEIGHTS_REPO}': {e}")
    st.stop()
st.sidebar.caption(
    "Images are processed in memory only. Nothing is stored or logged. "
    "DICOM headers (patient data) are never displayed."
)
if b.meta:
    st.sidebar.json(b.meta, expanded=False)

up = st.file_uploader("Mammogram view (DICOM, PNG or JPG)", type=["dcm", "dicom", "png", "jpg", "jpeg"])
if not up:
    st.info("Upload one mammogram view, then outline the region you want analysed.")
    st.stop()

a = M.load_array(up)
H, W = a.shape
if min(H, W) < 512:
    st.warning("This image is small for a mammogram; results may be unreliable.")

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Select the region of interest")
    x0, x1 = st.slider("Left to right (%)", 0, 100, (40, 60))
    y0, y1 = st.slider("Top to bottom (%)", 0, 100, (40, 60))
    box = (
        int(y0 * H / 100),
        int(x0 * W / 100),
        max(int(y1 * H / 100), int(y0 * H / 100) + 8),
        max(int(x1 * W / 100), int(x0 * W / 100) + 8),
    )
    lo, hi = np.percentile(a, [0.5, 99.5])
    prev = Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)).convert("RGB")
    sc = 700 / max(H, W)
    prev = prev.resize((int(W * sc), int(H * sc)))
    ImageDraw.Draw(prev).rectangle([box[1] * sc, box[0] * sc, box[3] * sc, box[2] * sc], outline=(255, 60, 60), width=3)
    st.image(prev, use_container_width=True)

gray = M.prep_gray(M.roi_crop(a, box))
x = M.normalise(gray)
base = M.analyse(b, x)

with right:
    st.subheader("2. Check the concepts the model sees")
    st.caption(
        "If you disagree with a concept, change it. The risk updates because the decision layer "
        "uses only these concepts."
    )
    overrides = {}
    for h, classes in b.heads.items():
        default = int(np.argmax(base["probs"][h]))
        choice = st.selectbox(
            f"{h}  (model confidence {base['probs'][h][default]:.0%})",
            classes,
            index=default,
            key=f"{h}-{up.name}-{box}",
        )
        if classes.index(choice) != default:
            overrides[h] = classes.index(choice)
    res = M.analyse(b, x, overrides) if overrides else base

st.divider()
st.subheader("3. Result")
colour = {"Suspicious": "red", "Likely benign": "green", "Uncertain": "orange"}[res["verdict"]]
m1, m2, m3 = st.columns(3)
m1.markdown(f"### :{colour}[{res['verdict']}]")
m2.metric("Model risk score", f"{res['p']:.0%}", f"{res['p'] - res['base_p']:+.0%} vs average lesion in training data")
m3.write(
    "Uncertain means the model cannot separate benign from malignant at the chosen error level. "
    "Refer for expert review."
)
if overrides:
    st.info(f"Result reflects your corrections to: {', '.join(overrides)}.")
if res["ood"]:
    st.error(
        "This region looks unlike the training data (different scanner, view, or no clear lesion). "
        "Do not rely on this result."
    )

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Why: contribution of each concept (exact, additive in log-odds)**")
    df = pd.Series(res["contrib"]).sort_values(key=np.abs, ascending=False).head(8)
    st.bar_chart(df.sort_values())
    st.caption("Positive pushes toward malignant, negative toward benign, relative to an average lesion.")
with c2:
    head = st.selectbox(
        "Where did the model look for this concept?",
        list(b.heads),
        index=list(b.heads).index("margins") if "margins" in b.heads else 0,
    )
    k = int(np.argmax(base["probs"][head]))
    cam = M.gradcam(b.model, x, head, k)
    rgb = (0.55 * np.stack([gray] * 3, -1) + 0.45 * colormaps["jet"](cam)[..., :3]).clip(0, 1)
    st.image(rgb, caption=f"{head}: {b.heads[head][k]} (Grad-CAM, approximate)", use_container_width=True)

st.caption(
    "Limits: trained on one public dataset of digitised film mammograms, region-based, not validated "
    "prospectively or across scanners, populations or breast density groups. Report any disagreement "
    "between this tool and the clinical read to the project team."
)
