"""
app.py  –  Raman CNN Dashboard  (Streamlit)
============================================
Run:  streamlit run app.py

Tabs
----
1. Training Monitor   – live loss/accuracy curves from training_log.csv
2. Model Info         – architecture, class distribution
3. Evaluation         – confusion matrix, classification report
4. Saliency Explorer  – per-class Integrated-Gradient maps
5. Live Inference     – classify a new spectrum from the database
"""

import os, json, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Raman CNN Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT      = "outputs"
LOG_CSV  = os.path.join(OUT, "logs",  "training_log.csv")
REG_CSV  = os.path.join(OUT, "logs",  "key_spectral_regions.csv")
SAL_NPZ  = os.path.join(OUT, "logs",  "saliency_maps.npz")
CFG_JSON = os.path.join(OUT, "model", "model_config.json")
MODEL_VS_BASELINE_CSV = os.path.join(OUT, "logs", "model_vs_baseline_models.csv")
BEST_PT  = os.path.join(OUT, "model", "best_model.pt")
CM_PNG   = os.path.join(OUT, "figures", "confusion_matrix.png")
TC_PNG   = os.path.join(OUT, "figures", "training_curves.png")
CD_PNG   = os.path.join(OUT, "figures", "class_distribution.png")
MIX_CFG_JSON = os.path.join(OUT, "model", "model_config_mixture.json")
MIX_BEST_PT  = os.path.join(OUT, "model", "best_model_mixture.pt")
MIX_LOG_CSV  = os.path.join(OUT, "logs", "training_log_mixture.csv")
MIX_CM_PNG   = os.path.join(OUT, "figures", "confusion_matrix_mixture.png")
MIX_CM_CSV   = os.path.join(OUT, "logs", "confusion_matrix_mixture.csv")
MIX_SAL_NPZ  = os.path.join(OUT, "logs", "saliency_maps_mixture.npz")
MIX_REG_CSV  = os.path.join(OUT, "logs", "key_spectral_regions_mixture.csv")
MIX_SAL_HM_PNG = os.path.join(OUT, "figures", "saliency_heatmap_all_mixture.png")

META_CSV    = "ramanbiolib/db/metadata_db.csv"
SPECTRA_CSV = "ramanbiolib/db/raman_spectra_db.csv"

# ── Helper: parse list strings from CSV ───────────────────────────────────────
def parse_list(s):
    return [float(v) for v in str(s).strip("[]").split(", ") if v]


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def latest_single_metrics():
    # Prefer the latest evaluated CNN row; fallback to model_config.json.
    if os.path.exists(MODEL_VS_BASELINE_CSV):
        try:
            df = pd.read_csv(MODEL_VS_BASELINE_CSV)
            if not df.empty:
                row = df[df["model"] == "cnn_multiscale_real_only"]
                if row.empty:
                    row = df[df["model"].astype(str).str.contains("cnn", case=False, na=False)]
                if row.empty:
                    row = df.iloc[[0]]
                row = row.iloc[0]
                return _safe_float(row.get("test_acc")), _safe_float(row.get("macro_f1")), "model_vs_baseline_models.csv"
        except Exception:
            pass
    if os.path.exists(CFG_JSON):
        try:
            cfg = json.load(open(CFG_JSON))
            return _safe_float(cfg.get("test_acc")), _safe_float(cfg.get("test_macro_f1")), "model_config.json"
        except Exception:
            pass
    return None, None, None

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Raman CNN Dashboard")
st.sidebar.markdown("**ramanbiolib** — 1D CNN + Integrated Gradients")
st.sidebar.divider()

# Training status badge
training_done = os.path.exists(LOG_CSV)
if training_done:
    test_acc, test_macro_f1, source_name = latest_single_metrics()
    acc_str = f"{test_acc:.1%}" if isinstance(test_acc, (int, float)) else "N/A"
    st.sidebar.success(f"✅ Training complete\nTest acc: **{acc_str}**")
    if isinstance(test_macro_f1, (int, float)):
        st.sidebar.caption(f"Macro-F1: {test_macro_f1:.3f}")
    if source_name:
        st.sidebar.caption(f"Source: {source_name}")
else:
    st.sidebar.warning("⏳ Training in progress…")
    st.sidebar.caption("Dashboard auto-refreshes every 10 s")

if os.path.exists(BEST_PT):
    size_mb = os.path.getsize(BEST_PT) / 1e6
    st.sidebar.info(f"💾 best_model.pt  ({size_mb:.1f} MB)")

st.sidebar.divider()
tab_names = ["📈 Training Monitor", "🏗️ Model Info",
             "📊 Evaluation", "🌡️ Saliency Explorer", "⚡ Live Inference",
             "🧪 Mixture Evaluation", "🧪 Mixture Saliency", "🧪 Mixture Inference"]
selected = st.sidebar.radio("Navigate", tab_names)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Monitor
# ═══════════════════════════════════════════════════════════════════════════════
if selected == tab_names[0]:
    st.title("📈 Training Monitor")

    if not os.path.exists(LOG_CSV):
        # Show partial progress from best_model.pt existence
        st.info("Training is running in the background. This page auto-refreshes every 10 seconds.")
        col1, col2 = st.columns(2)
        col1.metric("Model checkpoint", "✅ saved" if os.path.exists(BEST_PT) else "⏳ waiting")
        col2.metric("Class distribution", "✅ saved" if os.path.exists(CD_PNG) else "⏳ waiting")
        if os.path.exists(CD_PNG):
            st.image(CD_PNG, caption="Class Distribution", use_container_width=True)
        st.info("Training log will appear here once training completes.")
        time.sleep(10)
        st.rerun()
    else:
        df = pd.read_csv(LOG_CSV)
        epochs = list(range(1, len(df) + 1))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Cross-Entropy Loss", "Classification Accuracy"])
        fig.add_trace(go.Scatter(x=epochs, y=df["train_loss"], name="Train loss", line=dict(color="#4C8BF5")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=df["val_loss"],   name="Val loss",   line=dict(color="#F5A623")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=df["train_acc"],  name="Train acc",  line=dict(color="#4C8BF5")), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=df["val_acc"],    name="Val acc",    line=dict(color="#F5A623")), row=1, col=2)
        fig.update_xaxes(title_text="Epoch")
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final train loss", f"{df['train_loss'].iloc[-1]:.4f}")
        col2.metric("Final val loss",   f"{df['val_loss'].iloc[-1]:.4f}")
        col3.metric("Best train acc",   f"{df['train_acc'].max():.3f}")
        col4.metric("Best val acc",     f"{df['val_acc'].max():.3f}")

        st.dataframe(df.round(4), use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Info
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[1]:
    st.title("🏗️ Model & Data Info")

    # Class distribution
    if os.path.exists(CD_PNG):
        st.subheader("Class Distribution")
        st.image(CD_PNG, use_container_width=True)

    # Dataset stats
    try:
        meta = pd.read_csv(META_CSV)
        spec = pd.read_csv(SPECTRA_CSV,
                           converters={"wavenumbers": parse_list, "intensity": parse_list})
        merged = spec.merge(meta[["id","type"]].drop_duplicates("id"), on="id")
        merged["class"] = merged["type"].str.split("/").str[0]
        KEEP = ["Proteins","Lipids","Saccharides","AminoAcids","PrimaryMetabolites","NucleicAcids"]
        merged = merged[merged["class"].isin(KEEP)]
        counts = merged["class"].value_counts().reset_index()
        counts.columns = ["Class","Spectra"]

        fig = px.bar(counts, x="Class", y="Spectra", color="Class",
                     title="Spectra per class (ramanbiolib)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        wn = np.array(merged["wavenumbers"].iloc[0])
        col1, col2, col3 = st.columns(3)
        col1.metric("Total spectra (6 classes)", len(merged))
        col2.metric("Wavenumber points", len(wn))
        col3.metric("Wavenumber range", f"{wn[0]:.0f}–{wn[-1]:.0f} cm⁻¹")
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")

    # Architecture
    st.subheader("CNN Architecture")
    st.code("""
RamanCNN1D  (input: batch × 1 × 1351)
│
├─ Block 1: Conv1D(1→32, k=15) ×2  + BN + ReLU + MaxPool(4) + Dropout(0.25)
├─ Block 2: Conv1D(32→64, k=11) ×2 + BN + ReLU + MaxPool(4) + Dropout(0.25)
├─ Block 3: Conv1D(64→128, k=7)    + BN + ReLU + MaxPool(4) + Dropout(0.25)
│
├─ Flatten → Linear(→256) + ReLU + Dropout(0.4)
└─ Linear(256 → 6 classes)

Parameters : 831,654
Optimizer  : Adam  (lr=1e-3, wd=1e-4)
Scheduler  : CosineAnnealingLR (T_max=80)
Batch size : 32  (WeightedRandomSampler for class balance)
Augment    : ×15 Gaussian noise (σ=0.015) + amplitude scaling (0.85–1.15)
""", language="text")

    if os.path.exists(CFG_JSON):
        cfg = json.load(open(CFG_JSON))
        st.json(cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[2]:
    st.title("📊 Evaluation")

    if not training_done:
        st.info("⏳ Waiting for training to complete…")
        time.sleep(10); st.rerun()
    else:
        if os.path.exists(CM_PNG):
            st.subheader("Confusion Matrix")
            st.image(CM_PNG, use_container_width=True)

            # also compute ROC/AUC points from the confusion matrix CSV
            cm_csv = os.path.join(OUT, "logs", "confusion_matrix_single.csv")
            if os.path.exists(cm_csv):
                try:
                    df_cm = pd.read_csv(cm_csv)
                    if {"class", "tp", "fp", "tn", "fn"}.issubset(df_cm.columns):
                        roc_df = df_cm.copy()
                    else:
                        true_col = df_cm.columns[0]
                        matrix = df_cm.set_index(true_col)
                        matrix.index = matrix.index.astype(str).str.replace("true_", "", regex=False)
                        matrix.columns = matrix.columns.astype(str).str.replace("pred_", "", regex=False)
                        cm = matrix.to_numpy(dtype=float)
                        total = float(cm.sum())
                        roc_rows = []
                        for idx, class_name in enumerate(matrix.index):
                            tp = float(cm[idx, idx])
                            fn = float(cm[idx, :].sum() - tp)
                            fp = float(cm[:, idx].sum() - tp)
                            tn = float(total - tp - fn - fp)
                            roc_rows.append({
                                "class": class_name,
                                "tp": tp,
                                "fp": fp,
                                "tn": tn,
                                "fn": fn,
                            })
                        roc_df = pd.DataFrame(roc_rows)
                    roc_df["tpr"] = roc_df["tp"] / np.maximum(roc_df["tp"] + roc_df["fn"], 1e-9)
                    roc_df["fpr"] = roc_df["fp"] / np.maximum(roc_df["fp"] + roc_df["tn"], 1e-9)
                    roc_df["auc"] = 0.5 * (roc_df["tpr"] + (1 - roc_df["fpr"]))

                    st.subheader("ROC points from confusion matrix")
                    fig_roc = px.scatter(roc_df, x="fpr", y="tpr", text="class",
                                         title="ROC (one point per class)",
                                         labels={"fpr":"False positive rate",
                                                 "tpr":"True positive rate"})
                    # annotate AUC on hover
                    fig_roc.update_traces(marker_size=12,
                                          hovertemplate="%{text}<br>FPR=%{x:.2f}<br>TPR=%{y:.2f}<br>AUC≈%{customdata[0]:.3f}",
                                          customdata=roc_df[["auc"]].values)
                    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                      line=dict(dash="dash", color="gray"))
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate ROC plot: {e}")

        if os.path.exists(TC_PNG):
            st.subheader("Training Curves")
            st.image(TC_PNG, use_container_width=True)

        if os.path.exists(REG_CSV):
            st.subheader("Top Spectral Regions per Class (from Integrated Gradients)")
            df_reg = pd.read_csv(REG_CSV)
            st.dataframe(df_reg, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Saliency Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[3]:
    st.title("🌡️ Saliency Explorer (Integrated Gradients)")

    if not os.path.exists(SAL_NPZ):
        st.info("⏳ Saliency maps not yet generated. Waiting for training to complete…")
        time.sleep(10); st.rerun()
    else:
        data = np.load(SAL_NPZ, allow_pickle=True)
        wn   = data["wavenumbers"]
        classes = list(data["class_names"])

        sel_class = st.selectbox("Select molecular class", classes)
        sal = data[sel_class]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

        # Load mean spectrum
        try:
            spec_df = pd.read_csv(SPECTRA_CSV,
                                  converters={"wavenumbers": parse_list, "intensity": parse_list})
            meta_df = pd.read_csv(META_CSV)
            merged  = spec_df.merge(meta_df[["id","type"]].drop_duplicates("id"), on="id")
            merged["class"] = merged["type"].str.split("/").str[0]
            X_cls = np.array(merged[merged["class"]==sel_class]["intensity"].tolist())
            mean_spec = X_cls.mean(axis=0) if len(X_cls) > 0 else np.zeros_like(wn)
        except Exception:
            mean_spec = np.zeros_like(wn)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=wn, y=mean_spec, name="Mean spectrum",
                                 line=dict(color="#4C8BF5", width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=wn, y=sal_n, name="|IG| saliency",
                                 fill="tozeroy", fillcolor="rgba(220,20,60,0.25)",
                                 line=dict(color="crimson", width=1)), secondary_y=True)
        fig.update_xaxes(title_text="Wavenumber (cm⁻¹)")
        fig.update_yaxes(title_text="Intensity", secondary_y=False)
        fig.update_yaxes(title_text="Normalised |IG|", secondary_y=True)
        fig.update_layout(title=f"Saliency Map — {sel_class}", height=450,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap of all classes
        st.subheader("All-class saliency heatmap")
        sal_matrix = np.array([(data[c]-data[c].min())/(data[c].max()-data[c].min()+1e-9) for c in classes])
        step = 5
        fig2 = px.imshow(sal_matrix[:, ::step],
                         x=[f"{v:.0f}" for v in wn[::step]],
                         y=classes,
                         color_continuous_scale="hot",
                         aspect="auto",
                         title="Integrated-Gradient Saliency (all classes)",
                         labels={"x": "Wavenumber (cm⁻¹)", "color": "|IG|"})
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        if os.path.exists(REG_CSV):
            df_reg = pd.read_csv(REG_CSV)
            st.subheader(f"Top wavenumber regions — {sel_class}")
            st.dataframe(df_reg[df_reg["class"]==sel_class].reset_index(drop=True),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Inference
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[4]:
    st.title("⚡ Live Inference")

    if not os.path.exists(BEST_PT):
        st.info("⏳ Model checkpoint not yet available.")
        time.sleep(5); st.rerun()

    try:
        import torch, torch.nn as nn

        def _normalise_per_spectrum(x):
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            return (x - x_min) / max(x_max - x_min, 1e-8)

        def preprocess_single_spectrum(intensity, cfg):
            x = np.asarray(intensity, dtype=np.float32).copy()
            if bool(cfg.get("use_log_scale", True)):
                x = np.log1p(np.clip(x, 0.0, None)).astype(np.float32)

            window = int(cfg.get("savgol_window", 0) or 0)
            poly = int(cfg.get("savgol_poly", 2) or 2)
            if window >= 3:
                if window % 2 == 0:
                    window += 1
                max_win = x.shape[0] if x.shape[0] % 2 == 1 else (x.shape[0] - 1)
                window = min(window, max_win)
                if window >= 3:
                    poly = min(max(0, poly), window - 1)
                    x = savgol_filter(x, window_length=window, polyorder=poly).astype(np.float32)

            base = _normalise_per_spectrum(x).astype(np.float32)
            channels = [base]
            deriv = int(cfg.get("derivative_channels", 0) or 0)
            if deriv >= 1:
                d1 = np.gradient(base).astype(np.float32)
                channels.append(_normalise_per_spectrum(d1).astype(np.float32))
            if deriv >= 2:
                d2 = np.gradient(np.gradient(base)).astype(np.float32)
                channels.append(_normalise_per_spectrum(d2).astype(np.float32))

            if len(channels) == 1:
                x_model = channels[0][None, None, :]
            else:
                x_model = np.stack(channels, axis=0)[None, :, :]
            return torch.tensor(x_model, dtype=torch.float32)

        class MultiScaleConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch, kernels=(9, 21, 41), pool=2, dropout=0.15):
                super().__init__()
                branch_ch = max(8, out_ch // len(kernels))
                self.branches = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(in_ch, branch_ch, kernel_size=k, padding=k // 2, bias=False),
                        nn.BatchNorm1d(branch_ch),
                        nn.ReLU(inplace=True)
                    )
                    for k in kernels
                ])
                fused_ch = branch_ch * len(kernels)
                self.fuse = nn.Sequential(
                    nn.Conv1d(fused_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(pool),
                    nn.Dropout(dropout)
                )

            def forward(self, x):
                x_multi = torch.cat([branch(x) for branch in self.branches], dim=1)
                return self.fuse(x_multi)

        class RamanCNN1D(nn.Module):
            def __init__(
                self,
                n_classes=6,
                in_channels=1,
                kernels=(9, 21, 41),
                channels=(64, 96, 128),
                classifier_hidden=128,
                conv_dropout=0.15,
                dense_dropout=0.15
            ):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True)
                )
                self.block1 = MultiScaleConvBlock(32, int(channels[0]), kernels=kernels, pool=2, dropout=conv_dropout)
                self.block2 = MultiScaleConvBlock(int(channels[0]), int(channels[1]), kernels=kernels, pool=2, dropout=conv_dropout)
                self.block3 = MultiScaleConvBlock(int(channels[1]), int(channels[2]), kernels=kernels, pool=2, dropout=conv_dropout)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(int(channels[2]), int(classifier_hidden)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dense_dropout),
                    nn.Linear(int(classifier_hidden), int(n_classes))
                )

            def _fwd(self, x):
                x = self.stem(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                return self.global_pool(x).squeeze(-1)

            def forward(self, x):
                return self.classifier(self._fwd(x))

        @st.cache_resource
        def load_model():
            cfg = json.load(open(CFG_JSON)) if os.path.exists(CFG_JSON) else {}
            default_classes = ["AminoAcids","Lipids","NucleicAcids","PrimaryMetabolites","Proteins","Saccharides"]
            class_names = cfg.get("class_names", default_classes)
            kernels = tuple(cfg.get("multiscale_kernels", [9, 21, 41]))
            channels = tuple(cfg.get("cnn_channels", [64, 96, 128]))
            m = RamanCNN1D(
                n_classes=int(cfg.get("n_classes", len(class_names))),
                in_channels=int(cfg.get("input_channels", 1)),
                kernels=kernels,
                channels=channels,
                classifier_hidden=int(cfg.get("classifier_hidden", 128)),
                conv_dropout=float(cfg.get("conv_dropout", 0.15)),
                dense_dropout=float(cfg.get("dense_dropout", 0.15))
            )
            m.load_state_dict(torch.load(BEST_PT, map_location="cpu"))
            m.eval()
            return m, class_names, cfg

        model, CLASS_NAMES, model_cfg = load_model()

        # Load DB spectra
        spec_df = pd.read_csv(SPECTRA_CSV,
                              converters={"wavenumbers":parse_list,"intensity":parse_list})
        meta_df = pd.read_csv(META_CSV)
        merged = spec_df.merge(meta_df[["id","type"]].drop_duplicates("id"), on="id")
        merged["class"] = merged["type"].str.split("/").str[0]
        KEEP = ["Proteins","Lipids","Saccharides","AminoAcids","PrimaryMetabolites","NucleicAcids"]
        merged = merged[merged["class"].isin(KEEP)]

        st.subheader("Select a spectrum from the database")
        component_list = merged["component"].tolist()
        selected_comp = st.selectbox("Component", component_list)
        row = merged[merged["component"] == selected_comp].iloc[0]
        intensity = np.array(row["intensity"], dtype=np.float32)
        wn = np.array(row["wavenumbers"])
        true_class = row["class"]

        # Plot spectrum
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wn, y=intensity, mode="lines", name=selected_comp,
                                 line=dict(color="#4C8BF5", width=1.5)))
        fig.update_layout(title=f"Spectrum: {selected_comp}",
                          xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity",
                          height=350)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🔍 Run Inference", type="primary"):
            x = preprocess_single_spectrum(intensity, model_cfg)
            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1).squeeze().numpy()
                pred_idx = int(probs.argmax())
                pred_class = CLASS_NAMES[pred_idx]

            col1, col2 = st.columns(2)
            match = pred_class == true_class
            col1.metric("Predicted class", pred_class,
                        delta="✓ correct" if match else f"✗ true: {true_class}",
                        delta_color="normal" if match else "inverse")
            col2.metric("Confidence", f"{probs[pred_idx]*100:.1f}%")

            fig_prob = px.bar(
                x=CLASS_NAMES, y=probs*100,
                labels={"x":"Class","y":"Probability (%)"},
                title="Class probabilities",
                color=CLASS_NAMES,
                height=300
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    except Exception as e:
        st.error(f"Inference error: {e}")
        st.info("Ensure the model checkpoint (best_model.pt) exists and training has completed.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Mixture Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[5]:
    st.title("🧪 Mixture Evaluation")
    st.caption("Artifacts from: python train_cnn_raman.py --task mixture")

    if not os.path.exists(MIX_CFG_JSON):
        st.info("Mixture model config not found yet. Train with: python train_cnn_raman.py --task mixture")
    else:
        cfg = json.load(open(MIX_CFG_JSON))
        st.metric("Mixture test macro-F1", f"{cfg.get('test_macro_f1', 'N/A')}")

        if os.path.exists(MIX_LOG_CSV):
            dfm = pd.read_csv(MIX_LOG_CSV)
            epochs = list(range(1, len(dfm) + 1))
            figm = make_subplots(rows=1, cols=2, subplot_titles=["BCE Loss", "Macro-F1"])
            figm.add_trace(go.Scatter(x=epochs, y=dfm["train_loss"], name="Train loss", line=dict(color="#4C8BF5")), row=1, col=1)
            figm.add_trace(go.Scatter(x=epochs, y=dfm["val_loss"], name="Val loss", line=dict(color="#F5A623")), row=1, col=1)
            figm.add_trace(go.Scatter(x=epochs, y=dfm["train_f1"], name="Train macro-F1", line=dict(color="#4C8BF5")), row=1, col=2)
            figm.add_trace(go.Scatter(x=epochs, y=dfm["val_f1"], name="Val macro-F1", line=dict(color="#F5A623")), row=1, col=2)
            figm.update_xaxes(title_text="Epoch")
            figm.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(figm, use_container_width=True)

        if os.path.exists(MIX_CM_PNG):
            st.subheader("Mixture confusion matrix (one-vs-rest per class)")
            st.image(MIX_CM_PNG, use_container_width=True)
        if os.path.exists(MIX_CM_CSV):
            st.subheader("Mixture confusion counts")
            st.dataframe(pd.read_csv(MIX_CM_CSV), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Mixture Saliency Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[6]:
    st.title("🧪 Mixture Saliency Explorer")
    st.caption("Separate from the original saliency explorer")

    if not os.path.exists(MIX_SAL_NPZ):
        st.info("Mixture saliency maps not found yet. Train with: python train_cnn_raman.py --task mixture")
    else:
        mix_data = np.load(MIX_SAL_NPZ, allow_pickle=True)
        wn = mix_data["wavenumbers"]
        classes = list(mix_data["class_names"])

        sel_class = st.selectbox("Select mixture class", classes)
        sal = mix_data[sel_class]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wn, y=sal_n, name="Mixture |IG| saliency",
            fill="tozeroy", fillcolor="rgba(220,20,60,0.25)",
            line=dict(color="crimson", width=1)
        ))
        fig.update_layout(
            title=f"Mixture Saliency — {sel_class}",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Normalised |IG|",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("All-class mixture saliency heatmap")
        sal_matrix = np.array([
            (mix_data[c] - mix_data[c].min()) / (mix_data[c].max() - mix_data[c].min() + 1e-9)
            for c in classes
        ])
        step = 5
        fig2 = px.imshow(
            sal_matrix[:, ::step],
            x=[f"{v:.0f}" for v in wn[::step]],
            y=classes,
            color_continuous_scale="hot",
            aspect="auto",
            title="Integrated-Gradient Saliency (mixture model)",
            labels={"x": "Wavenumber (cm⁻¹)", "color": "|IG|"}
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        if os.path.exists(MIX_SAL_HM_PNG):
            st.image(MIX_SAL_HM_PNG, caption="Saved mixture saliency heatmap", use_container_width=True)
        if os.path.exists(MIX_REG_CSV):
            df_mix_reg = pd.read_csv(MIX_REG_CSV)
            st.subheader(f"Top mixture regions — {sel_class}")
            st.dataframe(
                df_mix_reg[df_mix_reg["class"] == sel_class].reset_index(drop=True),
                use_container_width=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Mixture Inference (multi-label)
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[7]:
    st.title("🧪 Mixture Inference (Multi-label)")
    st.caption("Requires artifacts from: python train_cnn_raman.py --task mixture")

    if not os.path.exists(MIX_BEST_PT) or not os.path.exists(MIX_CFG_JSON):
        st.info("Mixture model not found yet. Train with: python train_cnn_raman.py --task mixture")
    else:
        try:
            import torch, torch.nn as nn

            class RamanCNN1D(nn.Module):
                def __init__(self, input_len=1351, n_classes=6):
                    super().__init__()
                    self.block1 = nn.Sequential(
                        nn.Conv1d(1, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.Conv1d(32, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.MaxPool1d(4), nn.Dropout(0.25))
                    self.block2 = nn.Sequential(
                        nn.Conv1d(32, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.Conv1d(64, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.MaxPool1d(4), nn.Dropout(0.25))
                    self.block3 = nn.Sequential(
                        nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
                        nn.MaxPool1d(4), nn.Dropout(0.25))
                    dummy = torch.zeros(1, 1, input_len)
                    flat = self._fwd(dummy).shape[1]
                    self.classifier = nn.Sequential(
                        nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.4),
                        nn.Linear(256, n_classes))
                def _fwd(self, x):
                    return self.block3(self.block2(self.block1(x))).view(x.size(0), -1)
                def forward(self, x):
                    return self.classifier(self._fwd(x))

            @st.cache_resource
            def load_mixture_model():
                cfg = json.load(open(MIX_CFG_JSON))
                model = RamanCNN1D(cfg["input_len"], cfg["n_classes"])
                model.load_state_dict(torch.load(MIX_BEST_PT, map_location="cpu"))
                model.eval()
                return model, cfg

            mix_model, mix_cfg = load_mixture_model()
            class_names = mix_cfg["class_names"]

            spec_df = pd.read_csv(
                SPECTRA_CSV,
                converters={"wavenumbers": parse_list, "intensity": parse_list}
            )
            meta_df = pd.read_csv(META_CSV)
            merged = spec_df.merge(meta_df[["id", "type"]].drop_duplicates("id"), on="id")
            merged["class"] = merged["type"].str.split("/").str[0]
            merged = merged[merged["class"].isin(class_names)].reset_index(drop=True)

            st.subheader("Build a synthetic mixture from existing spectra")
            n_comp = st.slider("Number of components", min_value=2, max_value=4, value=2, step=1)
            threshold = st.slider("Prediction threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)

            selected_rows = []
            weights = []
            cols = st.columns(n_comp)
            for i in range(n_comp):
                with cols[i]:
                    cls = st.selectbox(f"Class {i+1}", options=class_names, index=i % len(class_names), key=f"mix_cls_{i}")
                    subset = merged[merged["class"] == cls]
                    comp = st.selectbox(f"Component {i+1}", options=subset["component"].tolist(), key=f"mix_comp_{i}")
                    wt = st.number_input(f"Weight {i+1}", min_value=0.0, value=float(1.0 / n_comp), step=0.05, key=f"mix_wt_{i}")
                    row = subset[subset["component"] == comp].iloc[0]
                    selected_rows.append(row)
                    weights.append(float(wt))

            w = np.array(weights, dtype=np.float32)
            w = np.ones_like(w) / len(w) if w.sum() <= 0 else (w / w.sum())
            wn = np.array(selected_rows[0]["wavenumbers"], dtype=np.float32)
            mix_intensity = np.zeros_like(np.array(selected_rows[0]["intensity"], dtype=np.float32))
            true_labels = set()
            for wi, row in zip(w, selected_rows):
                mix_intensity += wi * np.array(row["intensity"], dtype=np.float32)
                true_labels.add(row["class"])

            fig_mix = go.Figure()
            fig_mix.add_trace(go.Scatter(x=wn, y=mix_intensity, mode="lines", name="Synthetic mixture",
                                         line=dict(color="#4C8BF5", width=1.5)))
            fig_mix.update_layout(
                title="Synthetic mixture spectrum",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Intensity",
                height=320
            )
            st.plotly_chart(fig_mix, use_container_width=True)

            if st.button("Run Mixture Inference", type="primary"):
                x = torch.tensor(mix_intensity).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    logits = mix_model(x)
                    probs = torch.sigmoid(logits).squeeze().numpy()

                pred_labels = [c for c, p in zip(class_names, probs) if p >= threshold]
                if not pred_labels:
                    pred_labels = [class_names[int(np.argmax(probs))]]

                st.write("True classes (from selected components):", ", ".join(sorted(true_labels)))
                st.write("Predicted classes:", ", ".join(pred_labels))

                prob_df = pd.DataFrame({"Class": class_names, "Probability (%)": probs * 100.0})
                fig_prob = px.bar(
                    prob_df, x="Class", y="Probability (%)", color="Class",
                    title="Per-class probabilities (sigmoid)", height=300
                )
                st.plotly_chart(fig_prob, use_container_width=True)
        except Exception as e:
            st.error(f"Mixture inference error: {e}")
