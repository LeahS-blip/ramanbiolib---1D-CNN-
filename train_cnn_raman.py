"""
train_cnn_raman.py
==================
Standalone Python script to reproduce the CNN classification pipeline
from cnn_raman_classification.ipynb.

Usage:
    python train_cnn_raman.py
    python train_cnn_raman.py --single-synthetic-samples 5000
    python train_cnn_raman.py --single-synthetic-samples 8000
    python train_cnn_raman.py --task mixture
    python train_cnn_raman.py --task mixture --mixture-samples 12000 --mixture-epochs 60

Outputs (written to outputs/):
    model/best_model.pt          -- best checkpoint by val-accuracy
    model/final_model.pt         -- weights at end of training
    model/model_config.json      -- architecture / hyperparameter metadata
    logs/training_log.csv        -- per-epoch loss & accuracy
    logs/key_spectral_regions.csv-- top-5 wavenumber regions per class
    logs/saliency_maps.npz       -- raw Integrated-Gradient arrays
    figures/class_distribution.png
    figures/training_curves.png
    figures/confusion_matrix.png
    figures/saliency_<class>.png (one per molecular class)
    figures/saliency_heatmap_all.png
"""

import os, json, warnings, argparse, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             multilabel_confusion_matrix, roc_curve, auc)
from scipy.signal import find_peaks, savgol_filter

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def augment_spectrum(x):
    noise = torch.randn_like(x) * 0.01
    scale = torch.rand(1).item() * 0.1 + 0.95
    shift = torch.randint(-3, 3, (1,)).item()
    x = x * scale + noise
    return torch.roll(x, shifts=shift, dims=-1)


def run_mixture_training(
    n_samples=12000,
    max_components=3,
    noise_std=0.01,
    epochs=60,
    batch_size=32,
    threshold=0.5
):
    """Train a multi-label CNN on synthetic spectral mixtures."""
    print("Running mixture training pipeline (--task mixture)")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(seed)

    for d in ('outputs/figures', 'outputs/logs', 'outputs/model'):
        os.makedirs(d, exist_ok=True)

    def parse_list_local(s):
        return [float(v) for v in s.strip('[]').split(', ')]

    spectra_df = pd.read_csv(
        'ramanbiolib/db/raman_spectra_db.csv',
        converters={'wavenumbers': parse_list_local, 'intensity': parse_list_local}
    )
    meta_df = pd.read_csv('ramanbiolib/db/metadata_db.csv')
    meta_unique = meta_df[['id', 'type']].drop_duplicates(subset='id')
    df = spectra_df.merge(meta_unique, on='id')
    df['class'] = df['type'].str.split('/').str[0]

    keep_classes = ['Proteins', 'Lipids', 'Saccharides',
                    'AminoAcids', 'PrimaryMetabolites', 'NucleicAcids']
    df = df[df['class'].isin(keep_classes)].reset_index(drop=True)
    class_names = sorted(df['class'].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    seq_len = len(df['intensity'].iloc[0])
    wavenumbers = np.array(df['wavenumbers'].iloc[0], dtype=np.float32)

    spectra_by_class = {
        c: np.array(df[df['class'] == c]['intensity'].tolist(), dtype=np.float32)
        for c in class_names
    }

    max_components = int(max(2, max_components))
    max_components = int(min(max_components, len(class_names)))
    n_samples = int(max(1, n_samples))
    epochs = int(max(1, epochs))
    batch_size = int(max(1, batch_size))
    threshold = float(np.clip(threshold, 0.0, 1.0))

    def synthesize_mixtures(n_samples_local, max_components_local, noise_std_local):
        X_mix = np.zeros((n_samples_local, seq_len), dtype=np.float32)
        Y_mix = np.zeros((n_samples_local, len(class_names)), dtype=np.float32)
        for i in range(n_samples_local):
            k = int(rng.integers(2, max_components_local + 1))
            picked = rng.choice(class_names, size=k, replace=False)
            weights = rng.dirichlet(np.ones(k)).astype(np.float32)
            mix = np.zeros(seq_len, dtype=np.float32)
            for cls, w in zip(picked, weights):
                idx = int(rng.integers(0, len(spectra_by_class[cls])))
                mix += w * spectra_by_class[cls][idx]
                Y_mix[i, class_to_idx[cls]] = 1.0
            mix += rng.normal(0, noise_std_local, seq_len).astype(np.float32)
            X_mix[i] = np.clip(mix, 0.0, None)
        return X_mix, Y_mix

    X, Y = synthesize_mixtures(
        n_samples_local=n_samples,
        max_components_local=max_components,
        noise_std_local=float(noise_std)
    )
    # if holdout provided, use that as test set
    if 'holdout_X' in locals() and holdout_X is not None:
        X_train, y_train = X, Y
        X_test, y_test = holdout_X, holdout_y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.20, random_state=seed, shuffle=True
        )

    class RamanMixDataset(Dataset):
        def __init__(self, X_arr, y_arr, train=False):
            self.X = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y_arr, dtype=torch.float32)
            self.train = bool(train)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            spectrum = self.X[idx]
            if self.train:
                spectrum = augment_spectrum(spectrum)
            return spectrum, self.y[idx]

    train_ds = RamanMixDataset(X_train, y_train, train=True)
    test_ds = RamanMixDataset(X_test, y_test, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class RamanCNN1D(nn.Module):
        def __init__(self, input_len=1351, n_classes=6):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv1d(1, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(4), nn.Dropout(0.25)
            )
            self.block2 = nn.Sequential(
                nn.Conv1d(32, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(4), nn.Dropout(0.25)
            )
            self.block3 = nn.Sequential(
                nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(4), nn.Dropout(0.25)
            )
            dummy = torch.zeros(1, 1, input_len)
            flat = self._fwd(dummy).shape[1]
            self.classifier = nn.Sequential(
                nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, n_classes)
            )
        def _fwd(self, x):
            return self.block3(self.block2(self.block1(x))).view(x.size(0), -1)
        def forward(self, x):
            return self.classifier(self._fwd(x))

    model = RamanCNN1D(input_len=seq_len, n_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_path = 'outputs/model/best_model_mixture.pt'

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_true, tr_pred = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * len(yb)
            probs = torch.sigmoid(logits)
            tr_pred.append((probs > threshold).detach().cpu().numpy())
            tr_true.append(yb.detach().cpu().numpy())
        scheduler.step()

        model.eval()
        va_loss = 0.0
        va_true, va_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += float(loss.item()) * len(yb)
                probs = torch.sigmoid(logits)
                va_pred.append((probs > threshold).cpu().numpy())
                va_true.append(yb.cpu().numpy())

        tr_true = np.vstack(tr_true).astype(int)
        tr_pred = np.vstack(tr_pred).astype(int)
        va_true = np.vstack(va_true).astype(int)
        va_pred = np.vstack(va_pred).astype(int)

        tr_f1 = f1_score(tr_true, tr_pred, average='macro', zero_division=0)
        va_f1 = f1_score(va_true, va_pred, average='macro', zero_division=0)
        history['train_loss'].append(tr_loss / len(train_ds))
        history['val_loss'].append(va_loss / len(test_ds))
        history['train_f1'].append(float(tr_f1))
        history['val_f1'].append(float(va_f1))

        if va_f1 > best_val_f1:
            best_val_f1 = float(va_f1)
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  train_loss={history['train_loss'][-1]:.4f} "
                f"val_loss={history['val_loss'][-1]:.4f} "
                f"train_f1={tr_f1:.3f} val_f1={va_f1:.3f}"
            )

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs > threshold).astype(int)
    test_macro_f1 = f1_score(y_test.astype(int), y_pred, average='macro', zero_division=0)

    # Mixture confusion matrices and ROC curves (one-vs-rest per class)
    mcm = multilabel_confusion_matrix(y_test.astype(int), y_pred)
    fig_cm, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    cm_rows = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, cls_name in enumerate(class_names):
        # Confusion matrix subplot
        cm = mcm[i]  # [[tn, fp], [fn, tp]]
        ax = axes[i]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{cls_name} (CM)', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1], labels=['Pred 0', 'Pred 1'])
        ax.set_yticks([0, 1], labels=['True 0', 'True 1'])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cm[r, c]), ha='center', va='center', color='black', fontsize=10)
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        cm_rows.append({'class': cls_name, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    
    # ROC curves in remaining subplots
    for i, cls_name in enumerate(class_names):
        ax = axes[len(class_names) + i]
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{cls_name} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{cls_name} (ROC)', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
    
    fig_cm.suptitle('Mixture Multi-label: Confusion Matrices and ROC Curves (one-vs-rest)', fontsize=13, fontweight='bold')
    fig_cm.tight_layout()
    fig_cm.savefig('outputs/figures/confusion_matrix_mixture.png', dpi=150, bbox_inches='tight')
    plt.close(fig_cm)
    pd.DataFrame(cm_rows).to_csv('outputs/logs/confusion_matrix_mixture.csv', index=False)

    # Mixture Integrated Gradients saliency maps
    def integrated_gradients_mix(mdl, x, target_class, n_steps=50):
        baseline = torch.zeros_like(x)
        alphas = torch.linspace(0, 1, n_steps, device=device)
        interpolated = torch.stack([baseline + a * (x - baseline) for a in alphas]).squeeze(1)
        interpolated.requires_grad_(True)
        logits_int = mdl(interpolated)
        logits_int[:, target_class].sum().backward()
        avg_grads = interpolated.grad.mean(dim=0)
        ig = ((x - baseline).squeeze() * avg_grads.squeeze()).detach().cpu().numpy()
        return ig

    def class_mean_saliency_mix(mdl, X_cls, class_idx, n_samples=30, n_steps=50):
        if len(X_cls) == 0:
            return np.zeros(seq_len, dtype=np.float32)
        idx = rng.choice(len(X_cls), size=min(n_samples, len(X_cls)), replace=False)
        vals = []
        mdl.eval()
        for j in idx:
            x = torch.tensor(X_cls[j], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            ig = integrated_gradients_mix(mdl, x, class_idx, n_steps=n_steps)
            vals.append(np.abs(ig))
        return np.mean(vals, axis=0).astype(np.float32)

    saliency_maps_mix = {}
    mean_spectra_mix = {}
    for i, cls_name in enumerate(class_names):
        mask = y_test[:, i] > 0.5
        X_cls = X_test[mask]
        saliency_maps_mix[cls_name] = class_mean_saliency_mix(model, X_cls, i)
        mean_spectra_mix[cls_name] = X_cls.mean(axis=0) if len(X_cls) else np.zeros(seq_len, dtype=np.float32)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, cls_name in enumerate(class_names):
        sal = saliency_maps_mix[cls_name]
        spec = mean_spectra_mix[cls_name]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(wavenumbers, spec, color=colors[i], lw=1.5, label='Mean mixture spectrum')
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Intensity', color=colors[i])
        ax1.tick_params(axis='y', labelcolor=colors[i])
        ax2 = ax1.twinx()
        ax2.fill_between(wavenumbers, sal_n, alpha=0.35, color='crimson', label='IG saliency')
        ax2.set_ylabel('Normalised |IG|', color='crimson')
        ax2.tick_params(axis='y', labelcolor='crimson')
        ax1.set_title(f'Mixture Saliency Map - {cls_name}')
        plt.tight_layout()
        plt.savefig(f'outputs/figures/saliency_mixture_{cls_name.lower()}.png', dpi=150)
        plt.close(fig)

    sal_matrix_mix = np.array([
        (saliency_maps_mix[c] - saliency_maps_mix[c].min()) /
        (saliency_maps_mix[c].max() - saliency_maps_mix[c].min() + 1e-9)
        for c in class_names
    ])
    step = 10
    wn_ds = wavenumbers[::step]
    sd_ds = sal_matrix_mix[:, ::step]
    fig_hm, ax_hm = plt.subplots(figsize=(14, 4))
    im = ax_hm.imshow(
        sd_ds, aspect='auto', cmap='hot',
        extent=[wn_ds[0], wn_ds[-1], len(class_names) - 0.5, -0.5]
    )
    ax_hm.set_yticks(range(len(class_names)))
    ax_hm.set_yticklabels(class_names, fontsize=11)
    ax_hm.set_xlabel('Wavenumber (cm⁻¹)')
    ax_hm.set_title('Integrated-Gradient Saliency Heatmap (mixture classes)')
    plt.colorbar(im, ax=ax_hm, label='Normalised |IG|')
    plt.tight_layout()
    plt.savefig('outputs/figures/saliency_heatmap_all_mixture.png', dpi=150)
    plt.close(fig_hm)

    window = 20
    summary_rows = []
    for cls_name in class_names:
        sal = saliency_maps_mix[cls_name]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
        peaks, _ = find_peaks(sal_n, prominence=0.10, distance=15)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(sal_n))])
        ranked = peaks[np.argsort(sal_n[peaks])[::-1]]
        for pk in ranked[:5]:
            wn = wavenumbers[pk]
            summary_rows.append({
                'class': cls_name,
                'center_cm': int(wn),
                'range': f'{int(wn - window)}-{int(wn + window)} cm^-1',
                'saliency_score': round(float(sal_n[pk]), 4)
            })
    pd.DataFrame(summary_rows).to_csv('outputs/logs/key_spectral_regions_mixture.csv', index=False)
    np.savez(
        'outputs/logs/saliency_maps_mixture.npz',
        wavenumbers=wavenumbers,
        class_names=np.array(class_names),
        **{cls: saliency_maps_mix[cls] for cls in class_names}
    )

    pd.DataFrame(history).to_csv('outputs/logs/training_log_mixture.csv', index=False)
    torch.save(model.state_dict(), 'outputs/model/final_model_mixture.pt')
    cfg = {
        'task': 'mixture_multilabel',
        'input_len': int(seq_len),
        'n_classes': len(class_names),
        'class_names': class_names,
        'threshold_default': threshold,
        'synthetic_samples': int(len(X)),
        'max_components': int(max_components),
        'noise_std': float(noise_std),
        'batch_size': batch_size,
        'wavenumber_range': [int(wavenumbers[0]), int(wavenumbers[-1])],
        'epochs': epochs,
        'best_val_macro_f1': round(best_val_f1, 4),
        'test_macro_f1': round(float(test_macro_f1), 4)
    }
    with open('outputs/model/model_config_mixture.json', 'w') as fh:
        json.dump(cfg, fh, indent=2)

    print("Saved: outputs/logs/training_log_mixture.csv")
    print("Saved: outputs/figures/confusion_matrix_mixture.png")
    print("Saved: outputs/logs/confusion_matrix_mixture.csv")
    print("Saved: outputs/logs/saliency_maps_mixture.npz")
    print("Saved: outputs/logs/key_spectral_regions_mixture.csv")
    print("Saved: outputs/figures/saliency_heatmap_all_mixture.png")
    print("Saved: outputs/model/best_model_mixture.pt")
    print("Saved: outputs/model/model_config_mixture.json")
    print(f"Final test macro-F1: {test_macro_f1:.4f}")


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--task', choices=['single', 'mixture'], default='single')
_parser.add_argument('--use-merged', action='store_true', help='Load data from data/merged/X.npy and y.npy instead of internal DB')
_parser.add_argument('--single-epochs', type=int, default=150)
_parser.add_argument('--single-batch-size', type=int, default=128)
_parser.add_argument('--single-lr', type=float, default=1e-3)
_parser.add_argument('--single-weight-decay', type=float, default=1e-5)
_parser.add_argument('--single-conv-dropout', type=float, default=0.15)
_parser.add_argument('--single-dense-dropout', type=float, default=0.15)
_parser.add_argument('--single-loss', choices=['ce', 'focal'], default='focal')
_parser.add_argument('--single-focal-gamma', type=float, default=3.0)
_parser.add_argument('--single-focal-gamma-grid', type=str, default='2.0,3.0,4.0')
_parser.add_argument('--single-aug-factor', type=int, default=15)
_parser.add_argument('--single-shift-max', type=int, default=4)
_parser.add_argument('--single-aug-noise-std', type=float, default=0.012)
_parser.add_argument('--single-aug-scale-min', type=float, default=0.92)
_parser.add_argument('--single-aug-scale-max', type=float, default=1.08)
_parser.add_argument('--single-aug-stretch-max', type=float, default=0.03)
_parser.add_argument('--single-savgol-window', type=int, default=0)
_parser.add_argument('--single-savgol-poly', type=int, default=2)
_parser.add_argument('--single-derivative-channels', type=int, choices=[0, 1, 2], default=1)
_parser.add_argument(
    '--single-log-scale',
    type=int,
    choices=[0, 1],
    default=1,
    help='Apply log1p intensity transform before per-spectrum normalization.'
)
_parser.add_argument(
    '--single-real-only',
    type=int,
    choices=[0, 1],
    default=1,
    help='1 disables synthetic sample generation and trains/CVs on real spectra only.'
)
_parser.add_argument(
    '--single-synthetic-samples',
    type=int,
    default=0,
    help='Number of train-only synthetic single-label spectra to generate (e.g., 5000, 8000).'
)
_parser.add_argument('--single-synthetic-noise-std', type=float, default=0.01)
_parser.add_argument('--single-synthetic-max-components', type=int, default=3)
_parser.add_argument('--single-synthetic-peak-perturb-std', type=float, default=0.08)
_parser.add_argument('--single-val-size', type=float, default=0.15)
_parser.add_argument('--single-early-stop-patience', type=int, default=15)
_parser.add_argument('--single-kfolds', type=int, default=5)
_parser.add_argument('--single-balanced-sampler', type=int, choices=[0, 1], default=0)
_parser.add_argument('--single-class-weights', type=int, choices=[0, 1], default=0)
_parser.add_argument('--single-skip-interpretability', action='store_true')
_parser.add_argument('--single-skip-baselines', action='store_true')
_parser.add_argument('--single-continue-from-best', action='store_true')
_parser.add_argument('--single-extra-epochs', type=int, default=0)
_parser.add_argument('--single-resume-path', type=str, default='outputs/model/best_model.pt')
_parser.add_argument('--mixture-samples', type=int, default=12000)
_parser.add_argument('--mixture-max-components', type=int, default=3)
_parser.add_argument('--mixture-noise-std', type=float, default=0.01)
_parser.add_argument('--mixture-epochs', type=int, default=60)
_parser.add_argument('--mixture-batch-size', type=int, default=128)
_parser.add_argument('--mixture-threshold', type=float, default=0.5)
_args, _ = _parser.parse_known_args()
if _args.task == 'mixture':
    run_mixture_training(
        n_samples=_args.mixture_samples,
        max_components=_args.mixture_max_components,
        noise_std=_args.mixture_noise_std,
        epochs=_args.mixture_epochs,
        batch_size=_args.mixture_batch_size,
        threshold=_args.mixture_threshold
    )
    sys.exit(0)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

for d in ('outputs/figures', 'outputs/logs', 'outputs/model'):
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & prepare data
# ─────────────────────────────────────────────────────────────────────────────

def parse_list(s):
    return [float(v) for v in s.strip('[]').split(', ')]

# Optionally load merged arrays produced by scripts/merge_datasets.py
if _args.use_merged and Path('data/merged/X.npy').exists():
    print('Loading merged dataset from data/merged')
    X_raw = np.load('data/merged/X.npy')
    y_raw = np.load('data/merged/y.npy')
    try:
        import json as _jsonu
        classes_map = _jsonu.load(open('data/merged/classes.json'))
        # classes_map is name->index; produce ordered class list by index
        CLASS_NAMES = [None] * len(classes_map)
        for k, v in classes_map.items():
            CLASS_NAMES[v] = k
    except Exception:
        CLASS_NAMES = [str(i) for i in range(len(np.unique(y_raw)))]
    N_CLASSES = len(CLASS_NAMES)
    SEQ_LEN = X_raw.shape[1]
    # create synthetic wavenumbers matching default grid
    wavenumbers = np.linspace(400.0, 1800.0, SEQ_LEN).astype(np.float32)
    print(f'Loaded merged dataset: {X_raw.shape[0]} samples, {SEQ_LEN} points')
    # Construct a lightweight dataframe compatible with downstream code
    df = pd.DataFrame({
        'class': [CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else 'unknown' for lbl in y_raw],
        'wavenumbers': [list(wavenumbers) for _ in range(len(y_raw))],
        'intensity': [list(x) for x in X_raw]
    })
    # Ensure consistent types
    df['class'] = df['class'].astype(str)
else:
    spectra_df = pd.read_csv(
        'ramanbiolib/db/raman_spectra_db.csv',
        converters={'wavenumbers': parse_list, 'intensity': parse_list}
    )
    meta_df = pd.read_csv('ramanbiolib/db/metadata_db.csv')

    meta_unique = meta_df[['id', 'type']].drop_duplicates(subset='id')
    df = spectra_df.merge(meta_unique, on='id')
    df['class'] = df['type'].str.split('/').str[0]

print('Full dataset:', df.shape)
print(df['class'].value_counts().to_string(), '\n')

# ── Class distribution plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
counts = df['class'].value_counts()
ax.bar(counts.index, counts.values,
       color=plt.cm.tab10(np.linspace(0, 1, len(counts))))
ax.set_xlabel('Molecular Class', fontsize=12)
ax.set_ylabel('Number of Spectra', fontsize=12)
ax.set_title('Class Distribution in ramanbiolib Spectra DB', fontsize=14)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/class_distribution.png', dpi=150)
plt.close()
print('Saved: outputs/figures/class_distribution.png')

# ── Filter to top-6 classes ───────────────────────────────────────────────────
KEEP_CLASSES = ['Proteins', 'Lipids', 'Saccharides',
                'AminoAcids', 'PrimaryMetabolites', 'NucleicAcids']

if _args.use_merged:
    df_filt = df.reset_index(drop=True)
else:
    df_filt = df[df['class'].isin(KEEP_CLASSES)].reset_index(drop=True)
print('Filtered dataset:', df_filt.shape)
print(df_filt['class'].value_counts().to_string(), '\n')

X_raw       = np.array(df_filt['intensity'].tolist(),   dtype=np.float32)
wavenumbers = np.array(df_filt['wavenumbers'].iloc[0],  dtype=np.float32)

le          = LabelEncoder()
y_raw       = le.fit_transform(df_filt['class'])
CLASS_NAMES = list(le.classes_)
N_CLASSES   = len(CLASS_NAMES)
SEQ_LEN     = X_raw.shape[1]

print(f'Spectrum length  : {SEQ_LEN} points')
print(f'Wavenumber range : {wavenumbers[0]:.0f}-{wavenumbers[-1]:.0f} cm^-1')
print(f'Classes ({N_CLASSES})     : {CLASS_NAMES}\n')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data augmentation
# ─────────────────────────────────────────────────────────────────────────────

AUG_FACTOR = int(max(0, _args.single_aug_factor))
SHIFT_MAX = int(max(0, _args.single_shift_max))
NOISE_STD = float(max(0.0, _args.single_aug_noise_std))
SCALE_MIN = float(_args.single_aug_scale_min)
SCALE_MAX = float(_args.single_aug_scale_max)
if SCALE_MIN > SCALE_MAX:
    SCALE_MIN, SCALE_MAX = SCALE_MAX, SCALE_MIN
STRETCH_MAX = float(max(0.0, _args.single_aug_stretch_max))
SAVGOL_WINDOW = int(max(0, _args.single_savgol_window))
SAVGOL_POLY = int(max(0, _args.single_savgol_poly))
DERIVATIVE_CHANNELS = int(_args.single_derivative_channels)
USE_LOG_SCALE = bool(int(_args.single_log_scale))
REAL_ONLY = bool(int(_args.single_real_only))
SINGLE_SYNTHETIC_SAMPLES_REQUESTED = int(max(0, _args.single_synthetic_samples))
SINGLE_SYNTHETIC_SAMPLES = int(SINGLE_SYNTHETIC_SAMPLES_REQUESTED)
SINGLE_SYNTHETIC_NOISE_STD = float(max(0.0, _args.single_synthetic_noise_std))
SINGLE_SYNTHETIC_MAX_COMPONENTS = int(max(1, _args.single_synthetic_max_components))
SINGLE_SYNTHETIC_PEAK_PERTURB_STD = float(max(0.0, _args.single_synthetic_peak_perturb_std))
if REAL_ONLY and SINGLE_SYNTHETIC_SAMPLES > 0:
    print('single-real-only=1 -> ignoring single-synthetic-samples and training on real data only.')
    SINGLE_SYNTHETIC_SAMPLES = 0
SINGLE_RUN_TAG = f'single_syn_{SINGLE_SYNTHETIC_SAMPLES}'
SINGLE_RUN_DIR = os.path.join('outputs', 'runs', SINGLE_RUN_TAG)
SINGLE_RUN_FIG_DIR = os.path.join(SINGLE_RUN_DIR, 'figures')
SINGLE_RUN_LOG_DIR = os.path.join(SINGLE_RUN_DIR, 'logs')
SINGLE_RUN_MODEL_DIR = os.path.join(SINGLE_RUN_DIR, 'model')
for d in (SINGLE_RUN_FIG_DIR, SINGLE_RUN_LOG_DIR, SINGLE_RUN_MODEL_DIR):
    os.makedirs(d, exist_ok=True)
SINGLE_RUN_TRAIN_LOG_PATH = os.path.join(SINGLE_RUN_LOG_DIR, 'training_log.csv')
SINGLE_RUN_CONFUSION_CSV_PATH = os.path.join(SINGLE_RUN_LOG_DIR, 'confusion_matrix_single.csv')
SINGLE_RUN_CONFUSION_FIG_PATH = os.path.join(SINGLE_RUN_FIG_DIR, 'confusion_matrix.png')
SINGLE_RUN_CURVES_FIG_PATH = os.path.join(SINGLE_RUN_FIG_DIR, 'training_curves.png')
SINGLE_RUN_MODEL_CONFIG_PATH = os.path.join(SINGLE_RUN_MODEL_DIR, 'model_config.json')
SINGLE_RUN_BEST_MODEL_PATH = os.path.join(SINGLE_RUN_MODEL_DIR, 'best_model.pt')
SINGLE_RUN_FINAL_MODEL_PATH = os.path.join(SINGLE_RUN_MODEL_DIR, 'final_model.pt')
SINGLE_RUN_SUMMARY_PATH = os.path.join(SINGLE_RUN_LOG_DIR, 'run_summary.json')
SINGLE_RUN_COMPARISON_CSV = 'outputs/logs/single_synthetic_runs_summary.csv'
print(f'Run tag: {SINGLE_RUN_TAG}')
print(f'Per-run outputs: {SINGLE_RUN_DIR}')

def _apply_edge_shift(spec, shift):
    if shift == 0:
        return spec
    shifted = np.roll(spec, shift).astype(np.float32, copy=False)
    if shift > 0:
        shifted[:shift] = shifted[shift]
    else:
        shifted[shift:] = shifted[shift - 1]
    return shifted


def _apply_stretch(spec, stretch_factor):
    if abs(stretch_factor) < 1e-8:
        return spec
    n = spec.shape[0]
    x = np.arange(n, dtype=np.float32)
    center = 0.5 * (n - 1)
    denom = max(1.0 + float(stretch_factor), 1e-3)
    source_x = center + (x - center) / denom
    source_x = np.clip(source_x, 0.0, n - 1.0)
    return np.interp(x, source_x, spec, left=float(spec[0]), right=float(spec[-1])).astype(np.float32)


def augment_spectra(
    X,
    y,
    factor=AUG_FACTOR,
    noise_std=NOISE_STD,
    scale_min=SCALE_MIN,
    scale_max=SCALE_MAX,
    stretch_max=STRETCH_MAX,
    shift_max=SHIFT_MAX,
    seed=SEED
):
    """Raman augmentation: shift + Gaussian noise + amplitude scale + wavelength stretching."""
    X_aug, y_aug = [X.copy()], [y.copy()]
    rng = np.random.default_rng(seed)
    for _ in range(factor):
        X_new = np.empty_like(X, dtype=np.float32)
        scales = rng.uniform(scale_min, scale_max, size=X.shape[0]).astype(np.float32)
        stretches = rng.uniform(-stretch_max, stretch_max, size=X.shape[0]).astype(np.float32)
        shifts = rng.integers(-shift_max, shift_max + 1, size=X.shape[0]) if shift_max > 0 else np.zeros(X.shape[0], dtype=np.int32)
        noise = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        for i in range(X.shape[0]):
            spec = _apply_stretch(X[i], stretches[i])
            spec = spec * scales[i]
            spec = spec + noise[i]
            if shift_max > 0:
                spec = _apply_edge_shift(spec, int(shifts[i]))
            X_new[i] = np.clip(spec, 0.0, None)
        X_aug.append(X_new)
        y_aug.append(y.copy())
    return np.vstack(X_aug), np.concatenate(y_aug)


def synthesize_single_class_spectra(
    X,
    y,
    n_samples,
    noise_std=SINGLE_SYNTHETIC_NOISE_STD,
    max_components=SINGLE_SYNTHETIC_MAX_COMPONENTS,
    peak_perturb_std=SINGLE_SYNTHETIC_PEAK_PERTURB_STD,
    scale_min=SCALE_MIN,
    scale_max=SCALE_MAX,
    shift_max=SHIFT_MAX,
    seed=SEED
):
    """
    Class-conditional synthetic spectra for single-label training:
    weighted in-class peak mixing + peak perturbation + additive noise.
    """
    n_samples = int(max(0, n_samples))
    if n_samples == 0:
        seq_len = X.shape[1]
        return np.empty((0, seq_len), dtype=np.float32), np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    seq_len = int(X.shape[1])
    x_axis = np.arange(seq_len, dtype=np.float32)

    class_counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    class_probs = class_counts / np.maximum(class_counts.sum(), 1.0)
    class_to_indices = {
        cls_idx: np.where(y == cls_idx)[0]
        for cls_idx in range(N_CLASSES)
    }

    X_syn = np.empty((n_samples, seq_len), dtype=np.float32)
    y_syn = np.empty((n_samples,), dtype=np.int64)

    for i in range(n_samples):
        cls_idx = int(rng.choice(np.arange(N_CLASSES), p=class_probs))
        pool = class_to_indices[cls_idx]
        if pool.size == 0:
            cls_idx = int(np.argmax(class_counts))
            pool = class_to_indices[cls_idx]

        max_k = int(max(1, min(max_components, pool.size)))
        if max_k >= 2:
            k = int(rng.integers(2, max_k + 1))
        else:
            k = 1
        chosen = rng.choice(pool, size=k, replace=bool(pool.size < k))
        weights = rng.dirichlet(np.ones(k)).astype(np.float32)
        spec = np.zeros(seq_len, dtype=np.float32)
        for idx_local, w in zip(chosen, weights):
            spec += w * X[idx_local]

        peaks, _ = find_peaks(spec, prominence=max(float(spec.max()) * 0.03, 1e-6), distance=15)
        n_perturb = int(rng.integers(1, 4))
        for _ in range(n_perturb):
            if len(peaks) > 0:
                center = int(rng.choice(peaks))
            else:
                center = int(rng.integers(0, seq_len))
            width = float(rng.uniform(4.0, 18.0))
            rel_amp = float(rng.normal(0.0, peak_perturb_std))
            amp = rel_amp * float(spec[center] + 1e-6)
            bump = np.exp(-0.5 * ((x_axis - center) / width) ** 2).astype(np.float32)
            spec += amp * bump

        scale = float(rng.uniform(scale_min, scale_max))
        spec *= scale
        if shift_max > 0:
            shift = int(rng.integers(-shift_max, shift_max + 1))
            spec = _apply_edge_shift(spec, shift)
        spec += rng.normal(0.0, noise_std, size=seq_len).astype(np.float32)
        X_syn[i] = np.clip(spec, 0.0, None)
        y_syn[i] = cls_idx

    return X_syn, y_syn


def _normalise_per_spectrum(X):
    x_min = X.min(axis=1, keepdims=True)
    x_max = X.max(axis=1, keepdims=True)
    return (X - x_min) / np.maximum(x_max - x_min, 1e-8)


def _resolve_savgol_window(seq_len):
    if SAVGOL_WINDOW < 3:
        return 0
    window = SAVGOL_WINDOW if SAVGOL_WINDOW % 2 == 1 else (SAVGOL_WINDOW + 1)
    max_win = seq_len if seq_len % 2 == 1 else (seq_len - 1)
    window = min(window, max_win)
    if window < 3:
        return 0
    return window


def preprocess_spectra(X):
    Xp = np.asarray(X, dtype=np.float32).copy()
    if USE_LOG_SCALE:
        # Stabilize peak magnitude variation while preserving non-negativity.
        Xp = np.log1p(np.clip(Xp, 0.0, None)).astype(np.float32)
    window = _resolve_savgol_window(Xp.shape[1])
    if window > 0:
        poly = int(min(max(0, SAVGOL_POLY), window - 1))
        Xp = savgol_filter(Xp, window_length=window, polyorder=poly, axis=1).astype(np.float32)
    base = _normalise_per_spectrum(Xp).astype(np.float32)
    channels = [base]
    if DERIVATIVE_CHANNELS >= 1:
        d1 = np.gradient(base, axis=1).astype(np.float32)
        channels.append(_normalise_per_spectrum(d1).astype(np.float32))
    if DERIVATIVE_CHANNELS >= 2:
        d2 = np.gradient(np.gradient(base, axis=1), axis=1).astype(np.float32)
        channels.append(_normalise_per_spectrum(d2).astype(np.float32))
    if len(channels) == 1:
        return channels[0]
    return np.stack(channels, axis=1).astype(np.float32)


# split into training and test; respect holdout if provided
if _args.use_merged and 'holdout_X' in locals() and holdout_X is not None:
    X_train_raw, y_train_raw = X_raw, y_raw
    X_test, y_test = holdout_X, holdout_y
else:
    if _args.use_merged:
        X_train_raw, X_test, y_train_raw, y_test = train_test_split(
            X_raw, y_raw, test_size=0.25, stratify=None, random_state=SEED
        )
    else:
        X_train_raw, X_test, y_train_raw, y_test = train_test_split(
            X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=SEED
        )
X_test_raw = X_test.copy()
print(f'Raw split -> Train: {len(X_train_raw)}  Test (untouched): {len(X_test)}')

val_size = float(np.clip(_args.single_val_size, 0.05, 0.40))
if _args.use_merged:
    X_train_core_raw, X_val, y_train_core_raw, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=val_size, stratify=None, random_state=SEED
    )
else:
    X_train_core_raw, X_val, y_train_core_raw, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=val_size, stratify=y_train_raw, random_state=SEED
    )
print(f'Train/Val split inside 75% -> Train: {len(X_train_core_raw)}  Val: {len(X_val)}')

# Augment only train split to avoid train/test leakage.
X_train_real_aug, y_train_real_aug = augment_spectra(X_train_core_raw, y_train_core_raw, seed=SEED)
n_synth_added = 0
if (not REAL_ONLY) and SINGLE_SYNTHETIC_SAMPLES > 0:
    X_synth, y_synth = synthesize_single_class_spectra(
        X_train_core_raw,
        y_train_core_raw,
        n_samples=SINGLE_SYNTHETIC_SAMPLES,
        seed=SEED
    )
    X_train = np.vstack([X_train_real_aug, X_synth]).astype(np.float32)
    y_train = np.concatenate([y_train_real_aug, y_synth])
    n_synth_added = int(len(X_synth))
else:
    X_train = X_train_real_aug
    y_train = y_train_real_aug

print(
    f'After train-only augmentation: real_aug={len(X_train_real_aug)}  '
    f'synthetic={n_synth_added}  total={len(X_train)}\n'
)

# Feature-level preprocessing: optional Savitzky-Golay smoothing,
# per-spectrum normalisation, and optional derivative channels.
X_train = preprocess_spectra(X_train)
X_val = preprocess_spectra(X_val)
X_test = preprocess_spectra(X_test_raw)
X_raw_proc = preprocess_spectra(X_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CHANNELS = int(X_train.shape[1]) if X_train.ndim == 3 else 1
print(
    f'Train (augmented): {len(X_train)}  '
    f'Val (raw): {len(X_val)}  Test (raw holdout): {len(X_test)} | '
    f'input_channels={INPUT_CHANNELS}\n'
)


class RamanDataset(Dataset):
    def __init__(self, X, y, train=False):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        elif X_tensor.ndim != 3:
            raise ValueError(f'Expected 2D or 3D spectra tensor, got shape {tuple(X_tensor.shape)}')
        self.X = X_tensor
        self.y = torch.tensor(y, dtype=torch.long)
        self.train = bool(train)
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx):
        spectrum = self.X[idx]
        if self.train:
            spectrum = augment_spectrum(spectrum)
        return spectrum, self.y[idx]


train_ds = RamanDataset(X_train, y_train, train=True)
val_ds   = RamanDataset(X_val, y_val, train=False)
test_ds  = RamanDataset(X_test, y_test, train=False)

class_counts = np.bincount(y_train, minlength=N_CLASSES)
weights      = 1.0 / class_counts[y_train]
sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

BATCH        = int(max(1, _args.single_batch_size))
USE_BALANCED_SAMPLER = bool(int(_args.single_balanced_sampler))
USE_CLASS_WEIGHTS = bool(int(_args.single_class_weights))
SKIP_INTERPRETABILITY = bool(_args.single_skip_interpretability)
if USE_BALANCED_SAMPLER:
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

CONV_DROPOUT = float(np.clip(_args.single_conv_dropout, 0.0, 0.9))
DENSE_DROPOUT = float(np.clip(_args.single_dense_dropout, 0.0, 0.9))

# ─────────────────────────────────────────────────────────────────────────────
# 4. 1D CNN
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleConvBlock(nn.Module):
    """Parallel Conv1D branches at multiple kernel sizes, then channel fusion."""
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
    """Multi-scale 1D CNN with global average pooling. Input: (batch, C, L)."""
    def __init__(self, input_len=1351, n_classes=6, in_channels=1):
        super().__init__()
        multiscale_kernels = (9, 21, 41)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.block1 = MultiScaleConvBlock(32, 64, kernels=multiscale_kernels, pool=2, dropout=CONV_DROPOUT)
        self.block2 = MultiScaleConvBlock(64, 96, kernels=multiscale_kernels, pool=2, dropout=CONV_DROPOUT)
        self.block3 = MultiScaleConvBlock(96, 128, kernels=multiscale_kernels, pool=2, dropout=CONV_DROPOUT)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DENSE_DROPOUT),
            nn.Linear(128, n_classes)
        )

    def _fwd(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).squeeze(-1)
        return x

    def forward(self, x):
        return self.classifier(self._fwd(x))


model = RamanCNN1D(
    input_len=SEQ_LEN,
    n_classes=N_CLASSES,
    in_channels=INPUT_CHANNELS
).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model parameters: {n_params:,}\n')

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = int(max(1, _args.single_epochs))
LR = float(_args.single_lr)
WD = float(_args.single_weight_decay)
if str(_args.single_loss).lower() != 'focal':
    print("Warning: --single-loss=ce is deprecated; forcing focal loss.")
SINGLE_LOSS = 'focal'
FOCAL_GAMMA = float(max(0.0, _args.single_focal_gamma))
PATIENCE = int(max(1, _args.single_early_stop_patience))
OVERFIT_GAP_THRESHOLD = 0.12
OVERFIT_GAP_PATIENCE = max(3, PATIENCE // 3)
K_FOLDS = int(max(1, _args.single_kfolds))
# When using merged dataset, cap K_FOLDS to avoid "n_splits > members per class" error
if _args.use_merged:
    min_class_count = np.bincount(y_raw).min() if len(np.unique(y_raw)) > 0 else 1
    K_FOLDS = min(K_FOLDS, max(1, min_class_count - 1))
    if K_FOLDS == 1:
        K_FOLDS = 1  # Disable CV for merged data with small classes
RESUME_SINGLE = bool(_args.single_continue_from_best)
if RESUME_SINGLE and int(_args.single_extra_epochs) > 0:
    EPOCHS = int(_args.single_extra_epochs)


def make_balanced_class_weights(y_labels):
    counts = np.bincount(y_labels, minlength=N_CLASSES).astype(np.float32)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    weights = weights / np.maximum(weights.mean(), 1e-8)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def build_criterion(class_weights_tensor):
    weight_arg = class_weights_tensor if USE_CLASS_WEIGHTS else None
    return FocalLoss(alpha=weight_arg, gamma=FOCAL_GAMMA)

cv_rows = []
if K_FOLDS > 1 and (not _args.use_merged):
    print(f'\nRunning {K_FOLDS}-fold CV on the 75% train split...')
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y_train_raw), start=1):
        X_fold_train_raw = X_train_raw[tr_idx]
        y_fold_train_raw = y_train_raw[tr_idx]
        X_fold_val = X_train_raw[va_idx]
        y_fold_val = y_train_raw[va_idx]
        X_fold_train_real_aug, y_fold_train_real_aug = augment_spectra(X_fold_train_raw, y_fold_train_raw)
        if (not REAL_ONLY) and SINGLE_SYNTHETIC_SAMPLES > 0:
            fold_synth_n = int(round(
                SINGLE_SYNTHETIC_SAMPLES * (len(X_fold_train_raw) / max(len(X_train_core_raw), 1))
            ))
            X_fold_synth, y_fold_synth = synthesize_single_class_spectra(
                X_fold_train_raw,
                y_fold_train_raw,
                n_samples=fold_synth_n,
                seed=SEED + fold_idx
            )
            X_fold_train = np.vstack([X_fold_train_real_aug, X_fold_synth]).astype(np.float32)
            y_fold_train = np.concatenate([y_fold_train_real_aug, y_fold_synth])
        else:
            X_fold_train = X_fold_train_real_aug
            y_fold_train = y_fold_train_real_aug

        X_fold_train = preprocess_spectra(X_fold_train)
        X_fold_val = preprocess_spectra(X_fold_val)

        fold_train_ds = RamanDataset(X_fold_train, y_fold_train, train=True)
        fold_val_ds = RamanDataset(X_fold_val, y_fold_val, train=False)
        fold_counts = np.bincount(y_fold_train, minlength=N_CLASSES)
        fold_weights = 1.0 / np.maximum(fold_counts[y_fold_train], 1.0)
        fold_sampler = WeightedRandomSampler(fold_weights, len(fold_weights), replacement=True)
        if USE_BALANCED_SAMPLER:
            fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH, sampler=fold_sampler)
        else:
            fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH, shuffle=True)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=BATCH, shuffle=False)

        fold_input_channels = int(X_fold_train.shape[1]) if X_fold_train.ndim == 3 else 1
        fold_model = RamanCNN1D(
            input_len=SEQ_LEN,
            n_classes=N_CLASSES,
            in_channels=fold_input_channels
        ).to(DEVICE)
        fold_class_weights = make_balanced_class_weights(y_fold_train)
        fold_criterion = build_criterion(fold_class_weights)
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=LR, weight_decay=WD)
        fold_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fold_optimizer, T_max=EPOCHS)

        fold_best_f1 = 0.0
        fold_best_acc = 0.0
        fold_best_loss = float('inf')
        fold_best_epoch = 0
        fold_no_improve = 0
        fold_epochs_ran = 0

        for epoch in range(1, EPOCHS + 1):
            fold_model.train()
            fold_tr_preds = []
            fold_tr_labels = []
            for xb, yb in fold_train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                fold_optimizer.zero_grad()
                logits = fold_model(xb)
                loss = fold_criterion(logits, yb)
                loss.backward()
                fold_optimizer.step()
                fold_tr_preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
                fold_tr_labels.extend(yb.detach().cpu().numpy().tolist())
            fold_scheduler.step()

            fold_model.eval()
            v_loss = v_correct = v_total = 0
            fold_va_preds = []
            fold_va_labels = []
            with torch.no_grad():
                for xb, yb in fold_val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = fold_model(xb)
                    loss = fold_criterion(logits, yb)
                    v_loss += loss.item() * len(yb)
                    v_correct += (logits.argmax(1) == yb).sum().item()
                    v_total += len(yb)
                    fold_va_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                    fold_va_labels.extend(yb.cpu().numpy().tolist())

            fold_epochs_ran = epoch
            v_loss_epoch = v_loss / max(v_total, 1)
            v_acc = v_correct / max(v_total, 1)
            v_macro_f1 = f1_score(fold_va_labels, fold_va_preds, average='macro', zero_division=0)
            improved = (v_macro_f1 > fold_best_f1 + 1e-6)
            if improved:
                fold_best_f1 = v_macro_f1
                fold_best_loss = v_loss_epoch
                fold_best_acc = v_acc
                fold_best_epoch = epoch
                fold_no_improve = 0
            else:
                fold_no_improve += 1
            if fold_no_improve >= PATIENCE:
                break

        cv_rows.append({
            'fold': fold_idx,
            'best_val_macro_f1': float(fold_best_f1),
            'best_val_loss': float(fold_best_loss),
            'best_val_acc': float(fold_best_acc),
            'best_epoch': int(fold_best_epoch),
            'epochs_ran': int(fold_epochs_ran),
            'train_samples_raw': int(len(X_fold_train_raw)),
            'val_samples_raw': int(len(X_fold_val))
        })
        print(
            f"CV fold {fold_idx}/{K_FOLDS}: "
            f"best_val_macro_f1={fold_best_f1:.4f} best_val_loss={fold_best_loss:.4f} "
            f"(best_epoch={fold_best_epoch}, epochs_ran={fold_epochs_ran})"
        )

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv('outputs/logs/single_kfold_cv.csv', index=False)
    print('Saved: outputs/logs/single_kfold_cv.csv')
    print(
        f"CV val macro-F1 mean+/-std: {cv_df['best_val_macro_f1'].mean():.4f}+/-{cv_df['best_val_macro_f1'].std(ddof=0):.4f} | "
        f"val loss mean+/-std: {cv_df['best_val_loss'].mean():.4f}+/-{cv_df['best_val_loss'].std(ddof=0):.4f}"
    )

# Inverse-frequency class weights to improve minority-class recall.
class_weights = make_balanced_class_weights(y_train)
criterion = build_criterion(class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {'train_loss': [], 'train_acc': [], 'train_macro_f1': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}
best_val_f1 = 0.0
best_val_acc = 0.0
best_val_loss = float('inf')
best_ckpt_path = 'outputs/model/best_model.pt'
epochs_no_improve = 0
overfit_no_improve = 0

if RESUME_SINGLE:
    resume_path = _args.single_resume_path
    if os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        print(f'Resuming single-model training from: {resume_path}')
        model.eval()
        v_loss = v_correct = v_total = 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += len(yb)
                v_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                v_labels.extend(yb.cpu().numpy().tolist())
        best_val_loss = (v_loss / v_total) if v_total else float('inf')
        best_val_acc = (v_correct / v_total) if v_total else 0.0
        best_val_f1 = f1_score(v_labels, v_preds, average='macro', zero_division=0) if v_labels else 0.0
        print(f'Initial val loss/macro_f1 from checkpoint: {best_val_loss:.4f}/{best_val_f1:.4f}')
    else:
        print(f'Warning: resume checkpoint not found, starting fresh: {resume_path}')

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = t_correct = t_total = 0
    train_preds, train_labels = [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * len(yb)
        t_correct += (logits.argmax(1) == yb).sum().item()
        t_total   += len(yb)
        train_preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        train_labels.extend(yb.detach().cpu().numpy().tolist())
    scheduler.step()

    model.eval()
    v_loss = v_correct = v_total = 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            v_loss    += loss.item() * len(yb)
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total   += len(yb)
            val_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            val_labels.extend(yb.cpu().numpy().tolist())

    t_acc = t_correct / t_total
    v_acc = v_correct / v_total
    t_macro_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    v_macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    v_loss_epoch = v_loss / v_total
    history['train_loss'].append(t_loss / t_total)
    history['train_acc'].append(t_acc)
    history['train_macro_f1'].append(t_macro_f1)
    history['val_loss'].append(v_loss_epoch)
    history['val_acc'].append(v_acc)
    history['val_macro_f1'].append(v_macro_f1)

    improved = (v_macro_f1 > best_val_f1 + 1e-6)
    if improved:
        best_val_f1 = v_macro_f1
        best_val_loss = v_loss_epoch
        best_val_acc = v_acc
        torch.save(model.state_dict(), best_ckpt_path)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Overfitting guard: if train macro-F1 stays much higher than val macro-F1,
    # stop before memorization dominates.
    f1_gap = t_macro_f1 - v_macro_f1
    if epoch >= 10 and (not improved) and f1_gap > OVERFIT_GAP_THRESHOLD:
        overfit_no_improve += 1
    else:
        overfit_no_improve = 0

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{EPOCHS}  '
              f'train_loss={t_loss/t_total:.4f} macro_f1={t_macro_f1:.3f}  '
              f'val_loss={v_loss_epoch:.4f} macro_f1={v_macro_f1:.3f}')

    if epochs_no_improve >= PATIENCE:
        print(
            f'Early stopping at epoch {epoch} '
            f'(no val_macro_f1 improvement for {PATIENCE} epochs).'
        )
        break
    if overfit_no_improve >= OVERFIT_GAP_PATIENCE:
        print(
            f'Early stopping at epoch {epoch} '
            f'(overfitting guard: train-val macro-F1 gap > {OVERFIT_GAP_THRESHOLD:.2f} '
            f'for {OVERFIT_GAP_PATIENCE} epochs).'
        )
        break

print(f'\nBest val macro-F1/loss: {best_val_f1:.4f}/{best_val_loss:.4f}')
history_df = pd.DataFrame(history)
history_df.to_csv('outputs/logs/training_log.csv', index=False)
history_df.to_csv(SINGLE_RUN_TRAIN_LOG_PATH, index=False)
print('Saved: outputs/logs/training_log.csv')
print(f'Saved: {SINGLE_RUN_TRAIN_LOG_PATH}')

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], lw=2, label='Train')
axes[0].plot(history['val_loss'],   lw=2, label='Val')
axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
axes[0].legend()
axes[1].plot(history['train_macro_f1'], lw=2, label='Train')
axes[1].plot(history['val_macro_f1'],   lw=2, label='Val')
axes[1].set(xlabel='Epoch', ylabel='Macro-F1', title='Macro-F1')
axes[1].legend()
plt.tight_layout()
plt.savefig('outputs/figures/training_curves.png', dpi=150)
plt.savefig(SINGLE_RUN_CURVES_FIG_PATH, dpi=150)
plt.close()
print('Saved: outputs/figures/training_curves.png')
print(f'Saved: {SINGLE_RUN_CURVES_FIG_PATH}')

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

model.load_state_dict(torch.load('outputs/model/best_model.pt', map_location=DEVICE))
model.eval()
torch.save(model.state_dict(), SINGLE_RUN_BEST_MODEL_PATH)
print(f'Saved: {SINGLE_RUN_BEST_MODEL_PATH}')

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(DEVICE))
        probs = F.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
test_acc   = accuracy_score(all_labels, all_preds)
test_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f'\nTest macro-F1: {test_macro_f1:.4f} | Test accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)\n')
cls_report_text = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
print(cls_report_text)
cls_report_dict = classification_report(
    all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
)
class_rows = []
for cls_name in CLASS_NAMES:
    row = cls_report_dict.get(cls_name, {})
    class_rows.append({
        'class': cls_name,
        'precision': float(row.get('precision', 0.0)),
        'recall': float(row.get('recall', 0.0)),
        'f1_score': float(row.get('f1-score', 0.0)),
        'support': int(row.get('support', 0))
    })
pd.DataFrame(class_rows).to_csv('outputs/logs/classification_report_single.csv', index=False)
print('Saved: outputs/logs/classification_report_single.csv')

baseline_df = None
if not _args.single_skip_baselines:
    print('\nTraining baseline models on raw 75/25 split...')
    X_train_flat = X_train_raw.reshape(len(X_train_raw), -1)
    X_test_flat = X_test_raw.reshape(len(X_test_raw), -1)
    baseline_models = {
        'logreg_l2': make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight='balanced',
                solver='lbfgs'
            )
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            class_weight='balanced_subsample',
            n_jobs=-1
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            random_state=SEED
        )
    }
    baseline_rows = []
    for name, clf in baseline_models.items():
        clf.fit(X_train_flat, y_train_raw)
        pred = clf.predict(X_test_flat)
        baseline_rows.append({
            'model': name,
            'test_acc': float(accuracy_score(y_test, pred)),
            'macro_f1': float(f1_score(y_test, pred, average='macro', zero_division=0))
        })
    baseline_df = pd.DataFrame(baseline_rows).sort_values('test_acc', ascending=False)
    baseline_df.to_csv('outputs/logs/baseline_models.csv', index=False)
    print('Saved: outputs/logs/baseline_models.csv')
    print(baseline_df.to_string(index=False))
    cnn_row = pd.DataFrame([{
        'model': 'cnn_multiscale_real_only',
        'test_acc': float(test_acc),
        'macro_f1': float(test_macro_f1)
    }])
    compare_df = pd.concat([cnn_row, baseline_df], ignore_index=True)
    compare_df = compare_df.sort_values('macro_f1', ascending=False)
    compare_df.to_csv('outputs/logs/model_vs_baseline_models.csv', index=False)
    compare_df.to_csv(os.path.join(SINGLE_RUN_LOG_DIR, 'model_vs_baseline_models.csv'), index=False)
    print('Saved: outputs/logs/model_vs_baseline_models.csv')
    print(compare_df.to_string(index=False))

# ── Confusion matrix & ROC Curves ────────────────────────────────────────────
cm_vals = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(
    cm_vals,
    index=[f'true_{c}' for c in CLASS_NAMES],
    columns=[f'pred_{c}' for c in CLASS_NAMES]
)
cm_df.to_csv('outputs/logs/confusion_matrix_single.csv')
cm_df.to_csv(SINGLE_RUN_CONFUSION_CSV_PATH)
print('Saved: outputs/logs/confusion_matrix_single.csv')
print(f'Saved: {SINGLE_RUN_CONFUSION_CSV_PATH}')

# Create figure with subplots: confusion matrix and ROC curves
fig = plt.figure(figsize=(16, 7))

# Left subplot: Confusion matrix
ax_cm = plt.subplot(1, 2, 1)
ConfusionMatrixDisplay(cm_vals, display_labels=CLASS_NAMES).plot(
    ax=ax_cm, colorbar=False, cmap='Blues')
ax_cm.set_title('Confusion Matrix – Test Set', fontsize=13, fontweight='bold')
plt.setp(ax_cm.get_xticklabels(), rotation=30, ha='right')

# Right subplot: ROC Curves (One-vs-Rest)
ax_roc = plt.subplot(1, 2, 2)

# Binarize labels for multi-class ROC
y_bin = label_binarize(all_labels, classes=np.arange(N_CLASSES))

# Compute ROC curve and ROC area for each class
fpr_dict = dict()
tpr_dict = dict()
roc_auc_dict = dict()
colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

for i, class_name in enumerate(CLASS_NAMES):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    ax_roc.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=2,
                label=f'{class_name} (AUC = {roc_auc_dict[i]:.3f})')

# Plot diagonal line (random classifier)
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax_roc.set_title('ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=10)
ax_roc.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.savefig(SINGLE_RUN_CONFUSION_FIG_PATH, dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/figures/confusion_matrix.png')
print(f'Saved: {SINGLE_RUN_CONFUSION_FIG_PATH}')

# ─────────────────────────────────────────────────────────────────────────────
# 7. Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────

def integrated_gradients(model, x, target_class, n_steps=50):
    """
    Compute Integrated Gradients attribution for a single spectrum.
    x : Tensor (1, C, L) on DEVICE
    Returns np.ndarray (C, L) or (L,)
    """
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(n_steps, 1, 1, 1)
    diff = (x - baseline).unsqueeze(0)
    interpolated = (baseline.unsqueeze(0) + alphas * diff).squeeze(1)  # (n_steps, C, L)
    interpolated.requires_grad_(True)
    logits = model(interpolated)
    logits[:, target_class].sum().backward()
    avg_grads = interpolated.grad.mean(dim=0)   # (C, L)
    ig = ((x.squeeze(0) - baseline.squeeze(0)) * avg_grads).detach().cpu().numpy()
    return ig


def class_mean_saliency(model, X_cls, label, n_samples=15, n_steps=50):
    model.eval()
    igs = []
    idx = np.random.choice(len(X_cls), min(n_samples, len(X_cls)), replace=False)
    for i in idx:
        x_np = X_cls[i]
        if x_np.ndim == 1:
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        else:
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        ig = integrated_gradients(model, x, label, n_steps=n_steps)
        ig_abs = np.abs(ig)
        if ig_abs.ndim == 2:
            ig_abs = ig_abs.mean(axis=0)
        igs.append(ig_abs)
    return np.mean(igs, axis=0)


print('\nComputing Integrated Gradient saliency maps...')
saliency_maps  = {}
class_spectra  = {}

for cls_name in CLASS_NAMES:
    cls_idx = le.transform([cls_name])[0]
    mask    = y_raw == cls_idx
    X_cls   = X_raw_proc[mask]
    print(f'  {cls_name} ({len(X_cls)} spectra) ...', end=' ', flush=True)
    saliency_maps[cls_name] = class_mean_saliency(model, X_cls, cls_idx)
    class_spectra[cls_name] = X_raw[mask].mean(axis=0)
    print('done')

# ─────────────────────────────────────────────────────────────────────────────
# 8. Saliency overlay plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

for i, cls_name in enumerate(CLASS_NAMES):
    sal   = saliency_maps[cls_name]
    spec  = class_spectra[cls_name]
    sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(wavenumbers, spec, color=COLORS[i], lw=1.5, label='Mean spectrum')
    ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity (normalised)', color=COLORS[i], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COLORS[i])

    ax2 = ax1.twinx()
    ax2.fill_between(wavenumbers, sal_n, alpha=0.35, color='crimson', label='IG saliency')
    ax2.set_ylabel('Normalised |IG| saliency', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    peaks, _ = find_peaks(sal_n, prominence=0.15, distance=20)
    if len(peaks) > 0:
        top_pk = peaks[np.argsort(sal_n[peaks])[::-1][:3]]
        for pk in top_pk:
            ax1.axvline(wavenumbers[pk], color='grey', lw=0.8, ls='--')
            ax1.text(wavenumbers[pk] + 5, spec.max() * 0.9,
                     f'{wavenumbers[pk]:.0f}', fontsize=8, rotation=90)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc='upper right', fontsize=9)
    ax1.set_title(f'Saliency Map – {cls_name}', fontsize=14)
    plt.tight_layout()
    fp = f'outputs/figures/saliency_{cls_name.lower()}.png'
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f'Saved: {fp}')

# ── Aggregate heatmap ─────────────────────────────────────────────────────────
sal_matrix = np.array([
    (saliency_maps[c] - saliency_maps[c].min()) /
    (saliency_maps[c].max() - saliency_maps[c].min() + 1e-9)
    for c in CLASS_NAMES
])
step  = 10
wn_ds = wavenumbers[::step]
sd_ds = sal_matrix[:, ::step]

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(sd_ds, aspect='auto', cmap='hot',
               extent=[wn_ds[0], wn_ds[-1], len(CLASS_NAMES)-0.5, -0.5])
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_yticklabels(CLASS_NAMES, fontsize=11)
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_title('Integrated-Gradient Saliency Heatmap (all classes)', fontsize=13)
plt.colorbar(im, ax=ax, label='Normalised |IG|')
plt.tight_layout()
plt.savefig('outputs/figures/saliency_heatmap_all.png', dpi=150)
plt.close()
print('Saved: outputs/figures/saliency_heatmap_all.png')

# ─────────────────────────────────────────────────────────────────────────────
# 9. Key spectral regions summary
# ─────────────────────────────────────────────────────────────────────────────

WINDOW       = 20
summary_rows = []

for cls_name in CLASS_NAMES:
    sal   = saliency_maps[cls_name]
    sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    peaks, _ = find_peaks(sal_n, prominence=0.10, distance=15)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(sal_n)])
    ranked = peaks[np.argsort(sal_n[peaks])[::-1]]
    for pk in ranked[:5]:
        wn = wavenumbers[pk]
        summary_rows.append({
            'class': cls_name,
            'center_cm': int(wn),
            'range': f'{int(wn-WINDOW)}-{int(wn+WINDOW)} cm^-1',
            'saliency_score': round(float(sal_n[pk]), 4)
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('outputs/logs/key_spectral_regions.csv', index=False)
print('\nKey spectral regions:')
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Save artefacts
# ─────────────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), 'outputs/model/final_model.pt')
torch.save(model.state_dict(), SINGLE_RUN_FINAL_MODEL_PATH)

config = {
    'input_len': SEQ_LEN,
    'n_classes': N_CLASSES,
    'class_names': CLASS_NAMES,
    'wavenumber_range': [int(wavenumbers[0]), int(wavenumbers[-1])],
    'batch_size': BATCH,
    'learning_rate': LR,
    'weight_decay': WD,
    'conv_dropout': CONV_DROPOUT,
    'dense_dropout': DENSE_DROPOUT,
    'loss': SINGLE_LOSS,
    'focal_gamma': FOCAL_GAMMA,
    'use_balanced_sampler': USE_BALANCED_SAMPLER,
    'use_class_weights': USE_CLASS_WEIGHTS,
    'real_only': REAL_ONLY,
    'use_log_scale': USE_LOG_SCALE,
    'derivative_channels': DERIVATIVE_CHANNELS,
    'savgol_window': SAVGOL_WINDOW,
    'savgol_poly': SAVGOL_POLY,
    'input_channels': INPUT_CHANNELS,
    'augmentation_factor': AUG_FACTOR,
    'augmentation_shift_max': SHIFT_MAX,
    'single_synthetic_samples_requested': SINGLE_SYNTHETIC_SAMPLES_REQUESTED,
    'single_synthetic_samples_used': n_synth_added,
    'single_synthetic_noise_std': SINGLE_SYNTHETIC_NOISE_STD,
    'single_synthetic_max_components': SINGLE_SYNTHETIC_MAX_COMPONENTS,
    'single_synthetic_peak_perturb_std': SINGLE_SYNTHETIC_PEAK_PERTURB_STD,
    'multiscale_kernels': [9, 21, 41],
    'cnn_channels': [64, 96, 128],
    'classifier_hidden': 128,
    'epochs_target': EPOCHS,
    'epochs_ran': len(history['train_loss']),
    'skip_interpretability': SKIP_INTERPRETABILITY,
    'k_folds': K_FOLDS,
    'kfold_cv_ran': bool(K_FOLDS > 1),
    'kfold_cv_rows': int(len(cv_rows)),
    'kfold_val_macro_f1_mean': (round(float(np.mean([r['best_val_macro_f1'] for r in cv_rows])), 4) if cv_rows else None),
    'kfold_val_macro_f1_std': (round(float(np.std([r['best_val_macro_f1'] for r in cv_rows])), 4) if cv_rows else None),
    'kfold_val_loss_mean': (round(float(np.mean([r['best_val_loss'] for r in cv_rows])), 4) if cv_rows else None),
    'kfold_val_loss_std': (round(float(np.std([r['best_val_loss'] for r in cv_rows])), 4) if cv_rows else None),
    'resume_from_checkpoint': RESUME_SINGLE,
    'val_size_within_train': val_size,
    'best_val_macro_f1': round(float(best_val_f1), 4),
    'best_val_acc': round(best_val_acc, 4),
    'best_val_loss': round(float(best_val_loss), 4),
    'test_macro_f1': round(float(test_macro_f1), 4),
    'test_acc': round(float(test_acc), 4),
    'baselines_ran': bool(baseline_df is not None),
    'baseline_best_model': (str(baseline_df.iloc[0]['model']) if baseline_df is not None and len(baseline_df) else None),
    'baseline_best_test_acc': (round(float(baseline_df.iloc[0]['test_acc']), 4) if baseline_df is not None and len(baseline_df) else None),
    'baseline_best_macro_f1': (round(float(baseline_df.iloc[0]['macro_f1']), 4) if baseline_df is not None and len(baseline_df) else None)
}
with open('outputs/model/model_config.json', 'w') as fh:
    json.dump(config, fh, indent=2)
with open(SINGLE_RUN_MODEL_CONFIG_PATH, 'w') as fh:
    json.dump(config, fh, indent=2)

np.savez('outputs/logs/saliency_maps.npz',
         wavenumbers=wavenumbers,
         class_names=np.array(CLASS_NAMES),
         **{cls: saliency_maps[cls] for cls in CLASS_NAMES})

run_summary = {
    'timestamp_utc': datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
    'run_tag': SINGLE_RUN_TAG,
    'real_only': bool(REAL_ONLY),
    'use_log_scale': bool(USE_LOG_SCALE),
    'use_balanced_sampler': bool(USE_BALANCED_SAMPLER),
    'use_class_weights': bool(USE_CLASS_WEIGHTS),
    'single_synthetic_samples_requested': int(SINGLE_SYNTHETIC_SAMPLES_REQUESTED),
    'single_synthetic_samples_used': int(n_synth_added),
    'epochs_ran': int(len(history['train_loss'])),
    'best_val_macro_f1': round(float(best_val_f1), 4),
    'best_val_acc': round(float(best_val_acc), 4),
    'best_val_loss': round(float(best_val_loss), 4),
    'test_macro_f1': round(float(test_macro_f1), 4),
    'test_acc': round(float(test_acc), 4)
}
with open(SINGLE_RUN_SUMMARY_PATH, 'w') as fh:
    json.dump(run_summary, fh, indent=2)

summary_row = pd.DataFrame([run_summary])
if os.path.exists(SINGLE_RUN_COMPARISON_CSV):
    summary_existing = pd.read_csv(SINGLE_RUN_COMPARISON_CSV)
    summary_all = pd.concat([summary_existing, summary_row], ignore_index=True)
else:
    summary_all = summary_row
summary_all.to_csv(SINGLE_RUN_COMPARISON_CSV, index=False)

print('\n=== All artefacts saved ===')
print(f'Saved: {SINGLE_RUN_MODEL_CONFIG_PATH}')
print(f'Saved: {SINGLE_RUN_FINAL_MODEL_PATH}')
print(f'Saved: {SINGLE_RUN_SUMMARY_PATH}')
print(f'Saved: {SINGLE_RUN_COMPARISON_CSV}')
print(f'FINAL TEST MACRO-F1: {test_macro_f1:.4f}')
print(f'FINAL TEST ACCURACY: {test_acc*100:.2f}%')

