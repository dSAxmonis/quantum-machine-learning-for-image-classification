"""
exp2_recycled_qubit_attention_qcnn.py
======================================
TASK 2: Novel Architecture — Recycled-Qubit Attention & Correlation-Regularized QCNN (RQA-QCNN).
- Implements Zero-Cost Quantum Attention (QAttn) using 4-body Pauli-Z parity <Z1 Z2 Z3 Z4>
  measured on the pooled/dropped qubits.
- Implements Seong & Kim Feature Correlation Regularization: L_corr = (rho_mean - rho_target)^2
- Evaluates 4-way component ablation + rho_target sweep {0.0, 0.25, 0.50, 0.75}
- Uses Multi-Patch Pretrained ResNet-18 Spatial Grid (16 spatial partitions) with fast vectorized PyTorch simulation
- Generates Fig 3 (300 DPI Attention Heatmap Overlays)
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score

from qml_suite_core import (
    get_cifar10_5k_split, reduce_backbone_b,
    ENCODINGS, HybridQCNNModule,
    CACHE_DIR, RESULTS_DIR, FIGURES_DIR, DEVICE
)

# -------------------------------------------------------------
# Feature Correlation Regularizer (Seong & Kim)
# -------------------------------------------------------------
def feature_correlation_reg(v, target_corr=0.5):
    """
    Computes batch-averaged off-diagonal correlation among partition feature vectors
    and penalizes deviation from target_corr.
    v has shape (B, n_partitions, 8)
    """
    B, V, D = v.shape
    v_flat = v.reshape(B * V, D)
    v_centered = v_flat - v_flat.mean(dim=0, keepdim=True)
    cov = (v_centered.T @ v_centered) / (B * V - 1 + 1e-7)
    std = torch.sqrt(torch.diag(cov) + 1e-7)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-7)
    
    mask = ~torch.eye(D, dtype=torch.bool, device=v.device)
    mean_off_diag = corr[mask].mean()
    reg_loss = (mean_off_diag - target_corr) ** 2
    return mean_off_diag.item(), reg_loss

# -------------------------------------------------------------
# Spatial RQA-QCNN Architecture
# -------------------------------------------------------------
class SpatialRQA_Anchored(nn.Module):
    def __init__(self, num_classes=10, use_attn=True, lambda_corr=0.1, target_rho=0.75, tau=0.5):
        super().__init__()
        self.use_attn = use_attn
        self.lambda_corr = lambda_corr
        self.target_rho = target_rho
        self.tau = tau
        
        # 1x1 learned projection from 512 to 8
        self.proj = nn.Sequential(
            nn.Linear(512, 8),
            nn.LayerNorm(8)
        )
        self.qcnn = HybridQCNNModule(
            encoding_fn=ENCODINGS["AQHE-DualEnc"],
            n_qubits=8,
            n_out=4,
            reupload_scale=0.15,
            use_skip=False,
            multi_basis=True
        )
        self.fc_q = nn.Linear(12, num_classes)
        self.fc_c = nn.Linear(8, num_classes)
        
    def forward(self, s, x_glob):
        B = s.shape[0]
        v = self.proj(s) # (B, 16, 8)
        v_angles = torch.sigmoid(v) * np.pi
        
        # Vectorized QCNN simulation on all B*16 patches simultaneously
        hq_flat, parity_flat = self.qcnn.forward_circuit(v_angles.reshape(B * 16, 8))
        hq = hq_flat.reshape(B, 16, 12)
        parity = parity_flat.reshape(B, 16)
        
        if self.use_attn:
            alpha = F.softmax(parity / self.tau, dim=-1)
            w_hq = torch.sum(hq * alpha.unsqueeze(-1), dim=1)
        else:
            alpha = torch.ones(B, 16, device=s.device) / 16.0
            w_hq = torch.mean(hq, dim=1)
            
        logits = self.fc_q(w_hq) + self.fc_c(x_glob)
        return logits, alpha, v

# -------------------------------------------------------------
# Spatial Feature Extraction & Caching
# -------------------------------------------------------------
def get_spatial_features():
    cache_path = CACHE_DIR / "cifar10_spatial_16x512.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['s_tr'], data['s_val'], data['s_te'], data['y_tr'], data['y_val'], data['y_te']
        
    print("[*] Extracting 4x4x512 ResNet-18 spatial feature maps...")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = get_cifar10_5k_split(seed=42)
    weights = ResNet18_Weights.IMAGENET1K_V1
    base = resnet18(weights=weights).eval().to(DEVICE)
    stem = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4
    )
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def extract_sp(X_arr):
        feats = []
        B = 64
        for i in range(0, len(X_arr), B):
            batch = torch.tensor(X_arr[i:i+B], dtype=torch.float32).to(DEVICE)
            b_128 = F.interpolate(batch, size=(128, 128), mode='bilinear', align_corners=False)
            b_norm = norm(b_128)
            with torch.no_grad():
                out = stem(b_norm)
            out_p = out.permute(0, 2, 3, 1).reshape(-1, 16, 512)
            feats.append(out_p.cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)
        
    s_tr = extract_sp(X_tr)
    s_val = extract_sp(X_val)
    s_te = extract_sp(X_te)
    np.savez_compressed(cache_path, s_tr=s_tr, s_val=s_val, s_te=s_te, y_tr=y_tr, y_val=y_val, y_te=y_te)
    return s_tr, s_val, s_te, y_tr, y_val, y_te

# -------------------------------------------------------------
# Training and Evaluation Routine
# -------------------------------------------------------------
def train_and_eval_rqa(model, s_tr, x_tr, y_tr, s_val, x_val, y_val, s_te, x_te, y_te,
                       epochs=15, lr=0.01, lambda_corr=0.1, target_rho=0.5, batch_size=128):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    tr_ds = TensorDataset(torch.tensor(s_tr), torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(s_val), torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(s_te), torch.tensor(x_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
    
    loader_tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    loader_te = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    
    best_val_acc = -1.0
    best_weights = None
    
    for ep in range(epochs):
        model.train()
        for bs, bx, by in loader_tr:
            bs, bx, by = bs.to(DEVICE), bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            logits, _, v = model(bs, bx)
            ce_loss = F.cross_entropy(logits, by, label_smoothing=0.05)
            if lambda_corr > 0:
                _, reg = feature_correlation_reg(v, target_corr=target_rho)
                loss = ce_loss + lambda_corr * reg
            else:
                loss = ce_loss
            loss.backward()
            opt.step()
        sched.step()
        
        # Val
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for bs, bx, by in loader_val:
                bs, bx = bs.to(DEVICE), bx.to(DEVICE)
                logits, _, _ = model(bs, bx)
                v_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                v_trues.extend(by.numpy())
        vacc = accuracy_score(v_trues, v_preds)
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
        
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bs, bx, by in loader_te:
            bs, bx = bs.to(DEVICE), bx.to(DEVICE)
            logits, _, _ = model(bs, bx)
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            trues.extend(by.numpy())
            
    acc = accuracy_score(trues, preds) * 100.0
    f1 = f1_score(trues, preds, average="macro")
    return acc, f1

# -------------------------------------------------------------
# Main Task 2 Execution
# -------------------------------------------------------------
def run_rqa_experiments(fast_mode=False, device_str="cpu"):
    global DEVICE
    DEVICE = torch.device(device_str)
    print("\n" + "="*80)
    print(" TASK 2: RECYCLED-QUBIT ATTENTION & CORRELATION REGULARIZATION (RQA-QCNN)")
    print("="*80)
    
    # 1. Load spatial grid features
    s_tr, s_val, s_te, y_tr, y_val, y_te = get_spatial_features()
    
    # 2. Load global 8-d LDA features for Backbone B
    cache_b = CACHE_DIR / "backbone_b_resnet18_features.npz"
    if not cache_b.exists():
        from exp1_reproducible_pillar1_ablation import get_or_create_cached_embeddings
        get_or_create_cached_embeddings(fast_mode=False)
    data_b = np.load(cache_b)
    x_tr, x_val, x_te, _ = reduce_backbone_b(data_b['f_tr'], data_b['f_val'], data_b['f_te'], data_b['y_tr'])
    
    epochs = 4 if fast_mode else 12
    
    # ---------------------------------------------------------
    # Component Ablation
    # ---------------------------------------------------------
    rqa_variants = [
        ("Vanilla Parallel QCNN", False, 0.0, 0.75),
        ("Parallel QCNN + Corr Reg", False, 0.1, 0.75),
        ("Parallel QCNN + QAttn", True, 0.0, 0.75),
        ("Full RQA-QCNN", True, 0.1, 0.75),
    ]
    
    rqa_ablation_results = {}
    best_full_rqa_model = None
    print("\n[*] Running 4-Way Component Ablation of RQA-QCNN...")
    for var_name, use_attn, lambda_corr, target_rho in rqa_variants:
        print(f"\n---> Training {var_name} (Attn={use_attn}, LambdaCorr={lambda_corr}, TargetRho={target_rho})...")
        model = SpatialRQA_Anchored(num_classes=10, use_attn=use_attn, lambda_corr=lambda_corr, target_rho=target_rho).to(DEVICE)
        acc, f1 = train_and_eval_rqa(
            model, s_tr, x_tr, y_tr, s_val, x_val, y_val, s_te, x_te, y_te,
            epochs=epochs, lr=0.01, lambda_corr=lambda_corr, target_rho=target_rho
        )
        if var_name == "Full RQA-QCNN":
            best_full_rqa_model = model
        rqa_ablation_results[var_name] = {"accuracy": f"{acc:.2f}%", "f1": f"{f1:.4f}", "raw_acc": acc}
        print(f"    [✓] {var_name:25s} -> Test Acc: {acc:.2f}%, Macro-F1: {f1:.4f}")
        
    # ---------------------------------------------------------
    # Correlation Target Sweep
    # ---------------------------------------------------------
    print("\n[*] Sweeping Seong & Kim Target Correlation rho in {0.0, 0.25, 0.50, 0.75}...")
    rho_sweep_results = {}
    rho_list = [0.0, 0.50] if fast_mode else [0.0, 0.25, 0.50, 0.75]
    for rho in rho_list:
        m = SpatialRQA_Anchored(num_classes=10, use_attn=True, lambda_corr=0.1, target_rho=rho).to(DEVICE)
        acc_s, _ = train_and_eval_rqa(
            m, s_tr, x_tr, y_tr, s_val, x_val, y_val, s_te, x_te, y_te,
            epochs=epochs, lr=0.01, lambda_corr=0.1, target_rho=rho
        )
        rho_sweep_results[str(rho)] = acc_s
        print(f"    - Target Rho = {rho:.2f} -> Test Acc: {acc_s:.2f}%")
        
    # ---------------------------------------------------------
    # Attention Map Visualization (Figure 3)
    # ---------------------------------------------------------
    print("\n[*] Generating Attention Heatmap Visualization (Fig 3, 300 DPI)...")
    raw_cache = CACHE_DIR / "cifar10_5k_raw_images.npz"
    if raw_cache.exists():
        d_raw = np.load(raw_cache)
        X_te_raw = d_raw['X_te'][:4]
    else:
        (_, _), (_, _), (X_te_raw_all, _) = get_cifar10_5k_split(seed=42)
        X_te_raw = X_te_raw_all[:4]
        
    model_viz = best_full_rqa_model if best_full_rqa_model is not None else SpatialRQA_Anchored(num_classes=10, use_attn=True, lambda_corr=0.1, target_rho=0.5).to(DEVICE)
    model_viz.eval()
    with torch.no_grad():
        _, alpha_maps, _ = model_viz(
            torch.tensor(s_te[:4]).to(DEVICE),
            torch.tensor(x_te[:4], dtype=torch.float32).to(DEVICE)
        )
    alpha_maps = alpha_maps.cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=300)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(4):
        img_np = X_te_raw[i].transpose(1, 2, 0)
        img_np = np.clip((img_np * 0.25 + 0.45), 0.0, 1.0)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Input: {class_names[y_te[i]]}", fontsize=11, fontweight="bold")
        axes[0, i].axis("off")
        
        attn_grid = alpha_maps[i].reshape(4, 4)
        axes[1, i].imshow(img_np)
        axes[1, i].imshow(attn_grid, cmap="inferno", alpha=0.65, extent=[0, 32, 32, 0], interpolation="bicubic")
        axes[1, i].set_title("Recycled-Qubit QAttn", fontsize=10, fontweight="bold")
        axes[1, i].axis("off")
        
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "fig3_rqa_qcnn_attention_maps.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {fig3_path}")
    
    # Save to metrics_summary.json
    metrics_path = RESULTS_DIR / "metrics_summary.json"
    data_all = {}
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                data_all = json.load(f)
        except Exception:
            pass
            
    data_all["task2_rqa_qcnn"] = {
        "ablation_results": rqa_ablation_results,
        "rho_target_sweep": rho_sweep_results,
        "resnet_grid_accuracy": f"{rqa_ablation_results['Full RQA-QCNN']['raw_acc']:.2f}%"
    }
    with open(metrics_path, "w") as f:
        json.dump(data_all, f, indent=2)
    print(f"[✓] Task 2 metrics saved to: {metrics_path}")

run_experiment_2 = run_rqa_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run fast calibration mode")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, or mps")
    args = parser.parse_args()
    run_rqa_experiments(fast_mode=args.fast, device_str=args.device)
