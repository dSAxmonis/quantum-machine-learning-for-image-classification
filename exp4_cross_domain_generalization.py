#!/usr/bin/env python3
"""
TASK 4: CROSS-DOMAIN GENERALIZATION (PILLAR 5)
========================================================================================
Manuscript: "Hierarchical & Attention-Guided Quantum Convolutional Neural Networks
            for Natural Image Classification: Literature Auditing, Multi-Encoding
            Benchmarking, Recycled-Qubit Attention, and Quantum-Classical Ablation"
========================================================================================
Runs the exact Pillar 1 pipeline (Frozen Pretrained ResNet-18 -> Shrinkage LDA(8) ->
8-qubit AQHE-DualEnc QCNN vs. Quantum-Only vs. Classical RBF-SVM & Linear Head) across:
  1. Fashion-MNIST (10 classes: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)
  2. MedMNIST (BloodMNIST: 8 classes of blood cell microscope images)

Demonstrates cross-domain generalization without per-dataset hyperparameter re-tuning.
Saves metrics to ./results/metrics_summary.json.
========================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from qml_suite_core import (
    HybridQCNNModule,
    train_and_eval_model,
    set_all_seeds,
    ENCODINGS,
    DEVICE,
    RESULTS_DIR,
    CACHE_DIR
)


def get_fashion_mnist_splits(root="./data", total_samples=5000, seed=42, fast_mode=False):
    """
    Loads Fashion-MNIST, converts 1-channel 28x28 to 3-channel 32x32 RGB,
    and returns stratified splits.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_set = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    
    train_targets = np.array(train_set.targets)
    test_targets = np.array(test_set.targets)
    
    if fast_mode:
        n_train, n_val, n_test = 300, 100, 200
    else:
        n_train = int(total_samples * 0.72)
        n_val = int(total_samples * 0.08)
        n_test = total_samples - n_train - n_val
        
    sss_tr_val = StratifiedShuffleSplit(n_splits=1, train_size=n_train + n_val, random_state=seed)
    tr_val_idx, _ = next(sss_tr_val.split(np.zeros(len(train_targets)), train_targets))
    
    tr_val_targets = train_targets[tr_val_idx]
    sss_tr = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
    tr_sub, val_sub = next(sss_tr.split(np.zeros(len(tr_val_targets)), tr_val_targets))
    
    final_tr_idx = tr_val_idx[tr_sub]
    final_val_idx = tr_val_idx[val_sub]
    
    sss_te = StratifiedShuffleSplit(n_splits=1, train_size=min(n_test, len(test_targets)), random_state=seed)
    final_te_idx, _ = next(sss_te.split(np.zeros(len(test_targets)), test_targets))
    
    ds_train = Subset(train_set, final_tr_idx)
    ds_val = Subset(train_set, final_val_idx)
    ds_test = Subset(test_set, final_te_idx)
    
    print(f"[Fashion-MNIST] Subsets created: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}, Classes=10")
    return ds_train, ds_val, ds_test, 10


def get_medmnist_splits(root="./data", total_samples=5000, seed=42, dataset_name="bloodmnist", fast_mode=False):
    """
    Loads BloodMNIST (8 classes) or PathMNIST (9 classes), scales to 32x32 RGB,
    and returns stratified train, val, test subsets.
    """
    import medmnist
    from medmnist import INFO
    
    dataset_name = dataset_name.lower()
    DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tr_data = DataClass(split='train', download=True, root=root, transform=transform, as_rgb=True)
    val_data = DataClass(split='val', download=True, root=root, transform=transform, as_rgb=True)
    te_data = DataClass(split='test', download=True, root=root, transform=transform, as_rgb=True)
    
    n_classes = len(INFO[dataset_name]['label'])
    
    if fast_mode:
        n_train, n_val, n_test = 300, 100, 200
    else:
        n_train = int(total_samples * 0.72)
        n_val = int(total_samples * 0.08)
        n_test = total_samples - n_train - n_val
        
    def sample_subset(ds, target_n, s):
        labels = np.array([ds[i][1].item() if hasattr(ds[i][1], 'item') else int(ds[i][1]) for i in range(len(ds))])
        if len(ds) <= target_n:
            return ds
        sss = StratifiedShuffleSplit(n_splits=1, train_size=target_n, random_state=s)
        sub_idx, _ = next(sss.split(np.zeros(len(labels)), labels))
        return Subset(ds, sub_idx)
        
    ds_train = sample_subset(tr_data, n_train, seed)
    ds_val = sample_subset(val_data, n_val, seed)
    ds_test = sample_subset(te_data, n_test, seed)
    
    print(f"[{dataset_name.upper()}] Subsets created: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}, Classes={n_classes}")
    return ds_train, ds_val, ds_test, n_classes


def extract_and_reduce_dataset_features(ds_train, ds_val, ds_test, n_classes, dataset_name, device, cache_dir="./cache"):
    """
    Extracts 512-d ResNet-18 features, caches them, and applies Shrinkage LDA to 8 dimensions.
    If n_classes <= 8 (e.g. BloodMNIST has 8 classes -> max 7 LDA components), LDA(7) is augmented with 1 PCA component.
    """
    os.makedirs(cache_dir, exist_ok=True)
    total_len = len(ds_train) + len(ds_val) + len(ds_test)
    cache_path = os.path.join(cache_dir, f"features_resnet18_{dataset_name}_{total_len}.npz")
    
    if os.path.exists(cache_path):
        print(f"[{dataset_name}] Loading cached ResNet-18 features from {cache_path}...")
        dat = np.load(cache_path)
        X_tr_raw, y_tr = dat['X_tr'], dat['y_tr']
        X_val_raw, y_val = dat['X_val'], dat['y_val']
        X_te_raw, y_te = dat['X_te'], dat['y_te']
    else:
        print(f"[{dataset_name}] Extracting features using Pretrained ResNet-18 on {device}...")
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone.eval()
        backbone = backbone.to(device)
        
        def extract(ds):
            loader = DataLoader(ds, batch_size=64, shuffle=False)
            feats, targets = [], []
            with torch.no_grad():
                for imgs, tgts in loader:
                    imgs = imgs.to(device)
                    out = backbone(imgs)
                    feats.append(out.cpu().numpy())
                    if hasattr(tgts, 'numpy'):
                        targets.append(tgts.numpy().reshape(-1))
                    elif hasattr(tgts, 'item'):
                        targets.append(np.array([tgts.item()]))
                    else:
                        targets.append(np.array(tgts).reshape(-1))
            return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)
            
        X_tr_raw, y_tr = extract(ds_train)
        X_val_raw, y_val = extract(ds_val)
        X_te_raw, y_te = extract(ds_test)
        
        np.savez(cache_path, X_tr=X_tr_raw, y_tr=y_tr, X_val=X_val_raw, y_val=y_val, X_te=X_te_raw, y_te=y_te)
        print(f"[{dataset_name}] Saved raw features to {cache_path}.")
        
    # Standardize
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_te_scaled = scaler.transform(X_te_raw)
    
    target_dim = 8
    max_lda_dim = n_classes - 1
    
    if max_lda_dim >= target_dim:
        lda = LinearDiscriminantAnalysis(n_components=target_dim, solver='eigen', shrinkage='auto')
        X_tr_lda = lda.fit_transform(X_tr_scaled, y_tr)
        X_val_lda = lda.transform(X_val_scaled)
        X_te_lda = lda.transform(X_te_scaled)
    else:
        # e.g., BloodMNIST: 8 classes -> max 7 LDA components
        lda = LinearDiscriminantAnalysis(n_components=max_lda_dim, solver='eigen', shrinkage='auto')
        X_tr_lda_sub = lda.fit_transform(X_tr_scaled, y_tr)
        X_val_lda_sub = lda.transform(X_val_scaled)
        X_te_lda_sub = lda.transform(X_te_scaled)
        
        pca = PCA(n_components=target_dim - max_lda_dim)
        X_tr_pca_sub = pca.fit_transform(X_tr_scaled)
        X_val_pca_sub = pca.transform(X_val_scaled)
        X_te_pca_sub = pca.transform(X_te_scaled)
        
        X_tr_lda = np.concatenate([X_tr_lda_sub, X_tr_pca_sub], axis=1)
        X_val_lda = np.concatenate([X_val_lda_sub, X_val_pca_sub], axis=1)
        X_te_lda = np.concatenate([X_te_lda_sub, X_te_pca_sub], axis=1)
        
    # Scale to [0, pi] for quantum angle encoding
    minmax = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_q = minmax.fit_transform(X_tr_lda)
    X_val_q = minmax.transform(X_val_lda)
    X_te_q = minmax.transform(X_te_lda)
    
    return (X_tr_q, y_tr), (X_val_q, y_val), (X_te_q, y_te)


def evaluate_domain_pipeline(dataset_name, tr_data, val_data, te_data, n_classes, device, epochs=15, fast_mode=False):
    """
    Evaluates:
      1. Classical RBF-SVM (Upper-bound Non-linear Classical Classifier)
      2. Classical Linear Head (Direct LDA Linear Classifier)
      3. Pure Quantum QCNN (AQHE-DualEnc + Re-uploading 0.3, Skip=OFF)
      4. Full Hybrid QCNN (AQHE-DualEnc + Re-uploading 0.3 + Skip=ON)
    """
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = tr_data, val_data, te_data
    
    # 1. Classical RBF-SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_tr, y_tr)
    svm_preds = svm.predict(X_te)
    svm_acc = accuracy_score(y_te, svm_preds) * 100.0
    svm_f1 = f1_score(y_te, svm_preds, average='macro')
    print(f"[{dataset_name}] Classical RBF-SVM:      Acc = {svm_acc:.2f}%, Macro-F1 = {svm_f1:.4f}")
    
    domain_results = {
        "dataset": dataset_name,
        "n_classes": n_classes,
        "rbf_svm": {"test_acc": round(svm_acc, 2), "macro_f1": round(svm_f1, 4)}
    }
    
    # 2. Classical Skip Only (Linear Head)
    lin_clf = nn.Linear(8, n_classes).to(device)
    opt_c = torch.optim.Adam(lin_clf.parameters(), lr=5e-3)
    tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    for _ in range(15 if fast_mode else 30):
        opt_c.zero_grad()
        F.cross_entropy(lin_clf(tr_t), y_t).backward()
        opt_c.step()
    lin_preds = torch.argmax(lin_clf(torch.tensor(X_te, dtype=torch.float32).to(device)), dim=-1).cpu().numpy()
    lin_acc = accuracy_score(y_te, lin_preds) * 100.0
    lin_f1 = f1_score(y_te, lin_preds, average="macro")
    print(f"[{dataset_name}] Classical Linear Head:  Acc = {lin_acc:.2f}%, Macro-F1 = {lin_f1:.4f}")
    domain_results["classical_linear_head"] = {"test_acc": round(lin_acc, 2), "macro_f1": round(lin_f1, 4)}
    
    # 3. Quantum Only & 4. Full Hybrid
    configs = [
        ("Quantum Only", False, 0.3),
        ("Full Hybrid", True, 0.3)
    ]
    
    n_ep = 5 if fast_mode else epochs
    for name, use_skip, alpha in configs:
        model = HybridQCNNModule(
            encoding_fn=ENCODINGS["AQHE-DualEnc"],
            n_qubits=8,
            n_out=4,
            num_classes=n_classes,
            reupload_scale=alpha,
            use_skip=use_skip
        ).to(device)
        
        acc, f1, loss = train_and_eval_model(
            model=model,
            X_train=X_tr, y_train=y_tr,
            X_val=X_val, y_val=y_val,
            X_test=X_te, y_test=y_te,
            epochs=n_ep,
            batch_size=64,
            lr=0.01
        )
        
        acc_pct = acc * 100.0
        key = name.lower().replace(" ", "_")
        domain_results[key] = {
            "test_acc": round(acc_pct, 2),
            "macro_f1": round(f1, 4),
            "test_loss": round(loss, 4)
        }
        if name == "Full Hybrid":
            model_hybrid = model

    # 5. Feature-Noise Robustness Evaluation (sigma = 0.15)
    np.random.seed(42)
    X_te_noisy = np.clip(X_te + np.random.normal(0, 0.15, size=X_te.shape), 0.0, np.pi)
    
    # RBF-SVM Noisy
    svm_noisy_preds = svm.predict(X_te_noisy)
    svm_noisy_acc = accuracy_score(y_te, svm_noisy_preds) * 100.0
    
    # Classical MLP (8->32->C) Clean & Noisy
    mlp_clf = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, n_classes)).to(device)
    opt_m = torch.optim.AdamW(mlp_clf.parameters(), lr=0.01)
    mlp_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)
    for _ in range(n_ep):
        mlp_clf.train()
        for bx, by in mlp_loader:
            opt_m.zero_grad()
            F.cross_entropy(mlp_clf(bx.to(device)), by.to(device)).backward()
            opt_m.step()
    mlp_clf.eval()
    with torch.no_grad():
        mlp_clean_preds = torch.argmax(mlp_clf(torch.tensor(X_te, dtype=torch.float32).to(device)), dim=-1).cpu().numpy()
        mlp_noisy_preds = torch.argmax(mlp_clf(torch.tensor(X_te_noisy, dtype=torch.float32).to(device)), dim=-1).cpu().numpy()
    mlp_clean_acc = accuracy_score(y_te, mlp_clean_preds) * 100.0
    mlp_noisy_acc = accuracy_score(y_te, mlp_noisy_preds) * 100.0
    
    # Full Hybrid Noisy
    model_hybrid.eval()
    with torch.no_grad():
        hybrid_noisy_preds = torch.argmax(model_hybrid(torch.tensor(X_te_noisy, dtype=torch.float32).to(device)), dim=-1).cpu().numpy()
    hybrid_noisy_acc = accuracy_score(y_te, hybrid_noisy_preds) * 100.0
    
    domain_results["noise_robustness_sigma_0_15"] = {
        "hybrid_clean_acc": round(domain_results["full_hybrid"]["test_acc"], 2),
        "hybrid_noisy_acc": round(hybrid_noisy_acc, 2),
        "mlp_clean_acc": round(mlp_clean_acc, 2),
        "mlp_noisy_acc": round(mlp_noisy_acc, 2),
        "svm_clean_acc": round(svm_acc, 2),
        "svm_noisy_acc": round(svm_noisy_acc, 2)
    }
    print(f"[{dataset_name}] Noisy (sigma=0.15) Acc: Full Hybrid = {hybrid_noisy_acc:.2f}%, Classical MLP = {mlp_noisy_acc:.2f}%, Classical RBF-SVM = {svm_noisy_acc:.2f}%")
        print(f"[{dataset_name}] {name:22s} -> Acc: {acc_pct:.2f}%, F1: {f1:.4f}, Loss: {loss:.4f}")
        
    return domain_results


def run_experiment_4(fast=False, device_str="cpu"):
    """Main execution function for Task 4 Cross-Domain Generalization."""
    print("=" * 80)
    print("STARTING TASK 4: CROSS-DOMAIN GENERALIZATION (PILLAR 5)")
    print("=" * 80)
    
    set_all_seeds(42)
    device = torch.device(device_str)
    
    total_samples = 1000 if fast else 5000
    epochs = 5 if fast else 30
    
    # 1. Fashion-MNIST
    print("\n--- Benchmark 1: Fashion-MNIST ---")
    f_tr_ds, f_val_ds, f_te_ds, f_classes = get_fashion_mnist_splits(
        root="./data", total_samples=total_samples, seed=42, fast_mode=fast
    )
    f_tr, f_val, f_te = extract_and_reduce_dataset_features(
        f_tr_ds, f_val_ds, f_te_ds, f_classes, "fashion_mnist", device
    )
    res_fashion = evaluate_domain_pipeline(
        "Fashion-MNIST", f_tr, f_val, f_te, f_classes, device, epochs=epochs, fast_mode=fast
    )
    
    # 2. MedMNIST (BloodMNIST)
    print("\n--- Benchmark 2: MedMNIST (BloodMNIST Biomedical Image Generalization) ---")
    m_tr_ds, m_val_ds, m_te_ds, m_classes = get_medmnist_splits(
        root="./data", total_samples=total_samples, seed=42, dataset_name="bloodmnist", fast_mode=fast
    )
    m_tr, m_val, m_te = extract_and_reduce_dataset_features(
        m_tr_ds, m_val_ds, m_te_ds, m_classes, "bloodmnist", device
    )
    res_med = evaluate_domain_pipeline(
        "BloodMNIST", m_tr, m_val, m_te, m_classes, device, epochs=epochs, fast_mode=fast
    )
    
    cross_domain_summary = {
        "fashion_mnist": res_fashion,
        "medmnist": res_med
    }
    
    # Save to metrics_summary.json
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    all_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}
            
    all_metrics["exp4_cross_domain"] = cross_domain_summary
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"\n[DONE] Cross-Domain results saved to {metrics_path}")
    print("=" * 80)
    return cross_domain_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run fast test mode with 300/100/200 samples and 5 epochs")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device: cpu or cuda")
    args = parser.parse_args()
    
    run_experiment_4(fast=args.fast, device_str=args.device)
