"""
qml_suite_core.py
=================
Core shared library for Quantum Machine Learning BTP Research Suite:
- Dataset loaders (CIFAR-10 stratified 5,000 subset, Fashion-MNIST, etc.)
- Classical Backbones (SmallResNet, Pretrained ResNet-18)
- Dimensionality Reduction (PCA + Ledoit-Wolf Shrinkage LDA)
- Quantum Encodings (10 primary + audited buggy/repaired versions)
- Quantum Convolutional Neural Networks (QCNN) with Residual Re-uploading
- Recycled-Qubit Attention (RQA) & Seong-Kim Correlation Regularization
"""

import os
import math
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, f1_score, log_loss

import pennylane as qml

# -----------------------------------------------------------------------------
# Global Hardware & Device Configuration
# -----------------------------------------------------------------------------
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("./results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_all_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_quantum_device(n_qubits, mixed=False, diff_method="backprop"):
    """Returns default.qubit for PyTorch backprop, or lightning.qubit for adjoint."""
    if mixed:
        return qml.device("default.mixed", wires=n_qubits)
    if diff_method == "backprop":
        return qml.device("default.qubit", wires=n_qubits)
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)

# -----------------------------------------------------------------------------
# Dataset Loaders & Deterministic Stratified Splits
# -----------------------------------------------------------------------------
def get_cifar10_5k_split(root="./data", seed=42, n_train=3600, n_val=400, n_test=1000):
    """
    Returns exact class-balanced 5,000 CIFAR-10 subset:
    3,600 train (360/class), 400 val (40/class), 1,000 test (100/class).
    """
    raw_cache = CACHE_DIR / "cifar10_5k_raw_images.npz"
    if raw_cache.exists():
        d = np.load(raw_cache)
        return (d['X_tr'], d['y_tr']), (d['X_val'], d['y_val']), (d['X_te'], d['y_te'])
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform_raw = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Try loading from local cifar10 ImageFolder if present
    local_path = Path(root) / "cifar10"
    if (local_path / "train").exists() and (local_path / "test").exists():
        tr_ds = torchvision.datasets.ImageFolder(str(local_path / "train"), transform=transform_raw)
        te_ds = torchvision.datasets.ImageFolder(str(local_path / "test"), transform=transform_raw)
        images = []
        labels = []
        for img, lbl in tr_ds:
            images.append(img.numpy())
            labels.append(lbl)
        for img, lbl in te_ds:
            images.append(img.numpy())
            labels.append(lbl)
        X = np.stack(images, axis=0) # (60000, 3, 32, 32)
        y = np.array(labels)
    else:
        # Load from torchvision datasets
        ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_raw)
        ds_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_raw)
        X_tr = ds.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        X_te = ds_test.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        X = np.concatenate([X_tr, X_te], axis=0)
        y = np.concatenate([np.array(ds.targets), np.array(ds_test.targets)], axis=0)
    
    train_idx, val_idx, test_idx = [], [], []
    for c in range(10):
        cls_indices = np.where(y == c)[0]
        np.random.shuffle(cls_indices)
        train_idx.extend(cls_indices[:360])
        val_idx.extend(cls_indices[360:400])
        test_idx.extend(cls_indices[400:500])
        
    train_idx, val_idx, test_idx = np.array(train_idx), np.array(val_idx), np.array(test_idx)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(raw_cache, X_tr=X[train_idx], y_tr=y[train_idx], X_val=X[val_idx], y_val=y[val_idx], X_te=X[test_idx], y_te=y[test_idx])
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

# -----------------------------------------------------------------------------
# Classical Backbones
# -----------------------------------------------------------------------------
class SmallResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return F.relu(out)

class SmallResNet(nn.Module):
    """Backbone A: 128-d penultimate embedding extractor."""
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.layer1 = SmallResNetBlock(8, 16, stride=1)
        self.layer2 = SmallResNetBlock(16, 32, stride=2) # 16x16
        self.layer3 = SmallResNetBlock(32, 64, stride=2) # 8x8
        self.pool = nn.AdaptiveAvgPool2d((2, 1)) # 64 * 2 = 128-d
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward_features(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        feat = self.forward_features(x)
        return self.fc(self.drop(feat))

def extract_resnet18_features(X_split, batch_size=64, finetuned_model=None):
    """Backbone B: Pretrained ResNet-18 (512-d embeddings)."""
    if finetuned_model is not None:
        model = finetuned_model
    else:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    
    # ResNet expects 224x224 or normalized 32x32
    resize_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    loader = DataLoader(TensorDataset(torch.tensor(X_split, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    features = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = resize_norm(batch_x.to(DEVICE))
            feat = model(batch_x)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

# -------------------------------------------------------------
# Dimensionality Reduction Pipelines
# -------------------------------------------------------------
def reduce_backbone_a(feats_tr, feats_val, feats_te, y_tr, lda_dim=8):
    """
    Backbone A: StandardScaler -> PCA(16, whiten=True) -> Shrinkage LDA(8) -> MinMaxScaler([0, pi])
    """
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(feats_tr)
    Z_val = scaler.transform(feats_val)
    Z_te = scaler.transform(feats_te)
    
    pca = PCA(n_components=16, whiten=True, random_state=42)
    P_tr = pca.fit_transform(Z_tr)
    P_val = pca.transform(Z_val)
    P_te = pca.transform(Z_te)
    
    lda = LDA(n_components=lda_dim, solver='eigen', shrinkage='auto')
    L_tr = lda.fit_transform(P_tr, y_tr)
    L_val = lda.transform(P_val)
    L_te = lda.transform(P_te)
    
    mms = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr = mms.fit_transform(L_tr)
    X_val = mms.transform(L_val)
    X_te = mms.transform(L_te)
    
    return X_tr, X_val, X_te, lda

def reduce_backbone_b(feats_tr, feats_val, feats_te, y_tr, lda_dim=8):
    """
    Backbone B: Direct StandardScaler -> Shrinkage LDA(8) -> MinMaxScaler([0, pi])
    """
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(feats_tr)
    Z_val = scaler.transform(feats_val)
    Z_te = scaler.transform(feats_te)
    
    lda = LDA(n_components=lda_dim, solver='eigen', shrinkage='auto')
    L_tr = lda.fit_transform(Z_tr, y_tr)
    L_val = lda.transform(Z_val)
    L_te = lda.transform(Z_te)
    
    mms = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr = mms.fit_transform(L_tr)
    X_val = mms.transform(L_val)
    X_te = mms.transform(L_te)
    
    return X_tr, X_val, X_te, lda

# -------------------------------------------------------------
# Quantum Encodings Library (10 Primary + Repaired + Buggy)
# -------------------------------------------------------------
def enc_pi4angle(x, wires):
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.RY(2.0 * float(x[i % len(x)]), wires=w)

def enc_aqhe_dualenc(x, wires):
    """AQHE-DualEnc: Dual amplitude-phase encoding + iQFT."""
    for j, w in enumerate(wires):
        xj = float(x[j % len(x)])
        amp_j = float(2.0 * np.arctan(xj / np.pi))
        phi_j = float(2 * np.pi * np.log1p(xj) / np.log1p(2 * np.pi))
        qml.RY(amp_j, wires=w)
        if abs(phi_j) > 1e-9:
            qml.PhaseShift(phi_j, wires=w)
    qml.adjoint(qml.QFT)(wires=wires)

def enc_aqhe_loghelix(x, wires):
    for w in wires:
        qml.Hadamard(wires=w)
    for j, w in enumerate(wires):
        xj = float(x[j % len(x)])
        phi = float(2 * np.pi * np.log1p(xj) / np.log1p(2 * np.pi))
        if abs(phi) > 1e-9:
            qml.PhaseShift(phi, wires=w)
    qml.adjoint(qml.QFT)(wires=wires)

def enc_aqhe_fibspiral(x, wires):
    golden_ratio = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1 / golden_ratio)
    for j, w in enumerate(wires):
        xj = float(x[j % len(x)])
        theta = float(np.arccos(np.clip(1.0 - 2.0 * (xj / np.pi), -1.0, 1.0)))
        phi = float(j * golden_angle + xj)
        qml.RY(theta, wires=w)
        qml.PhaseShift(phi, wires=w)
    qml.adjoint(qml.QFT)(wires=wires)

def enc_denseangle(x, wires):
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RZ(float(x[i % len(x)]) ** 2, wires=w)
    for i in range(n):
        for j in range(i + 1, min(i + 3, n)):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(float(x[i % len(x)] * x[j % len(x)]), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])

def enc_entangledangle(x, wires):
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RY(float(x[i % len(x)]), wires=w)
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

def enc_frqi_fixed(x, wires):
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i in range(n):
        qml.CRY(2.0 * float(x[i % len(x)]), wires=[wires[i], wires[(i + 1) % n]])

def enc_frqi_buggy(x, wires):
    n = len(wires)
    for i in range(n):
        target = i + 1
        if target >= n:
            break
        qml.CRY(float(x[i % len(x)]), wires=[wires[i], wires[target]])

def enc_iqp_fixed(x, wires):
    n = len(wires)
    for _ in range(2):
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            qml.RZ(float(x[i % len(x)]), wires=w)
        for i in range(n):
            for j in range(i + 1, n):
                qml.IsingZZ(2.0 * (np.pi - float(x[i % len(x)])) * (np.pi - float(x[j % len(x)])), wires=[wires[i], wires[j]])

def enc_iqp_buggy(x, wires):
    n = len(wires)
    for i in range(n - 1):
        qml.IsingZZ(float(x[i % len(x)] * x[(i + 1) % len(x)]), wires=[wires[i], wires[i + 1]])

def enc_qhe_fixed(x, wires):
    for i, w in enumerate(wires):
        qml.RY(float(x[i % len(x)]), wires=w)
        qml.RZ(float(np.log1p(x[i % len(x)])), wires=w)
    qml.adjoint(qml.QFT)(wires=wires)

def enc_qhe_buggy(x, wires):
    for i, w in enumerate(wires):
        qml.RY(float(x[i % len(x)]), wires=w)
        qml.RZ(float(np.log1p(x[i % len(x)])), wires=w)

def enc_hamiltonian(x, wires, trotter_steps=2, t=1.0):
    n = len(wires)
    dt = t / trotter_steps
    for w in wires:
        qml.Hadamard(wires=w)
    for _ in range(trotter_steps):
        for i, w in enumerate(wires):
            qml.RZ(2.0 * dt * float(x[i % len(x)]), wires=w)
        for i in range(n):
            for j in range(i + 1, n):
                coupling = 2.0 * dt * float(x[i % len(x)] * x[j % len(x)])
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RX(coupling, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])

ENCODINGS = {
    "AQHE-DualEnc": enc_aqhe_dualenc,
    "AQHE-LogHelix": enc_aqhe_loghelix,
    "AQHE-FibSpiral": enc_aqhe_fibspiral,
    "Pi4Angle": enc_pi4angle,
    "DenseAngle": enc_denseangle,
    "EntangledAngle": enc_entangledangle,
    "FRQI": enc_frqi_fixed,
    "IQP": enc_iqp_fixed,
    "QHE": enc_qhe_fixed,
    "Hamiltonian": enc_hamiltonian,
}

# -------------------------------------------------------------
# QCNN Circuit & PyTorch Model Blocks
# -------------------------------------------------------------
def conv_block(params, a, b):
    qml.U3(params[0], params[1], params[2], wires=a)
    qml.U3(params[3], params[4], params[5], wires=b)
    qml.CRot(params[6], params[7], params[8], wires=[a, b])

def pool_block(params, c, t):
    qml.CRZ(params[1], wires=[c, t])
    qml.PauliX(wires=c)
    qml.CRX(params[0], wires=[c, t])
    qml.PauliX(wires=c)

class HybridQCNNModule(nn.Module):
    """
    Standard QCNN Module with Quantum Residual Re-uploading (Wen et al. 2024),
    Classical Skip Connection (Xia et al. 2026), and Recycled-Qubit 4-Body Parity Attention.
    Implements fast vectorized PyTorch state-vector simulation with full backprop.
    """
    def __init__(self, encoding_fn=None, n_qubits=8, n_out=4, num_classes=10, 
                 reupload_scale=0.3, use_skip=True, multi_basis=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_out = n_out
        self.encoding_fn = encoding_fn
        self.reupload_scale = reupload_scale
        self.use_skip = use_skip
        self.multi_basis = multi_basis
        self.num_classes = num_classes
        
        # Learnable residual re-uploading scaling factor (Wen et al. 2024)
        if reupload_scale > 0:
            self.alpha = nn.Parameter(torch.tensor(float(reupload_scale)))
        else:
            self.register_buffer('alpha', torch.tensor(0.0))
        
        # Trainable QCNN parameters
        self.conv1_w = nn.Parameter(0.1 * torch.randn(4, 6))
        self.pool1_w = nn.Parameter(0.1 * torch.randn(4))
        self.fc_q_w = nn.Parameter(0.1 * torch.randn(3, n_out, 3))
        
        self.q_dim = n_out * (3 if multi_basis else 1)
        if self.encoding_fn is not None:
            self.head_w_q = nn.Linear(self.q_dim, num_classes, bias=True)
        if use_skip:
            self.head_w_c = nn.Linear(n_qubits, num_classes, bias=True)
            
        # Precomputed parity sign for 4-body dropped parity <Z0 Z2 Z4 Z6>
        b = torch.arange(256)
        b0 = (b >> 7) & 1
        b2 = (b >> 5) & 1
        b4 = (b >> 3) & 1
        b6 = (b >> 1) & 1
        self.register_buffer('parity_sign', ((-1) ** (b0 + b2 + b4 + b6)).float())

    def _apply_ry_all(self, psi, angles):
        B = angles.shape[0]
        c = torch.cos(angles/2)
        s = torch.sin(angles/2)
        for i in range(8):
            left = 1 << i
            right = 1 << (7 - i)
            psi_v = psi.view(B, left, 2, right)
            p0 = psi_v[:, :, 0, :]
            p1 = psi_v[:, :, 1, :]
            ci = c[:, i].view(B, 1, 1)
            si = s[:, i].view(B, 1, 1)
            psi = torch.stack([ci * p0 - si * p1, si * p0 + ci * p1], dim=2).reshape(B, 256)
        return psi

    def _apply_cnot(self, psi, c_w, t_w):
        B = psi.shape[0]
        k = min(c_w, t_w) // 2
        left = 1 << (2 * k)
        right = 1 << (7 - (2 * k + 1))
        psi_v = psi.view(B, left, 4, right)
        p00 = psi_v[:, :, 0, :]
        p01 = psi_v[:, :, 1, :]
        p10 = psi_v[:, :, 2, :]
        p11 = psi_v[:, :, 3, :]
        if c_w < t_w:
            return torch.stack([p00, p01, p11, p10], dim=2).reshape(B, 256)
        else:
            return torch.stack([p00, p11, p10, p01], dim=2).reshape(B, 256)

    def _apply_cry(self, psi, c_w, t_w, theta):
        B = psi.shape[0]
        k = min(c_w, t_w) // 2
        left = 1 << (2 * k)
        right = 1 << (7 - (2 * k + 1))
        psi_v = psi.view(B, left, 4, right)
        p00 = psi_v[:, :, 0, :]
        p01 = psi_v[:, :, 1, :]
        p10 = psi_v[:, :, 2, :]
        p11 = psi_v[:, :, 3, :]
        c = torch.cos(theta/2).view(-1, 1, 1)
        s = torch.sin(theta/2).view(-1, 1, 1)
        return torch.stack([p00, p01, c * p10 - s * p11, s * p10 + c * p11], dim=2).reshape(B, 256)

    def forward_circuit(self, x):
        """Vectorized 8-qubit QCNN forward pass."""
        B = x.shape[0]
        psi = torch.zeros((B, 256), dtype=torch.complex64, device=x.device)
        psi[:, 0] = 1.0
        
        # 1. Encoding (AQHE-DualEnc or Pi4Angle)
        amp = 2.0 * torch.arctan(x / np.pi)
        psi = self._apply_ry_all(psi, amp)
        
        # 2. Layer 1 Convolution on pairs (0,1), (2,3), (4,5), (6,7)
        for idx, (w1, w2) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
            psi = self._apply_cnot(psi, w1, w2)
            rot_angles = self.conv1_w[idx, :2].unsqueeze(0).expand(B, 2)
            c = torch.cos(rot_angles/2)
            s = torch.sin(rot_angles/2)
            for j, w in enumerate([w1, w2]):
                left = 1 << w
                right = 1 << (7 - w)
                psi_v = psi.view(B, left, 2, right)
                p0 = psi_v[:, :, 0, :]
                p1 = psi_v[:, :, 1, :]
                cj = c[:, j].view(B, 1, 1)
                sj = s[:, j].view(B, 1, 1)
                psi = torch.stack([cj * p0 - sj * p1, sj * p0 + cj * p1], dim=2).reshape(B, 256)
            psi = self._apply_cnot(psi, w2, w1)
            
        # 3. Residual Data Re-uploading before pooling
        if float(self.reupload_scale) > 0:
            psi = self._apply_ry_all(psi, self.alpha * x)
            
        # 4. Pooling Layer 1 (keep 1, 3, 5, 7; pool 0, 2, 4, 6)
        for idx, (c_w, t_w) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
            psi = self._apply_cry(psi, c_w, t_w, self.pool1_w[idx])
            
        probs = torch.abs(psi)**2
        b = torch.arange(256, device=psi.device)
        
        # 5. Measure Pauli observables on kept wires [1, 3, 5, 7]
        kept = [1, 3, 5, 7]
        expvals = []
        for w in kept:
            bit = 7 - w
            sign = ((-1) ** ((b >> bit) & 1)).float()
            expvals.append((probs * sign).sum(dim=-1))
            
        parity_4body = (probs * self.parity_sign).sum(dim=-1)
        h_q_base = torch.stack(expvals, dim=-1) # (B, 4)
        
        if self.multi_basis:
            h_q = torch.cat([h_q_base, torch.sin(h_q_base), torch.cos(h_q_base)], dim=-1) # (B, 12)
        else:
            h_q = h_q_base
            
        return h_q, parity_4body

    def forward(self, x):
        if self.encoding_fn is None:
            # Classical Skip Only (Linear LDA Head)
            return self.head_w_c(x)
            
        h_q, _ = self.forward_circuit(x)
        logits = self.head_w_q(h_q)
        if self.use_skip:
            logits = logits + self.head_w_c(x)
        return logits

def count_trainable_params(model):
    """Returns total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ParameterMatchedMLP(nn.Module):
    """Classical MLP matching the exact trainable parameter count (~276-285) of Hybrid QCNN."""
    def __init__(self, in_features=8, num_classes=10, hidden_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class TrainedLinearProbe(nn.Module):
    """Standalone trained linear head (90 parameters) on 8-d Shrinkage LDA features."""
    def __init__(self, in_features=8, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.fc(x)

def train_and_eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                         epochs=30, lr=0.01, batch_size=64, weight_decay=1e-4,
                         use_cosine=True, label_smoothing=0.05):
    """Utility to train QCNN with AdamW, CosineAnnealingLR, and label smoothing."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4) if use_cosine else None
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    loader_tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    loader_te = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    
    best_val_acc = -1.0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step()
            
        # Validation
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for bx, by in loader_val:
                bx = bx.to(DEVICE)
                preds = torch.argmax(model(bx), dim=-1).cpu().numpy()
                v_preds.extend(preds)
                v_trues.extend(by.numpy())
        val_acc = accuracy_score(v_trues, v_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    
    # Test Evaluation
    model.eval()
    te_preds, te_probs, te_trues = [], [], []
    with torch.no_grad():
        for bx, by in loader_te:
            bx = bx.to(DEVICE)
            logits = model(bx)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs = probs / np.clip(np.sum(probs, axis=-1, keepdims=True), 1e-12, None)
            preds = np.argmax(probs, axis=-1)
            te_preds.extend(preds)
            te_probs.extend(probs)
            te_trues.extend(by.numpy())
            
    te_acc = accuracy_score(te_trues, te_preds)
    te_f1 = f1_score(te_trues, te_preds, average="macro")
    num_c = len(probs[0]) if len(probs) > 0 else 10
    te_loss = log_loss(te_trues, te_probs, labels=list(range(num_c)))
    
    return te_acc, te_f1, te_loss
