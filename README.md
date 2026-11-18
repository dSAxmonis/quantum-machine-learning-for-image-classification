# RQA-QCNN: Hierarchical and Attention-Guided Quantum Convolutional Neural Networks

Official research repository and reproducible benchmark suite for:
> **"Hierarchical and Attention-Guided Quantum Convolutional Neural Networks for Natural Image Classification: Literature Auditing, Multi-Encoding Benchmarking, Recycled-Qubit Attention, and Quantum-Classical Ablation"**

Authored by **Monis Husain**, Soumya, Ahan, and Ankita Bansal.  
*Department of Information Technology, Netaji Subhas University of Technology (NSUT), New Delhi, India.*

---

## 📌 Executive Summary & Key Results

- **Dimension Distillation:** A two-stage pipeline combining deep feature extraction (ResNet-18) with regularized Shrinkage-LDA (Ledoit-Wolf) compresses 512-dimensional semantic representations into an 8-dimensional Fisher-optimal subspace $\mathbf{x} \in [0, \pi]^8$.
- **4 Literature Bug Repairs:** Identified and corrected rank deficiencies, phase conjugation dropouts, and coordinate overflows across IQP, QHE, FRQI, and Graph Laplacian encodings, restoring full 256-basis-state superpositions.
- **Recycled-Qubit Attention (RQA-QCNN):** Non-local 4-qubit parity observables $\langle \bigotimes_{i \in \mathcal{Q}_{\text{dropped}}} Z_i \rangle$ extracted from mid-circuit pooling qubits drive spatial attention gating ($\tau=0.5$), boosting accuracy from $80.40\%$ to **$81.60\%$**.
- **Verified Quantum Attribution:** A controlled 5-way ablation across three random seeds confirms that the isolated quantum circuit achieves **$71.00\%$** test accuracy, while the Full Hybrid with learnable residual data re-uploading reaches **$82.40\% \pm 0.35\%$** (285 parameters), strictly outperforming parameter-matched classical MLPs (**$81.07\%$**, 276 parameters, $p < 0.01$).
- **Theoretical Diagnostics:** Pareto-optimal sweet spot achieved by Adaptive Quantum Haar Encoding (AQHE-DualEnc: Centered $\text{KTA} = 0.6440$, depth-6 variance $\text{Var}[\partial \mathcal{L}] = 0.0759$), avoiding the barren plateaus and kernel collapse of over-entangled IQP ($Q=0.981$, $\text{KTA}=0.2211$, $\text{Var}=0.0030$).
- **Intrinsic Noise Resilience:** Bounded unitary gate Lipschitz properties ($L \le 1$) limit accuracy degradation under Gaussian perturbations ($\sigma=0.15$) to only **$-3.0\%$** on CIFAR-10, outperforming classical MLPs ($-5.7\%$) and SVMs ($-3.9\%$).

---

## 📊 Benchmark Summary

| Model / Architecture | Qubits | Trainable Params | CIFAR-10 Clean | CIFAR-10 Noisy ($\sigma=0.15$) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Hybrid RQA-QCNN (Ours)** | **8** | **285** | **82.40%** | **79.40% (-3.0%)** |
| Parameter-Matched Classical MLP | -- | 276 | 81.07% | 75.80% (-5.3%) |
| Standard Classical MLP | -- | 618 | 82.80% | 77.10% (-5.7%) |
| Pure Quantum QCNN (No Skip) | 8 | 195 | 71.00% | 68.20% (-2.8%) |
| Untrained Skip Projection (Ablation) | -- | 90 | 26.43% | 21.10% (-5.3%) |

---

## 📁 Repository Structure

```text
├── README.md                              # Project documentation & benchmark overview
├── requirements.txt                       # Core dependencies
├── run_all_experiments.py                 # Master reproduction pipeline
├── literature_bug_fixes_audit.py          # Benchmark and repair audit of the 4 literature bugs
├── qml_suite_core.py                      # Core QML library: AQHE, RQA-QCNN, Shrinkage-LDA, ZNE
├── exp1_reproducible_pillar1_ablation.py  # 5-Way Quantum-Classical ablation experiment
├── exp2_recycled_qubit_attention_qcnn.py  # Recycled-Qubit parity attention gating experiment
├── exp3_quantum_diagnostics_and_noise.py  # Meyer-Wallach, KTA, expressibility, & ZNE noise
├── exp4_cross_domain_generalization.py    # Cross-domain transfer (CIFAR-10, Fashion, BloodMNIST)
├── main.tex                               # Two-column camera-ready journal manuscript
├── research_paper.tex                     # Publication LaTeX source
├── BTP_Journal_Paper_Overleaf.zip         # Self-contained Overleaf submission package
├── figures/                               # Publication-grade figures (PNG, 300+ DPI)
└── results/                               # Empirical metrics (JSON) and compiled LaTeX tables
```

---

## 🚀 Reproduction & Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/dSAxmonis/quantum-machine-learning-for-image-classification.git
cd quantum-machine-learning-for-image-classification
pip install -r requirements.txt
```

### 2. Audit Literature Bugs
```bash
python literature_bug_fixes_audit.py
```

### 3. Run End-to-End Experimental Suite
```bash
python run_all_experiments.py
```

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{husain2026hierarchical,
  title={Hierarchical and Attention-Guided Quantum Convolutional Neural Networks for Natural Image Classification: Literature Auditing, Multi-Encoding Benchmarking, Recycled-Qubit Attention, and Quantum-Classical Ablation},
  author={Husain, Monis and Soumya and Ahan and Bansal, Ankita},
  journal={arXiv preprint},
  year={2026}
}
```
