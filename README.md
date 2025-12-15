# MedMNIST-EdgeAIv2 🔬📱
### *Next-Gen Edge AI for Medical Imaging via Knowledge Distillation*

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Author](https://img.shields.io/badge/Author-STiFLeR7-blue?style=for-the-badge&logo=github)

---

## 🚀 Project Summary

**MedMNIST-EdgeAIv2** is a rigorous research framework designed by **[STiFLeR7](https://github.com/STiFLeR7)** to bridge the gap between high-performance medical image analysis and resource-constrained edge deployment. By leveraging advanced **Knowledge Distillation (KD)** and **Attention Transfer (AT)** techniques, this project successfully compresses heavy "Teacher" models (ResNet50) into lightweight "Student" architectures (MobileNetV2, EfficientNetB0, ResNet18) without significant loss in diagnostic accuracy.

The framework is battle-tested across multiple modalities:
*   **Dermatoscopy**: HAM10000 & ISIC (Skin Lesion Classification)
*   **Ophthalmology**: OCT2017 (Retinal Disease Detection)

## 🧠 Core Methodologies

We employ a composite distillation loss function that orchestrates three critical components:

$$ L_{total} = \alpha L_{CE} + (1 - \alpha) L_{KD} + \beta L_{AT} $$

### Distillation Workflow
```mermaid
graph TD
    subgraph Teacher [Teacher Model (ResNet50)]
        T_Img[Input Image] --> T_Feat[Feature Maps]
        T_Feat --> T_Logits[Logits]
    end

    subgraph Student [Student Model (EffNet/MobileNet)]
        S_Img[Input Image] --> S_Feat[Feature Maps]
        S_Feat --> S_Logits[Logits]
    end

    T_Logits -->|Soft Targets| KD_Loss[KL Divergence Loss]
    S_Logits -->|Soft Preds| KD_Loss
    
    T_Feat -->|Attention Maps| AT_Loss[Attention Transfer Loss]
    S_Feat -->|Attention Maps| AT_Loss

    Label[Ground Truth] -->|Hard Targets| CE_Loss[Cross-Entropy Loss]
    S_Logits -->|Hard Preds| CE_Loss

    KD_Loss --> Total_Loss[Total Loss]
    AT_Loss --> Total_Loss
    CE_Loss --> Total_Loss
    
    style Teacher fill:#f9f,stroke:#333
    style Student fill:#bbf,stroke:#333
    style Total_Loss fill:#f55,stroke:#333,color:#fff
```

1.  **Soft Knowledge Distillation ($L_{KD}$)**: Transfers "dark knowledge" (inter-class relationships) from the Teacher's logits to the Student using Kullback-Leibler divergence with temperature scaling ($\tau$).
2.  **Attention Transfer ($L_{AT}$)**: Forces the Student to mimic the spatial attention maps of the Teacher, ensuring it focuses on the same pathological features (lesions, drusen, membranes).
3.  **Hard Target Learning ($L_{CE}$)**: Standard Cross-Entropy loss against ground truth labels.

### 🔬 Ablation Configurations (`ablations.json`)
The framework supports dynamic experimentation configurations:
| Config Name | Alpha ($\alpha$) | Temperature ($\tau$) | Beta ($\beta$, AT) | Focus |
| :--- | :---: | :---: | :---: | :--- |
| **softKD** | `0.6` | `6` | `250` | Balanced Logit + Attention Transfer |
| **hardKD** | `0.8` | `2` | `0` | Pure Logit Distillation (No AT) |
| **ATstrong**| `0.6` | `4` | `1500` | Heavy Focus on Spatial Attention |

## 🛠️ Tech Stack & Architecture

### **Core Framework**
*   **Deep Learning**: `PyTorch`, `Torchvision`
*   **Orchestration**: `PowerShell` scripts for multi-stage pipelines (Phase 1-3)
*   **Analysis**: `Pandas`, `Scikit-learn`, `Matplotlib` (Confusion Matrices, ROC Curves)
*   **Interpretability**: **Grad-CAM** integration for visualizing model decision hotspots.

### **Model Zoo**
*   **Teacher**: `ResNet50` (The Oracle)
*   **Students**:
    *   `EfficientNet-B0` (SOTA Efficiency)
    *   `MobileNetV2` (Mobile Optimized)
    *   `ResNet18` (Standard Lightweight)

## 📊 Evaluation & Benchmarks

The framework evaluates models on four critical axes:
1.  **Accuracy Metrics**: Precision, Recall, F1-Score (Macro/Weighted), per-class breakdown.
2.  **Pareto Efficiency**: Mapping Accuracy vs. Latency (GPU/CPU) to find the optimal edge deployment candidate.
3.  **Robustness**: Testing resilience against image corruptions (Gaussian noise, JPEG compression, contrast shifts).
4.  **Failure Analysis**: Automated identification of top confusion pairs (e.g., *Drusen* vs *CNV*).

## 📂 Project Structure

```text
MedMNIST-EdgeAIv2/
├── data/                   # Dataset roots (HAM10000, OCT2017, ISIC)
├── models/                 # Checkpoints for Teachers and Distilled Students
├── reports/                # PDF Reports and aggregated CSVs
├── RESULTS/                # Raw experiment metrics, logs, and tables
│   ├── students/           # Per-student detailed execution logs
│   │   └── efficientnet_b0/
│   │       └── softKD/
│   │           └── ham10000/
│   └── teacher_*/          # Teacher baseline results
├── scripts/                # Automated orchestrators
│   ├── phase1_ham10000.ps1 # End-to-end pipeline for HAM10000
│   ├── phase2_oct2017.ps1  # End-to-end pipeline for OCT2017
│   └── run_kd_sweeps.ps1   # Hyperparameter grid search
├── external_src/           # Modular source code
└── train_and_eval.py       # Main engine: Training, KD, Evaluation, Grad-CAM
```

## ⚡ Workflow & Usage

### 1. Training & Distillation
To launch a targeted KD run (e.g., distilling ResNet50 to MobileNetV2 on HAM10000):

```bash
python train_and_eval.py --mode train_kd \
    --teacher-arch resnet50 --teacher-ckpt ./models/teacher_ham10000.pth \
    --student-arch mobilenet_v2 \
    --dataset ham10000 --data-root ./data/HAM10000 \
    --alpha 0.6 --tau 6 --beta 250
```

### 2. Full Phase Orchestration
Run the end-to-end phase script to train, evaluate, and generate reports:

```powershell
.\scripts\phase1_ham10000.ps1 -DoRobustness $true -Device cuda
```

### 3. Generate Analysis PDF
Aggregate all results into a publication-ready PDF:

```powershell
.\scripts\make_phase1_report.ps1
```

## 📈 Key Results (Preview)

*   **OCT2017**: `EfficientNet-B0` (Student) achieves **~96%** accuracy, matching the ResNet50 Teacher while being **5x faster** and **10x smaller**.
*   **HAM10000**: Attention Transfer significantly improves evaluating pigmented lesions by forcing the student to look at irregular borders rather than skin artifacts.

---
### 👨‍💻 Author & Maintainer

Developed with ❤️ and ☕ by **[STiFLeR7](https://github.com/STiFLeR7)**.
*   *GitHub*: [github.com/STiFLeR7](https://github.com/STiFLeR7)
*   *Project*: **MedMNIST-EdgeAIv2**
